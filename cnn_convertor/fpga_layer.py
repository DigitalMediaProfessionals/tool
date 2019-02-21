# -*- coding: utf-8 -*-
"""
    Copyright 2018 Digital Media Professionals Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import os
import math
import logging
import numpy as np
import itertools
from cnn_convertor import cnn_layer, cnn_exception, cnn_docgen
from cnn_convertor.cnn_layer import NodeType, get_conv_out_width_floor
from cnn_convertor import fpga_limitation
from enum import IntEnum, auto


MAX_RUN = 32
MAX_FC_VECTOR_SIZE = 16384
MAX_KERNEL_SIZE = 7
MAX_UNIFIED_BUFFER_SIZE = 640 * 1024  # 640KB
MEM_ALIGN = 128 // 8  # 128bit


def check_memalign():
    n = MEM_ALIGN
    assert n > 0
    while (n & 1) == 0:
        n >>= 1
    assert n == 1


check_memalign()


def set_max_fc_vector_size(n):
    global MAX_FC_VECTOR_SIZE
    MAX_FC_VECTOR_SIZE = n


def set_max_kernel_size(n):
    global MAX_KERNEL_SIZE
    MAX_KERNEL_SIZE = n


def set_ub_size(n):
    global MAX_UNIFIED_BUFFER_SIZE
    MAX_UNIFIED_BUFFER_SIZE = n


def get_actfunc(tpe):
    # 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU,
    # 6 = ReLU6
    return {
        NodeType.TanH: 1,
        NodeType.Sigmoid: 3,
        NodeType.PReLU: 4,
        NodeType.ELU: 5,
        NodeType.ReLU6: 6}.get(tpe, 2)


def make_align_size(size):
    r = size & (MEM_ALIGN - 1)
    if r != 0:
        size = (size & ~(MEM_ALIGN - 1)) + MEM_ALIGN
    return size


def divup(a, b):
    n, d = divmod(a, b)
    if d:
        n += 1
    return n


def calc_conv_tiles(node):
    if node.param.group > 1:
        return 1
    w = node.input_dim[0]
    h = node.input_dim[1]
    c = node.input_dim[2]
    m = node.output_dim[2]
    p = node.param.kernel_size[0]
    pad_lrtb = node.param.pad_lrtb
    stride = node.param.stride
    dilation = node.param.dilation
    c_blocks = (c >> 3) + (1 if c & 7 else 0)
    t = 0
    while True:
        t += 1
        tw = divup(w, t) + p - 1  # width of tile
        ow = get_conv_out_width_floor(tw, p, pad_lrtb[0], pad_lrtb[1],
                                      stride[0], dilation[0])
        oh = get_conv_out_width_floor(h, p, pad_lrtb[2], pad_lrtb[3],
                                      stride[1], dilation[1])
        os = ow * oh * min(8, m)  # output buffer size
        ts_1c = tw * h  # tile size for single channel
        ts_blk16 = ts_1c * min(8, c)
        ts_blk128 = (ts_blk16 >> 3) + (1 if ts_blk16 & 7 else 0)
        # Ensure size modulo 16 = 2, this to ensure 8 blocks
        # can be read in parallel from 16 cuts in 1x1 mode
        ts_blk128 += (2 - ts_blk128) & 15
        ts_128 = ts_blk128 * c_blocks
        # Ensure size modulo 16 = 0, this to ensure 8 blocks
        # can be read in parallel from 16 cuts in 1x1 mode
        ts_128 += (0 - ts_128) & 15
        ts = ts_128 << 3  # input tile size in UBUF (in float16)
        uu = ts + os  # unified buffer utilization
        if uu * 2 <= MAX_UNIFIED_BUFFER_SIZE:
            return t


def calc_pool_tiles(node):
    if node.type is NodeType.UpSampling:
        return 1
    w = node.input_dim[0]
    h = node.input_dim[1]
    c = node.input_dim[2]
    t = 0
    while True:
        t += 1
        uu, d = divmod(w * h * min(8, c), t)
        if d:
            uu += 1
        if uu * 2 <= MAX_UNIFIED_BUFFER_SIZE:
            return t


def merge_bn_scale(node, kernel_size, n_c, n_m):
    weight = node.weight
    bias = node.bias
    if node.bn_node is not None and node.bn_node.mean is not None:
        bn_mean = node.bn_node.mean
    else:
        bn_mean = np.zeros_like(bias)
    if node.bn_node is not None and node.bn_node.var is not None:
        bn_var = node.bn_node.var
    else:
        bn_var = np.full_like(bias, 1.0 - 0.00001)
    if node.sc_node is not None and node.sc_node.weight is not None:
        sc_weight = node.sc_node.weight
    else:
        sc_weight = np.ones_like(bias)
    if node.sc_node is not None and node.sc_node.bias is not None:
        sc_bias = node.sc_node.bias
    else:
        sc_bias = np.zeros_like(bias)
    e = 0.00001
    weight.shape = (n_m, n_c * kernel_size[1] * kernel_size[0])
    for i in range(n_m):
        assert np.min(bn_var[i]) >= 0, "Invalid bn_var[%d]=%s" % (i, bn_var[i])
        norm = sc_weight[i] / math.sqrt(bn_var[i] + e)
        weight[i, :] = weight[i, :] * norm
        bias[i] = (bias[i] - bn_mean[i]) * norm + sc_bias[i]
    weight.shape = (-1,)
    return weight, bias


def calc_kmeans(weight):
    weight.shape = (-1,)
    clusters = min(255, len(weight))
    # scikit version
#    from sklearn.cluster import KMeans
#    kmeans = KMeans(n_clusters=clusters, n_init=3,
#                    max_iter=10, tol=1e-6)
#    labels = kmeans.fit_predict(weight.reshape(-1, 1))
#    centers = kmeans.cluster_centers_

    # opencv version
    import cv2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 1e-6)
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(weight,
                                              clusters, None,
                                              criteria, 3, flags)
    labels.shape = (len(labels),)

    # sort the labels by frequency
#    freq = np.zeros(shape=(clusters,), dtype=np.uint32)
#    for i in labels:
#        freq[i] += 1
#    sort_freq = [(i, freq[i], centers[i])
#                 for i in range(clusters)]
#    sort_freq.sort(key=lambda tup: tup[1], reverse=True)
#    remap_table = [0 for i in range(clusters)]
#    centers = np.zeros(shape=(256,), dtype=np.float16)
#    for i in range(clusters):
#        remap_table[sort_freq[i][0]] = i + 1
#        centers[i + 1] = sort_freq[i][2]
#    for i, index in enumerate(labels):
#        labels[i] = remap_table[index]

    # add 0 as the first element
    centers = np.insert(centers.astype(np.float16), 0, 0.0)
    for i, index in enumerate(labels):
        labels[i] = index + 1

    return centers, labels


def get_kernel_size_for_weight(node):
    k = node.param.kernel_size
    p = max(k[0], k[1]) | 1
    return (p, p)


def pack_conv_weight(node, of, quantization):
    logging.info('Packing weight for node: %s.', node.name)
    if node.param.dilation[0] == 1 and node.param.dilation[1] == 1:
        _pack_conv_weight_nondil(node, of, quantization)
    else:
        _pack_conv_weight_dil(node, of, quantization)


def _pack_conv_weight_dil(node, of, quantization):
    # parameters
    n_c = node.input_dim[2]
    if node.param.group > 1:
        n_c = n_c // node.param.group
    n_m = node.output_dim[2]
    kernel_size = node.param.kernel_size[:]

    weight = node.weight
    bias = node.bias
    if weight is None or bias is None:
        if weight is None:
            weight = np.ones(
                (n_m, n_c, kernel_size[1], kernel_size[0]), np.float32)
        if bias is None:
            bias = np.zeros((n_m,), np.float32)
        node.set_weight_bias(weight, bias)
    if node.bn_node is not None or node.sc_node is not None:
        weight, bias = merge_bn_scale(node, kernel_size, n_c, n_m)
    if node.act_node and node.act_node.type == NodeType.PReLU:
        prelu = node.act_node.weight
    else:
        prelu = None

    if quantization:
        centers, labels = calc_kmeans(weight)
        weight_type = np.uint8
        dsize = 1
    else:
        labels = weight.astype(np.float16)
        weight_type = np.float16
        dsize = 2
    bias16 = bias.astype(np.float16)
    if prelu is not None:
        prelu16 = prelu.astype(np.float16)
    else:
        prelu16 = None
    labels.shape = (n_m, n_c, kernel_size[1], kernel_size[0])

    offs = 0
    if quantization:
        centers.tofile(of)
        of.write(b'\0\0' * (256 - centers.size))
        offs += 256 * 2
    # main: write to file
    for y in range(kernel_size[1]):
        for x in range(kernel_size[0]):
            for m_start in range(0, n_m, 8):
                m_stop = min(m_start + 8, n_m)
                use_zero = ((x != kernel_size[0] - 1) or
                            (y != kernel_size[1] - 1))

                def _write_bias(b):
                    if use_zero:
                        of.write(b'\0\0' * 8)
                    else:
                        b[m_start:m_stop].tofile(of)
                        pad = m_start + 8 - m_stop
                        of.write(b'\0\0' * pad)  # 32-byte align padding

                # Bias and PReLU
                _write_bias(bias16)
                offs += 8 * 2
                if prelu16 is not None:
                    _write_bias(prelu16)
                    offs += 8 * 2

                # Conv
                for c_start in range(0, n_c, 64):
                    c_stop = min(c_start + 64, n_c)
                    buffer = np.zeros(shape=(12, 6), dtype=weight_type)
                    for m in range(m_start, m_stop):
                        for c in range(c_start, c_stop):
                            t = c & 7
                            _x = ((c & 63) >> 3) % 3
                            _y = ((c & 63) >> 3) // 3
                            buffer[11 - (t >> 1) * 3 - _y, (t & 1) * 3 + _x]\
                                                        = labels[m, c, y, x]
                        buffer.tofile(of)
                        offs += buffer.size * dsize

            # 16-byte align padding
            if offs & 15:
                pad = 16 - offs & 15
                of.write(b'\0' * pad)
                offs += pad

    # final 16-byte align padding
    if offs & 15:
        pad = 16 - offs & 15
        of.write(b'\0' * pad)
        offs += pad


def _pack_conv_weight_nondil(node, of, quantization):
    n_c = node.input_dim[2]
    if node.param.group > 1:
        n_c = n_c // node.param.group
    n_m = node.output_dim[2]
    kernel_size = get_kernel_size_for_weight(node)

    weight = node.weight
    bias = node.bias
    if weight is None or bias is None:
        if weight is None:
            weight = np.ones(
                (n_m, n_c, kernel_size[1], kernel_size[0]), np.float32)
        if bias is None:
            bias = np.zeros((n_m,), np.float32)
        node.set_weight_bias(weight, bias)

    if node.bn_node is not None or node.sc_node is not None:
        weight, bias = merge_bn_scale(node, kernel_size, n_c, n_m)
    if node.act_node and node.act_node.type == NodeType.PReLU:
        prelu = node.act_node.weight
    else:
        prelu = None

    if quantization:
        centers, labels = calc_kmeans(weight)
        weight_type = np.uint8
    else:
        labels = weight.astype(np.float16)
        weight_type = np.float16
    bias16 = bias.astype(np.float16)
    if prelu is not None:
        prelu16 = prelu.astype(np.float16)
    else:
        prelu16 = None
    buffer = np.zeros(shape=(12, 6), dtype=weight_type)

    labels.shape = (n_m, n_c, kernel_size[1], kernel_size[0])
    if quantization:
        centers.tofile(of)
        of.write(b'\0\0' * (256 - centers.size))
    if kernel_size[0] == 7:
        for m_start in range(0, n_m, 8):
            m_stop = min(m_start + 8, n_m)

            bias16[m_start:m_stop].tofile(of)
            for i in range(m_stop, m_start + 8):
                of.write(b'\0\0')  # padding
            if prelu16 is not None:
                prelu16[m_start:m_stop].tofile(of)
                for i in range(m_stop, m_start + 8):
                    of.write(b'\0\0')  # padding

            for c_start in range(0, n_c, 8):
                c_stop = min(c_start + 8, n_c)
                for m in range(m_start, m_stop):
                    for c in range(c_start, c_stop):
                        for y in range(kernel_size[1]):
                            for x in range(6):
                                buffer[5 + y, x] = labels[m, c, y, x]
                        buffer[2, 5] = labels[m, c, 0, 6]
                        for y in range(3):
                            buffer[y, 3] = labels[m, c, y + 1, 6]
                            buffer[y, 0] = labels[m, c, y + 4, 6]
                        buffer.tofile(of)

    elif kernel_size[0] == 5:
        for m_start in range(0, n_m, 8):
            m_stop = min(m_start + 8, n_m)
            bias16[m_start:m_stop].tofile(of)
            for i in range(m_stop, m_start + 8):
                of.write(b'\0\0')
            if prelu16 is not None:
                prelu16[m_start:m_stop].tofile(of)
                for i in range(m_stop, m_start + 8):
                    of.write(b'\0\0')
            for c_start in range(0, n_c, 8):
                c_stop = min(c_start + 8, n_c)
                for m in range(m_start, m_stop):
                    for c in range(c_start, c_stop):
                        t = c % 2
                        if t == 0 and c == c_stop - 1:
                            buffer = np.zeros(shape=(12, 6), dtype=weight_type)
                        for y in range(kernel_size[1]):
                            for x in range(kernel_size[0]):
                                buffer[7 - t * 6 + y, x] = labels[m, c, y, x]
                        if t == 1 or t == 0 and c == c_stop - 1:
                            buffer.tofile(of)
    elif kernel_size[0] == 3:
        for m_start in range(0, n_m, 8):
            m_stop = min(m_start + 8, n_m)
            bias16[m_start:m_stop].tofile(of)
            for i in range(m_stop, m_start + 8):
                of.write(b'\0\0')
            if prelu16 is not None:
                prelu16[m_start:m_stop].tofile(of)
                for i in range(m_stop, m_start + 8):
                    of.write(b'\0\0')
            for c_start in range(0, n_c, 8):
                c_stop = min(c_start + 8, n_c)
                for m in range(m_start, m_stop):
                    if c_stop - c_start >= 1 and c_stop - c_start <= 7:
                        buffer = np.zeros(shape=(12, 6), dtype=weight_type)
                    for c in range(c_start, c_stop):
                        t = c % 8
                        for y in range(kernel_size[1]):
                            for x in range(kernel_size[0]):
                                buffer[9 - t // 2 * 3 + y, t % 2 * 3 + x] = (
                                    labels[m, c, y, x])
                    buffer.tofile(of)
    elif kernel_size[0] == 1:
        for m_start in range(0, n_m, 8):
            m_stop = min(m_start + 8, n_m)
            bias16[m_start:m_stop].tofile(of)
            for i in range(m_stop, m_start + 8):
                of.write(b'\0\0')
            if prelu16 is not None:
                prelu16[m_start:m_stop].tofile(of)
                for i in range(m_stop, m_start + 8):
                    of.write(b'\0\0')
            for c_start in range(0, n_c, 64):
                c_stop = min(c_start + 64, n_c)
                for m in range(m_start, m_stop):
                    if c_stop - c_start >= 1 and c_stop - c_start <= 63:
                        buffer = np.zeros(shape=(12, 6), dtype=weight_type)
                    for c in range(c_start, c_stop):
                        t = c % 8
                        x = c % 64 // 8 % 3
                        y = c % 64 // 8 // 3
                        buffer[11 - t // 2 * 3 - y, t % 2 * 3 + x] = (
                            labels[m, c, 0, 0])
                    buffer.tofile(of)
    else:
        logging.exception('Encountered unsupported kernel size %d.',
                          kernel_size[0])
        raise cnn_exception.ConvertError('Unsupported kernel size ' +
                                         str(kernel_size[0]))

    # add 0 padding so weight size will be 16-bytes aligned
    d = of.tell() & 15
    if d:
        logging.info("Added %d zeros to align weight size", 16 - d)
        np.zeros(16 - d, dtype=np.uint8).tofile(of)


def pack_fc_weight(node, conv_node, of, quantization):
    logging.info('Packing FC weight for node: %s.', node.name)

    if quantization:
        centers, labels = calc_kmeans(node.weight)
        labels = labels.astype(np.uint8)
    else:
        labels = node.weight.astype(np.float16)
    bias16 = node.bias.astype(np.float16)

    offs = 0
    if quantization:
        centers.tofile(of)
        offs += centers.nbytes
    if conv_node is not None:
        if len(conv_node.output_dim) == 3:
            w, h, c = conv_node.output_dim
        elif len(conv_node.output_dim) == 1:
            w, h, c = 1, 1, conv_node.output_dim[0]
        m = node.param.num_output
        if w != 1 or h != 1:
            logging.info('Reordering FC weight for node: %s.', node.name)
            labels.shape = (m, c, h, w)
            for n in range(m):
                for d in range(0, c, 8):
                    e = d + 8 if d + 8 < c else c
                    tr_index8 = labels[n, d:e, :, :].transpose(2, 1, 0)
                    labels[n, d:e, :, :] = tr_index8.reshape(e - d, h, w)
    labels.tofile(of)
    offs += labels.nbytes

    d = offs & 15  # bias must be 16-bytes aligned
    if d:
        logging.info("Added %d zeros to align bias", 16 - d)
        np.zeros(16 - d, dtype=np.uint8).tofile(of)
        offs += 16 - d

    bias16.tofile(of)
    offs += bias16.nbytes

    d = offs & 15  # add 0 padding so weight size will be 16-bytes aligned
    if d:
        logging.info("Added %d zeros to align bias size", 16 - d)
        np.zeros(16 - d, dtype=np.uint8).tofile(of)
        offs += 16 - d


def _get_weight_size(inc, outc, kernel, quantization, use_prelu):
    """
    get weight size of non-dilation convolution
    @param inc # of input channels
    @param outc # of output channels
    @param kernel Krenel size: max(kx, ky) | 1
    @param quantization Flag if quantization is enabled or not
    @param use_prelu Flag if PReLU follows after this CONV
    """
    if kernel == 7:
        pass
    if kernel == 5:
        inc = inc // 2 + inc % 2
    if kernel == 3:
        inc = inc // 8 + (0 if inc % 8 == 0 else 1)
    if kernel == 1:
        inc = inc // 64 + (0 if inc % 64 == 0 else 1)

    if quantization:
        wsize = 512 + 72 * outc * inc + 16 * ((outc + 7) // 8)
        wsize = (wsize + 0xf) & (~0xf)  # align to 16 bytes
    else:
        wsize = 144 * outc * inc + 16 * ((outc + 7) // 8)

    # add PReLU parameter size
    if use_prelu:
        wsize += 16 * ((outc + 7) // 8)
    return wsize


def _get_weight_size_dil(inc, outc, kx, ky, quantization, use_prelu=False):
    """
    get weight size of dilation convolution
    @param inc # of input channels
    @param outc # of output channels
    @param kx Kernel size in x axis
    @param ky Kernel size in y axis
    @param quantization Flag if quantization is enabled or not
    @param use_prelu Flag if PReLU follows after this CONV
    """
    wsize = 512 if quantization else 0
    _w = _get_weight_size(inc, outc, 1, quantization, use_prelu)
    for _ in range(kx * ky):
        wsize += _w
        if quantization:
            wsize -= 512
        d = wsize & 15
        if d:
            wsize += 16 - d

    return wsize


def get_weight_size(node, quantization):
    if node is None or node.type is not NodeType.Convolution:
        return 0

    inc = node.input_dim[2]
    if node.param.group > 1:
        inc //= node.param.group
    outc = node.output_dim[2]
    use_prelu = node.act_node and node.act_node.type == NodeType.PReLU

    if node.param.dilation[0] == 1 and node.param.dilation[1] == 1:
        kernel = get_kernel_size_for_weight(node)
        return _get_weight_size(inc, outc, kernel[0],
                                quantization, use_prelu)
    else:
        return _get_weight_size_dil(inc, outc, node.param.kernel_size[0],
                                    node.param.kernel_size[1], quantization,
                                    use_prelu)


def get_fc_weight_size(node, quantization):
    if len(node.input_dim) == 3:
        w, h, c = node.input_dim
    elif len(node.input_dim) == 1:
        w, h, c = 1, 1, node.input_dim[0]
    m = node.output_dim[0]
    if quantization:
        size = w * h * c * m + 512
    else:
        size = w * h * c * m * 2
    size = (size + 0xf) & (~0xf)  # align to 16 bytes
    size += m * 2
    size = (size + 0xf) & (~0xf)  # align to 16 bytes
    return size


def gen_header_header(of, name, custom_layer_config):
    of.write('/*\n'
             ' *  Copyright 2018 Digital Media Professionals Inc.\n\n'
             ' *  Licensed under the Apache License, Version 2.0 (the "License");\n'
             ' *  you may not use this file except in compliance with the License.\n'
             ' *  You may obtain a copy of the License at\n\n'
             ' *      http://www.apache.org/licenses/LICENSE-2.0\n\n'
             ' *  Unless required by applicable law or agreed to in writing, software\n'
             ' *  distributed under the License is distributed on an "AS IS" BASIS,\n'
             ' *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n'
             ' *  See the License for the specific language governing permissions and\n'
             ' *  limitations under the License.\n\n'
             ' *  This source code was generated using DMP-DV700 tools.\n'
             ' */\n'
             '#pragma once\n\n'
             '#include "dmp_network.h"\n\n')
    for custom_type, custom_config in custom_layer_config.items():
        of.write('struct custom_param_{0}'.format(custom_type))
        of.write(' {\n')
        for param_name, param_type in custom_config[0].items():
            if '[' in param_type:
                sub_index = param_type.find('[')
                of.write('  {0:5} {1}{2};\n'.format(
                    param_type[:sub_index], param_name, param_type[sub_index:]))
            else:
                of.write('  {0:5} {1};\n'.format(param_type, param_name))
        of.write('};\n\n')
    for custom_type in custom_layer_config:
        of.write(
            'void custom_callback_{0}(fpga_layer &layer, void *custom_param, uint8_t *io_ptr);\n\n'.format(custom_type))
    of.write('class C{0} : public CDMP_Network '.format(name))
    of.write('{\n'
             ' private:\n')


def gen_header_layer(of, n, layer, quantization):
    of.write('  /*!\n\n'
             '  Layer description\n\n'
             '  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |\n'
             '  | :- | :- | :-: | :-: | :-: | :-: | :-: |\n')
    of.write('  | {0} | FPGA-Layer | {1} | {2} | {3} | - | - |\n'.format(
             n, str(layer.type),
             str(layer.node_in.input_dim),
             str(layer.node_out.output_dim)))
    for i, run in enumerate(layer.run):
        if run.conv is not None:
            of.write('  | {0}-{1} | {2} | {3} | {4} | {5} | - | {6} |\n'.format(
                     n, i,
                     run.conv.name,
                     str(run.conv.type),
                     str(run.conv.input_dim),
                     str(run.conv.output_dim),
                     get_weight_size(run.conv, quantization)))
        if run.pool is not None:
            of.write('  | {0}-{1} | {2} | {3} | {4} | {5} | - | - |\n'.format(
                     n, i,
                     run.pool.name,
                     str(run.pool.type),
                     str(run.pool.input_dim),
                     str(run.pool.output_dim)))
    of.write('\n  */\n')
    of.write('  void Layer_{0}();\n'.format(n))


def gen_header_footer(of, name):
    of.write('\n'
             ' public:\n'
             '  virtual bool Initialize();\n')
    of.write('  C{0}();\n'.format(name))
    of.write('  virtual ~C{0}();\n'.format(name))
    of.write('};\n')


def gen_source_header(of, name, net):
    of.write('/*\n'
             ' *  Copyright 2018 Digital Media Professionals Inc.\n\n'
             ' *  Licensed under the Apache License, Version 2.0 (the "License");\n'
             ' *  you may not use this file except in compliance with the License.\n'
             ' *  You may obtain a copy of the License at\n\n'
             ' *      http://www.apache.org/licenses/LICENSE-2.0\n\n'
             ' *  Unless required by applicable law or agreed to in writing, software\n'
             ' *  distributed under the License is distributed on an "AS IS" BASIS,\n'
             ' *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n'
             ' *  See the License for the specific language governing permissions and\n'
             ' *  limitations under the License.\n\n'
             ' *  This source code was generated using DMP-DV700 tools.\n'
             ' */\n\n')
    of.write('#include "{0}_gen.h"\n'.format(name))
    of.write('\n\n')
    of.write('C{0}::C{0}() {{\n'.format(name))
    of.write('  // Empty by design\n}\n\n')
    of.write('C{0}::~C{0}() {{\n'.format(name))
    of.write('  // Empty by design\n}\n\n')
    of.write('bool C{0}::Initialize() '.format(name))
    of.write('{\n')
    of.write('  if (!ReserveMemory({0}, {1})) {{\n'.format(
        net.weight_size, net.buffer_size))
    of.write('    return false;\n')
    of.write('  }\n\n')
    of.write('  set_num_layers({0});\n'.format(len(net.layer)))
    of.write('  set_num_output_layers({0});\n\n'.format(net.num_output_layers))
    for n in range(len(net.layer)):
        of.write('  Layer_{0}();\n'.format(n))
    of.write('\n  return true;\n}\n\n')


def gen_source_conv(of, name, n, layer, quantization):
    global weight_offset

    of.write('//Layer_{0}: Convolution Layer\n'.format(n))
    layer_names = []
    for run in layer.run:
        if run.conv is not None:
            of.write('//  ->: {0}\n'.format(run.conv.name))
            layer_names.append(run.conv.name)
            if run.conv.bn_node:
                of.write('//  ->: {0}\n'.format(run.conv.bn_node.name))
                layer_names.append(run.conv.bn_node.name)
            if run.conv.sc_node:
                of.write('//  ->: {0}\n'.format(run.conv.sc_node.name))
                layer_names.append(run.conv.sc_node.name)
            if run.conv.act_node:
                of.write('//  ->: {0}\n'.format(run.conv.act_node.name))
                layer_names.append(run.conv.act_node.name)
        if run.pool is not None:
            of.write('//  ->: {0}\n'.format(run.pool.name))
            layer_names.append(run.pool.name)
    of.write('void C{0}::Layer_{1}() '.format(name, n))
    of.write('{\n')
    of.write('  dmp_dv_cmdraw_conv_v0& conf = get_layer({0}).conv_conf;\n'.format(n))
    of.write('  conf.header.size = sizeof(conf);\n')
    of.write('  conf.header.device_type = DMP_DV_DEV_CONV;\n')
    of.write('  conf.header.version = 0;\n')
    of.write('  // Topo: {0:032b}\n'.format(layer.topo))
    of.write('  conf.topo = 0x{0:X};  // [31:0] Output Destination of each run, 0 = UBUF, 1 = EXTMEM\n\n'.format(layer.topo))
    of.write('  // Input Configuration:\n')
    of.write('  conf.w = {0};  // Input Width\n'.format(layer.node_in.input_dim[0]))
    of.write('  conf.h = {0};  // Input Height\n'.format(layer.node_in.input_dim[1]))
    of.write('  conf.z = 1;  // Input Depth\n')
    of.write('  conf.c = {0};  // Input Channels\n'.format(layer.node_in.input_dim[2]))
    of.write('  conf.input_buf.mem = io_mem_;\n'
             '  conf.input_buf.offs = {0};\n\n'.format(layer.layer_in[0].output_addr_offset))
    of.write('  // Output Configuration:\n')
    of.write('  conf.output_buf.mem = io_mem_;\n'
             '  conf.output_buf.offs = {0};\n\n'.format(layer.output_addr_offset))
    if layer.run[0].pool and layer.run[0].pool.type is NodeType.Eltwise:
        of.write('  conf.eltwise_buf.mem = io_mem_;\n'
                 '  conf.eltwise_buf.offs = {0};  // Input byte address for elementwise add (0 = UBUF Input Buffer)\n'
                 '  conf.output_mode = 1;  // 0 = concat, 1 = eltwise add\n\n'.format(layer.layer_in[1].output_addr_offset))
    else:
        of.write('  conf.eltwise_buf.mem = NULL;\n'
                 '  conf.eltwise_buf.offs = 0;  // Input byte address for elementwise add (0 = UBUF Input Buffer)\n'
                 '  conf.output_mode = 0;  // 0 = concat, 1 = eltwise add\n\n')
    of.write('  // Runs Configuration:\n')
    of.write('  // ->{0} run(s)\n'.format(len(layer.run)))
    for i, run in enumerate(layer.run):
        is_conv = run.conv is not None and run.conv.type is NodeType.Convolution
        is_lrn = run.conv is not None and run.conv.type is NodeType.LRN
        if is_conv:
            p = run.conv.param.kernel_size[0]
            if run.conv.param.kernel_size[1] != run.conv.param.kernel_size[0]:
                p |= (run.conv.param.kernel_size[1] << 8)
            conv_enable = (1 if run.conv.param.group <= 1 else 3)
        else:
            p = 1
            conv_enable = 0
        conv_pad = run.conv.param.pad_fpga if is_conv else 0
        conv_stride = (run.conv.param.stride[0] |
                       (run.conv.param.stride[1] << 8) if is_conv else 0x0101)
        if is_conv and ((run.conv.param.dilation[0] & 0xfe) or
                        (run.conv.param.dilation[1] & 0xfe)):
            conv_dilation = ((run.conv.param.dilation[0] & 0xff |
                             ((run.conv.param.dilation[1] & 0xff) << 8))
                             if is_conv else 0)
        else:
            # For non dilation CONV, use 0 to enble HW's non-dilation conv mode
            conv_dilation = 0

        pool_enable = (1 if run.pool else 0)
        pool_size = (run.pool.param.kernel_size[0] | (
            run.pool.param.kernel_size[1] << 8) if run.pool else 0)
        pool_stride = (run.pool.param.stride[0] | (
            run.pool.param.stride[1] << 8) if run.pool else 0x101)
        pool_pad = run.pool.param.pad_fpga if run.pool else 0
        actfunc = 0
        actparam = 0
        pool_avg_param = 0
        node_in = run.pool
        if run.conv is not None:
            node_in = run.conv
            if node_in.act_node:
                actfunc = get_actfunc(node_in.act_node.type)
                actparam = np.float16(node_in.act_node.param.relu_param).view(np.uint16)
        node_out = run.conv
        if run.pool is not None:
            node_out = run.pool
            if ((node_out.type is NodeType.Pooling and
                 node_out.param.pool != 0) or
                    node_out.type is NodeType.Eltwise):
                pool_enable = 2
                if run.pool.param.split_pool_divisor is None:
                    pool_avg_param = np.float16(
                        1.0 / (run.pool.param.kernel_size[0] *
                               run.pool.param.kernel_size[1])).view(np.uint16)
                else:
                    pool_avg_param = np.float16(
                        1.0 / run.pool.param.split_pool_divisor).view(np.uint16)
            if node_out.type is NodeType.UpSampling:
                pool_enable = 4
            if node_out.type is NodeType.Power:
                pool_enable = 2
                pool_avg_param = np.float16(
                    node_out.param.scale).view(np.uint16)
        m = node_out.output_dim[2]
        # Detect if this is the case of pool node being merged into concat node
        if node_in != node_out and node_in not in node_out.input_nodes:
            m = node_in.output_dim[2]
        of.write('  //--------------------------------------------------\n')
        of.write('  //RUN : {0}\n'.format(i))
        of.write('  //--------------------------------------------------\n')
        if run.conv is not None:
            of.write('  //->: {0}\n'.format(run.conv.name))
            if run.conv.bn_node:
                of.write('  //->: {0}\n'.format(run.conv.bn_node.name))
            if run.conv.sc_node:
                of.write('  //->: {0}\n'.format(run.conv.sc_node.name))
            if run.conv.act_node:
                of.write('  //->: {0}\n'.format(run.conv.act_node.name))
        if run.pool is not None:
            of.write('  //->: {0}\n'.format(run.pool.name))
        of.write('  conf.run[{0}].m = {1};  // Output Channels\n'.format(i, m))
        of.write('  conf.run[{0}].conv_enable = {1};  // 1 = Enabled, 0 = Disabled\n'.format(i, conv_enable))
        of.write('  conf.run[{0}].p = 0x{1:X};  // Filter Width and Height\n'.format(i, p))
        of.write('  conf.run[{0}].pz = 1;  // Filter Depth\n'.format(i))
        of.write('  conf.run[{0}].weight_buf.mem = weights_mem_;\n'
                 '  conf.run[{0}].weight_buf.offs = {1};\n'.format(i, weight_offset))
        of.write('  conf.run[{0}].weight_fmt = {1};  // Weight format (0 = random access blocks, 1 = compact stream, 3 = 8-bit qunatized stream)\n'.format(i, ((3 if quantization else 1) if is_conv else 0)))
        of.write('  conf.run[{0}].conv_pad = 0x{1:X};  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding\n'.format(i, conv_pad))
        of.write('  conf.run[{0}].conv_stride = 0x{1:X};  // bits [7:0] = X stride, bits [15:8] = Y stride\n'.format(i, conv_stride))
        of.write('  conf.run[{0}].conv_dilation = 0x{1:X};  // bits [7:0] = X dilation, bits [15:8] = Y dilation\n'.format(i, conv_dilation))
        of.write('  conf.run[{0}].pool_enable = {1};  // 0 = disabled, 1 = max pooling, 2 = average pooling\n'.format(i, pool_enable))
        of.write('  conf.run[{0}].pool_size = 0x{1:X};  // bits [7:0] = width, bits [15:8] = height\n'.format(i, pool_size))
        of.write('  conf.run[{0}].pool_stride = 0x{1:X};  // bits [7:0] = X stride, bits [15:8] = Y stride\n'.format(i, pool_stride))
        of.write('  conf.run[{0}].pool_pad = 0x{1:X};  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding\n'.format(i, pool_pad))
        of.write('  conf.run[{0}].pool_avg_param = 0x{1:X};  // Usually set to 1/pool_size^2 in FP16 format when using average pooling (average pooling assumes square size)\n'.format(i, pool_avg_param))
        of.write('  conf.run[{0}].actfunc = {1};  // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6\n'.format(i, actfunc))
        of.write('  conf.run[{0}].actfunc_param = 0x{1:X};  // Leaky ReLU parameter (NOTE: 0x2E66 is 0.1 in FP16)\n'.format(i, actparam))
        of.write('  conf.run[{0}].rectifi_en = 0;  // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)\n'.format(i))
        of.write('  conf.run[{0}].lrn = 0x{1:X};  // [0] : 1 = LRN enable, 0 = LRN disable, [1] : 1 = incl. power func, 0 = excl., [8:11] = x^2 scale factor log2\n'.format(i, (0x503 if is_lrn else 0)))
        weight_offset += get_weight_size(run.conv, quantization)


def gen_source_fc(of, name, n, layer, quantization):
    global weight_offset
    node = layer.node_in
    if len(node.input_dim) == 3:
        w, h, c = node.input_dim
    elif len(node.input_dim) == 1:
        w, h, c = 1, 1, node.input_dim[0]
    m = node.output_dim[0]
    size = get_fc_weight_size(node, quantization)
    actfunc = 0
    actparam = 0
    if node.act_node:
        if node.act_node.type == NodeType.ReLU:
            actfunc = 1
            if node.act_node.param.relu_param != 0.0:
                actfunc = 3
                actparam = np.float16(node.act_node.param.relu_param).view(np.uint16)
        elif node.act_node.type == NodeType.TanH:
            actfunc = 2
        elif node.act_node.type == NodeType.Sigmoid:
            actfunc = 4
        else:
            raise ValueError("Unsupported activation: %s" % node.act_node.type)
    of.write('// Layer_{0}: Fully Connected Layer\n'.format(n))
    of.write('//	->: {0}\n'.format(node.name))
    of.write('void C{0}::Layer_{1}() '.format(name, n))
    of.write('{\n')
    of.write('  dmp_dv_cmdraw_fc_v0& conf = get_layer({0}).fc_conf;\n'.format(n))
    of.write('  conf.header.size = sizeof(conf);\n')
    of.write('  conf.header.version = 0;\n')
    of.write('  conf.header.device_type = DMP_DV_DEV_FC;\n')
    of.write('  conf.input_size = {0};\n'.format(w * h * c))
    of.write('  conf.output_size = {0};\n'.format(m))
    of.write('  conf.weight_buf.mem = weights_mem_;\n'
             '  conf.weight_buf.offs = {0};\n'.format(weight_offset))
    of.write('  conf.input_buf.mem = io_mem_;\n'
             '  conf.input_buf.offs = {0};\n'.format(layer.layer_in[0].output_addr_offset))
    of.write('  conf.output_buf.mem = io_mem_;\n'
             '  conf.output_buf.offs = {0};\n'.format(layer.output_addr_offset))
    of.write('  conf.weight_fmt = {0};  // 0 = unquantized weight matrix, 1 = qunatized\n'.format((1 if quantization else 0)))
    of.write('  conf.actfunc = {0};  // Activation Function: 0 = None, 1 = ReLU, 2 = Tanh, 3 = Leaky ReLU, 4 = Sigmoid, 5 = PReLU (PReLU must be used with POST-OP=1)\n'.format(actfunc))
    of.write('  conf.actfunc_param = 0x{0:X};  // Leaky ReLU parameter (in FP16 format), 0 = non-leaky\n'.format(actparam))
    weight_offset += size


def _is_input_hw_layout(layer):
    for inl in layer.layer_in:
        if inl.type is LayerType.Convolution:
            return True
        elif inl.type is LayerType.Concatenate:
            if _is_input_hw_layout(inl):
                return True
            # TODO: raise exception if multipe layouts exists in input

    return False


def gen_source_layer(of, name, n, layer, quantization):
    global output_index
    type_map = {LayerType.Input: 'LT_INPUT',
                LayerType.Convolution: 'LT_CONV',
                LayerType.InnerProduct: 'LT_FC',
                LayerType.Flatten: 'LT_FLATTEN',
                LayerType.Concatenate: 'LT_CONCAT',
                LayerType.CopyConcatenate: 'LT_COPY_CONCAT',
                LayerType.SoftMax: 'LT_SOFTMAX',
                LayerType.Custom: 'LT_CUSTOM'}

    if layer.type is LayerType.Convolution:
        gen_source_conv(of, name, n, layer, quantization)
        of.write('\n')
    elif layer.type is LayerType.InnerProduct:
        gen_source_fc(of, name, n, layer, quantization)
        of.write('\n')
    else:
        if layer.type is LayerType.Input:
            of.write('//Layer_{0}: Input Layer\n'.format(n))
        elif layer.type is LayerType.Concatenate:
            of.write('//Layer_{0}: Concatenate Layer\n'.format(n))
        elif layer.type is LayerType.CopyConcatenate:
            of.write('//Layer_{0}: CopyConcatenate Layer\n'.format(n))
        elif layer.type is LayerType.Flatten:
            of.write('//Layer_{0}: Flatten Layer\n'.format(n))
        elif layer.type is LayerType.SoftMax:
            of.write('//Layer_{0}: SoftMax Layer\n'.format(n))
        else:
            of.write('//Layer_{0}: Custom Layer\n'.format(n))
        node = layer.node_in
        of.write('//	->: {0}\n'.format(node.name))
        of.write('void C{0}::Layer_{1}() '.format(name, n))
        of.write('{\n')

    if layer.type is LayerType.Custom:
        custom_param = layer.node_in.param.custom_param
        of.write('  static custom_param_{0} custom_param = {{\n'.format(
            custom_param[2]))
        for param in custom_param[0].values():
            if type(param) is list:
                of.write('    { ')
                for value in param:
                    of.write('{0}, '.format(value))
                of.write(' },\n')
            elif type(param) is bool:
                of.write('    {0},\n'.format('true' if param else 'false'))
            else:
                of.write('    {0},\n'.format(param))
        of.write('  };\n\n')
    elif layer.type is LayerType.CopyConcatenate:
        of.write('  static fpga_layer *input_layers[] = {\n')
        for layer_in in layer.layer_in:
            of.write('    &layers_[{0}],\n'.format(layer_in.index))
        of.write('  };\n\n')

    of.write('  fpga_layer& layer = get_layer({0});\n'.format(n))
    of.write('  layer.name = "{0}";\n'.format(layer.node_out.name))
    of.write('  layer.type = {0};\n'.format(type_map[layer.type]))
    of.write('  layer.input_offs = {0};\n'.format(
        layer.layer_in[0].output_addr_offset))
    of.write('  layer.output_offs = {0};\n'.format(
        layer.output_addr_offset))
    of.write('  layer.output_size = {0};\n'.format(
        layer.node_out.output_size))
    dim = layer.node_in.input_dim
    for i in range(len(dim)):
        of.write('  layer.input_dim[{0}] = {1};\n'.format(i, dim[i]))
    of.write('  layer.input_dim_size = {0};\n'.format(len(dim)))
    dim = layer.node_out.output_dim
    for i in range(len(dim)):
        of.write('  layer.output_dim[{0}] = {1};\n'.format(i, dim[i]))
    of.write('  layer.output_dim_size = {0};\n'.format(len(dim)))
    of.write('  layer.is_output = {0};\n'.format('true' if layer.is_output else 'false'))
    osize = 1
    for d in layer.node_out.output_dim:
        osize *= d
    of.write('  layer.is_f32_output = {0};\n'.format('true' if layer.node_out.output_size / osize == 4 else 'false'))
    of.write('  layer.is_input_hw_layout = {0};\n'.format('true' if _is_input_hw_layout(layer) else 'false'))
    if layer.type is LayerType.SoftMax:
        axis = layer.node_in.param.axis
        if axis < 0:
            axis = len(layer.node_in.input_dim) + axis
        of.write('  layer.softmax_axis = {0};\n'.format(axis))
    elif layer.type is LayerType.Custom:
        of.write('  layer.custom_proc_ptr = &custom_callback_{0};\n'.format(layer.node_in.param.custom_param[2]))
        of.write('  layer.custom_param = &custom_param;\n')
    elif layer.type is LayerType.CopyConcatenate:
        of.write('  layer.input_layer_num = {0};\n'.format(len(layer.layer_in)))
        of.write('  layer.input_layers = input_layers;\n')
    if layer.is_output:
        of.write('  output_layers_[{0}] = &layer;\n'.format(output_index))
        output_index += 1
    of.write('}}//end of  Layer_{0}\n\n'.format(n))


class FPGARun(object):
    def __init__(self):
        self.conv = None
        self.pool = None


class LayerType(IntEnum):
    Input = auto()
    Convolution = auto()
    InnerProduct = auto()
    Flatten = auto()
    Concatenate = auto()
    CopyConcatenate = auto()
    SoftMax = auto()
    Custom = auto()
    Other = auto()

    def __str__(self):
        return self.name


class FPGALayer(object):
    def __init__(self, nodes):
        self.type = LayerType.Other
        self.run = []
        self.node_in = nodes[0]
        self.node_out = nodes[-1]
        self.output_addr_offset = 0
        self.is_output = False
        self.layer_in = []
        self.index = -1

        concat_node = None
        # append runs
        run = FPGARun()
        for node in nodes:
            if node.type in (NodeType.Convolution, NodeType.LRN):
                if run.conv:
                    self.run.append(run)
                    run = FPGARun()
                run.conv = node
            elif node.type is NodeType.Pooling:
                if concat_node:
                    for prev_run in self.run:
                        if prev_run.conv in concat_node.input_nodes:
                            prev_run.pool = node
                else:
                    if run.conv and run.conv not in node.input_nodes:
                        self.run.append(run)
                        run = FPGARun()
                    run.pool = node
                    self.run.append(run)
                    run = FPGARun()
            elif node.type in (NodeType.UpSampling, NodeType.Eltwise,
                               NodeType.Power):
                if run.conv:
                    self.run.append(run)
                    run = FPGARun()
                run.pool = node
                self.run.append(run)
                run = FPGARun()
            elif node.type is NodeType.Input:
                self.type = LayerType.Input
            elif node.type is NodeType.InnerProduct:
                self.type = LayerType.InnerProduct
            elif node.type is NodeType.Flatten:
                self.type = LayerType.Flatten
            elif node.type is NodeType.Concat:
                concat_node = node
                if (run.conv or run.pool):
                    self.run.append(run)
                    run = FPGARun()
                if len(self.run) == 0:
                    need_copy = False
                    # ignore the last input node
                    for node_in in node.input_nodes[:-1]:
                        if (node_in.output_dim[-1] % 8 != 0) or \
                           (node_in.output_size !=
                                make_align_size(node_in.output_size)):
                            need_copy = True
                            break
                    if need_copy:
                        self.type = LayerType.CopyConcatenate
                    else:
                        self.type = LayerType.Concatenate
            elif node.type is NodeType.SoftMax:
                self.type = LayerType.SoftMax
            elif node.type is NodeType.Custom:
                self.type = LayerType.Custom
        if run.conv or run.pool:
            self.run.append(run)

        if len(self.run) > 0:
            self.type = LayerType.Convolution

        # determine layer parameters
        topo = 0
        max_tiles = 1
        for i, run in enumerate(self.run):
            node_out = run.conv
            if (run.pool):
                node_out = run.pool
            if (node_out == self.node_out or
                    (concat_node and node_out in concat_node.input_nodes) or
                    (run.conv is not None and
                        any(x != 1 for x in run.conv.param.dilation[0:2]))):
                topo |= (1 << i)
            if run.conv is not None:
                tiles = calc_conv_tiles(run.conv)
                if tiles != 1:
                    topo |= (1 << i)
                max_tiles = max(tiles, max_tiles)
                if run.pool is not None:
                    tiles = calc_pool_tiles(run.pool)
                    max_tiles = max(tiles, max_tiles)
        if len(self.run) > 0 and topo == 0:
            topo = 1
        self.topo = topo
        self.tiles = max_tiles


class FPGANetwork(object):
    def __init__(self, net: cnn_layer.Network=None, quantization=True):
        self.layer = []
        self.output_layer = []
        self.num_output_layers = 0
        self.num_conv_layers = 0
        self.num_fc_layers = 0
        self.weight_size = 0
        self.buffer_size = 0
        self.custom_layer_config = {}
        self.quantization = quantization
        self.tensorflow_backend = net.tensorflow_backend
        if type(net) is cnn_layer.Network:
            self.original_net = net
            self.custom_layer_config = net.custom_layer
            self.check_limitation(net)
            self.convert_network(net)

    def check_limitation(self, net: cnn_layer.Network) -> None:
        limit = fpga_limitation.Limitation()
        conv_types = [NodeType.Convolution, NodeType.LRN, NodeType.Pooling,
                      NodeType.UpSampling]
        for node in net.traverse_list:
            if node.type in conv_types:
                if node.input_dim[0] > limit.max_conv_width:
                    msg = ("The input width {1:d} of layer {0:s} "
                           "exceeds maximum supported by FPGA {2:d}").format(
                               node.name, node.input_dim[0],
                               limit.max_conv_width)
                    logging.error(msg)
                    raise cnn_exception.ConvertError(msg)
                if node.input_dim[1] > limit.max_conv_height:
                    msg = ("The input height {1:d} of layer {0:s} "
                           "exceeds maximum supported by FPGA {2:d}").format(
                               node.name, node.input_dim[1],
                               limit.max_conv_height)
                    logging.error(msg)
                    raise cnn_exception.ConvertError(msg)
                if node.input_dim[2] > limit.max_conv_channel:
                    msg = ("The input channels {1:d} of layer {0:s} "
                           "exceed maximum supported by FPGA {2:d}").format(
                               node.name, node.input_dim[2],
                               limit.max_conv_channel)
                    logging.error(msg)
                    raise cnn_exception.ConvertError(msg)
                kernel_size = node.param.kernel_size
                if max(kernel_size) > limit.max_conv_kernel:
                    msg = ("The kernel size {1:d} of layer {0:s} "
                           "exceeds maximum supported by FPGA {2:d}").format(
                               node.name, max(kernel_size),
                               limit.max_conv_kernel)
                    logging.error(msg)
                    raise cnn_exception.ConvertError(msg)
            if node.type is NodeType.InnerProduct:
                if node.input_dim[-1] > limit.max_fc_channel:
                    msg = ("The input channels {1:d} of layer {0:s} "
                           "exceed maximum supported by FPGA {2:d}".format(
                               node.name, node.input_dim[-1],
                               limit.max_fc_channel))
                    logging.error(msg)
                    raise cnn_exception.ConvertError(msg)

    def pad_weight_matrix(self, conv_node):
        filter_sizew = conv_node.param.kernel_size[0]
        filter_sizeh = conv_node.param.kernel_size[1]
        kernel_size = get_kernel_size_for_weight(conv_node)
        padw = kernel_size[0] - filter_sizew
        padh = kernel_size[1] - filter_sizeh
        if padw != 0 or padh != 0:
            c = conv_node.input_dim[2]
            if conv_node.param.group > 1:
                c //= conv_node.param.group
            m = conv_node.output_dim[2]
            weight = conv_node.weight
            weight.shape = (m, c, filter_sizeh, filter_sizew)
            weight = np.pad(weight, ((0, 0), (0, 0), (padh, 0), (0, padw)),
                            'constant')
            conv_node.weight = weight

    def convert_network(self, net: cnn_layer.Network) -> None:
        tl = net.traverse_list
        converted_node = []
        layer_start_index = -1
        end_index = len(tl)
        index = 0

        while index < end_index:
            ignore = False
            node = tl[index]
            if index > 0:
                prev_node_type = tl[index - 1].type
            else:
                prev_node_type = None
            if (node.type is NodeType.Convolution or
                    node.type is NodeType.LRN):
                self.pad_weight_matrix(node)
                pass
            elif node.type is NodeType.Pooling:
                # Test if the pool node can merge with previous convolution node
                if (prev_node_type == NodeType.Convolution and
                        node.param.pool == 0 and
                        layer_start_index != -1 and
                        calc_conv_tiles(tl[index - 1]) == 1 and
                        calc_pool_tiles(node) == 1 and
                        node in tl[index - 1].output_nodes):
                    index += 1
                    converted_node.append(node)
                    continue
            elif node.type is NodeType.UpSampling:
                pass
            elif (node.type is NodeType.Concat and
                  (node.param.axis == 0 or
                   len(node.input_dim) == 3 and node.param.axis == 2)):
                node.output_size = sum(
                        [x.output_size for x in node.input_nodes])
            elif node.type is NodeType.Eltwise:
                pass
            elif node.type is NodeType.InnerProduct:
                pass
            elif node.type is NodeType.Flatten:
                node.output_size = node.input_nodes[0].output_size
                dim = node.input_dim
                if (dim[-1] <= 8 or
                        (len(dim) == 3 and dim[0] == 1 and dim[1] == 1)):
                    index += 1
                    converted_node.append(node)
                    continue
            elif node.type is NodeType.Reshape:
                node.output_size = node.input_nodes[0].output_size
                index += 1
                converted_node.append(node)
                continue
            elif node.type is NodeType.Custom:
                pass
            elif node.type is NodeType.Input:
                pass
            elif node.type is NodeType.SoftMax:
                pass
            else:
                ignore = True

            if (len(converted_node) > 0 and
                    not set(node.input_nodes).issubset(set(converted_node))):
                ignore = True

            if layer_start_index != -1:
                layer = FPGALayer(tl[layer_start_index:index])
                self.layer.append(layer)
                layer_start_index = -1
            if not ignore:
                converted_node.append(node)
                layer_start_index = index

            index += 1
            # handle branch, try to merge simple braches into single layer
            can_merge = True
            while len(node.output_nodes) > 1 and can_merge:
                if layer_start_index != -1:
                    layer = FPGALayer(tl[layer_start_index:index])
                    self.layer.append(layer)
                    layer_start_index = -1
                # find concat node index
                concat_index = index
                # stop at concat node or another branch node
                while (tl[concat_index].type != NodeType.Concat and
                       len(tl[concat_index].output_nodes) == 1):
                    concat_index += 1
                # test if find a concat node, if found another branch node,
                # this is not a simple branch
                if (tl[concat_index].type == NodeType.Concat and
                        tl[concat_index].param.axis ==
                        len(tl[concat_index].output_dim) - 1):
                    # make sure all branching paths are merged to this node
                    # and run depth of each path <= 2
                    for node_out in node.output_nodes:
                        run_depth = 0
                        while run_depth < 3 and node_out != tl[concat_index]:
                            run_depth += 1
                            if not node_out.output_nodes:
                                can_merge = False
                                break
                            if node_out.type is NodeType.Convolution:
                                node_out = node_out.output_nodes[0]
                                if (node_out.type is not NodeType.Pooling or
                                        node_out.param.pool != 0):
                                    continue
                            elif node_out.type is NodeType.Pooling:
                                pass
                            else:
                                run_depth = 100
                            node_out = node_out.output_nodes[0]
                        if not can_merge or run_depth > 2 or run_depth == 0:
                            can_merge = False
                            break
                    # test if all channels of input nodes are dividable by 8
                    if can_merge:
                        for node_in in tl[concat_index].input_nodes:
                            if node_in.output_dim[2] % 8 != 0:
                                can_merge = False
                                break
                    if can_merge:
                        # handle specil case: pool node immediately after
                        # the concat node can be merge into the same layer
                        node_next = tl[concat_index + 1]
                        if (node_next.type is NodeType.Pooling and
                            node_next.param.pool == 0 and
                            all(t.type is NodeType.Convolution
                                for t in tl[concat_index].input_nodes)):
                            concat_index += 1
                        converted_node.extend(tl[index:concat_index + 1])
                        layer = FPGALayer(tl[index:concat_index + 1])
                        self.layer.append(layer)
                        index = concat_index + 1
                        node = tl[concat_index]
                else:
                    break

        # append the last layer
        if layer_start_index != -1:
            layer = FPGALayer(tl[layer_start_index:index])
            self.layer.append(layer)

        # determine output layers
        output_nodes = net.output_nodes[:]
        for layer in self.layer:
            if layer.node_out in output_nodes:
                layer.is_output = True
                output_nodes.remove(layer.node_out)
        i = 0
        while i < len(output_nodes):
            node = output_nodes[i]
            i += 1
            for node_in in node.input_nodes:
                out_node_reached = False
                if node_in in converted_node:
                    for layer in reversed(self.layer):
                        if layer.node_out is node_in:
                            out_node_reached = True
                            layer.is_output = True
                            break
                if not out_node_reached:
                    output_nodes.append(node_in)

        self.connect_layers()

        # remove unnecessary layers
        for layer in self.layer[:]:
            if layer.type is LayerType.Input:
                self.layer.remove(layer)

    def connect_layers(self):
        """Set layer output addresses."""
        logging.info('Converted layer info')
        logging.info("{:22s} {:22s} {:12s} {:5s} {:18s} {:8s} {:s}".format(
            'Input Node', 'Output Node', 'Node Type', 'Live Range',
            'Output Dimension', 'Addr', 'Size'))

        class LayerLiveRange(object):
            def __init__(self, layer, index):
                self.layer = layer
                self.birth_index = index
                self.death_index = index
                self.output_concat_lr = None
                self.allocated = False
        live_ranges = []
        weight_size = 0
        for index, layer in enumerate(self.layer):
            # calc weight and increment the counter
            if layer.type is LayerType.Convolution:
                self.num_conv_layers += 1
                for run in layer.run:
                    weight_size += get_weight_size(run.conv, self.quantization)
            elif layer.type is LayerType.InnerProduct:
                self.num_fc_layers += 1
                weight_size += get_fc_weight_size(layer.node_in,
                                                  self.quantization)

            # add to output_layer
            if layer.is_output:
                self.num_output_layers += 1
                self.output_layer.append(layer)

            # create LayerLiveRange
            lr = LayerLiveRange(layer, index)
            for node_in, lr_in in itertools.product(layer.node_in.input_nodes,
                                                    live_ranges):
                if node_in is lr_in.layer.node_out:
                    lr.layer.layer_in.append(lr_in.layer)
                    if lr_in.death_index < index:
                        lr_in.death_index = index
            if lr.layer.is_output:
                lr.death_index = len(self.layer) - 1
            live_ranges.append(lr)

        # handle concat layer and update live_ranges
        def _lr_is_partof_concat(lr, concat_lr):
            if concat_lr.layer.type is not LayerType.Concatenate:
                return False
            while lr:
                if lr is concat_lr:
                    return True
                lr = lr.output_concat_lr
            return False

        for index, lr in enumerate(reversed(live_ranges)):
            index = len(live_ranges) - index
            if lr.layer.type is LayerType.Concatenate:
                for prev_lr in live_ranges[:index]:
                    if prev_lr.layer.node_out in lr.layer.node_in.input_nodes:
                        prev_lr.output_concat_lr = lr

        for index, lr in enumerate(reversed(live_ranges)):
            index = len(live_ranges) - index
            if lr.layer.type is LayerType.Concatenate:
                for prev_lr in live_ranges[:index]:
                    if _lr_is_partof_concat(prev_lr, lr):
                        max_di = max(prev_lr.death_index, lr.death_index)
                        prev_lr.death_index = max_di
                        lr.death_index = max_di
                        min_di = min(prev_lr.birth_index, lr.birth_index)
                        prev_lr.birth_index = min_di
                        lr.birth_index = min_di

        def get_dst_concat_lr(lr):
            """
            search final concat node Life Range
            If lr.output_concat_lr is None, lr is returned.
            """
            _lr = lr
            while _lr.output_concat_lr:
                _lr = _lr.output_concat_lr
            return _lr

        allocated_size = 0  # size to be allocated
        for index, lr in enumerate(reversed(live_ranges)):
            index = len(live_ranges) - 1 - index

            if lr.output_concat_lr:
                offset = 0
                for node in lr.output_concat_lr.layer.node_in.input_nodes:
                    if node == lr.layer.node_out:
                        break
                    offset += node.output_size
                lr.layer.output_addr_offset = lr.output_concat_lr.layer.output_addr_offset + offset
                lr.allocated = True

            necessary_size = make_align_size(lr.layer.node_out.output_size)
            if not lr.allocated:
                current_live_ranges = [_lr for _lr in live_ranges
                                       if not
                                       (_lr.birth_index > lr.death_index or
                                        _lr.death_index < lr.birth_index)]
                current_live_ranges = sorted(
                                    current_live_ranges,
                                    key=(lambda x: x.layer.output_addr_offset))

                # find if can re-use empty spaces in current live ranges
                empty_found = False
                current_offset = 0
                for clr in current_live_ranges:
                    if not clr.allocated:
                        continue

                    empty_found = (clr.layer.output_addr_offset
                                   - current_offset >= necessary_size)
                    if empty_found:
                        lr.layer.output_addr_offset = current_offset
                        break
                    else:
                        # to next, update current_offset
                        increment_size = make_align_size(
                            clr.layer.node_out.output_size)
                        if clr.output_concat_lr:
                            increment_size = make_align_size(
                                clr.output_concat_lr.layer.node_out.output_size)
                        current_offset = (clr.layer.output_addr_offset +
                                          increment_size)

                # if not, put it in the end of current buffer
                if not empty_found:
                    lr.layer.output_addr_offset = current_offset
                    if current_offset + necessary_size > allocated_size:
                        allocated_size = current_offset + necessary_size
                lr.allocated = True

        for lr in live_ranges:
            logging.info("{:22s} {:22s} {:12s} {:02d} {:02d} {:18s} {:08X} {:08X}{:s}".format(
                lr.layer.node_in.name, lr.layer.node_out.name,
                str(lr.layer.type),
                lr.birth_index, lr.death_index,
                str(lr.layer.node_out.output_dim),
                lr.layer.output_addr_offset,
                lr.layer.node_out.output_size,
                ('*' if lr.layer.is_output else '')))
        logging.info('allocated size:{:d}'.format(allocated_size))
        self.weight_size = weight_size
        self.buffer_size = allocated_size

    def output_header(self, of, name) -> None:
        gen_header_header(of, name, self.custom_layer_config)
        for n, layer in enumerate(self.layer):
            gen_header_layer(of, n, layer, self.quantization)
        gen_header_footer(of, name)

    def output_source(self, of, name) -> None:
        global weight_offset
        global output_index
        global is_tensorflow
        weight_offset = 0
        output_index = 0
        is_tensorflow = self.tensorflow_backend
        gen_source_header(of, name, self)
        for n, layer in enumerate(self.layer):
            layer.index = n
            gen_source_layer(of, name, n, layer, self.quantization)

    def output_weights(self, of) -> None:
        prev_node = None
        for layer in self.layer:
            if layer.type is LayerType.Convolution:
                for run in layer.run:
                    if (run.conv is not None and
                            run.conv.type is NodeType.Convolution):
                        pack_conv_weight(run.conv, of, self.quantization)
            elif layer.type is LayerType.InnerProduct:
                pack_fc_weight(layer.node_in, prev_node, of, self.quantization)
            prev_node = layer.node_out

    def output_network(self, output_folder: str, network_name: str,
                       gensrc: bool, gendoc: bool, gengraph: bool,
                       graphviz_path: str) -> None:
        output_folder = os.path.join(output_folder, network_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if gensrc:
            header_file_name = os.path.join(output_folder,
                                            network_name + '_gen.h')
            source_file_name = os.path.join(output_folder,
                                            network_name + '_gen.cpp')
        weight_file_name = os.path.join(output_folder,
                                        network_name + '_weights.bin')
        if gensrc:
            hf = open(header_file_name, "w")
            self.output_header(hf, network_name)
            hf.close()
            sf = open(source_file_name, "w")
            self.output_source(sf, network_name)
            sf.close()
        if gendoc:
            cnn_docgen.output_doc(output_folder, network_name, gengraph,
                                  graphviz_path, self)
        with open(weight_file_name, "wb") as bf:
            self.output_weights(bf)
