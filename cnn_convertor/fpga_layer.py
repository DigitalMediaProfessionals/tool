# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
 Copyright(c) 2017 by Digital Media Professionals Inc.
 All rights reserved.
------------------------------------------------------------
"""
import os
import math
import logging
import numpy as np
from cnn_convertor import cnn_layer, cnn_exception, cnn_docgen
from cnn_convertor.cnn_layer import NodeType
from enum import IntEnum, auto


MAX_RUN = 32
MAX_UNIFIED_BUFFER_SIZE = 640 * 1024  # 640KB
MEM_ALIGN = 128 // 8  # 128bit


def get_actfunc(tpe):
    # 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU,
    # 6 = ReLU6
    return {
        NodeType.TanH: 1,
        NodeType.Sigmoid: 3,
        NodeType.ELU: 5}.get(tpe, 2)


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


def get_conv_out_width(width, kx, pad, stride):
    return (pad + width + pad - kx) // stride + 1


def calc_conv_tiles(node):
    w = node._input_dim[0]
    h = node._input_dim[1]
    c = node._input_dim[2]
    m = node._output_dim[2]
    p = node._param.kernel_size[0]
    pad = node._param.pad
    stride = node._param.stride
    c_blocks = (c >> 3) + (1 if c & 7 else 0)
    t = 0
    while True:
        t += 1
        tw = divup(w, t) + p - 1  # width of tile
        ow = get_conv_out_width(tw, p, pad[0], stride[0])
        oh = get_conv_out_width(h, p, pad[1], stride[1])
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
    if node._type is NodeType.UpSampling:
        return 1
    w = node._input_dim[0]
    h = node._input_dim[1]
    c = node._input_dim[2]
    t = 0
    while True:
        t += 1
        uu, d = divmod(w * h * min(8, c), t)
        if d:
            uu += 1
        if uu * 2 <= MAX_UNIFIED_BUFFER_SIZE:
            return t


def merge_bn_scale(node, kernel_size, n_c, n_m):
    weight = node._weight
    bias = node._bias
    if node._bn_node is not None:
        bn_mean = node._bn_node._mean
        bn_var = node._bn_node._var
    else:
        bn_mean = np.zeros_like(bias)
        bn_var = np.full_like(bias, 1.0 - 0.00001)
    if node._sc_node is not None:
        sc_weight = node._sc_node._weight
        sc_bias = node._sc_node._bias
    else:
        sc_weight = np.ones_like(bias)
        sc_bias = np.zeros_like(bias)
    e = 0.00001
    weight.shape = (n_m, n_c * kernel_size[1] * kernel_size[0])
    for i in range(n_m):
        norm = sc_weight[i] / math.sqrt(bn_var[i] + e)
        weight[i, :] = weight[i, :] * norm
        bias[i] = (bias[i] - bn_mean[i]) * norm + sc_bias[i]
    weight.shape = (-1,)
    return weight, bias


def calc_kmeans(weight):
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


def pack_weight(node, of, quantization):
    logging.info('Packing weight for node: %s.', node._name)

    n_c = node._input_dim[2]
    if node._param.group > 1:
        n_c = n_c // node._param.group
    n_m = node._output_dim[2]
    kernel_size = node._param.kernel_size

    weight = node._weight
    bias = node._bias
    if weight is None and bias is None:
        weight = np.ones((n_m, n_c, kernel_size[1], kernel_size[0]), np.float32)
        bias = np.zeros((n_m, n_c), np.float32)
        node.set_weight_bias(weight, bias)

    if node._bn_node is not None or node._sc_node is not None:
        weight, bias = merge_bn_scale(node, kernel_size, n_c, n_m)

    if quantization:
        centers, labels = calc_kmeans(weight)
        weight_type = np.uint8
    else:
        labels = weight.astype(np.float16)
        weight_type = np.float16
    bias16 = bias.astype(np.float16)
    buffer = np.zeros(shape=(12, 6), dtype=weight_type)

    labels.shape = (n_m, n_c, kernel_size[1], kernel_size[0])
    if quantization:
        centers.tofile(of)
    if kernel_size[0] == 7:
        for m_start in range(0, n_m, 8):
            m_stop = min(m_start + 8, n_m)
            bias16[m_start:m_stop].tofile(of)
            for i in range(m_stop, m_start + 8):
                of.write(b'\0\0')
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
        raise cnn_exception.ConvertError('Unsupported kernel size' +
                                         kernel_size[0])


def pack_fc_weight(node, conv_node, of):
    logging.info('Packing FC weight for node: %s.', node._name)
    centers, labels = calc_kmeans(node._weight)
    index8 = labels.astype(np.uint8)
    bias16 = node._bias.astype(np.float16)

    centers.tofile(of)
    if conv_node is not None:
        if len(conv_node._output_dim) == 3:
            w, h, c = conv_node._output_dim
        elif len(conv_node._output_dim) == 1:
            w, h, c = 1, 1, conv_node._output_dim[0]
        m = node._param.num_output
        if w != 1 or h != 1:
            logging.info('Reordering FC weight for node: %s.', node._name)
            index8.shape = (m, c, h, w)
            for n in range(m):
                for d in range(0, c, 8):
                    e = d + 8 if d + 8 < c else c
                    tr_index8 = index8[n, d:e, :, :].transpose(2, 1, 0)
                    index8[n, d:e, :, :] = tr_index8.reshape(e - d, h, w)
    index8.tofile(of)
    bias16.tofile(of)


def get_weight_size(node, quantization):
    if node is None or node._type is not NodeType.Convolution:
        return 0
    c = node._input_dim[2]
    if node._param.group > 1:
        c //= node._param.group
    m = node._output_dim[2]
    k = node._param.kernel_size
    if k[0] == 7:
        pass
    if k[0] == 5:
        c = c // 2 + c % 2
    if k[0] == 3:
        c = c // 8 + (0 if c % 8 == 0 else 1)
    if k[0] == 1:
        c = c // 64 + (0 if c % 64 == 0 else 1)
    if quantization:
        weight_size = 512 + 72 * m * c + 16 * ((m + 7) // 8)
    else:
        weight_size = 144 * m * c + 16 * ((m + 7) // 8)
    return weight_size


def get_fc_weight_size(node):
    if len(node._input_dim) == 3:
        w, h, c = node._input_dim
    elif len(node._input_dim) == 1:
        w, h, c = 1, 1, node._input_dim[0]
    m = node._output_dim[0]
    size = w * h * c * m + m * 2 + 512
    return size


def gen_header_header(of, name, custom_layer_config):
    of.write('/*\n'
             '*------------------------------------------------------------\n'
             '* Copyright(c) 2017 by Digital Media Professionals Inc.\n'
             '* All rights reserved.\n'
             '*------------------------------------------------------------\n'
             '* This source code was generated using DMP-DV700 tools\n'
             '* Digital Media Professionals Inc.\n'
             '*------------------------------------------------------------\n'
             '*/\n\n'
             '#pragma once\n'
             '#include "dmp_network.h"\n\n')
    for custom_type, custom_config in custom_layer_config.items():
        of.write ('struct custom_param_{0}\n'.format(custom_type))
        of.write('{\n')
        for param_name, param_type in custom_config[0].items():
            if '[' in param_type:
                sub_index = param_type.find('[')
                of.write('    {0:5} {1}{2};\n'.format(param_type[:sub_index], param_name, param_type[sub_index:]))
            else:
                of.write('    {0:5} {1};\n'.format(param_type, param_name))
        of.write('};\n\n')
    for custom_type in custom_layer_config:
        of.write('void custom_callback_{0}(fpga_layer &layer, void *custom_param);\n\n'.format(custom_type))
    of.write('class C{0} : public CDMP_Network\n'.format(name))
    of.write('{\n'
             '    private:\n')


def gen_header_layer(of, n, layer, quantization):
    of.write('/*!\n\n'
             'Layer description\n\n'
             '| ID | Layers | Type | Dim In | Dim Out | Param | Mem |\n'
             '| :- | :- | :-: | :-: | :-: | :-: | :-: |\n')
    of.write('| {0} | FPGA-Layer | {1} | {2} | {3} | - | - |\n'.format(
             n, str(layer.type),
             str(layer.node_in._input_dim),
             str(layer.node_out._output_dim)))
    for i, run in enumerate(layer.run):
        if run.conv is not None:
            of.write('| {0}-{1} | {2} | {3} | {4} | {5} | - | {6} |\n'.format(
                     n, i,
                     run.conv._name,
                     str(run.conv._type),
                     str(run.conv._input_dim),
                     str(run.conv._output_dim),
                     get_weight_size(run.conv, quantization)))
        if run.pool is not None:
            of.write('| {0}-{1} | {2} | {3} | {4} | {5} | - | - |\n'.format(
                     n, i,
                     run.pool._name,
                     str(run.pool._type),
                     str(run.pool._input_dim),
                     str(run.pool._output_dim)))
    of.write('\n*/\n')
    of.write('        void Layer_{0}();\n'.format(n))


def gen_header_footer(of, name):
    of.write('\n\n'
             '    public:\n'
             '        unsigned int get_total_layer_count();\n'
             '        unsigned int get_output_layer_count();\n'
             '        unsigned int get_convolution_layer_count();\n'
             '        unsigned int get_innerproduct_layer_count();\n'
             '        int initialize();\n')
    of.write('        C{0}();\n'.format(name))
    of.write('        ~C{0}();\n'.format(name))
    of.write('\n\n};\n')


def gen_source_header(of, name, net):
    of.write('/*\n'
             '*------------------------------------------------------------\n'
             '* Copyright(c) 2017 by Digital Media Professionals Inc.\n'
             '* All rights reserved.\n'
             '*------------------------------------------------------------\n'
             '* This source code was generated using DMP-DV700 tools\n'
             '* Digital Media Professionals Inc.\n'
             '*------------------------------------------------------------\n'
             '*/\n\n')
    of.write('#include "{0}_gen.h"\n'.format(name))
    of.write('\n\n\n')
    of.write('C{0}::C{0}()\n'.format(name))
    of.write('{\n\n}\n\n')
    of.write('C{0}::~C{0}()\n'.format(name))
    of.write('{\n\n}\n\n')
    of.write('unsigned int C{0}::get_total_layer_count()\n'.format(name))
    of.write('{\n'
             '    return num_layers;\n'
             '}\n\n')
    of.write('unsigned int C{0}::get_output_layer_count()\n'.format(name))
    of.write('{\n'
             '    return num_output_layers;\n'
             '}\n\n')
    of.write('unsigned int C{0}::get_convolution_layer_count()\n'.format(name))
    of.write('{\n'
             '    return num_conv_layers;\n'
             '}\n\n')
    of.write('unsigned int C{0}::get_innerproduct_layer_count()\n'.format(name))
    of.write('{\n'
             '    return num_fc_layers;\n'
             '}\n\n')
    of.write('int C{0}::initialize()\n'.format(name))
    of.write('{\n')
    of.write('    num_layers = {0};\n'.format(len(net._layer)))
    of.write('    num_output_layers = {0};\n'.format(net.num_output_layers))
    of.write('    num_conv_layers = {0};\n'.format(net.num_conv_layers))
    of.write('    num_fc_layers = {0};\n'.format(net.num_fc_layers))
    of.write('    weight_size = {0};\n'.format(net.weight_size))
    of.write('    buffer_size = {0};\n'.format(net.buffer_size))
    of.write('    layers.resize(num_layers);\n'
             '    output_layers.resize(num_output_layers);\n'
             '    conv_layers.resize(num_conv_layers);\n'
             '    fc_layers.resize(num_fc_layers);\n'
             '    memory_size_request.resize(2);\n\n'
             '    //set_default_convolution_layers_parameters();\n')
    for n in range(len(net._layer)):
        of.write('    Layer_{0}();\n'.format(n))
    of.write('\n'
             '    //Add 2 memory size requests. One for weights, the other for io buffers\n'
             '    memory_size_request[0] = weight_size;\n'
             '    memory_size_request[1] = buffer_size;\n\n'
             '    return 0;\n'
             '}\n\n')


def gen_source_conv(of, name, n, layer, quantization):
    global weight_offset
    global conv_index
    global is_tensorflow
    of.write('//Layer_{0}: Convolution Layer\n'.format(n))
    for run in layer.run:
        if run.conv is not None:
            of.write('//  ->: {0}\n'.format(run.conv._name))
            if run.conv._bn_node:
                of.write('//  ->: {0}\n'.format(run.conv._bn_node._name))
            if run.conv._sc_node:
                of.write('//  ->: {0}\n'.format(run.conv._sc_node._name))
            if run.conv._act_node:
                of.write('//  ->: {0}\n'.format(run.conv._act_node._name))
        if run.pool is not None:
            of.write('//  ->: {0}\n'.format(run.pool._name))
    of.write('void C{0}::Layer_{1}()\n'.format(name, n))
    of.write('{\n')
    of.write('    struct top_conv_conf& _conf = get_conv_layer({0});\n'.format(conv_index))
    of.write('    //Topo: {0:032b}\n'.format(layer.topo))
    of.write('    _conf.hw.header.topo = 0x{0:X}; // [31:0] Output Destination of each run, 0 = UBUF, 1 = EXTMEM\n\n'.format(layer.topo))
    of.write('    //Input Configuration:\n')
    of.write('    _conf.hw.input.w = {0}; // Input Width\n'.format(layer.node_in._input_dim[0]))
    of.write('    _conf.hw.input.h = {0}; // Input Height\n'.format(layer.node_in._input_dim[1]))
    of.write('    _conf.hw.input.z = 1; // Input Depth\n')
    of.write('    _conf.hw.input.c = {0}; // Input Channels\n'.format(layer.node_in._input_dim[2]))
    of.write('    _conf.hw.input.input_base_addr = 0x{0:08X}; // Input byte address\n'.format(layer.layer_in[0].output_addr_offset))
    of.write('    _conf.hw.input.tiles = {0}; // Number of horizontal tiles (supported with restrictions)\n\n'.format(layer.tiles))
    of.write('    //Output Configuration:\n')
    if len(layer.node_out._output_dim) == 3:
        of.write('    _conf.sw.output.w = {0}; // Output Width\n'.format(layer.node_out._output_dim[0]))
        of.write('    _conf.sw.output.h = {0}; // Output Height\n'.format(layer.node_out._output_dim[1]))
        of.write('    _conf.sw.output.z = 1; // Output Depth\n')
        of.write('    _conf.sw.output.m = {0}; // Output Channels\n'.format(layer.node_out._output_dim[2]))
    else:
        of.write('    _conf.sw.output.w = 1; // Output Width\n')
        of.write('    _conf.sw.output.h = 1; // Output Height\n')
        of.write('    _conf.sw.output.z = 1; // Output Depth\n')
        of.write('    _conf.sw.output.m = {0}; // Output Channels\n'.format(layer.node_out._output_dim[0]))
    of.write('    _conf.hw.output.output_base_addr = 0x{0:08X}; // Output byte address\n'.format(layer.output_addr_offset))
    of.write('    _conf.hw.output.eltwise_base_addr = 0xDEADBEEF; // Input byte address for elementwise add (0 = UBUF Input Buffer)\n'
             '    _conf.hw.output.output_mode =0; // 0 = concat, 1 = eltwise add\n\n'
             '    //Runs Configuration:\n')
    of.write('    //->{0} run(s)\n'.format(len(layer.run)))
    for i, run in enumerate(layer.run):
        is_conv = run.conv is not None and run.conv._type is NodeType.Convolution
        is_lrn = run.conv is not None and run.conv._type is NodeType.LRN
        p = (run.conv._param.kernel_size[0] if is_conv else 1)
        if is_conv:
            conv_enable = (1 if run.conv._param.group <= 1 else 3)
        else:
            conv_enable = 0
        conv_pad = (run.conv._param.pad[0] | (run.conv._param.pad[1] << 16) if is_conv else 0)
        conv_pad |= conv_pad << 8
        conv_stride = (run.conv._param.stride[0] | (run.conv._param.stride[1] << 8) if is_conv else 0x101)
        pool_enable = (1 if run.pool else 0)
        pool_size = (run.pool._param.kernel_size[0] | (run.pool._param.kernel_size[1] << 8) if run.pool else 0)
        pool_stride = (run.pool._param.stride[0] | (run.pool._param.stride[1] << 8) if run.pool else 0x101)
        pool_pad = (run.pool._param.pad[0] | (run.pool._param.pad[1] << 16) if run.pool else 0)
        pool_pad |= pool_pad << 8
        actfunc = 0
        actparam = 0
        pool_avg_param = 0
        node_in = run.pool
        if run.conv is not None:
            node_in = run.conv
            if node_in._act_node:
                actfunc = get_actfunc(node_in._act_node._type)
                actparam = np.float16(node_in._act_node._param.relu_param)
                actparam = actparam.data[1] << 8 | actparam.data[0]
        node_out = run.conv
        if run.pool is not None:
            node_out = run.pool
            if ((node_out._type is NodeType.Pooling and
                 node_out._param.pool != 0) or
                    node_out._type is NodeType.Eltwise):
                pool_enable = 2
                pool_avg_param = np.float16(
                    1.0 / (run.pool._param.kernel_size[0] *
                           run.pool._param.kernel_size[1])).view(np.uint16)
            if node_out._type is NodeType.UpSampling:
                pool_enable = 4
            if node_out._type is NodeType.Power:
                pool_enable = 2
                pool_avg_param = np.float16(
                    node_out._param.scale).view(np.uint16)
        m = node_out._output_dim[2]
        # adjust pad size if backend was tensorflow
        if is_tensorflow:
            if run.conv is not None:
                if ((conv_pad & 0xFF) * 2 + run.conv._input_dim[0] - p) > (run.conv._output_dim[0] - 1) * (conv_stride & 0xFF):
                    conv_pad -= 1
                if (((conv_pad & 0xFF0000) >> 16) * 2 + run.conv._input_dim[1] - p) > (run.conv._output_dim[1] - 1) * ((conv_stride & 0xFF00) >> 8):
                    conv_pad -= 0x10000
            if run.pool is not None and run.pool._type is NodeType.Pooling:
                if ((pool_pad & 0xFF) * 2 + run.pool._input_dim[0] - run.pool._param.kernel_size[0]) > (run.pool._output_dim[0] - 1) * (pool_stride & 0xFF):
                    pool_pad -= 1
                if (((pool_pad & 0xFF0000) >> 16) * 2 + run.pool._input_dim[1] - run.pool._param.kernel_size[1]) > (run.pool._output_dim[1] - 1) * ((pool_stride & 0xFF00) >> 8):
                    pool_pad -= 0x10000
        # detect if this is the case of pool node being merged into concat node
        if node_in != node_out and node_in not in node_out._input_nodes:
            m = node_in._output_dim[2]
        of.write('    //--------------------------------------------------\n')
        of.write('    //RUN : {0}\n'.format(i))
        of.write('    //--------------------------------------------------\n')
        if run.conv is not None:
            of.write('    //->: {0}\n'.format(run.conv._name))
            if run.conv._bn_node:
                of.write('    //->: {0}\n'.format(run.conv._bn_node._name))
            if run.conv._sc_node:
                of.write('    //->: {0}\n'.format(run.conv._sc_node._name))
            if run.conv._act_node:
                of.write('    //->: {0}\n'.format(run.conv._act_node._name))
        if run.pool is not None:
            of.write('    //->: {0}\n'.format(run.pool._name))
        of.write('    _conf.sw.run[{0}].in_w = {1}; // Optional: Input width (not used by HW - discovered on the fly)\n'.format(i, node_in._input_dim[0]))
        of.write('    _conf.sw.run[{0}].in_h = {1}; // Optional: Input height (not used by HW - discovered on the fly)\n'.format(i, node_in._input_dim[1]))
        of.write('    _conf.sw.run[{0}].in_c = {1}; // Optional: Input Channels (not used by HW - discovered on the fly)\n'.format(i, node_in._input_dim[2]))
        of.write('    _conf.sw.run[{0}].out_w = {1}; // Optional: Output width (not used by HW - discovered on the fly)\n'.format(i, node_out._output_dim[0]))
        of.write('    _conf.sw.run[{0}].out_h = {1}; // Optional: Output height (not used by HW - discovered on the fly)\n'.format(i, node_out._output_dim[1]))
        of.write('    _conf.hw.run[{0}].m = {1}; // Output Channels\n'.format(i, m))
        of.write('    _conf.hw.run[{0}].conv_enable = {1}; // 1 = Enabled, 0 = Disabled\n'.format(i, conv_enable))
        of.write('    _conf.hw.run[{0}].p = {1}; // Filter Width and Height\n'.format(i, p))
        of.write('    _conf.hw.run[{0}].pz = 1; // Filter Depth\n'.format(i))
        of.write('    _conf.hw.run[{0}].weight_base_addr = 0x{1:08X}; // Filter Weight and Bias byte address\n'.format(i, weight_offset))
        of.write('    _conf.hw.run[{0}].weight_fmt = {1}; // Weight format (0 = random access blocks, 1 = compact stream, 3 = 8-bit qunatized stream)\n'.format(i, ((3 if quantization else 1) if is_conv else 0)))
        of.write('    _conf.sw.run[{0}].weight_size = {1}; // Actual size in bytes of LUT, weights and bias (in bytes)\n'.format(i, get_weight_size(run.conv, quantization)))
        of.write('    _conf.hw.run[{0}].conv_pad = 0x{1:X}; // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding\n'.format(i, conv_pad))
        of.write('    _conf.hw.run[{0}].conv_stride = 0x{1:X}; // bits [7:0] = X stride, bits [15:8] = Y stride\n'.format(i, conv_stride))
        of.write('    _conf.hw.run[{0}].conv_dilation = 0x0; // bits [7:0] = X dilation, bits [15:8] = Y dilation\n'.format(i))
        of.write('    _conf.hw.run[{0}].pool_enable = {1};  // 0 = disabled, 1 = max pooling, 2 = average pooling\n'.format(i, pool_enable))
        of.write('    _conf.hw.run[{0}].pool_size = 0x{1:X}; // bits [7:0] = width, bits [15:8] = height\n'.format(i, pool_size))
        of.write('    _conf.hw.run[{0}].pool_stride =0x{1:X}; // bits [7:0] = X stride, bits [15:8] = Y stride\n'.format(i, pool_stride))
        of.write('    _conf.hw.run[{0}].pool_pad = 0x{1:X}; // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding\n'.format(i, pool_pad))
        of.write('    _conf.hw.run[{0}].pool_avg_param = 0x{1:X}; // Must be set to 1/pool_size^2 in FP16 format when using average pooling (average pooling assumes square size)\n'.format(i, pool_avg_param))
        of.write('    _conf.hw.run[{0}].actfunc = {1}; // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6\n'.format(i, actfunc))
        of.write('    _conf.hw.run[{0}].actfunc_param = 0x{1:X}; // Leaky ReLU parameter (NOTE: 0x2E66 is 0.1 in FP16)\n'.format(i, actparam))
        of.write('    _conf.hw.run[{0}].rectifi_en = 0; // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)\n'.format(i))
        of.write('    _conf.hw.run[{0}].lrn= 0x{1:X}; // [0] : 1 = LRN enable, 0 = LRN disable, [1] : 1 = incl. power func, 0 = excl., [8:11] = x^2 scale factor log2\n'.format(i, (1027 if is_lrn else 0)))
        of.write('    _conf.hw.run[{0}].ALIGN_0 = 0;//Some comments needed here\n'.format(i))
        weight_offset += get_weight_size(run.conv, quantization)
    conv_index += 1


def gen_source_fc(of, name, n, layer):
    global weight_offset
    global fc_index
    node = layer.node_in
    if len(node._input_dim) == 3:
        w, h, c = node._input_dim
    elif len(node._input_dim) == 1:
        w, h, c = 1, 1, node._input_dim[0]
    m = node._output_dim[0]
    size = get_fc_weight_size(node)
    actfunc = 0
    actparam = 0
    if node._act_node:
        if node._act_node._type == NodeType.ReLU:
            actfunc = 0x10
            if node._act_node._param.relu_param != 0.0:
                actfunc = 0x30
                actparam = np.float16(node._act_node._param.relu_param)
                actparam = actparam.data[1] << 8 | actparam.data[0]
        else:
            actfunc = 0x20
    of.write('//Layer_{0}: Fully Connected Layer\n'.format(n))
    of.write('//	->: {0}\n'.format(node._name))
    of.write('void C{0}::Layer_{1}()\n'.format(name, n))
    of.write('{\n')
    of.write('    struct top_fc_conf& cf = get_ip_layer({0});\n'.format(fc_index))
    of.write('    cf.hw.input_size = {0};\n'.format(w * h * c))
    of.write('    cf.hw.output_size = {0};\n'.format(m))
    of.write('    cf.sw.total_size = {0}; //from tool // Actual size in bytes of LUT, weights and bias (in bytes)\n'.format(size))
    of.write('    cf.hw.stride = cf.hw.input_size;\n'
             '    cf.hw.bias_size = 2 * cf.hw.output_size; // bias size (in bytes) = 2 times the output size\n')
    of.write('    cf.hw.param_base_addr = 0x{0:08X}; //base address\n'.format(weight_offset))
    of.write('    cf.hw.weight_addr = 0x{0:08X}; //weight address = param_base_addr + 2*256 (lut size/float16/2bytes)\n'.format(weight_offset + 2 * 256))
    of.write('    cf.hw.bias_addr = 0x{0:08X}; //bias address =  weight_addr + stride*input size\n'.format(weight_offset + 2 * 256 + w * h * c * m))
    of.write('    cf.hw.input_base_addr = 0x{0:08X};\n'.format(layer.layer_in[0].output_addr_offset))
    of.write('    cf.hw.output_base_addr = 0x{0:08X};\n'.format(layer.output_addr_offset))
    of.write('    cf.hw.param_fmt = 1; // 0 = unquantized weight matrix, 1 = qunatized\n')
    of.write('    cf.hw.actfunc = 0x{0:X}; // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6\n'.format(actfunc))
    of.write('    cf.hw.actfunc_param = 0x{0:X}; // Leaky ReLU parameter (in FP16 format), 0 = non-leaky\n'.format(actparam))
    weight_offset += size
    fc_index += 1


def gen_source_layer(of, name, n, layer, quantization):
    global output_index
    type_map = { LayerType.Input: 'LT_INPUT',
                 LayerType.Convolution: 'LT_CONV',
                 LayerType.InnerProduct: 'LT_FC',
                 LayerType.Flatten: 'LT_FLATTEN',
                 LayerType.Concatenate: 'LT_CONCAT',
                 LayerType.SoftMax: 'LT_SOFTMAX',
                 LayerType.Custom: 'LT_CUSTOM' }

    if layer.type is LayerType.Convolution:
        gen_source_conv(of, name, n, layer, quantization)
        of.write('\n')
    elif layer.type is LayerType.InnerProduct:
        gen_source_fc(of, name, n, layer)
        of.write('\n')
    else:
        if layer.type is LayerType.Input:
            of.write('//Layer_{0}: Input Layer\n'.format(n))
        elif layer.type is LayerType.Concatenate:
            of.write('//Layer_{0}: Concatenate Layer\n'.format(n))
        elif layer.type is LayerType.Flatten:
            of.write('//Layer_{0}: Flatten Layer\n'.format(n))
        elif layer.type is LayerType.SoftMax:
            of.write('//Layer_{0}: SoftMax Layer\n'.format(n))
        else:
            of.write('//Layer_{0}: Custom Layer\n'.format(n))
        node = layer.node_in
        of.write('//	->: {0}\n'.format(node._name))
        of.write('void C{0}::Layer_{1}()\n'.format(name, n))
        of.write('{\n')

    if layer.type is LayerType.Custom:
        custom_param = layer.node_in._param.custom_param
        of.write ('    static custom_param_{0} custom_param = {{\n'.format(custom_param[2]))
        for param in custom_param[0].values():
            if type(param) is list:
                of.write('        { ')
                for value in param:
                    of.write('{0}, '.format(value))
                of.write(' },\n')
            elif type(param) is bool:
                of.write('        {0},\n'.format('true' if param else 'false'))
            else:
                of.write('        {0},\n'.format(param))
        of.write('    };\n\n')

    of.write('    struct fpga_layer& layer = layers[{0}];\n'.format(n))
    of.write('    layer.type = {0};\n'.format(type_map[layer.type]))
    if layer.type is LayerType.Convolution:
        of.write('    layer.hw_conf = (void*)&_conf;\n')
    elif layer.type is LayerType.InnerProduct:
        of.write('    layer.hw_conf = (void*)&cf;\n')
    else:
        of.write('    layer.hw_conf = (void*)0;\n')
    of.write('    layer.addr_cpu_input = 0x0;\n')
    of.write('    layer.addr_cpu_output = 0x0;\n')
    of.write('    layer.addr_offset_input = 0x{:08X};\n'.format(layer.layer_in[0].output_addr_offset))
    of.write('    layer.addr_offset_output = 0x{:08X};\n'.format(layer.output_addr_offset))
    of.write('    layer.output_size = {0};\n'.format(layer.node_out._output_size))
    dim = layer.node_in._input_dim
    for i in range(len(dim)):
        of.write('    layer.input_dim[{0}] = {1};\n'.format(i, dim[i]))
    of.write('    layer.input_dim_size = {0};\n'.format(len(dim)))
    dim = layer.node_out._output_dim
    for i in range(len(dim)):
        of.write('    layer.output_dim[{0}] = {1};\n'.format(i, dim[i]))
    of.write('    layer.output_dim_size = {0};\n'.format(len(dim)))
    of.write('    layer.is_output = {0};\n'.format('true' if layer.is_output else 'false'))
    osize = 1
    for d in layer.node_out._output_dim:
        osize *= d
    of.write('    layer.is_f32_output = {0};\n'.format('true' if layer.node_out._output_size / osize == 4 else 'false'))
    of.write('    layer.is_input_hw_layout = {0};\n'.format('true' if layer.layer_in[0].type is LayerType.Convolution else 'false'))
    if layer.type is LayerType.SoftMax:
        axis = layer.node_in._param.axis
        if axis < 0:
            axis = len(layer.node_in._input_dim) + axis
        of.write('    layer.softmax_axis = {0};\n'.format(axis))
    elif layer.type is LayerType.Custom:
        of.write('    layer.custom_proc_ptr = &custom_callback_{0};\n'.format(layer.node_in._param.custom_param[2]))
        of.write('    layer.custom_param = &custom_param;\n')
    if layer.is_output:
        of.write('    output_layers[{0}] = &layer;\n'.format(output_index))
        output_index += 1
    of.write('}}//end of  Layer_{0}\n\n'.format(n))


class FPGARun:
    def __init__(self):
        self.conv = None
        self.pool = None


class LayerType(IntEnum):
    Input = auto()
    Convolution = auto()
    InnerProduct = auto()
    Flatten = auto()
    Concatenate = auto()
    SoftMax = auto()
    Custom = auto()
    Other = auto()

    def __str__(self):
        return self.name


class FPGALayer:
    def __init__(self, nodes):
        self.type = LayerType.Other
        self.run = []
        self.node_in = nodes[0]
        self.node_out = nodes[-1]
        self.output_addr_offset = 0
        self.is_output = False
        self.layer_in = []

        concat_node = None
        # append runs
        run = FPGARun()
        for node in nodes:
            if node._type in (NodeType.Convolution, NodeType.LRN):
                if run.conv:
                    self.run.append(run)
                    run = FPGARun()
                run.conv = node
            elif node._type is NodeType.Pooling:
                if concat_node:
                    for prev_run in self.run:
                        if prev_run.conv in concat_node._input_nodes:
                            prev_run.pool = node
                else:
                    if run.conv and run.conv not in node._input_nodes:
                        self.run.append(run)
                        run = FPGARun()
                    run.pool = node
                    self.run.append(run)
                    run = FPGARun()
            elif node._type in (NodeType.UpSampling, NodeType.Eltwise,
                                NodeType.Power):
                if run.conv:
                    self.run.append(run)
                    run = FPGARun()
                run.pool = node
                self.run.append(run)
                run = FPGARun()
            elif node._type is NodeType.Input:
                self.type = LayerType.Input
            elif node._type is NodeType.InnerProduct:
                self.type = LayerType.InnerProduct
            elif node._type is NodeType.Flatten:
                self.type = LayerType.Flatten
            elif node._type is NodeType.Concat:
                concat_node = node
                if (run.conv or run.pool):
                    self.run.append(run)
                    run = FPGARun()
                if len(self.run) == 0:
                    self.type = LayerType.Concatenate
            elif node._type is NodeType.SoftMax:
                self.type = LayerType.SoftMax
            elif node._type is NodeType.Custom:
                self.type = LayerType.Custom
        if run.conv or run.pool:
            self.run.append(run)

        if len(self.run) > 0:
            self.type = LayerType.Convolution

        # determine layer parameters
        topo = 0
        max_tiles = 0
        for i, run in enumerate(self.run):
            node_out = run.conv
            if (run.pool):
                node_out = run.pool
            if (node_out == self.node_out or
                    (concat_node and node_out in concat_node._input_nodes)):
                topo |= (1 << i)
            if run.conv is not None:
                tiles = calc_conv_tiles(run.conv)
                max_tiles = max(tiles, max_tiles)
            if run.pool is not None:
                tiles = calc_pool_tiles(run.pool)
                max_tiles = max(tiles, max_tiles)
        if len(self.run) > 0 and topo == 0:
            topo = 1
        self.topo = topo
        self.tiles = max_tiles


class FPGANetwork:
    def __init__(self, net: cnn_layer.Network=None, quantization=True):
        self._layer = []
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
            self.custom_layer_config = net._custom_layer
            self.convert_network(net)

    def convert_network(self, net: cnn_layer.Network) -> None:
        tl = net._traverse_list
        converted_node = []
        layer_start_index = -1
        end_index = len(tl)
        index = 0

        while index < end_index:
            ignore = False
            node = tl[index]
            if index > 0:
                prev_node_type = tl[index - 1]._type
            else:
                prev_node_type = None
            if node._type in (NodeType.Convolution, NodeType.LRN):
                pass
            elif node._type is NodeType.Pooling:
                # Test if the pool node
                # can merge with previous convolution node
                if (prev_node_type == NodeType.Convolution and
                        node._param.pool == 0 and
                        calc_conv_tiles(tl[index - 1]) == 1 and
                        calc_pool_tiles(node) == 1):
                    index += 1
                    converted_node.append(node)
                    continue
            elif node._type in (NodeType.UpSampling, NodeType.Power):
                pass
            elif (node._type is NodeType.Concat and
                  (node._param.axis == 0 or
                   len(node._input_dim) == 3 and node._param.axis == 2)):
                pass
            elif node._type is NodeType.Eltwise:
                pass
            elif node._type is NodeType.InnerProduct:
                pass
            elif node._type is NodeType.Flatten:
                dim = node._input_dim
                if (dim[-1] <= 8 or
                        (len(dim) == 3 and dim[0] == 1 and dim[1] == 1)):
                    index += 1
                    converted_node.append(node)
                    continue
            elif node._type is NodeType.Reshape:
                index += 1
                converted_node.append(node)
                continue
            elif node._type is NodeType.Custom:
                pass
            elif node._type is NodeType.Input:
                pass
            elif node._type is NodeType.SoftMax:
                pass
            else:
                ignore = True

            if (len(converted_node) > 0 and
                    not set(node._input_nodes).issubset(set(converted_node))):
                ignore = True

            if layer_start_index != -1:
                layer = FPGALayer(tl[layer_start_index:index])
                self._layer.append(layer)
                layer_start_index = -1
            if not ignore:
                converted_node.append(node)
                layer_start_index = index

            index += 1
            # handle branch, try to merge simple braches into single layer
            can_merge = True
            while len(node._output_nodes) > 1 and can_merge:
                if layer_start_index != -1:
                    layer = FPGALayer(tl[layer_start_index:index])
                    self._layer.append(layer)
                    layer_start_index = -1
                # find concat node index
                concat_index = index
                # stop at concat node or another branch node
                while (tl[concat_index]._type != NodeType.Concat and
                       len(tl[concat_index]._output_nodes) == 1):
                    concat_index += 1
                # test if find a concat node, if found another branch node,
                # this is not a simple branch
                if (tl[concat_index]._type == NodeType.Concat and
                        tl[concat_index]._param.axis ==
                        len(tl[concat_index]._output_dim) - 1):
                    # make sure all branching paths are merged to this node
                    # and run depth of each path <= 2
                    for node_out in node._output_nodes:
                        run_depth = 0
                        while run_depth < 3 and node_out != tl[concat_index]:
                            run_depth += 1
                            if node_out._type is NodeType.Convolution:
                                node_out = node_out._output_nodes[0]
                                if (node_out._type is not NodeType.Pooling or
                                        node_out._param.pool != 0):
                                    continue
                            elif node_out._type is NodeType.Pooling:
                                pass
                            else:
                                run_depth = 100
                            node_out = node_out._output_nodes[0]
                        if run_depth > 2:
                            can_merge = False
                            break
                    # test if all channels of input nodes are dividable by 8
                    if can_merge:
                        for node_in in tl[concat_index]._input_nodes:
                            if node_in._output_dim[2] % 8 != 0:
                                can_merge = False
                                break
                    if can_merge:
                        # handle specil case: pool node immediately after
                        # the concat node can be merge into the same layer
                        node_next = tl[concat_index + 1]
                        if (node_next._type is NodeType.Pooling and
                            node_next._param.pool == 0 and
                            all(t._type is NodeType.Convolution
                                for t in tl[concat_index]._input_nodes)):
                            concat_index += 1
                        converted_node.extend(tl[index:concat_index + 1])
                        layer = FPGALayer(tl[index:concat_index + 1])
                        self._layer.append(layer)
                        index = concat_index + 1
                        node = tl[concat_index]
                else:
                    break

        # append the last layer
        if layer_start_index != -1:
            layer = FPGALayer(tl[layer_start_index:index])
            self._layer.append(layer)

        if self._layer[-1].node_out is net._traverse_list[-1]:
            self._layer[-1].is_output = True
        else:
            output_nodes = [net._traverse_list[-1]]
            i = 0
            while i < len(output_nodes):
                node = output_nodes[i]
                i += 1
                for node_in in node._input_nodes:
                    out_node_reached = False
                    if node_in in converted_node:
                        for layer in reversed(self._layer):
                            if layer.node_out is node_in:
                                out_node_reached = True
                                layer.is_output = True
                                break
                    if not out_node_reached:
                        output_nodes.append(node_in)

        self.connect_layers()

        # remove unnecessary layers
        for layer in self._layer[:]:
            if layer.type is LayerType.Input:
                self._layer.remove(layer)

    def connect_layers(self):
        """Set layer output addresses."""
        logging.info('Converted layer info')
        logging.info("{:22s} {:22s} {:12s} {:5s} {:18s} {:8s} {:s}".format(
            'Input Node', 'Output Node', 'Node Type', 'Range',
            'Output Dimension', 'Addr', 'Size'))

        class LayerLiveRange:
            def __init__(self, layer, index):
                self.layer = layer
                self.birth_index = index
                self.death_index = index
                self.output_concat_lr = None
                self.allocated = False
        live_ranges = []
        weight_size = 0
        for index, layer in enumerate(self._layer):
            if layer.type is LayerType.Convolution:
                self.num_conv_layers += 1
                for run in layer.run:
                    weight_size += get_weight_size(run.conv, self.quantization)
            elif layer.type is LayerType.InnerProduct:
                self.num_fc_layers += 1
                weight_size += get_fc_weight_size(layer.node_in)
            if layer.is_output:
                self.num_output_layers += 1
                self.output_layer.append(layer)
            lr = LayerLiveRange(layer, index)
            node_in = layer.node_in
            for lr_in in live_ranges:
                if lr_in.layer.node_out in node_in._input_nodes:
                    lr.layer.layer_in.append(lr_in.layer)
                    if lr_in.death_index < index:
                        lr_in.death_index = index
            if lr.layer.is_output:
                lr.death_index = len(self._layer) - 1
            live_ranges.append(lr)

        # handle concat layer
        for index, lr in enumerate(live_ranges):
            if lr.layer.type is LayerType.Concatenate:
                for prev_lr in live_ranges[:index]:
                    if prev_lr.layer.node_out in lr.layer.node_in._input_nodes:
                        prev_lr.output_concat_lr = lr
                        prev_lr.death_index = lr.death_index

        current_live_ranges = []
        allocated_size = 0

        for index, lr in enumerate(live_ranges):
            necessary_size = make_align_size(lr.layer.node_out._output_size)
            if lr.output_concat_lr:
                offset = 0
                for node in lr.output_concat_lr.layer.node_in._input_nodes:
                    if node == lr.layer.node_out:
                        break
                    offset += node._output_size
                if lr.output_concat_lr.allocated:
                    lr.allocated = True
                    lr.layer.output_addr_offset = (
                        lr.output_concat_lr.layer.output_addr_offset + offset)
                else:
                    necessary_size = make_align_size(
                        lr.output_concat_lr.layer.node_out._output_size)

            # update current live ranges
            for clr in current_live_ranges[:]:
                if clr.death_index < index:
                    current_live_ranges.remove(clr)

            if not lr.allocated:
                # find if can re-use empty spaces in current live ranges
                layer_allocated = False
                current_offset = 0
                for i, clr in enumerate(current_live_ranges):
                    if (clr.layer.output_addr_offset - current_offset >=
                            necessary_size):
                        layer_allocated = True
                        lr.layer.output_addr_offset = current_offset
                        if lr.output_concat_lr:
                            current_live_ranges.insert(i, lr.output_concat_lr)
                        else:
                            current_live_ranges.insert(i, lr)
                        break
                    else:
                        increment_size = make_align_size(
                            clr.layer.node_out._output_size)
                        if clr.output_concat_lr:
                            increment_size = make_align_size(
                                clr.output_concat_lr.layer.node_out._output_size)
                        current_offset = (clr.layer.output_addr_offset +
                                          increment_size)

                # if not, put it in the end of current buffer
                if not layer_allocated:
                    if lr.output_concat_lr:
                        current_live_ranges.append(lr.output_concat_lr)
                    else:
                        current_live_ranges.append(lr)
                    lr.layer.output_addr_offset = current_offset
                    if current_offset + necessary_size > allocated_size:
                        allocated_size = current_offset + necessary_size

                lr.allocated = True
                if lr.output_concat_lr and not lr.output_concat_lr.allocated:
                    lr.output_concat_lr.allocated = True
                    lr.output_concat_lr.layer.output_addr_offset = lr.layer.output_addr_offset
                    lr.layer.output_addr_offset += offset

            logging.info("{:22s} {:22s} {:12s} {:02d} {:02d} {:18s} {:08X} {:08X}{:s}".format(
                lr.layer.node_in._name, lr.layer.node_out._name,
                str(lr.layer.type),
                lr.birth_index, lr.death_index,
                str(lr.layer.node_out._output_dim),
                lr.layer.output_addr_offset,
                lr.layer.node_out._output_size,
                ('*' if lr.layer.is_output else '')))
        logging.info('allocated size:{:d}'.format(allocated_size))
        self.weight_size = weight_size
        self.buffer_size = allocated_size

    def output_header(self, of, name) -> None:
        gen_header_header(of, name, self.custom_layer_config)
        for n, layer in enumerate(self._layer):
            gen_header_layer(of, n, layer, self.quantization)
        gen_header_footer(of, name)

    def output_source(self, of, name) -> None:
        global weight_offset
        global output_index
        global conv_index
        global fc_index
        global is_tensorflow
        weight_offset = 0
        output_index = 0
        conv_index = 0
        fc_index = 0
        is_tensorflow = self.tensorflow_backend
        gen_source_header(of, name, self)
        for n, layer in enumerate(self._layer):
            gen_source_layer(of, name, n, layer, self.quantization)

    def output_weights(self, of) -> None:
        prev_node = None
        for layer in self._layer:
            if layer.type is LayerType.Convolution:
                for run in layer.run:
                    if (run.conv is not None and
                            run.conv._type is NodeType.Convolution):
                        pack_weight(run.conv, of, self.quantization)
            elif layer.type is LayerType.InnerProduct:
                pack_fc_weight(layer.node_in, prev_node, of)
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
