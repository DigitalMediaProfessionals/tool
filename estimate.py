#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
 Copyright(c) 2017 by Digital Media Professionals Inc.
 All rights reserved.
------------------------------------------------------------
"""
"""
Estimates network performance from data stored in DB.
"""
import sys
if (sys.version_info.major < 3 or
        (sys.version_info.major == 3 and sys.version_info.minor < 6)):
    raise ValueError("python3.6 or greater is required")

import argparse
import logging
import numpy as np
import os

from cnn_convertor import fpga_layer, cnn_layer, parser_caffe, parser_keras
from cnn_convertor.fpga_layer import \
    LayerType, NodeType, get_weight_size, get_fc_weight_size


def patch_path():
    thisdir = os.path.dirname(__file__)
    if not len(thisdir) or thisdir == ".":
        path = ".."
    else:
        updir = os.path.dirname(thisdir)
        path = updir if len(updir) else "."
    if path not in sys.path:
        sys.path.insert(0, path)


patch_path()
from perf_cache import PerfCache, ConvConf, FCConf, SMConf


class Result(object):
    @property
    def time_ms(self):
        raise NotImplementedError()


class ResultZero(Result):
    @property
    def time_ms(self):
        return 0.0


class ResultUnknown(Result):
    @property
    def time_ms(self):
        return None


class ResultConf(Result):
    def __init__(self):
        super(ResultConf, self).__init__()
        self._time_ms = None
        self._succeeded = None
        self._error_message = None
        self._ts_last_run = None
        self.conf = None

    def estimate(self, cache, conf):
        """Estimates execution time from DB cache.
        """
        found = False
        if not cache.fetch_info(conf):
            logging.debug("Record not found, adding the request")
            cache.add_request(conf)
        elif conf.succeeded:
            found = True
            time_ms = conf.time_ms
        if not found:
            time_ms = cache.estimate_time_ms(conf)

        self.time_ms = time_ms
        self.succeeded = conf.succeeded
        self.error_message = conf.error_message
        self.ts_last_run = conf.ts_last_run

        self.conf = conf

    @property
    def time_ms(self):
        return self._time_ms

    @time_ms.setter
    def time_ms(self, value):
        self._time_ms = value

    @property
    def succeeded(self):
        return self._succeeded

    @succeeded.setter
    def succeeded(self, value):
        self._succeeded = value

    @property
    def error_message(self):
        return self._error_message

    @error_message.setter
    def error_message(self, value):
        self._error_message = value

    @property
    def ts_last_run(self):
        return self._ts_last_run

    @ts_last_run.setter
    def ts_last_run(self, value):
        self._ts_last_run = value


class ResultConv(ResultConf):
    pass


class ResultFC(ResultConf):
    pass


class ResultSM(ResultConf):
    pass


def estimate_conv_run(run, layer, cache, quantization):
    """Estimates performance of convolutional layer
    treating each run as a separate layer.
    """
    result = ResultConv()
    conf = ConvConf()

    conf.topo = layer.topo
    conf.w = layer.node_in._input_dim[0]
    conf.h = layer.node_in._input_dim[1]
    conf.z = 1
    conf.c = layer.node_in._input_dim[2]
    conf.input_circular_offset = 0
    conf.input_tiles = layer.tiles
    conf.output_mode = 0
    is_conv = run.conv is not None and run.conv._type is NodeType.Convolution
    conf.p = (run.conv._param.kernel_size[0] if is_conv else 1)
    conf.pz = 1
    if is_conv:
        conf.conv_enable = (1 if run.conv._param.group <= 1 else 3)
    else:
        conf.conv_enable = 0
    conf.conv_pad[0] = run.conv._param.pad[0] if is_conv else 0
    conf.conv_pad[1] = run.conv._param.pad[0] if is_conv else 0
    conf.conv_pad[2] = run.conv._param.pad[1] if is_conv else 0
    conf.conv_pad[3] = run.conv._param.pad[1] if is_conv else 0
    conf.conv_stride[0] = run.conv._param.stride[0] if is_conv else 1
    conf.conv_stride[1] = run.conv._param.stride[1] if is_conv else 1
    pool_enable = (1 if run.pool else 0)
    conf.pool_size[0] = run.pool._param.kernel_size[0] if run.pool else 0
    conf.pool_size[1] = run.pool._param.kernel_size[1] if run.pool else 0
    conf.pool_stride[0] = run.pool._param.stride[0] if run.pool else 1
    conf.pool_stride[1] = run.pool._param.stride[1] if run.pool else 1
    conf.pool_pad[0] = run.pool._param.pad[0] if run.pool else 0
    conf.pool_pad[1] = run.pool._param.pad[0] if run.pool else 0
    conf.pool_pad[2] = run.pool._param.pad[1] if run.pool else 0
    conf.pool_pad[3] = run.pool._param.pad[1] if run.pool else 0
    actfunc = 0
    actfunc_param = 0
    pool_avg_param = 0
    node_in = run.pool
    if run.conv is not None:
        node_in = run.conv
        if node_in._act_node:
            actfunc = (1 if node_in._act_node._type is NodeType.TanH else 2)
            actfunc_param = node_in._act_node._param.relu_param
    node_out = run.conv
    if run.pool is not None:
        node_out = run.pool
        if node_out._type is NodeType.Pooling and node_out._param.pool != 0:
            pool_enable = 2
            pool_avg_param = 1.0 / (run.pool._param.kernel_size[0] *
                                    run.pool._param.kernel_size[1])
        if node_out._type is NodeType.UpSampling:
            pool_enable = 4
    conf.pool_enable = pool_enable
    conf.actfunc = actfunc
    conf.actfunc_param = actfunc_param
    conf.pool_avg_param = pool_avg_param
    conf.conv_dilation[0] = 0
    conf.conv_dilation[1] = 0
    conf.m = node_out._output_dim[2]
    conf.weight_fmt = ((3 if quantization else 1) if is_conv else 0)
    conf.rectifi_en = 0
    conf.input_size = conf.w * conf.h * conf.c * 2
    conf.output_size = (node_out._output_dim[0] * node_out._output_dim[1] *
                        node_out._output_dim[2] * 2)
    conf.weights_size = get_weight_size(run.conv, quantization)

    # Detect if this is the case of pool node being merged into concat node
    if node_in != node_out and node_in not in node_out._input_nodes:
        conf.m = node_in._output_dim[2]

    result.estimate(cache, conf)

    return result


def estimate_conv(layer, cache, quantization):
    """Estimates performance of convolutional layer.
    """
    res = []
    for run in layer.run:
        res.append(estimate_conv_run(run, layer, cache, quantization))
    chain_info = None
    if len(layer.run) > 1:
        chain_info = cache.estimate_conv_chain(list(r.conf for r in res))
    return chain_info, res


def estimate_fc(layer, cache, quantization):
    """Estimates performance of fully connected layer.
    """
    if not quantization:
        logging.warning(
            "Treating non-quantized weight as quantized in execution time "
            "estimation for Fully Connected layer")

    result = ResultFC()
    conf = FCConf()

    node = layer.node_in
    if len(node._input_dim) == 3:
        w, h, c = node._input_dim
    elif len(node._input_dim) == 1:
        w, h, c = 1, 1, node._input_dim[0]
    m = node._output_dim[0]
    actfunc = 0
    actfunc_param = 0
    if node._act_node:
        if node._act_node._type == NodeType.ReLU:
            actfunc = 0x10
            if node._act_node._param.relu_param != 0.0:
                actfunc = 0x30
                actfunc_param = node._act_node._param.relu_param
        else:
            actfunc = 0x20

    conf.input_size = w * h * c
    conf.output_size = m
    conf.param_fmt = 1
    conf.actfunc = actfunc
    conf.actfunc_param = actfunc_param
    conf.weights_size = get_fc_weight_size(node)

    result.estimate(cache, conf)

    return None, [result]


def estimate_sm(layer, cache):
    """Estimates performance of softmax layer.
    """
    result = ResultSM()
    conf = SMConf()

    axis = layer.node_in._param.axis
    if axis < 0:
        axis = len(layer.node_in._input_dim) + axis
    conf.size = layer.node_in._input_dim[axis]

    result.estimate(cache, conf)

    # Adjust time as DB stores only timing
    # for single dimension softmax application
    if result.time_ms is not None:
        result.time_ms *= (np.prod(layer.node_in._input_dim) /
                           layer.node_in._input_dim[axis])

    return None, [result]


def estimate_network(net_def: str, net_type):
    quantization = True

    network = cnn_layer.Network({})
    if net_type == 'CAFFE':
        parser_caffe.parse_caffe_def2(network, net_def)
    elif net_type == 'KERAS':
        parser_keras.parse_keras_network2(network, net_def, None)
    network.build_traverse_list()
    network.calc_inout_sizes()
    fpga_net = fpga_layer.FPGANetwork(network, quantization)

    results = []

    cache = PerfCache()

    for layer in fpga_net._layer:
        if layer.type is LayerType.Convolution:
            logging.debug("%s: Convolution", layer)
            res = estimate_conv(layer, cache, fpga_net.quantization)
        elif layer.type is LayerType.InnerProduct:
            logging.debug("%s: Fully Connected", layer)
            res = estimate_fc(layer, cache, fpga_net.quantization)
        elif layer.type is LayerType.Input:
            logging.debug("%s: Input: time is 0", layer)
            res = None, [ResultZero()]
        elif layer.type is LayerType.Concatenate:
            logging.debug("%s: Concatenate: time is 0", layer)
            res = None, [ResultZero()]
        elif layer.type is LayerType.Flatten:
            logging.debug("%s: Flatten: time is 0", layer)
            res = None, [ResultZero()]
        elif layer.type is LayerType.SoftMax:
            logging.debug("%s: SoftMax", layer)
            res = estimate_sm(layer, cache)
        else:
            logging.debug("%s: Custom", layer)
            res = None, [ResultUnknown()]
        results.append(res)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="DNN execution time on FPGA estimate test")
    parser.add_argument("NETWORK_DEF", type=str, help="network definition")
    args = parser.parse_args()
    path = os.path.abspath(args.NETWORK_DEF)
    _root, ext = os.path.splitext(path)
    with open(path, "r") as of:
        net_def = of.read()
    # Guess network framework
    if ext == "json":
        net_type = "KERAS"
    elif ext == ".prototxt":
        net_type = "CAFFE"
    else:
        # Guess from file content
        if net_def[0] == '{':
            net_type = "KERAS"
        else:
            net_type = "CAFFE"
    results = estimate_network(net_def, net_type)
    total_time = 0.0
    n_estimated = 0
    n_total = 0
    for group, res in results:
        if group is not None and group.succeeded:
            n_total += 1
            dt = group.time_ms
            print(dt)
            if dt is not None:
                total_time += dt
                n_estimated += 1
        else:
            for r in res:
                n_total += 1
                dt = r.time_ms
                print(dt)
                if dt is not None:
                    total_time += dt
                    n_estimated += 1

    print("Estimated for %d layers out of %d (%.0f%%)" %
          (n_estimated, n_total, 100.0 * n_estimated / n_total))
    if n_estimated == n_total:
        print("Time is %.1f msec" % total_time)
    else:
        print("Time is greater than %.1f msec" % total_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
