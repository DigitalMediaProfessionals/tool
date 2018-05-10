#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
Estimates network performance from data stored in DB.
"""
import sys
if (sys.version_info.major < 3 or
        (sys.version_info.major == 3 and sys.version_info.minor < 6)):
    raise ValueError("python3.6 or greater is required")

import argparse
import logging
import os

from cnn_convertor import fpga_layer, cnn_layer, parser_caffe, parser_keras
from cnn_convertor.fpga_layer import LayerType


def patch_path():
    head = __file__
    for _i in range(2):
        head, _tail = os.path.split(head)
        if not len(head):
            head = ".."
            break
    if head not in sys.path:
        sys.path.insert(0, head)


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


def estimate_conv(layer, cache):
    """Estimates performance of convolutional layer.
    """
    result = ResultConv()

    conf = ConvConf()
    conf.w = layer.node_in._input_dim[0]
    conf.h = layer.node_in._input_dim[1]
    conf.z = 1
    conf.c = layer.node_in._input_dim[2]
    # TODO: confinue from here.

    time_ms = cache.estimate_time_ms(conf)
    result.time_ms = time_ms
    result.succeeded = conf.succeeded
    result.error_message = conf.error_message
    result.ts_last_run = conf.ts_last_run

    return result


def estimate_fc(layer, cache):
    """Estimates performance of fully connected layer.
    """
    result = ResultFC()
    return result


def estimate_sm(layer, cache):
    """Estimates performance of softmax layer.
    """
    result = ResultSM()
    return result


def estimate_network(net_def: str, net_type):
    network = cnn_layer.Network({})
    if net_type == 'CAFFE':
        parser_caffe.parse_caffe_def2(network, net_def)
    elif net_type == 'KERAS':
        parser_keras.parse_keras_network2(network, net_def, None)
    network.build_traverse_list()
    network.calc_inout_sizes()
    fpga_net = fpga_layer.FPGANetwork(network, False)

    results = []

    cache = PerfCache()

    for layer in fpga_net._layer:
        if layer.type is LayerType.Convolution:
            logging.debug("%s: Convolution", layer)
            result = estimate_conv(layer, cache)
        elif layer.type is LayerType.InnerProduct:
            logging.debug("%s: Fully Connected", layer)
            result = estimate_fc(layer, cache)
        elif layer.type is LayerType.Input:
            logging.debug("%s: Input: time is 0", layer)
            result = ResultZero()
        elif layer.type is LayerType.Concatenate:
            logging.debug("%s: Concatenate: time is 0", layer)
            result = ResultZero()
        elif layer.type is LayerType.Flatten:
            logging.debug("%s: Flatten: time is 0", layer)
            result = ResultZero()
        elif layer.type is LayerType.SoftMax:
            logging.debug("%s: SoftMax", layer)
            result = estimate_sm(layer, cache)
        else:
            logging.debug("%s: Custom", layer)
            result = ResultUnknown()
        results.append((layer, result))

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
        # guess from file content
        if net_def[0] == '{':
            net_type = "KERAS"
        else:
            net_type = "CAFFE"
    results = estimate_network(net_def, net_type)
    print(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
