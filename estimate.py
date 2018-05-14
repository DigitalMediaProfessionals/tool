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
    thisdir = os.path.dirname(__file__)
    if not len(thisdir):
        thisdir = "."
    path = "%s/../db" % thisdir
    if path not in sys.path:
        sys.path.insert(0, path)


patch_path()
from cache import PerfCache, ConvConf, FCConf, SMConf


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


class ResultConv(Result):
    def __init__(self):
        super(ResultConv, self).__init__()
        self._time_ms = 0.0

    @property
    def time_ms(self):
        return self._time_ms


def estimate_conv(layer):
    """Estimates performance of convolutional layer.
    """
    result = ResultConv()

    # TODO: continue from here.

    return result


def estimate_fc(layer):
    """Estimates performance of fully connected layer.
    """
    result = ResultUnknown()
    return result


def estimate_sm(layer):
    """Estimates performance of softmax layer.
    """
    result = ResultUnknown()
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

    for layer in fpga_net._layer:
        if layer.type is LayerType.Convolution:
            logging.debug("%s: Convolution", layer)
            result = estimate_conv(layer)
        elif layer.type is LayerType.InnerProduct:
            logging.debug("%s: Fully Connected", layer)
            result = estimate_fc(layer)
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
            result = estimate_sm(layer)
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
