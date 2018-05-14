# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:22:16 2018

@author: zonghong.lyu
"""

from cnn_convertor import fpga_layer, cnn_layer, parser_caffe, parser_keras

def estimate_network(net_def: str, net_type):
    network = cnn_layer.Network({})
    if net_type == 'CAFFE':
        parser_caffe.parse_caffe_def2(network, net_def)
    elif net_type == 'KERAS':
        parser_keras.parse_keras_network2(network, net_def, None)
    network.build_traverse_list()
    network.calc_inout_sizes()
    fpga_net = fpga_layer.FPGANetwork(network, False)
    # TODO: implement estimation of each layer, return the converted net for now
    return fpga_net

if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description="DNN execution time on FPGA estimate test")
    parser.add_argument("NETWORK_DEF", type=str, help="network definition")
    args = parser.parse_args()
    path = os.path.abspath(args.NETWORK_DEF)
    root, ext = os.path.splitext(path)
    with open(path, "r") as of:
        net_def = of.read()
    # guess network framework
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
    net = estimate_network(net_def, net_type)
    import pdb
    pdb.set_trace()
