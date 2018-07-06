# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
 Copyright(c) 2017 by Digital Media Professionals Inc.
 All rights reserved.
------------------------------------------------------------
"""
import logging
from cnn_convertor import cnn_layer, parser_caffe, parser_keras

def parse_network(
        network_def: str,
        network_data: str,
        network_type: str,
        custom_layer: list, 
        debug
    ) -> cnn_layer.Network:
    network = cnn_layer.Network(custom_layer)
    
    logging.info('Start parsing. Network type:' + network_type)
    if network_type == 'CAFFE':
        parser_caffe.parse_caffe_def(network, network_def)
        network.build_traverse_list()
        network.calc_inout_sizes()
        parser_caffe.parse_caffe_data(network, network_data)
    elif network_type == 'KERAS':
        parser_keras.parse_keras_network(network, network_data)
        network.build_traverse_list()
        network.calc_inout_sizes()
    return network
    