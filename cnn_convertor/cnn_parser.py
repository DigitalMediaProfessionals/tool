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
import logging
from cnn_convertor import cnn_layer, parser_caffe, parser_keras


def parse_network(

    network_def: str,
    network_data: str,
    network_type: str,
    custom_layer: list,
    dim_override: tuple
) -> cnn_layer.Network:
    network = cnn_layer.Network(custom_layer, dim_override)

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
