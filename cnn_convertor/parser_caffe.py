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
from cnn_convertor import cnn_layer, caffe_pb2, cnn_exception
from google.protobuf import text_format
import numpy as np
NodeType = cnn_layer.NodeType


def get_tuple(param):
    try:
        if len(param) == 0:
            return 0, 0
        elif len(param) == 1:
            return param[0], param[0]
        else:
            return param[0], param[1]
    except TypeError:
        return param, param


def get_pad(param):
    try:
        if len(param) == 0:
            return 0, 0, 0, 0
        if len(param) == 1:
            return param[0], param[0], param[0], param[0]
        if len(param) == 2:
            return param[0], param[0], param[1], param[1]
        if len(param) == 4:
            return param[0], param[1], param[2], param[3]
        raise ValueError("Unsupported pad=%s" % param)
    except TypeError:
        assert int(param) == param
        return param, param, param, param


def parse_caffe_def2(network: cnn_layer.Network, netdef: str):
    type_map = {
        'Convolution': NodeType.Convolution,
        'InnerProduct': NodeType.InnerProduct,
        'Scale': NodeType.Scale,
        'BatchNorm': NodeType.BatchNorm,
        'LRN': NodeType.LRN,
        'Concat': NodeType.Concat,
        'Eltwise': NodeType.Eltwise,
        'Pooling': NodeType.Pooling,
        'Upsample': NodeType.UpSampling,
        'Power': NodeType.Power,
        'ReLU': NodeType.ReLU,
        'PReLU': NodeType.PReLU,
        'TanH': NodeType.TanH,
        'ELU': NodeType.ELU,
        'Sigmoid': NodeType.Sigmoid,
        'Input': NodeType.Input,
        'Data': NodeType.Data,
        'Dropout': NodeType.DropOut,
        'Softmax': NodeType.SoftMax,
        'Flatten': NodeType.Flatten,
        'Reshape': NodeType.Reshape,
    }

    caffe_net = caffe_pb2.NetParameter()
    try:
        text_format.Parse(netdef, caffe_net)
    except Exception as e:
        logging.exception(
            "Exception occurred while parsing Input network: %s", e)
        raise

    top_map = {}
    # Handle fixed size input
    if len(caffe_net.input) == 1:
        node = cnn_layer.LayerNode(caffe_net.input[0], NodeType.Input)
        network.append_input_node(node)
        if len(caffe_net.input_dim) == 4:
            dim = (caffe_net.input_dim[3],
                   caffe_net.input_dim[2],
                   caffe_net.input_dim[1])
        elif len(caffe_net.input_shape) == 1:
            dim = (caffe_net.input_shape[0].dim[3],
                   caffe_net.input_shape[0].dim[2],
                   caffe_net.input_shape[0].dim[1])
        node.set_input_dim(dim)
        node.set_output_dim(dim)
        network.debug_node = caffe_net
        top_map[caffe_net.input[0]] = node
    # Handle each layer node
    parsed_nodes = []
    for i, layer in enumerate(caffe_net.layer):
        logging.debug('Handling layer %d, Name: %s, Type %s',
                      i, layer.name, layer.type)
        if layer.type not in type_map:
            logging.exception('Encountered unsupported layer format %s.',
                              layer.type)
            raise cnn_exception.ParseError('Unsupported type:' + layer.type)

        node_type = type_map[layer.type]
        if node_type in (NodeType.ReLU, NodeType.TanH, NodeType.ELU,
                         NodeType.Sigmoid, NodeType.BatchNorm, NodeType.Scale,
                         NodeType.PReLU):
            node = cnn_layer.LayerNode(layer.name, node_type)
            up_node = top_map[layer.bottom[0]]
            if node_type == NodeType.BatchNorm:
                up_node.set_bn_node(node)
            elif node_type == NodeType.Scale:
                up_node.set_scale_node(node)
            else:
                if node_type == NodeType.PReLU:
                    pass
                if node_type == NodeType.ReLU:
                    param = cnn_layer.NodeParam()
                    param.relu_param = layer.relu_param.negative_slope
                    node.set_param(param)
                up_node.set_activation_node(node)
            # If this is not in-place layer, add the up_node to the top_map
            # Using the output label of this layer too
            if layer.top[0] != layer.bottom[0]:
                top_map[layer.top[0]] = up_node
            continue
        elif node_type in (NodeType.DropOut, NodeType.Data):
            # Ignore data and drop out layer
            continue

        input_nodes = []

        # search for exsisting input and output nodes
        for label in layer.bottom:
            if label in top_map:
                input_nodes.append(top_map[label])
        node = cnn_layer.LayerNode(layer.name, node_type, input_nodes)
        parsed_nodes.append(node)

        # add this node to top_map and bottom_map
        for label in layer.top:
            if label in top_map:
                raise cnn_exception.ParseError(
                    "Ill-formed layer. name: %s" % layer.name)
            top_map[label] = node

        if node_type == NodeType.Input:
            network.append_input_node(node)
            dim = (layer.input_param.shape[0].dim[3],
                   layer.input_param.shape[0].dim[2],
                   layer.input_param.shape[0].dim[1])
            node.set_input_dim(dim)
            node.set_output_dim(dim)
            network.debug_node = caffe_net
        elif node_type == NodeType.Convolution:
            param = cnn_layer.NodeParam()
            param.num_output = int(layer.convolution_param.num_output)
            param.kernel_size = get_tuple(layer.convolution_param.kernel_size)
            param.pad_lrtb = get_pad(layer.convolution_param.pad)
            param.stride = get_tuple(layer.convolution_param.stride)
            param.group = int(layer.convolution_param.group)
            node.set_param(param)
        elif node_type == NodeType.Pooling:
            param = cnn_layer.NodeParam()
            param.pool = int(layer.pooling_param.pool)
            param.kernel_size = get_tuple(layer.pooling_param.kernel_size)
            param.pad_lrtb = get_pad(layer.pooling_param.pad)
            param.stride = get_tuple(layer.pooling_param.stride)
            param.is_global = layer.pooling_param.global_pooling
            node.set_param(param)
        elif node_type == NodeType.UpSampling:
            param = cnn_layer.NodeParam()
            param.kernel_size = 2, 2
            node.set_param(param)
        elif node_type == NodeType.Power:
            if layer.power_param.power != 1 or layer.power_param.shift != 0:
                raise ValueError(
                    "Power layer is supported only with "
                    "power = 1 and shift = 0, got %s and %s" %
                    (layer.power_param.power, layer.power_param.shift))
            param = cnn_layer.NodeParam()
            param.scale = float(layer.power_param.scale)
            node.set_param(param)
        elif node_type == NodeType.InnerProduct:
            param = cnn_layer.NodeParam()
            param.num_output = int(layer.inner_product_param.num_output)
            node.set_param(param)
        elif node_type == NodeType.LRN:
            param = cnn_layer.NodeParam()
            node.set_param(param)
        elif node_type == NodeType.Reshape:
            param = cnn_layer.NodeParam()
            dims = layer.reshape_param.shape.dim
            param.reshape_param = (dims[3], dims[2], dims[1])
            node.set_param(param)

    for node in parsed_nodes:
        if len(node.output_nodes) == 0:
            network.append_output_node(node)


def parse_caffe_def(
    network: cnn_layer.Network,
    network_def: str
):
    logging.info('Parsing Caffe network definitions.')
    with open(network_def, 'r') as fin:
        parse_caffe_def2(network, fin.read())


def search_caffe_layer(layers, name):
    caffe_layer = [x for x in layers if x.name == name]
    if len(caffe_layer) == 0:
        logging.exception('Layer with name:' + name +
                          ' not found in input data network.')
        raise cnn_exception.ParseError('Layer ' + name +
                                       ' not found')
    return caffe_layer[0]


def parse_caffe_data(
    network: cnn_layer.Network,
    network_data: str
):
    logging.info('Parsing Caffe network data.')
    caffe_net = caffe_pb2.NetParameter()
    try:
        with open(network_data, 'rb') as fin:
            caffe_net.ParseFromString(fin.read())
    except Exception as e:
        logging.exception(
            "Exception occurred while parsing Input network: %s", e)
        raise

    network.debug_node = caffe_net
    if len(caffe_net.layer) != 0:
        layers = caffe_net.layer
    else:
        layers = caffe_net.layers

    for layer in network.traverse_list:
        if (layer.type is not NodeType.Convolution and
                layer.type is not NodeType.InnerProduct):
            continue
        caffe_layer = search_caffe_layer(layers, layer.name)
        weight = np.float32(caffe_layer.blobs[0].data)
        if len(caffe_layer.blobs) > 1:
            bias = np.float32(caffe_layer.blobs[1].data)
        else:
            bias = np.zeros((layer.output_dim[2]))
        layer.set_weight_bias(weight, bias)

        # set parameters for BatchNorm node and Scale node
        if layer.bn_node:
            caffe_layer = search_caffe_layer(layers, layer.bn_node.name)
            mean = np.float32(caffe_layer.blobs[0].data)
            var = np.float32(caffe_layer.blobs[1].data)
            scale = caffe_layer.blobs[2].data[0]
            scale = (0 if scale == 0.0 else 1.0 / scale)
            mean *= scale
            var *= scale
            layer.bn_node.set_mean_var(mean, var)
        if layer.sc_node:
            caffe_layer = search_caffe_layer(layers, layer.sc_node.name)
            weight = np.float32(caffe_layer.blobs[0].data)
            bias = np.float32(caffe_layer.blobs[1].data)
            layer.sc_node.set_weight_bias(weight, bias)
