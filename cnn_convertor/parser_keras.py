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
from collections import OrderedDict
from cnn_convertor import cnn_layer, cnn_exception
import h5py
import json
import numpy as np



NodeType = cnn_layer.NodeType
padding = (0, 0)


def get_padding():
    global padding
    ret = tuple(padding)
    padding = (0, 0)
    return ret


def set_inplace_node(node, config):
    activation = config['activation']
    input_node = None
    if activation == 'relu':
        node_type = NodeType.ReLU
    elif activation == 'tanh':
        node_type = NodeType.TanH
    elif activation == 'softmax':
        node_type = NodeType.SoftMax
        input_node = node
    in_node = cnn_layer.LayerNode(node.name + '_' + activation,
                                  node_type, input_node)
    if input_node is None:
        node.set_activation_node(in_node)


def get_weights(netweight, layer_name, need_flip, weight_entry):
    weights = []
    weight_names = []
    for i in range(len(weight_entry)):
        weight_names.append(None)
    wg = netweight[layer_name]
    for wn in wg.attrs['weight_names']:
        for i, entry in enumerate(weight_entry):
            if entry in wn.decode('utf8')[len(layer_name):]:
                weight_names[i] = wn
                break
        if i == len(weight_entry):
            # This entry is not found, assert
            logging.error('weight entry %s is not parsed!', wn.decode('utf8'))
    for name in weight_names:
        if name is not None:
            shape = wg[name].shape
            w = wg[name][()]
            if len(shape) == 4:
                w = np.transpose(w, (3, 2, 0, 1))
                if need_flip:
                    new_shape = (w.shape[0], w.shape[1], -1)
                    w = np.flip(w.reshape(new_shape), 2)
            if len(shape) == 2:
                w = np.transpose(w)
            w = w.reshape((-1,))
        else:
            w = None
        weights.append(w)
    return weights


def parse_keras_network2(network, net_def, netweight, need_flip=False):
    type_map = {
        'Conv2D': NodeType.Convolution,
        'DepthwiseConv2D': NodeType.Convolution,
        'SeparableConv2D': NodeType.Convolution,
        'Dense': NodeType.InnerProduct,
        'LRN': NodeType.LRN,
        'Concatenate': NodeType.Concat,
        'Add': NodeType.Eltwise,
        'MaxPooling2D': NodeType.Pooling,
        'AveragePooling2D': NodeType.Pooling,
        'GlobalMaxPooling2D': NodeType.Pooling,
        'GlobalAveragePooling2D': NodeType.Pooling,
        'UpSampling2D': NodeType.UpSampling,
        'InputLayer': NodeType.Input,
        'Softmax': NodeType.SoftMax,
        'Flatten': NodeType.Flatten,
        'Reshape': NodeType.Reshape,
    }
    netdef = json.loads(net_def)
    is_sequential = (netdef['class_name'] == 'Sequential')
    if type(netdef['config']) is list:
        layers = netdef['config']
    else:
        layers = netdef['config']['layers']
    network.debug_node = netweight

    # get data_format parameter
    for layer in layers:
        if 'data_format' in layer['config']:
            if layer['config']['data_format'] == 'channels_first':
                is_channel_first = True
            else:
                is_channel_first = False
            break

    top_map = {}
    prev_node = None
    # Handle each layer node
    for i, layer in enumerate(layers):
        layer_type = layer['class_name']
        config = layer['config']
        layer_name = config['name']
        logging.debug('Handling layer %d, Name: %s, Type %s',
                      i, layer_name, layer_type)

        # if the first node is not input node, create a dummy input node
        if i == 0 and layer_type != 'InputLayer':
            node = cnn_layer.LayerNode('Input', NodeType.Input, None)
            prev_node = node
            network.append_input_node(node)
            shape = config['batch_input_shape']
            if is_channel_first:
                dim = (shape[3], shape[2], shape[1])
            else:
                dim = (shape[2], shape[1], shape[3])
            node.set_input_dim(dim)
            node.set_output_dim(dim)
        if is_sequential:
            input_nodes = prev_node
            up_node = prev_node
        else:
            # search for exsisting input and output nodes
            input_nodes = []
            if len(layer['inbound_nodes']) > 0:
                inbound_nodes = [x[0] for x in layer['inbound_nodes'][0]]
                for label in inbound_nodes:
                    if label in top_map:
                        input_nodes.append(top_map[label])
                up_node = input_nodes[0]

        if (layer_type == 'Dropout'):
            top_map[layer_name] = up_node
            continue
        elif layer_type == 'Activation':
            activation = config['activation']
            if activation == 'softmax':
                node_type = NodeType.SoftMax
            else:
                if activation == 'relu' or activation == 'relu6':
                    node_type = NodeType.ReLU
                elif activation == 'tanh':
                    node_type = NodeType.TanH
                node = cnn_layer.LayerNode(layer_name, node_type)
                up_node.set_activation_node(node)
                top_map[layer_name] = up_node
                continue
        elif layer_type == 'LeakyReLU':
            node = cnn_layer.LayerNode(layer_name, NodeType.ReLU)
            param = cnn_layer.NodeParam()
            param.relu_param = config['alpha']
            node.set_param(param)
            up_node.set_activation_node(node)
            top_map[layer_name] = up_node
            continue
        elif layer_type == 'BatchNormalization':
            # handle case that the up_node is not a convolution node
            if up_node is None or up_node.type is not NodeType.Convolution:
                up_node = cnn_layer.LayerNode(layer_name, NodeType.Convolution,
                                              input_nodes)
                param = cnn_layer.NodeParam()
                # For keras, output is not set for depthwise convolution
                # skip setting num_output and set it
                # when calculating in_out sizes
                param.kernel_size = (1, 1)
                param.pad = (0, 0)
                param.keras_padding = 'same'
                param.stride = (1, 1)
                param.group = 1
                up_node.set_param(param)
                if netweight is not None:
                    up_node.set_weight_bias(None, None)
                prev_node = up_node
            if netweight is not None:
                weights = get_weights(netweight, layer_name, need_flip,
                                      ['gamma', 'beta',
                                       'moving_mean', 'moving_variance'])
            node = cnn_layer.LayerNode(layer_name, NodeType.BatchNorm)
            up_node.set_bn_node(node)
            if netweight is not None:
                node.set_mean_var(weights[2], weights[3])
            node = cnn_layer.LayerNode(layer_name, NodeType.Scale)
            up_node.set_scale_node(node)
            if netweight is not None:
                node.set_weight_bias(weights[0], weights[1])
            top_map[layer_name] = up_node
            continue
        elif layer_type == 'ZeroPadding2D':
            # there is no padding layer if caffe so just extract padding info
            global padding
            pad = config['padding']
            if type(pad) is not list:
                padding = (pad, pad)
            elif type(pad[0]) is not list:
                padding = (pad[1], pad[0])
            else:
                padding = (pad[0][0], pad[1][0])
            top_map[layer_name] = up_node
            continue
        elif layer_type == 'Merge':
            mode = config['mode']
            if mode == 'concat':
                node_type = NodeType.Concat
            elif mode == 'add':
                node_type = NodeType.Eltwise
        elif layer_type == 'Layer' and 'batch_input_shape' in config:
            node_type = NodeType.Input
        elif layer_type in network.custom_layer:
            custom_config = network.custom_layer[layer_type]
            # set parameter type if it is the first time
            if type(custom_config[0]) is list:
                c_type_map = {int: 'int', bool: 'bool', float: 'float'}
                param_list = OrderedDict()
                for param_name in custom_config[0]:
                    type_0 = type(config[param_name])
                    if type_0 is list:
                        type_1 = type(config[param_name][0])
                        list_size = len(config[param_name])
                        c_type = '{:s}[{:d}]'.format(c_type_map[type_1],
                                                     list_size)
                    elif type_0 in c_type_map:
                        c_type = c_type_map[type_0]
                    param_list[param_name] = c_type
                custom_config = (param_list, custom_config[1])
                network.custom_layer[layer_type] = custom_config
            node_type = NodeType.Custom
        else:
            if layer_type not in type_map:
                logging.warning('Ignoring unknown layer, Name: %s, Type: %s',
                                layer_name, layer_type)
                top_map[layer_name] = up_node
                continue
            node_type = type_map[layer_type]
        node = cnn_layer.LayerNode(layer_name, node_type, input_nodes)
        prev_node = node

        # add this node to top_map
        if layer_name in top_map:
            raise cnn_exception.ParseError(
                'Ill-formed layer. name:' + layer_name)
        top_map[layer_name] = node

        if node_type == NodeType.Input:
            network.append_input_node(node)
            shape = config['batch_input_shape']
            if is_channel_first:
                dim = (shape[3], shape[2], shape[1])
            else:
                dim = (shape[2], shape[1], shape[3])
            node.set_input_dim(dim)
            node.set_output_dim(dim)
        elif node_type == NodeType.Convolution:
            param = cnn_layer.NodeParam()
            # For keras, output is not set for depthwise convolution
            # skip setting num_output and set it when calculating in_out sizes
            if (layer_type != 'DepthwiseConv2D' and
                    layer_type != 'SeparableConv2D'):
                param.num_output = config['filters']
            param.kernel_size = tuple(config['kernel_size'])
            param.pad = get_padding()
            param.keras_padding = config['padding']
            param.stride = tuple(config['strides'])
            if (layer_type == 'DepthwiseConv2D' or
                    layer_type == 'SeparableConv2D'):
                param.group = config['depth_multiplier']
                if param.group > 1:
                    logging.error('Depthwise/Separable Convolution with'
                                  '\'depth_multiplier\' is not supported.')
                    raise cnn_exception.ParseError('Unsupported param')
            node.set_param(param)

            if layer_type == 'SeparableConv2D':
                point_node = cnn_layer.LayerNode(layer_name + '_point',
                                                 node_type, node)
                prev_node = point_node
                top_map[layer_name] = point_node
                param = cnn_layer.NodeParam()
                param.num_output = config['filters']
                point_node.set_param(param)

            if config['activation'] != 'linear':
                set_inplace_node(prev_node, config)
            if netweight is not None:
                if layer_type == 'SeparableConv2D':
                    weights = get_weights(netweight, layer_name, need_flip,
                                          ['depthwise_kernel',
                                           'pointwise_kernel', 'bias'])
                    node.set_weight_bias(weights[0], None)
                    point_node.set_weight_bias(weights[1], weights[2])
                else:
                    weights = get_weights(netweight, layer_name, need_flip,
                                          ['kernel', 'bias'])
                    node.set_weight_bias(weights[0], weights[1])
        elif node_type == NodeType.Pooling:
            param = cnn_layer.NodeParam()
            if layer_type in ('MaxPooling2D', 'GlobalMaxPooling2D'):
                param.pool = 0
            else:
                param.pool = 1
            if layer_type in ('MaxPooling2D', 'AveragePooling2D'):
                param.kernel_size = tuple(config['pool_size'])
                param.pad = get_padding()
                param.keras_padding = config['padding']
                param.stride = tuple(config['strides'])
            else:
                param.is_global = True
            node.set_param(param)
        elif node_type == NodeType.UpSampling:
            param = cnn_layer.NodeParam()
            param.kernel_size = tuple(config['size'])
            node.set_param(param)
        elif node_type == NodeType.InnerProduct:
            param = cnn_layer.NodeParam()
            param.num_output = config['units']
            node.set_param(param)
            if config['activation'] != 'linear':
                set_inplace_node(node, config)
            if netweight is not None:
                weights = get_weights(netweight, layer_name, need_flip,
                                      ['kernel', 'bias'])
                node.set_weight_bias(weights[0], weights[1])
        elif node_type == NodeType.Reshape:
            param = cnn_layer.NodeParam()
            param.reshape_param = tuple(config['target_shape'])
            node.set_param(param)
        elif node_type == NodeType.Concat:
            param = cnn_layer.NodeParam()
            if 'axis' in config:
                param.axis = config['axis']
            elif 'concat_axis' in config:
                param.axis = config['concat_axis']
            if param.axis > 0:
                param.axis -= 1
            if is_channel_first and param.axis >= 0:
                param.axis = 2 - param.axis
            node.set_param(param)
        elif node_type == NodeType.SoftMax:
            param = cnn_layer.NodeParam()
            if 'axis' in config:
                param.axis = config['axis']
            else:
                param.axis = -1
            node.set_param(param)
        elif node_type == NodeType.Custom:
            param = cnn_layer.NodeParam()
            custom_config = network.custom_layer[layer_type]
            custom_param = (
                OrderedDict({x: config[x] for x in custom_config[0]}),
                custom_config[1], layer_type)
            param.custom_param = custom_param
            node.set_param(param)
    network.set_output_node(prev_node)


def parse_keras_network(network, network_data):
    logging.info('Parsing Keras network.')

    try:
        keras_net = h5py.File(network_data, 'r')
    except Exception as e:
        logging.exception(
            'Exception occurred while opening Input network: %s', e)
        raise

    version = keras_net.attrs['keras_version'].decode('utf-8')
    major_version = int(version[:version.find('.')])
    if major_version < 2:
        logging.error('Keras version (< 2.0.0) not supported.')
        raise cnn_exception.ParseError('Unsupported Keras version')
    backend = keras_net.attrs['backend'].decode('utf-8')
    if backend == 'theano':
        need_flip = True
    elif backend == 'tensorflow':
        need_flip = False
        network.tensorflow_backend = True
    else:
        logging.error('Keras backend not supported.')
        raise cnn_exception.ParseError('Unsupported Keras backend')

    netdef = keras_net.attrs['model_config']
    netweight = keras_net['model_weights']
    parse_keras_network2(network, netdef, netweight, need_flip)
