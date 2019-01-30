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


def set_inplace_node(node, config):
    activation = config['activation']
    input_node = None
    if activation == 'relu':
        node_type = NodeType.ReLU
    elif activation == 'tanh':
        node_type = NodeType.TanH
    elif activation == 'sigmoid':
        node_type = NodeType.Sigmoid
    elif activation == 'elu':
        node_type = NodeType.ELU
    elif activation == 'softmax':
        node_type = NodeType.SoftMax
        input_node = node
    in_node = cnn_layer.LayerNode(node.name + '_' + activation,
                                  node_type, input_node)
    if input_node is None:
        node.act_node = in_node
        return None
    else:
        return in_node


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
            if len(shape) == 3:
                w = np.transpose(w, (2, 1, 0))
            if len(shape) == 2:
                w = np.transpose(w)
            w = w.reshape((-1,))
        else:
            w = None
        weights.append(w)
    return weights


def parse_keras_network2(network, net_def, netweight, need_flip=False):
    type_map = {
        'Conv1D': NodeType.Convolution,
        'Conv2D': NodeType.Convolution,
        'DepthwiseConv1D': NodeType.Convolution,
        'DepthwiseConv2D': NodeType.Convolution,
        'SeparableConv1D': NodeType.Convolution,
        'SeparableConv2D': NodeType.Convolution,
        'Dense': NodeType.InnerProduct,
        'LRN': NodeType.LRN,
        'Concatenate': NodeType.Concat,
        'Add': NodeType.Eltwise,
        'MaxPooling1D': NodeType.Pooling,
        'MaxPooling2D': NodeType.Pooling,
        'AveragePooling1D': NodeType.Pooling,
        'AveragePooling2D': NodeType.Pooling,
        'GlobalMaxPooling1D': NodeType.Pooling,
        'GlobalAveragePooling1D': NodeType.Pooling,
        'GlobalMaxPooling2D': NodeType.Pooling,
        'GlobalAveragePooling2D': NodeType.Pooling,
        'UpSampling1D': NodeType.UpSampling,
        'UpSampling2D': NodeType.UpSampling,
        'InputLayer': NodeType.Input,
        'Softmax': NodeType.SoftMax,
        'Flatten': NodeType.Flatten,
        'Reshape': NodeType.Reshape,
        'Dropout': NodeType.DropOut,
        'ZeroPadding1D': NodeType.Padding,
        'ZeroPadding2D': NodeType.Padding,
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

    node_map = OrderedDict()
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
            network.append_input_node(node)
            shape = config['batch_input_shape']
            # handle FC only model
            if len(shape) == 2:
                dim = (shape[1],)
            # handle 1D input dimensions
            elif len(shape) == 3:
                if is_channel_first:
                    dim = (shape[2], 1, shape[1])
                else:
                    dim = (shape[1], 1, shape[2])
            else:
                if is_channel_first:
                    dim = (shape[3], shape[2], shape[1])
                else:
                    dim = (shape[2], shape[1], shape[3])
            node.input_dim = dim
            node_map[""] = node

        if is_sequential:
            input_nodes = node_map.values()[-1]
        else:
            # search for exsisting input and output nodes
            input_nodes = []
            if len(layer['inbound_nodes']) > 0:
                inbound_nodes = [x[0] for x in layer['inbound_nodes'][0]]
                for label in inbound_nodes:
                    if label in node_map:
                        input_nodes.append(node_map[label])

        if (layer_type == 'Dropout'):
            top_map[layer_name] = cnn_layer.LayerNode(layer_name,
                                                      NodeType.DropOut,
                                                      input_nodes)
            continue
        elif layer_type == 'Activation':
            activation = config['activation']
            if activation == 'softmax':
                node_type = NodeType.SoftMax
            else:
                if activation == 'relu':
                    node_type = NodeType.ReLU
                elif activation == 'tanh':
                    node_type = NodeType.TanH
                elif activation == 'sigmoid':
                    node_type = NodeType.Sigmoid
                elif activation == "elu":
                    node_type = NodeType.ELU
                elif activation == 'relu6':
                    node_type = NodeType.ReLU6
                node = cnn_layer.LayerNode(layer_name, node_type, input_nodes)
                node_map[layer_name] = node
                continue
        elif layer_type == 'ReLU':
            if 'threshold' in config and config['threshold'] != 0.0:
                logging.error('ReLU layer can not support \'threshold\' != 0.')
                raise cnn_exception.ParseError('Unsupported Layer')
            if ('max_value' in config and config['max_value'] == 6 and
                    ('negative_slope' not in config or
                     config['negative_slope'] == 0)):
                node_type = NodeType.ReLU6
            elif 'max_value' not in config or config['max_value'] is None:
                node_type = NodeType.ReLU
            else:
                logging.error('ReLU layer with unsupported parameters.')
                raise cnn_exception.ParseError('Unsupported Layer')
            node = cnn_layer.LayerNode(layer_name, node_type, input_nodes)
            if 'negative_slope' in config:
                param = cnn_layer.NodeParam()
                param.relu_param = config['negative_slope']
                node.param = param
            node_map[layer_name] = node
            continue
        elif layer_type == 'LeakyReLU':
            node = cnn_layer.LayerNode(layer_name, NodeType.ReLU, input_nodes)
            param = cnn_layer.NodeParam()
            param.relu_param = config['alpha']
            node.param = param
            node_map[layer_name] = node
            continue
        elif layer_type == 'PReLU':
            # check if it is using shared axis for width and height axes
            if 'shared_axes' in config:
                shared_axes = config['shared_axes']
            else:
                shared_axes = None
            if (is_channel_first and shared_axes == [2, 3] or
                    not is_channel_first and shared_axes == [1, 2]):
                node = cnn_layer.LayerNode(layer_name, NodeType.PReLU,
                                           input_nodes)
                if netweight is not None:
                    weights = get_weights(netweight, layer_name, need_flip,
                                          ['alpha'])
                    node.set_weight_bias(weights[0], None)
                node_map[layer_name] = node
            else:
                logging.error('PReLU layer must set its shared_axes to'
                              'width and height axes.')
                raise cnn_exception.ParseError('Unsupported Layer')
            continue
        elif layer_type == 'BatchNormalization':
            node = cnn_layer.LayerNode(layer_name, NodeType.BatchNorm,
                                       input_nodes)
            if netweight is not None:
                weights = get_weights(netweight, layer_name, need_flip,
                                      ['gamma', 'beta',
                                       'moving_mean', 'moving_variance'])
                node.set_mean_var(weights[2], weights[3])
                node.set_weight_bias(weights[0], weights[1])
            node_map[layer_name] = node
            continue
        elif layer_type == 'ZeroPadding1D':
            # there is no padding layer if caffe so just extract padding info
            pad = config['padding']
            try:
                if len(pad) == 2:
                    if not all(int(x) == x for x in pad):
                        raise ValueError(
                            "Unsupported Keras ZeroPadding1D: %s" % pad)
                    padding = [int(pad[0]), int(pad[1]), 0, 0]
                elif len(pad) == 1:
                    if int(pad[0]) != pad[0]:
                        raise ValueError(
                            "Unsupported Keras ZeroPadding1D: %s" % pad)
                    padding = [int(pad[0])] * 4
                elif len(pad) == 0:
                    padding = [0, 0, 0, 0]
                else:
                    raise ValueError(
                        "Unsupported Keras ZeroPadding2D: %s" % pad)
            except TypeError:
                if int(pad) != pad:
                    raise ValueError(
                        "Unsupported Keras ZeroPadding2D: %s" % pad)
                padding = [int(pad)] * 4
            node = cnn_layer.LayerNode(layer_name, NodeType.Padding,
                                       input_nodes)
            param = cnn_layer.NodeParam()
            param.pad_lrtb = padding
            node.param = param
            node_map[layer_name] = node
            continue
        elif layer_type == 'ZeroPadding2D':
            # there is no padding layer if caffe so just extract padding info
            pad = config['padding']
            try:
                if len(pad) == 4:
                    if not all(int(x) == x for x in pad):
                        raise ValueError(
                            "Unsupported Keras ZeroPadding2D: %s" % pad)
                    padding = list(int(x) for x in pad)
                elif len(pad) == 2:
                    padding = [0, 0, 0, 0]
                    try:
                        if len(pad[0]) == 2:
                            padding[0] = pad[0][0]
                            padding[1] = pad[0][1]
                        elif len(pad[0]) == 1:
                            padding[0] = pad[0][0]
                            padding[1] = pad[0][0]
                        elif len(pad[0]) == 0:
                            padding[0] = 0
                            padding[1] = 0
                        else:
                            raise ValueError(
                                "Unsupported Keras ZeroPadding2D: %s" % pad)
                    except TypeError:
                        if int(pad[0]) != pad[0]:
                            raise ValueError(
                                "Unsupported Keras ZeroPadding2D: %s" % pad)
                        padding[0] = int(pad[0])
                        padding[1] = int(pad[0])
                    try:
                        if len(pad[1]) == 2:
                            padding[2] = pad[1][0]
                            padding[3] = pad[1][1]
                        elif len(pad[1]) == 1:
                            padding[2] = pad[1][0]
                            padding[3] = pad[1][0]
                        elif len(pad[1]) == 0:
                            padding[2] = 0
                            padding[3] = 0
                        else:
                            raise ValueError(
                                "Unsupported Keras ZeroPadding2D: %s" % pad)
                    except TypeError:
                        if int(pad[1]) != pad[1]:
                            raise ValueError(
                                "Unsupported Keras ZeroPadding2D: %s" % pad)
                        padding[2] = int(pad[1])
                        padding[3] = int(pad[1])
                elif len(pad) == 1:
                    if int(pad[0]) != pad[0]:
                        raise ValueError(
                            "Unsupported Keras ZeroPadding2D: %s" % pad)
                    padding = [0, 0, 0, 0]
                elif len(pad) == 0:
                    padding = [0, 0, 0, 0]
                else:
                    raise ValueError(
                        "Unsupported Keras ZeroPadding2D: %s" % pad)
            except TypeError:
                if int(pad) != pad:
                    raise ValueError(
                        "Unsupported Keras ZeroPadding2D: %s" % pad)
                padding = [int(pad), int(pad), int(pad), int(pad)]
            node = cnn_layer.LayerNode(layer_name, NodeType.Padding,
                                       input_nodes)
            param = cnn_layer.NodeParam()
            param.pad_lrtb = padding
            node.param = param
            node_map[layer_name] = node
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
                raise cnn_exception.ParseError(
                        'Unknown layer, Name: %s, Type: %s',
                        layer_name, layer_type)
            node_type = type_map[layer_type]
        node = cnn_layer.LayerNode(layer_name, node_type, input_nodes)

        # add this node to node_map
        if layer_name in node_map:
            raise cnn_exception.ParseError(
                'Ill-formed layer. name:' + layer_name)
        node_map[layer_name] = node

        if node_type == NodeType.Input:
            network.append_input_node(node)
            shape = config['batch_input_shape']
            # handle 1D input dimensions
            if len(shape) == 3:
                if is_channel_first:
                    dim = (shape[2], 1, shape[1])
                else:
                    dim = (shape[1], 1, shape[2])
            else:
                if is_channel_first:
                    dim = (shape[3], shape[2], shape[1])
                else:
                    dim = (shape[2], shape[1], shape[3])
            node.input_dim = dim
        elif node_type == NodeType.Convolution:
            is_1D = (layer_type[-2] == '1')
            param = cnn_layer.NodeParam()
            # For keras, output is not set for depthwise convolution
            # skip setting num_output and set it when calculating in_out sizes
            if (layer_type[:-2] != 'DepthwiseConv' and
                    layer_type[:-2] != 'SeparableConv'):
                param.num_output = config['filters']
            if is_1D:
                param.kernel_size = (config['kernel_size'][0], 1)
            else:
                param.kernel_size = (config['kernel_size'][1],
                                     config['kernel_size'][0])
            param.keras_padding = config['padding']
            param.dilation = config['dilation_rate']
            if is_1D:
                param.stride = (config['strides'][0], 1)
            else:
                param.stride = (config['strides'][1], config['strides'][0])
            if (layer_type[:-2] == 'DepthwiseConv' or
                    layer_type[:-2] == 'SeparableConv'):
                param.group = config['depth_multiplier']
                if param.group > 1:
                    logging.error('Depthwise/Separable Convolution with'
                                  '\'depth_multiplier\' is not supported.')
                    raise cnn_exception.ParseError('Unsupported param')
            node.param = param

            if layer_type[:-2] == 'SeparableConv':
                point_node = cnn_layer.LayerNode(layer_name + '_point',
                                                 node_type, node)
                node_map[layer_name] = point_node
                param = cnn_layer.NodeParam()
                param.num_output = config['filters']
                point_node.param = param

            if config['activation'] != 'linear':
                inplace_node = set_inplace_node(node_map.values()[-1], config)
                if inplace_node:
                    node_map[inplace_node.name] = inplace_node
            if netweight is not None:
                if layer_type[:-2] == 'SeparableConv':
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
            if layer_type in ('MaxPooling1D', 'MaxPooling2D',
                              'GlobalMaxPooling1D', 'GlobalMaxPooling2D'):
                param.pool = 0
            else:
                param.pool = 1
            if layer_type in ('MaxPooling1D', 'AveragePooling1D'):
                param.kernel_size = (config['pool_size'][0], 1)
                param.keras_padding = config['padding']
                param.stride = (config['strides'][0], 1)
            elif layer_type in ('MaxPooling2D', 'AveragePooling2D'):
                param.kernel_size = (config['pool_size'][1],
                                     config['pool_size'][0])
                param.keras_padding = config['padding']
                param.stride = (config['strides'][1], config['strides'][0])
            else:
                param.is_global = True
            node.param = param
        elif node_type == NodeType.UpSampling:
            is_1D = (layer_type[-2] == '1')
            param = cnn_layer.NodeParam()
            if is_1D:
                param.kernel_size = (config['size'][0], 1)
            else:
                param.kernel_size = (config['size'][1], config['size'][0])
            node.param = param
        elif node_type == NodeType.InnerProduct:
            param = cnn_layer.NodeParam()
            param.num_output = config['units']
            node.param = param
            if config['activation'] != 'linear':
                inplace_node = set_inplace_node(node, config)
                if inplace_node:
                    node_map[inplace_node.name] = inplace_node
            if netweight is not None:
                weights = get_weights(netweight, layer_name, need_flip,
                                      ['kernel', 'bias'])
                node.set_weight_bias(weights[0], weights[1])
        elif node_type == NodeType.Reshape:
            param = cnn_layer.NodeParam()
            param.reshape_param = tuple(config['target_shape'])
            node.param = param
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
            node.param = param
        elif node_type == NodeType.SoftMax:
            param = cnn_layer.NodeParam()
            if 'axis' in config:
                param.axis = config['axis']
            else:
                param.axis = -1
            node.param = param
        elif node_type == NodeType.Custom:
            param = cnn_layer.NodeParam()
            custom_config = network.custom_layer[layer_type]
            custom_param = (
                OrderedDict({x: config[x] for x in custom_config[0]}),
                custom_config[1], layer_type)
            param.custom_param = custom_param
            node.param = param

    _set_node_output(node_map)
    _manipulate_node_graph(node_map)
    for node in node_map.values():
        if len(node.output_nodes) == 0:
            network.append_output_node(node)


def _set_node_output(node_map):
    nodes = list(node_map.values())
    finished = []
    while nodes:
        node = nodes.pop()
        for in_n in node.input_nodes:
            in_n.output_nodes.append(node)
            if in_n not in finished and in_n not in nodes:
                nodes.append(in_n)
        finished.append(node)


def _manipulate_node_graph(node_map):
    act_types = (
            NodeType.ReLU,
            NodeType.PReLU,
            NodeType.TanH,
            NodeType.ELU,
            NodeType.Sigmoid,
            NodeType.ReLU6,)

    def _replace_node(old, new=None):
        """
        If new is None, just remove `old`
        """
        def _aux(t, old, new, old_list):
            """
            @param t Target list
            """
            index = t.index(old)
            if new is None:
                _l = t[:]
                t.clear()
                t.extend(_l[:index] + old_list + _l[index + 1:])
            else:
                t[index] = new

        if new is not None:
            new.input_nodes = old.input_nodes[:]
            new.output_nodes = old.output_nodes[:]

        for _out in old.output_nodes:
            _aux(_out.input_nodes, old, new, old.input_nodes)
        for _in in old.input_nodes:
            _aux(_in.output_nodes, old, new, old.output_nodes)

    def _create_dummy_conv_node(node):
        # dummy convolution
        node = cnn_layer.LayerNode(node.name, NodeType.Convolution,
                                   node.input_nodes)
        param = cnn_layer.NodeParam()
        # For keras, output is not set for depthwise convolution
        # skip setting num_output and set it
        # when calculating in_out sizes
        param.kernel_size = (1, 1)
        param.pad_lrtb = 0, 0, 0, 0
        param.keras_padding = "same"
        param.stride = (1, 1)
        param.group = 1
        node.param = param
        return node

    for node in node_map.values():
        if node.type is NodeType.BatchNorm:
            # Merge BatchNormalization to previous Convolution
            assert(len(node.input_nodes) == 1)
            create_dummy = (not
                            (node.input_nodes[0].type is NodeType.Convolution
                             and len(node.input_nodes[0].output_nodes) == 1))
            if create_dummy:
                base_node = _create_dummy_conv_node(node)
            else:
                base_node = node.input_nodes[0]

            bn_node = cnn_layer.LayerNode(node.name, NodeType.BatchNorm)
            bn_node.set_mean_var(node.mean, node.var)
            sc_node = cnn_layer.LayerNode(node.name, NodeType.Scale)
            sc_node.set_weight_bias(node.weight, node.bias)
            base_node.bn_node = bn_node
            base_node.sc_node = sc_node
            _replace_node(node, base_node if create_dummy else None)

        elif node.type in act_types:
            # Merge Activation to previous Convolution
            assert(len(node.input_nodes) == 1)
            _in = node.input_nodes[0]
            create_dummy = not (_in.type in [NodeType.Convolution,
                                             NodeType.InnerProduct]
                                and len(_in.output_nodes) == 1)
            if create_dummy:
                base_node = _create_dummy_conv_node(node)
            else:
                base_node = node.input_nodes[0]

            base_node.act_node = node
            _replace_node(node, base_node if create_dummy else None)

        elif node.type is NodeType.DropOut:
            # Ignore Dropout
            assert(len(node.input_nodes) == 1)
            _replace_node(node, None)
        elif node.type is NodeType.Padding:
            assert(len(node.input_nodes) == 1)
            for _out in node.output_nodes:
                if _out.type is not NodeType.Convolution:
                    raise cnn_exception.ParseError(
                        "Padding Layer '{}' must be followed by Convolution"
                        .format(node.name))
                _out.param.pad_lrtb = node.param.pad_lrtb
            _replace_node(node)


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
