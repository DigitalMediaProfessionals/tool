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
import math
import logging
from cnn_convertor import cnn_exception
from typing import List, Union, Tuple
from enum import IntEnum, auto


class NodeType(IntEnum):
    NonSupported = auto()
    Convolution = auto()
    InnerProduct = auto()
    Scale = auto()
    BatchNorm = auto()
    LRN = auto()
    Concat = auto()
    Eltwise = auto()
    Pooling = auto()
    UpSampling = auto()
    Power = auto()
    ReLU = auto()
    TanH = auto()
    Input = auto()
    Data = auto()
    DropOut = auto()
    SoftMax = auto()
    Flatten = auto()
    Reshape = auto()
    Custom = auto()
    ELU = auto()
    Sigmoid = auto()

    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)

    def __str__(self):
        return self.name


class NodeParam(object):
    def __init__(self):
        self.num_output = 0
        self.kernel_size = (1, 1)
        self.pad = (0, 0)
        self.keras_padding = None
        self.stride = (1, 1)
        self.pool = 0  # 0:max, 1:avg
        self.group = 1
        self.relu_param = 0.0
        self.is_global = False
        self.reshape_param = ()
        self.axis = -1
        self.custom_param = None
        self.scale = 1.0

    def __repr__(self):
        ret = ("[num_output:%d, kernel_size:%s, pad:%s, stride:%s, pool:%d, "
               "group:%d, relu_param:%f, is_global:%d]")
        return ret % (self.num_output, self.kernel_size, self.pad,
                      self.stride, self.pool, self.group, self.relu_param,
                      self.is_global)


class LayerNode(object):
    """Represent one layer node in a CNN.
    """

    def __init__(
        self,
        name: str,
        node_type: NodeType,
        input_node: Union['LayerNode', List['LayerNode']]=None,
        output_node: Union['LayerNode', List['LayerNode']]=None,
    ) -> None:
        """Construct a LayerNode

        Note:
            The newly created node will be automatically appended to the
            output nodes of all input_node if it is not None, and also to the
            input nodes of all output_node if it is not None.

        Args:
            name: Name of the node.
            node_type: Type of the node.
            input_node: A single LayerNode if there is only one input,
                or a list of LayerNode if there are multiple input nodes.
            output_node:A single LayerNode if there is only one output,
                or a list of LayerNode if there are multiple output nodes.
        """
        self.name = name
        self.type = node_type
        if input_node is None:
            input_node = []
        elif type(input_node) is not list:
            input_node = [input_node]
        if output_node is None:
            output_node = []
        elif type(output_node) is not list:
            output_node = [output_node]
        self.input_nodes = input_node
        self.output_nodes = output_node
        for node in input_node:
            node.output_nodes.append(self)
        for node in output_node:
            node.input_nodes.append(self)
        self.bn_node = None
        self.sc_node = None
        self.act_node = None
        self.param = NodeParam()

    def __repr__(self):
        ret = "['%s', %s, _param:%s]"
        return ret % (
            self.name,
            self.type,
            str(self.param))

    def set_bn_node(self, node: 'LayerNode'):
        """ Set the batched normalization node for convolution node."""
        self.bn_node = node

    def set_scale_node(self, node: 'LayerNode'):
        """ Set the scale node for convolution node."""
        self.sc_node = node

    def set_activation_node(self, node: 'LayerNode'):
        """ Set the activation node for convolution Node."""
        self.act_node = node

    def set_input_dim(self, input_dim: Tuple[int]):
        self.input_dim = input_dim

    def set_output_dim(self, output_dim: Tuple[int]):
        self.output_dim = output_dim
        self.output_size = 2
        if self.type is NodeType.Custom or self.type is NodeType.SoftMax:
            self.output_size = 4
        for n in output_dim:
            self.output_size *= n

    def set_param(self, param: NodeParam):
        if param.kernel_size == (0, 0):
            param.kernel_size = (1, 1)
        if param.stride == (0, 0):
            param.stride = (1, 1)
        self.param = param

    def set_weight_bias(self, weight, bias):
        """ Set weight and bias blobs data for convolution or scale node."""
        self.weight = weight
        self.bias = bias

    def set_mean_var(self, mean, var):
        """ Set mean and variance blobs data for BatchNorm node."""
        self.mean = mean
        self.var = var


class Network(object):
    """Represents a CNN.
    """

    def __init__(self, custom_layer, dim_override) -> None:
        """Construct an empty CNN.
        """
        self.input_nodes = []
        self.output_node = None
        self.debug_node = None
        self.custom_layer = custom_layer
        self.dim_override = dim_override
        self.tensorflow_backend = False

    def build_traverse_list(self) -> None:
        pending = [self.output_node]
        tlist = []
        while len(pending) > 0:
            node = pending.pop()
            tlist.append(node)
            for in_n in node.input_nodes:
                can_append = True
                for out_n in in_n.output_nodes:
                    if out_n not in tlist:
                        can_append = False
                        break
                if can_append:
                    pending.append(in_n)
        tlist.reverse()
        self.traverse_list = tlist

    def append_input_node(self, node):
        self.input_nodes.append(node)

    def set_output_node(self, node):
        self.output_node = node

    def split_pool_node(self, node):
        # only handle case where kernel_size[0] and [1] are equal now
        candidates = [7, 6, 5, 4, 3, 2]
        insert_index = self.traverse_list.index(node) + 1
        k_size = node.param.kernel_size[0]
        split_size = 0
        for c in candidates:
            if k_size % c == 0:
                split_size = c
                break
        if split_size == 0:
            logging.exception('Handling node %s, pool size not supported',
                              node.name)
            raise cnn_exception.ConvertError('Pool size unsupported')
        remain_size = node.param.kernel_size[0] // split_size
        remain_dim = (node.input_dim[0] // split_size,
                      node.input_dim[1] // split_size,
                      node.input_dim[2])

        remain_node = LayerNode(node.name + '_' + str(remain_size),
                                NodeType.Pooling)
        remain_node.input_nodes = [node]
        remain_node.output_nodes = node.output_nodes
        for node_out in node.output_nodes:
            node_out.input_nodes = [remain_node]
        node.output_nodes = [remain_node]
        node.name = node.name + '_' + str(split_size)

        remain_node.input_dim = remain_dim
        remain_node.set_output_dim(node.output_dim)
        node.set_output_dim(remain_dim)

        param = NodeParam()
        param.pool = node.param.pool
        param.kernel_size = (remain_size, remain_size)
        param.pad = (0, 0)
        param.stride = node.param.stride
        remain_node.set_param(param)
        node.param.kernel_size = (split_size, split_size)
        node.param.stride = (split_size, split_size)

        self.traverse_list.insert(insert_index, remain_node)

        if remain_size > 7:
            self.split_pool_node(remain_node)

    def insert_flatten_node(self, node, dim):
        insert_index = self.traverse_list.index(node)
        flat_node = LayerNode(node.name + '_flatten', NodeType.Flatten)
        flat_node.input_nodes = node.input_nodes
        flat_node.output_nodes = [node]
        for node_in in node.input_nodes:
            node_in.output_nodes = [flat_node]
        node.input_nodes = [flat_node]
        flat_node.input_dim = dim
        size = 1
        for n in dim:
            size *= n
        flat_node.set_output_dim((size,))
        self.traverse_list.insert(insert_index, flat_node)
        return flat_node

    def calc_inout_sizes(self):
        def get_output_xy(dim: Tuple[int],
                          param: NodeParam,
                          is_pool: bool) -> Tuple[int]:
            w, h = dim[0], dim[1]
            w += param.pad[0] * 2
            h += param.pad[1] * 2
            if param.keras_padding == 'same':
                if w % param.stride[0] == 0:
                    pw = param.pad[0] + \
                        max(param.kernel_size[0] - param.stride[0], 0)
                else:
                    pw = param.pad[0] + \
                        max(param.kernel_size[0] - w % param.stride[0], 0)
                if h % param.stride[1] == 0:
                    ph = param.pad[1] + \
                        max(param.kernel_size[1] - param.stride[1], 0)
                else:
                    ph = param.pad[1] + \
                        max(param.kernel_size[1] - h % param.stride[1], 0)
                param.pad = ((pw + 1) // 2, (ph + 1) // 2)
                w += pw
                h += ph
            w = ((w - param.kernel_size[0]) / param.stride[0]) + 1
            h = ((h - param.kernel_size[1]) / param.stride[1]) + 1
            if is_pool:
                w = math.ceil(w)
                h = math.ceil(h)
                # adjust padding
                padx = param.pad[0]
                while ((dim[0] + padx - param.kernel_size[0])
                       % param.stride[0]) > padx:
                    padx += 1
                pady = param.pad[1]
                while ((dim[1] + pady - param.kernel_size[1])
                       % param.stride[1]) > pady:
                    pady += 1
                param.pad = (padx, pady)
            else:
                w = math.floor(w)
                h = math.floor(h)
            return w, h

        # Override input dimension if dim_override is given
        if self.dim_override:
            dim = self.dim_override
            for node in self.input_nodes:
                new_dim = (dim[0], dim[1], node.input_dim[2])
                node.set_input_dim(new_dim)
                node.set_output_dim(new_dim)

        tr_list = self.traverse_list[:]
        for node in tr_list:
            if node.type == NodeType.Input:
                continue
            # For Keras, DepthwiseConvolution node don't have output_size set.
            # Set it here
            dim = node.input_nodes[0].output_dim
            if node.param.num_output == 0:
                node.param.num_output = dim[-1] * node.param.group
                node.param.group = dim[-1]
            if node.type == NodeType.Convolution:
                node.set_input_dim(dim)
                if len(dim) == 3:
                    dim = (get_output_xy(dim, node.param, False) +
                           (node.param.num_output,))
                else:
                    raise cnn_exception.ConvertError('Invalid dimension')
                node.set_output_dim(dim)
            elif node.type == NodeType.InnerProduct:
                node.set_input_dim(dim)
                dim = (node.param.num_output,)
                node.set_output_dim(dim)
            elif node.type == NodeType.Pooling:
                node.set_input_dim(dim)
                if node.param.is_global:
                    node.param.kernel_size = (dim[0], dim[1])
                    dim = (1, 1, dim[2])
                else:
                    dim = get_output_xy(dim, node.param, True) + (dim[2],)
                node.set_output_dim(dim)
                # split pool node if kernel size > 7
                if (node.param.kernel_size[0] > 7 or
                        node.param.kernel_size[1] > 7):
                    self.split_pool_node(node)
            elif node.type == NodeType.UpSampling:
                node.set_input_dim(dim)
                dim = (dim[0] * node.param.kernel_size[0],
                       dim[1] * node.param.kernel_size[1], dim[2])
                node.set_output_dim(dim)
            elif node.type == NodeType.Power:
                node.set_input_dim(dim)
                node.set_output_dim(dim)
            elif node.type == NodeType.Concat:
                axis = node.param.axis
                c = 0
                for in_n in node.input_nodes:
                    c += in_n.output_dim[axis]
                temp_dim = list(dim)
                temp_dim[axis] = c
                dim = tuple(temp_dim)
                node.set_input_dim(dim)
                node.set_output_dim(dim)
                if axis < 0:
                    node.param.axis = len(dim) + axis
            elif node.type == NodeType.Flatten:
                node.set_input_dim(dim)
                size = 1
                for n in dim:
                    size *= n
                node.set_output_dim((size,))
            elif node.type == NodeType.Reshape:
                if len(dim) == 3 and dim[0] > 1 and dim[1] > 1:
                    flat_node = self.insert_flatten_node(node, dim)
                    dim = flat_node.output_dim
                node.set_input_dim(dim)
                node.set_output_dim(node.param.reshape_param)
            elif node.type == NodeType.Custom:
                node.set_input_dim(dim)
                dim = node.param.custom_param[1](node.param.custom_param[0],
                                                 dim)
                node.set_output_dim(dim)
            else:
                node.set_input_dim(dim)
                node.set_output_dim(dim)


# Test script
if __name__ == '__main__':
    t1 = LayerNode('test1', NodeType.Convolution)
    t2 = LayerNode('test2', NodeType.ReLU, input_node=t1)
