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


def get_conv_out_width(width, kx, pad_left, pad_right, stride):
    return (pad_left + width + pad_right - kx) // stride + 1


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
    PReLU = auto()
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
    ReLU6 = auto()

    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)

    def __str__(self):
        return self.name


class NodeParam(object):
    def __init__(self):
        self.num_output = 0
        self.kernel_size = (1, 1)
        self._pad = [0, 0, 0, 0]
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
        self.split_pool_divisor = None
        self._dilation = [0, 0]

    @property
    def dilation(self):
        assert len(self._dilation) == 2
        return self._dilation

    @dilation.setter
    def dilation(self, value):
        assert len(self._dilation) == 2
        try:
            assert all(int(x) == x for x in value)
            if len(value) == 2:
                self._dilation[0] = int(value[0])
                self._dilation[1] = int(value[1])
            elif len(value) == 1:
                self._dilation[0] = int(value[0])
                self._dilation[1] = int(value[0])
            elif len(value) == 0:
                self._dilation[0] = 0
                self._dilation[1] = 0
            else:
                raise ValueError("Invalid value for dilation: %s" % value)
        except TypeError:
            assert int(value) == value
            self._pad[0] = int(value)
            self._pad[1] = int(value)
            self._pad[2] = int(value)
            self._pad[3] = int(value)
        assert len(self._dilation) == 4

    @property
    def pad(self):
        raise ValueError("Getter for NodeParam.pad is disabled")

    @pad.setter
    def pad(self, value):
        raise ValueError("Setter for NodeParam.pad is disabled")

    @property
    def pad_lrtb(self):
        """Returns reference to list with 4 elements with padding.
        """
        assert len(self._pad) == 4
        return self._pad

    @pad_lrtb.setter
    def pad_lrtb(self, value):
        assert len(self._pad) == 4
        try:
            assert all(int(x) == x for x in value)
            if len(value) == 4:
                self._pad[0] = int(value[0])
                self._pad[1] = int(value[1])
                self._pad[2] = int(value[2])
                self._pad[3] = int(value[3])
            elif len(value) == 2:
                self._pad[0] = int(value[0])
                self._pad[1] = int(value[0])
                self._pad[2] = int(value[1])
                self._pad[3] = int(value[1])
            elif len(value) == 1:
                self._pad[0] = int(value[0])
                self._pad[1] = int(value[0])
                self._pad[2] = int(value[0])
                self._pad[3] = int(value[0])
            elif len(value) == 0:
                self._pad[0] = 0
                self._pad[1] = 0
                self._pad[2] = 0
                self._pad[3] = 0
            else:
                raise ValueError("Invalid value for pad_lrtb: %s" % value)
        except TypeError:
            assert int(value) == value
            self._pad[0] = int(value)
            self._pad[1] = int(value)
            self._pad[2] = int(value)
            self._pad[3] = int(value)
        assert len(self._pad) == 4

    @property
    def pad_fpga(self):
        """Returns integer in FPGA hardware format for padding.
        """
        assert len(self.pad_lrtb) == 4
        return (self.pad_lrtb[0] | (self.pad_lrtb[1] << 8) |
                (self.pad_lrtb[2] << 16) | (self.pad_lrtb[3] << 24))

    @pad_fpga.setter
    def pad_fpga(self, value):
        raise ValueError("Setter for pad_fpga is disabled")

    def __repr__(self):
        ret = ("[num_output:%d, kernel_size:%s, pad:%s, stride:%s, pool:%d, "
               "group:%d, relu_param:%f, is_global:%d]")
        return ret % (self.num_output, self.kernel_size, self.pad_lrtb,
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
        self.output_nodes = []
        self.debug_node = None
        self.custom_layer = custom_layer
        self.dim_override = dim_override
        self.tensorflow_backend = False

    def build_traverse_list(self) -> None:
        pending = self.output_nodes[:]
        pending.reverse()
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

    def append_output_node(self, node):
        self.output_nodes.append(node)

    def _get_split_size(self, node, kernel_size):
        if kernel_size == 1:
            return 1, 0
        if node.param.pool == 0:    # max pooling
            candidates = [3, 2]
        else:   # avg pooling
            candidates = [7, 6, 5, 4, 3, 2]
        split_size = 0
        kernel_padding = 0
        for _i in range(candidates[0] - 1):
            for c in candidates:
                if (node.param.kernel_size[0] + kernel_padding) % c == 0:
                    split_size = c
                    break
            if split_size == 0:
                if node.param.pool == 0:
                    msg = ("Support for max pooling of size "
                           "%d x %d is not implemented" %
                           (node.param.kernel_size[0],
                            node.param.kernel_size[1]))
                    logging.exception(msg)
                    raise cnn_exception.ConvertError(msg)
                kernel_padding += 1
            else:
                break
        if split_size == 0:
            raise cnn_exception.ConvertError(
                "Possible implementation error detected: "
                "control should not reach this line")
        return split_size, kernel_padding

    def split_pool_node(self, node, level=0, remaining_divisor=None):
        if (not node.param.is_global and
                (node.param.stride[0] != node.param.kernel_size[0] or
                 node.param.stride[1] != node.param.kernel_size[1])):
            raise cnn_exception.ConvertError(
                "Support for pooling size %dx%d with stride %dx%d "
                "is not implemented" %
                (node.param.kernel_size[0], node.param.kernel_size[1],
                 node.param.stride[0], node.param.stride[1]))

        insert_index = self.traverse_list.index(node) + 1
        split_w, pad_w = self._get_split_size(node, node.param.kernel_size[0])
        split_h, pad_h = self._get_split_size(node, node.param.kernel_size[1])

        if level == 0:
            assert node.param.split_pool_divisor is None and \
                remaining_divisor is None
            remaining_divisor = (node.param.kernel_size[0] *
                                 node.param.kernel_size[1])
        node.param.split_pool_divisor = split_w * split_h
        remaining_divisor /= node.param.split_pool_divisor
        remain_w = (node.param.kernel_size[0] + pad_w) // split_w
        remain_h = (node.param.kernel_size[1] + pad_h) // split_h

        mx, my = 1, 1
        while mx or my:
            dx, mx = divmod(node.param.pad_lrtb[0] + node.input_dim[0] +
                            node.param.pad_lrtb[1], split_w)
            dy, my = divmod(node.param.pad_lrtb[2] + node.input_dim[1] +
                            node.param.pad_lrtb[3], split_h)
            if mx:
                node.param.pad_lrtb[1] += 1
            if my:
                node.param.pad_lrtb[3] += 1

        remain_dim = (dx, dy, node.input_dim[2])

        remain_node = LayerNode(node.name + '_' + str(remain_w) + '_'
                                + str(remain_h), NodeType.Pooling)
        remain_node.input_nodes = [node]
        remain_node.output_nodes = node.output_nodes
        for node_out in node.output_nodes:
            node_out.input_nodes = [remain_node]
        node.output_nodes = [remain_node]
        node.name = node.name + '_' + str(split_w) + '_' + str(split_h)

        remain_node.input_dim = remain_dim
        remain_node.set_output_dim(node.output_dim)
        node.set_output_dim(remain_dim)

        param = NodeParam()
        param.pool = node.param.pool
        param.kernel_size = (remain_w, remain_h)
        param.stride = (remain_w, remain_h)
        param.split_pool_divisor = remaining_divisor
        remain_node.set_param(param)
        node.param.kernel_size = (split_w, split_h)
        node.param.stride = (split_w, split_h)

        self.traverse_list.insert(insert_index, remain_node)

        if node.param.pool == 0:    # max pooling
            kernel_size_limit = 3
        else:   # avg pooling
            kernel_size_limit = 7
        if remain_w > kernel_size_limit or remain_h > kernel_size_limit:
            self.split_pool_node(remain_node, level + 1, remaining_divisor)

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

            if param.is_global:
                raise cnn_exception.ConvertError(
                    "get_output_xy() must not be called on "
                    "global pooling layers")

            w, h = dim[0], dim[1]

            if param.keras_padding == "same":
                ow = math.ceil(float(w) / param.stride[0])
                oh = math.ceil(float(h) / param.stride[1])

                # Increase padding if necessary
                while get_conv_out_width(
                        w, param.kernel_size[0], param.pad_lrtb[0],
                        param.pad_lrtb[1], param.stride[0]) < ow:
                    param.pad_lrtb[0 if param.pad_lrtb[0] < param.pad_lrtb[1]
                                   else 1] += 1
                while get_conv_out_width(
                        h, param.kernel_size[1], param.pad_lrtb[2],
                        param.pad_lrtb[3], param.stride[1]) < oh:
                    param.pad_lrtb[2 if param.pad_lrtb[2] < param.pad_lrtb[3]
                                   else 3] += 1
                # Decrease padding if necessary
                while get_conv_out_width(
                        w, param.kernel_size[0], param.pad_lrtb[0],
                        param.pad_lrtb[1], param.stride[0]) > ow:
                    param.pad_lrtb[0 if param.pad_lrtb[0] >= param.pad_lrtb[1]
                                   else 1] -= 1
                while get_conv_out_width(
                        h, param.kernel_size[1], param.pad_lrtb[2],
                        param.pad_lrtb[3], param.stride[1]) > oh:
                    param.pad_lrtb[2 if param.pad_lrtb[2] >= param.pad_lrtb[3]
                                   else 3] -= 1
                assert get_conv_out_width(
                        w, param.kernel_size[0], param.pad_lrtb[0],
                        param.pad_lrtb[1], param.stride[0]) == ow
                assert get_conv_out_width(
                        h, param.kernel_size[1], param.pad_lrtb[2],
                        param.pad_lrtb[3], param.stride[1]) == oh
            elif param.keras_padding == "causal":
                # TODO: dilation is ignored for now
                param.pad_lrtb[0] += param.kernel_size[0] - 1

            ow = (float(param.pad_lrtb[0] + w + param.pad_lrtb[1] -
                        param.kernel_size[0]) / param.stride[0]) + 1
            oh = (float(param.pad_lrtb[2] + h + param.pad_lrtb[3] -
                        param.kernel_size[1]) / param.stride[1]) + 1

            if is_pool and param.keras_padding is None:
                # Handle non-Keras (Caffe) padding separately
                ow = math.ceil(ow)
                oh = math.ceil(oh)
                # Increase padding if necessary
                while get_conv_out_width(
                        w, param.kernel_size[0], param.pad_lrtb[0],
                        param.pad_lrtb[1], param.stride[0]) < ow:
                    param.pad_lrtb[0 if param.pad_lrtb[0] < param.pad_lrtb[1]
                                   else 1] += 1
                while get_conv_out_width(
                        h, param.kernel_size[1], param.pad_lrtb[2],
                        param.pad_lrtb[3], param.stride[1]) < oh:
                    param.pad_lrtb[2 if param.pad_lrtb[2] < param.pad_lrtb[3]
                                   else 3] += 1
                assert get_conv_out_width(
                        w, param.kernel_size[0], param.pad_lrtb[0],
                        param.pad_lrtb[1], param.stride[0]) == ow
                assert get_conv_out_width(
                        h, param.kernel_size[1], param.pad_lrtb[2],
                        param.pad_lrtb[3], param.stride[1]) == oh
            else:
                ow = math.floor(ow)
                oh = math.floor(oh)

            return ow, oh

        # Override input dimension if dim_override is given
        if self.dim_override:
            dim = self.dim_override
            for node in self.input_nodes:
                new_dim = (dim[0], dim[1], node.input_dim[2])
                node.set_input_dim(new_dim)

        tr_list = self.traverse_list[:]
        for node in tr_list:
            if node.type == NodeType.Input:
                # Detect if the input dimension is not undefined
                if (len(node.input_dim) > 1 and
                    (node.input_dim[0] is None or node.input_dim[1] is None)):
                    msg = ("Network with undefined input dimension"
                           "is not supported.")
                    logging.exception(msg)
                    raise cnn_exception.ConvertError(msg)
                node.set_output_dim(node.input_dim)
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
                if node.param.pool == 0:    # max pooling
                    kernel_size_limit = 3
                else:   # avg pooling
                    kernel_size_limit = 7
                # split pool node if kernel size > limit
                if (node.param.kernel_size[0] > kernel_size_limit or
                        node.param.kernel_size[1] > kernel_size_limit):
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
