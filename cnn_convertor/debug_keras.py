import keras
from keras import layers
from keras import models
from keras.layers import Input
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope, deserialize_keras_object

import h5py
import json


def build_dict(seq, key="name"):
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))

# info_by_name = build_dict(lst, key="name")
# tom_info = info_by_name.get("Tom")
# {'index': 1, 'id': '2345', 'name': 'Tom'}

def deserialize(config, custom_objects=None):
    """Instantiate a layer from a config dictionary.

    # Arguments
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names)
            of custom (non-Keras) objects to class/functions

    # Returns
        Layer instance (may be Model, Sequential, Layer...)
    """
    from .. import models
    globs = globals()  # All layers.
    globs['Model'] = models.Model
    globs['Sequential'] = models.Sequential
    return deserialize_keras_object(config,
                                    module_objects=globs,
                                    custom_objects=custom_objects,
                                    printable_module_name='layer')




def layer_split(network_def, network_data, network_type,
                                    custom_layer):
    print('qwerty')
    print('qwerty')

    network_def = 'C:\\Alex\\Work\\fpga_perf\\tool\\network\\mobilenet.h5'

    f = h5py.File(network_def, mode='r')
    model_config = f.attrs.get('model_config')
    model_config = json.loads(model_config.decode('utf-8'))

    globs = globals()  # All layers.
    globs['Model'] = models.Model
    globs['Sequential'] = models.Sequential
    custom_objects = {'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}

    globs['Conv2D']= layers.Conv2D
    globs['relu6']=keras.applications.mobilenet.relu6
    globs['DepthwiseConv2D']=keras.applications.mobilenet.DepthwiseConv2D
    
    model_load = load_model(network_def, custom_objects=custom_objects)

    for i, layer in enumerate(model_config['config']['layers']):
        layer_type = layer['class_name']
        layer_config = layer['config']
        layer_method=getattr(keras.layers, layer['class_name'])
        
        if layer_type == 'InputLayer' and i==0:
            layer_config['name']="input"
        with CustomObjectScope({'relu6': relu6}):
            input_layer=Input(shape=model_load.layers[i].input_shape[1:])
        output_layer = layer_method(**layer_config)(input_layer)
        network = Model(inputs=input_layer, outputs = output_layer)
        network.save('net_'+str(i)+'.h5')

    # zxc= deserialize_keras_object(config,
    #                                 module_objects=globs,
    #                                 custom_objects=custom_objects,
    #                                 printable_module_name='layer')

    # globs['Conv2D']= layers.Conv2D
    # layer1 = model_config['config']['layers'][1]
    # zxc= deserialize_keras_object(layer1, module_objects=globs, custom_objects=custom_objects, printable_module_name='layer')


    # asd = load_model(network_def, custom_objects=custom_objects)


    print('qwerty')
    print('qwerty')



    






def calc_inout_sizes(self):
        def get_output_xy(dim: Tuple[int],
                          param: NodeParam,
                          is_pool: bool) -> Tuple[int]:
            w, h = dim[0], dim[1]
            w += param.pad[0] * 2
            h += param.pad[1] * 2
            if param.keras_padding == 'same':
                if w % param.stride[0] == 0:
                    pw = param.pad[0] + max(param.kernel_size[0] - param.stride[0], 0)
                else:
                    pw = param.pad[0] + max(param.kernel_size[0] - w % param.stride[0], 0)
                if h % param.stride[1] == 0:
                    ph = param.pad[1] + max(param.kernel_size[1] - param.stride[1], 0)
                else:
                    ph = param.pad[1] + max(param.kernel_size[1] - h % param.stride[1], 0)
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
                while ((dim[0] + padx - param.kernel_size[0]) % param.stride[0]) > padx:
                    padx += 1
                pady = param.pad[1]
                while ((dim[1] + pady - param.kernel_size[1]) % param.stride[1]) > pady:
                    pady += 1
                param.pad = (padx, pady)
            else:
                w = math.floor(w)
                h = math.floor(h)
            return w, h

        tr_list = self._traverse_list[:]
        for node in tr_list:
            if node._type == NodeType.Input:
                continue
            # For Keras, DepthwiseConvolution node don't have output_size set.
            # Set it here
            dim = node._input_nodes[0]._output_dim
            if node._param.num_output == 0:
                node._param.num_output = dim[-1] * node._param.group
                node._param.group = dim[-1]
            if node._type == NodeType.Convolution:
                node.set_input_dim(dim)
                if len(dim) == 3:
                    dim = (get_output_xy(dim, node._param, False) +
                           (node._param.num_output,))
                else:
                    raise cnn_exception.ConvertError('Invalid dimension')
                node.set_output_dim(dim)
            elif node._type == NodeType.InnerProduct:
                node.set_input_dim(dim)
                dim = (node._param.num_output,)
                node.set_output_dim(dim)
            elif node._type == NodeType.Pooling:
                node.set_input_dim(dim)
                if node._param.is_global:
                    node._param.kernel_size = (dim[0], dim[1])
                    dim = (1, 1, dim[2])
                else:
                    dim = get_output_xy(dim, node._param, True) + (dim[2],)
                node.set_output_dim(dim)
                # split pool node if kernel size > 7
                if (node._param.kernel_size[0] > 7 or
                        node._param.kernel_size[1] > 7):
                    self.split_pool_node(node)
            elif node._type == NodeType.UpSampling:
                node.set_input_dim(dim)
                dim = (dim[0] * node._param.kernel_size[0],
                       dim[1] * node._param.kernel_size[1], dim[2])
                node.set_output_dim(dim)
            elif node._type == NodeType.Power:
                node.set_input_dim(dim)
                node.set_output_dim(dim)
            elif node._type == NodeType.Concat:
                axis = node._param.axis
                c = 0
                for in_n in node._input_nodes:
                    c += in_n._output_dim[axis]
                temp_dim = list(dim)
                temp_dim[axis] = c
                dim = tuple(temp_dim)
                node.set_input_dim(dim)
                node.set_output_dim(dim)
                if axis < 0:
                    node._param.axis = len(dim) + axis
            elif node._type == NodeType.Flatten:
                node.set_input_dim(dim)
                size = 1
                for n in dim:
                    size *= n
                node.set_output_dim((size,))
            elif node._type == NodeType.Reshape:
                if len(dim) == 3 and dim[0] > 1 and dim[1] > 1:
                    flat_node = self.insert_flatten_node(node, dim)
                    dim = flat_node._output_dim
                node.set_input_dim(dim)
                node.set_output_dim(node._param.reshape_param)
            elif node._type == NodeType.Custom:
                node.set_input_dim(dim)
                dim = node._param.custom_param[1](node._param.custom_param[0],
                                                  dim)
                node.set_output_dim(dim)
            else:
                node.set_input_dim(dim)
                node.set_output_dim(dim)