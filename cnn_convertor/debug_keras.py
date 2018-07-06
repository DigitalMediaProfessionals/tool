import keras
from keras import layers
from keras import models
from keras.layers import Input
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope, deserialize_keras_object
from keras.layers import deserialize as layer_from_config


import os.path
import h5py
import json
import numpy as np
import tensorflow as tf
from scipy import misc

from keras import backend as K
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def get_input(layer_name):
		filename = 'output_'+layer_name+'.npy'
		if os.path.isfile(filename):
			data = np.load('output_'+layer_name+'.npy')
		else:
			print('NO INPUT DATA FOUND')	
			data = misc.imread('image_019.jpg')
			data = np.asarray([data])
		return data



def layer_split(fpga_network):
	print('wewqt')

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
	model_load_weights={}
	for layer in model_load.layers:
		model_load_weights[layer.name]=layer.get_weights()
	# model_load1 = keras.models.model_from_config(model_config, custom_objects=custom_objects)  #use the other one for  real weights	
	print('qwerty')
	print('qwerty')
	# model_weights = model_load.get_weights()
	
	fpga_network_layers={}

	for layer in fpga_network._layer:
		K.clear_session()
		first_layer = layer.node_in
		last_layer = layer.node_out
		input_dim = first_layer._input_dim
		input_nodes = first_layer._input_nodes
		name = first_layer._name
		
		keras_input = Input(shape=input_dim)
		keras_layer = model_load.get_layer(name)
		keras_layer_class = keras_layer.__class__
		keras_layer_config = keras_layer.get_config()
		with CustomObjectScope({'relu6': relu6}):
			keras_out_layer = keras_layer_class(**keras_layer_config)(keras_input)
			# keras_out_layer.set_weights(keras_layer.get_weights())

		if first_layer._bn_node:
			sub_layer_name = first_layer._bn_node._name
			keras_layer = model_load.get_layer(sub_layer_name)
			keras_layer_class = keras_layer.__class__
			keras_layer_config = keras_layer.get_config()
			with CustomObjectScope({'relu6': relu6}):
				keras_out_layer = keras_layer_class(**keras_layer_config)(keras_out_layer)
				# keras_out_layer.set_weights(keras_layer.get_weights())

		if first_layer._act_node:
			sub_layer_name = first_layer._act_node._name
			keras_layer = model_load.get_layer(sub_layer_name)
			keras_layer_class = keras_layer.__class__
			keras_layer_config = keras_layer.get_config()
			with CustomObjectScope({'relu6': relu6}):
				keras_out_layer = keras_layer_class(**keras_layer_config)(keras_out_layer)
				# keras_out_layer.set_weights(keras_layer.get_weights())

		keras_model = Model(inputs = keras_input, outputs = keras_out_layer)
		for layer in keras_model.layers:
			layer_name = layer.name
			model_load_weights = model_load_weights[layer_name]
			layer.set_weights(model_load_weights)


		input_data=[]
		input_nodes =  layer.layer_in
		for node in input_nodes:
			input_node_name = node.node_in._name
			input_data.append(get_input(input_node_name))
		print("Dfs")
		if input_data:
			prediction = keras_model.predict(input_data)
			np.save('output_'+name+'.npy', prediction)

		keras_model.save('keras_networks/'+name+'.h5')

	print('sadfsad')















def layer_split_old(network_def, network_data, network_type,
									custom_layer):
	print('qwerty')
	print('qwerty')
	import re
	regex = r"^[^/:]*"

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
	
	# model_load = load_model(network_def, custom_objects=custom_objects)	
	model_load = keras.models.model_from_config(model_config, custom_objects=custom_objects)  #use the other one for  real weights	
	print('qwerty')
	print('qwerty')
	model_weights = model_load.get_weights()
	
	fpga_network_layers={}


	for i, layer in enumerate(model_config['config']['layers']):
		K.clear_session()
		layer_type = layer['class_name']
		layer_name = layer['name']


		if layer_type in ('BatchNormalization', 'Activation', 'Dropout', 'Reshape') :
			top_layer = layer
			top_layer_type = layer_type
			if layer_type=='Activation':
				if layer['config']['activation']=='softmax':
					fpga_network_layers[layer_name] = [layer]
				else:
					while top_layer_type in ('BatchNormalization', 'Activation', 'Dropout', 'Reshape') :
						top_layer_name = top_layer['inbound_nodes'][0][0][0]
						for l in model_config['config']['layers']:
							if l['name'] == top_layer_name:
								top_layer = l
								break
						top_layer_type = top_layer['class_name']
					# top_layer_name = top_layer['name']
					fpga_network_layers[top_layer_name].append(layer)
			else:
			# search for existing input and output nodes
				while top_layer_type in ('BatchNormalization', 'Activation', 'Dropout', 'Reshape') :
					top_layer_name = top_layer['inbound_nodes'][0][0][0]
					for l in model_config['config']['layers']:
						if l['name'] == top_layer_name:
							top_layer = l
							break
					top_layer_type = top_layer['class_name']
				# top_layer_name = top_layer['name']
				fpga_network_layers[top_layer_name].append(layer)
		else:
			input_shape = model_load.layers[i].input_shape[1:]
			input_layer = Input(shape=input_shape)
			fpga_network_layers[layer_name] = [layer]

	print('qwerty')
	print('qwerty')

	outputs_layer_map = {}
	for key, value in fpga_network_layers.items():
		outputs_layer_map[value[-1]['name']] = key	

	network_outputs={}
	for key, value in list(fpga_network_layers.items())[1:]:
		print(0)
		print(value)
		first_layer_name = value[0]['name']
		input_shape = model_load.get_layer(first_layer_name).input_shape
		input_layer=Input(shape=input_shape[1:])
		print(1)
		network_output = input_layer
		for layer in value:
			layer_config = layer['config']
			layer_method=getattr(keras.layers, layer['class_name'])
			with CustomObjectScope({'relu6': relu6}):
				network_output = layer_method(**layer_config)(network_output)
		print(2)
		model = Model(inputs = input_layer, outputs = network_output)
		print(2.5)
		for layer in model.layers[1:]:
			layer.set_weights(model_load.get_layer(layer.name).get_weights())
		print(3)
		inbound_nodes = []
		inbound_node_names = [x[0] for x in layer['inbound_nodes'][0]]
		for inbound_node_name in inbound_node_names:
			inbound_node = outputs_layer_map[inbound_node_name]
			inbound_node_data = network_outputs['inbound_node']
			inbound_nodes.append(inbound_node_data)
		print(4)
		layer_output = model.predict(inbound_nodes)
		

		










def layer_split1d(network_def, network_data, network_type,
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
	print('qwerty')
	print('qwerty')
	model_weights = model_load.get_weights()
	

	for i, layer in enumerate(model_config['config']['layers']):
		K.clear_session()
		layer_type = layer['class_name']
		layer_config = layer['config']
		layer_method=getattr(keras.layers, layer['class_name'])
		
		if layer_type == 'InputLayer' and i==0:
			layer_config['name']="input"
			input_data = get_input(i, layer_type, shape=model_load.layers[i].input_shape[1:] )
		else:
			input_data = get_input(i, layer_type, shape=model_load.layers[i].input_shape[1:] )
		input_layer=Input(shape=model_load.layers[i].input_shape[1:])
		with CustomObjectScope({'relu6': relu6}):
			output_layer = layer_method(**layer_config)(input_layer)
		network = Model(inputs=input_layer, outputs = output_layer)
		network_weights = network.get_weights()
		if len(network_weights)>0:
			updated_weights=[]
			for w in range(len(network_weights)):
				updated_weights.append(model_weights[0])
				model_weights.pop(0)
			network.set_weights(updated_weights)


		layer_output = network.predict(input_data)
		np.save('output_'+str(i)+'.npy', layer_output)
		print(i)
		print(layer_type)
		print(layer_config)
		network.save('keras_layers/net_'+str(i)+'.h5')


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




# model_load_outputs =[]
# for layer in model_load.layers:
# 	model_load_outputs.append(layer.output)

# new_model = Model(inputs = model_load.input, outputs = model_load_outputs)
	



