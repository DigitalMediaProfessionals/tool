import keras
from keras import layers
from keras import models
from keras.layers import Input, DepthwiseConv2D, Conv2D
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope, deserialize_keras_object
from keras.layers import deserialize as layer_from_config
import pathlib
import sys

import os.path
import h5py
import json
import numpy as np
import tensorflow as tf
from scipy import misc
import re
import cv2 
from keras.utils import plot_model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
keras.backend.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

used_input=0

def get_input(layer_name, network_folder_name, input_params):
	layer_filename = layer_name.replace('/','_') 
	file_regex = "^\d{0,4}_"+layer_filename+"\.npy"
	output_file=None
	for filename in os.listdir(network_folder_name+'/keras_outputs'):
		if re.match(file_regex, filename):
			file=filename
			output_file = network_folder_name+"/keras_outputs/" + filename

	for filename in os.listdir(network_folder_name+'/keras_outputs_float16'):
		if re.match(file_regex, filename):
			file16=filename
			output_file_16 = network_folder_name+"/keras_outputs_float16/" + file16

	if output_file:
		data = np.load(output_file)
		if output_file_16:
			data16 = np.load(output_file_16).astype(np.float16)
		else:
			print('Converting Float32 input for '+layer_name)
			data16 = data.astype(np.float16)
		
	else:
		if used_input==1:
			print('ERROR: INPUT IMAGE ALREADY USED')
			sys.exit()
		else:
			print('USING INPUT IMAGE')	
			r_offs 		= input_params['r_offs']
			g_offs		= input_params['g_offs']
			b_offs		= input_params['b_offs']
			scale 		= input_params['scale']


			if input_params['integer_test'] ==1:
				model_input_shape = input_params['model_input_shape']
				data=np.random.randint(0,2,size=model_input_shape)
				cv2.imwrite(network_folder_name+'integer_img_original.jpg', data)
			elif input_params['random_input'] == 1:
				model_input_shape = input_params['model_input_shape']
				data=np.random.randint(0,255,size=model_input_shape)
				cv2.imwrite(network_folder_name+'random_img_original.jpg', data)
			else:
				input_file 	= input_params['input_file']
				data = cv2.imread(input_file)
				cv2.imwrite(network_folder_name+'debug_img_original.jpg', data)

			data=data+[r_offs, g_offs, b_offs]
			data=data*scale
			cv2.imwrite(network_folder_name+'input_img_processed.jpg', data)
			data = np.asarray([data.astype(np.float32)])

			print('Converting Float32 input for '+layer_name)
			data16 = data.astype(np.float16)
			
			# pose
			# data = cv2.imread('im1.jpg')
			# data = np.asarray([data])
			#mobilenet
			# data = cv2.imread('image_019.jpg')
			# data = np.asarray([data])
			# data = (data.astype(np.float32)-127.5)*0.0078431
			#squeezenet
			# data = cv2.imread('image_019.jpg')
			# data = cv2.resize(data, dsize=(227, 227), interpolation=cv2.INTER_CUBIC)
			# data = np.asarray([data])
			# data = (data.astype(np.float32)-128)*1

			file = 'input.npy'
			data.dump(network_folder_name+'input.npy')

	return data, file, data16

def reorder(dims):
	return (dims[1], dims[0], dims[2])

def process_inputs(params):
	inputs={}
	inputs['input_file'] = params.input_file 
	inputs['integer_test'] = params.integer_test
	inputs['random_input'] = params.random_input
	inputs['r_offs'] = params.r_offs
	inputs['g_offs'] = params.g_offs
	inputs['b_offs'] = params.b_offs
	inputs['scale'] = params.scale
	return inputs

# def layer_split(fpga_network, network_def, input_file,r_offs=0, g_offs=0, b_offs=0,scale=1, transpose=1,  network_folder_name=None):
def layer_split(fpga_network, network_def, **kwargs):
	# network_def = keras_net
	input_params = process_inputs(kwargs['input_params'])
	network_name = os.path.basename(network_def).split('.')[0]
	network_folder_name = 'debug/' + os.path.basename(network_def).split('.')[0]+'/'
	pathlib.Path(network_folder_name+'keras_outputs').mkdir(parents=True, exist_ok=True)
	pathlib.Path(network_folder_name+'keras_networks').mkdir(parents=True, exist_ok=True)
	pathlib.Path(network_folder_name+'PLACE_FPGA_DUMPS_HERE').mkdir(parents=True, exist_ok=True)
	pathlib.Path(network_folder_name+'keras_outputs_float16').mkdir(parents=True, exist_ok=True)
	# network_def = 'C:\\Alex\\Work\\fpga_perf\\tool\\network\\mobilenet.h5'

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
	input_params['model_input_shape'] = model_load.input_shape[1:]
	if input_params['integer_test'] !=1:
		model_load.save(network_folder_name+network_name+'_original_model.h5')
	model_load_weights={}
	for layer in model_load.layers:
		model_load_weights[layer.name]=layer.get_weights()
	# model_load1 = keras.models.model_from_config(model_config, custom_objects=custom_objects)  #use the other one for  real weights	
	# print('qwerty')
	# print('qwerty')
	# model_weights = model_load.get_weights()
	plot_file = network_folder_name+'/'+network_name+'_model.png'
	plot_model(model_load, to_file=plot_file, show_shapes=True)


	fpga_network_layers={}
	keras_input_map={}

	i=0
	for layer in fpga_network.layer:
		K.clear_session()

		first_layer = layer.node_in
		last_layer = layer.node_out

		
		name = first_layer.name
		
		if name[-6:]=="_point":
			print('pointlayer')
			continue
		
		input_nodes = first_layer.input_nodes
		if len(input_nodes)>1:
			input_dims = []
			for input_node in input_nodes:
				input_dims.append(reorder(input_node.output_dim))
			keras_input = []
			for input_dim in input_dims:
				keras_input.append(Input(shape=input_dim))
		else:
			if len(first_layer.input_dim)==3:
				input_dim = reorder(first_layer.input_dim)
			elif len(first_layer.input_dim)==1:
				input_dim = first_layer.input_dim

			keras_input = Input(shape=input_dim)

		if model_load.get_layer(name).__class__.__name__ == 'SeparableConv2D':
			print('SEPCONV')
			sepconv_layer = model_load.get_layer(name)
			sepconv_weights = model_load_weights[name]

			depthconfig=sepconv_layer.get_config()
			pointconfig=sepconv_layer.get_config()
			

			unused_depth_args = ['filters', 'pointwise_initializer', 'pointwise_regularizer', 'pointwise_constraint', 'bias_initializer']
			for arg in unused_depth_args:
				try:
					del depthconfig[arg]
				except:
					pass
			unused_point_args = ['name', 'kernel_size', 'strides', 'depth_multiplier', 'depthwise_initializer', 'pointwise_initializer', 'depthwise_regularizer', 'pointwise_regularizer', 'depthwise_constraint', 'pointwise_constraint', 'bias_initializer']
			for arg in unused_point_args:
				try:
					del pointconfig[arg]
				except:
					pass
			
			with CustomObjectScope({'relu6': relu6}):
				depth_layer = keras.layers.DepthwiseConv2D(**depthconfig)(keras_input)
			depth_model = Model(inputs=keras_input, outputs = depth_layer)
			depth_bias_shape = depth_model.get_weights()[1].shape
			depth_weights = []
			depth_weights.append(sepconv_weights[0])
			depth_weights.append(np.zeros(depth_bias_shape))
			depth_model.set_weights(depth_weights)

			point_name = name+'_point'
			point_layer_fpga = fpga_network.layer[i+1]

			node_layers = []
			node_layers.append(point_layer_fpga.node_in)
			if point_layer_fpga.node_in == point_layer_fpga.node_out:
				pass
			else:
				node_layers.append(point_layer_fpga.node_out)


			point_input = Input(shape=depth_model.output_shape[1:])
			keras_out_layer = point_input
			for node_layer in node_layers:

				if node_layer==point_layer_fpga.node_in:
					with CustomObjectScope({'relu6': relu6}):
						keras_out_layer = Conv2D(name=point_name, kernel_size=(1,1), **pointconfig)(keras_out_layer)
				else:
					node_layer_name = node_layer.name
					try:
						keras_layer = model_load.get_layer(node_layer_name)
					except:
						print('model load error')
					keras_layer_class = keras_layer.__class__
					keras_layer_config = keras_layer.get_config()
					with CustomObjectScope({'relu6': relu6}):
						keras_out_layer = keras_layer_class(**keras_layer_config)(keras_out_layer)
						# keras_out_layer.set_weights(keras_layer.get_weights())
			
				if node_layer.bn_node:
					sub_layer_name = node_layer.bn_node.name
					keras_layer = model_load.get_layer(sub_layer_name)
					keras_layer_class = keras_layer.__class__
					keras_layer_config = keras_layer.get_config()
					with CustomObjectScope({'relu6': relu6}):
						keras_out_layer = keras_layer_class(**keras_layer_config)(keras_out_layer)
						# keras_out_layer.set_weights(keras_layer.get_weights())

				if node_layer.act_node:
					sub_layer_name = node_layer.act_node.name
					keras_layer = model_load.get_layer(sub_layer_name)
					keras_layer_class = keras_layer.__class__
					keras_layer_config = keras_layer.get_config()
					with CustomObjectScope({'relu6': relu6}):
						keras_out_layer = keras_layer_class(**keras_layer_config)(keras_out_layer)
						# keras_out_layer.set_weights(keras_layer.get_weights())
			point_model = Model(inputs=point_input, outputs = keras_out_layer)
			# point_model.set_weights(sepconv_weights[1:3])


			for keras_model_layer in point_model.layers:
				keras_model_layer_name = keras_model_layer.name
				if keras_model_layer_name==point_name:
					point_model.set_weights(sepconv_weights[1:3])
				else:
					try:
						keras_model_layer_weights = model_load_weights[keras_model_layer_name]
						keras_model_layer.set_weights(keras_model_layer_weights)
					except:
						pass

			input_data=[]
			input_data16=[]
			input_files=[]
			input_nodes =  layer.layer_in
			for node in input_nodes:
				input_node_name = node.node_in.name
				data, filename, data16 = get_input(input_node_name, network_folder_name, input_params)
				input_data.append(data)
				input_files.append(filename)
				input_data16.append(data16)

			depth_predict = depth_model.predict(input_data)
			name = name.replace('/','_') 
			depth_predict.dump(network_folder_name+'keras_outputs/'+str(i).zfill(3)+'_'+name+'.npy')
			depth_model.save(network_folder_name+'keras_networks/'+str(i).zfill(3)+'_'+name+'.h5')
			keras_input_map[str(i).zfill(3)+'_'+name]=input_files

			depth_predict16 = depth_model.predict(input_data16).astype(np.float16)
			depth_predict16.dump(network_folder_name+'keras_outputs_float16/'+str(i).zfill(3)+'_'+name+'.npy')
			
			i+=1
			
			point_predict = point_model.predict(depth_predict)
			point_name = point_name.replace('/','_') 
			point_predict.dump(network_folder_name+'keras_outputs/'+str(i).zfill(3)+'_'+point_name+'.npy')
			point_model.save(network_folder_name+'keras_networks/'+str(i).zfill(3)+'_'+point_name+'.h5')
			keras_input_map[str(i).zfill(3)+'_'+point_name]=[str(i-1).zfill(3)+'_'+name+'.npy']

			point_predict16 = point_model.predict(depth_predict16).astype(np.float16)
			point_predict16.dump(network_folder_name+'keras_outputs_float16/'+str(i).zfill(3)+'_'+point_name+'.npy')

			i+=1

		else:
			node_layers = []
			out_nodes=[]
			input_nodes=[]
			for out_node in layer.node_out.output_nodes:
				out_nodes.append(out_node)
			

			for in_node in layer.node_in.input_nodes:
				input_nodes.append(in_node)
				if layer.node_in==layer.node_out:
					node_layers.append(layer.node_in)
					continue
				next_nodes=in_node.output_nodes
				if next_nodes[0] not in node_layers:
						node_layers.append(next_nodes[0])
				while len(next_nodes)>0:
					if next_nodes[0] in out_nodes:
						break
					if next_nodes[0] not in node_layers:
						node_layers.append(next_nodes[0])
					for out in next_nodes[0].output_nodes:
						if out not in out_nodes:
							next_nodes.append(out)
					next_nodes.remove(next_nodes[0])
			# next(fpga_network.layer, None)
			# continue


			
			# if layer.node_in == layer.node_out:
			# 	node_layers.append(layer.node_in)
			# elif layer.node_in.output_nodes[0] == layer.node_out:
			# 	node_layers.append(layer.node_in)
			# 	node_layers.append(layer.node_out)
			# elif layer.node_in.output_nodes[0].type.name == 'Concat':

			# else:
			# 	for inner_output in layer.node_in.output_nodes:


			# else:
			# 	node_layers.append(layer.node_out)

			# keras_out_layer = keras_input
			keras_layers={}
			model_inputs={}
			model_inputs_list=[]
			# for input_node in input_nodes:
			# 	model_inputs[input_node.name] = Input(shape=input_node.output_dim)
			
			for node_layer in node_layers:	
				node_layer_name = node_layer.name
				print(node_layer_name)
				# output_layer_name = node_layer_name
				node_layer_inputs = []
				for input_node in node_layer.input_nodes:
					if input_node in input_nodes:
						if input_node.name in model_inputs:
							input_layer = model_inputs[input_node.name]	
						else:
							# dim=input_node.output_dim
							if len(input_node.output_dim)==3:
								dim = reorder(input_node.output_dim)
							elif len(input_node.output_dim)==1:
								dim = input_node.output_dim
							input_layer=Input(shape=dim)
							model_inputs[input_node.name] = input_layer
							model_inputs_list.append(input_layer)
						node_layer_inputs.append(input_layer)

					else:
						node_layer_inputs.append(keras_layers[input_node.name])
				if len(node_layer_inputs)==1:
					node_layer_inputs=node_layer_inputs[0]
			

				keras_layer = model_load.get_layer(node_layer_name)

				keras_layer_class = keras_layer.__class__
				keras_layer_config = keras_layer.get_config()
				with CustomObjectScope({'relu6': relu6}):
					keras_layers[node_layer_name] = keras_layer_class(**keras_layer_config)(node_layer_inputs)
					# keras_out_layer.set_weights(keras_layer.get_weights())
			
				if node_layer.bn_node:
					sub_layer_name = node_layer.bn_node.name
					keras_layer = model_load.get_layer(sub_layer_name)
					keras_layer_class = keras_layer.__class__
					keras_layer_config = keras_layer.get_config()
					with CustomObjectScope({'relu6': relu6}):
						keras_layers[node_layer_name] = keras_layer_class(**keras_layer_config)(keras_layers[node_layer_name])
						# keras_out_layer.set_weights(keras_layer.get_weights())

				if node_layer.act_node:
					sub_layer_name = node_layer.act_node.name
					keras_layer = model_load.get_layer(sub_layer_name)
					keras_layer_class = keras_layer.__class__
					keras_layer_config = keras_layer.get_config()
					with CustomObjectScope({'relu6': relu6}):
						keras_layers[node_layer_name] = keras_layer_class(**keras_layer_config)(keras_layers[node_layer_name])
						# keras_out_layer.set_weights(keras_layer.get_weights())

			keras_model = Model(inputs = model_inputs_list, outputs = keras_layers[node_layer_name])
			for keras_model_layer in keras_model.layers:
				keras_model_layer_name = keras_model_layer.name
				try:
					keras_model_layer_weights = model_load_weights[keras_model_layer_name]
					keras_model_layer.set_weights(keras_model_layer_weights)
				except:
					pass


			input_data=[]
			input_data16=[]
			input_files=[]
			input_nodes =  layer.layer_in
			for node in input_nodes:
				input_node_name = node.node_in.name
				data, filename, data16 = get_input(input_node_name, network_folder_name, input_params)
				if data.ndim==2:
					if data.shape[0]==1:
						data=np.asarray([[data]])
						data16=np.asarray([[data16]])
				if data.ndim==4:
					if data.shape[0]==1:
						if data.shape[1]==1:
							if data.shape[2]==1:
								if keras_model.input_shape[1]==data.shape[3]:
									data = data[0][0]
									data16 = data16[0][0]
				input_data.append(data)
				input_files.append(filename)
				input_data16.append(data16)

			name = name.replace('/','_')
			try:
				prediction = keras_model.predict(input_data)
				prediction.dump(network_folder_name+'keras_outputs/'+str(i).zfill(3)+'_'+name+'.npy')

				prediction16 = keras_model.predict(input_data16).astype(np.float16)
				prediction16.dump(network_folder_name+'keras_outputs_float16/'+str(i).zfill(3)+'_'+name+'.npy')

			except:
				print("prediction error")
			

			keras_model.save(network_folder_name+'keras_networks/'+str(i).zfill(3)+'_'+name+'.h5')
			keras_input_map[str(i).zfill(3)+'_'+name]=input_files
			i+=1
	
	map_json = json.dumps(keras_input_map)		
	f = open(network_folder_name+'/'+network_name+'_input_map.json','w')
	f.write(map_json)
	f.close()

	print('Done.')



