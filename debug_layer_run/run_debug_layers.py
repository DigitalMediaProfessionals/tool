import keras
from keras import layers
from keras import models
from keras.layers import Input, DepthwiseConv2D, Conv2D
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope, deserialize_keras_object
from keras.layers import deserialize as layer_from_config
import pathlib
import glob
import os.path
import argparse
import os
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
import benchmark


keras.backend.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


custom_objects = {'relu6': keras.layers.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}


def remap(arr, dim):
    if len(dim) < 3:
        return arr
    sub_arrs = []
    for step in range(0, dim[2], 8):
        step_end = dim[2] if step + 8 > dim[2] else step + 8
        sub_arr = arr[dim[0] * dim[1] * step : dim[0] * dim[1] * step_end]
        sub_arr.shape = (dim[1], dim[0], step_end - step)
        sub_arr = np.transpose(sub_arr, axes = (1, 0, 2))
        sub_arrs.append(sub_arr)
    return np.concatenate(tuple(sub_arrs), axis = 2)


parser = argparse.ArgumentParser()
parser.add_argument("INPUT_FOLDER", type=str, help="Input ini file")

	
args = parser.parse_args()
network_debug_folder = os.path.abspath(args.INPUT_FOLDER)+'\\'


layers_folder = network_debug_folder + 'keras_networks\\'
keras_outputs_folder = network_debug_folder + 'keras_outputs\\'
fpga_outputs_folder = network_debug_folder + 'PLACE_FPGA_DUMPS_HERE\\'
debug_output_folder=network_debug_folder+'fpga_dump_debug_outputs\\'

pathlib.Path(debug_output_folder).mkdir(parents=True, exist_ok=True)

if len(glob.glob(network_debug_folder+'*map.json'))>0:
    network_map_file = glob.glob(network_debug_folder+'*map.json')[0]
    network_map = json.load(open(network_map_file))
else:
    network_map_file = None
    network_map = None

layer_files = glob.glob(layers_folder+'/*')
keras_files = glob.glob(keras_outputs_folder+'/*')
keras_input = np.load(network_debug_folder+'input.npy')
fpga_files = glob.glob(fpga_outputs_folder+'/*')

layer_files = sorted(layer_files)
keras_files = sorted(keras_files)
fpga_files = sorted(fpga_files)

fpga_regex = "layer_input.bin$"
r=re.compile(fpga_regex)
fpga_input_file = list(filter(lambda x: r.search(x), fpga_files))
fpga_files = list(filter(lambda x: not r.search(x), fpga_files))
if len(fpga_files)>100:
    fpga_layers=[]
    for i in range(10):
        filenum=('0'+str(i))
        fpga_layers.append(fpga_outputs_folder+'layer'+filenum+'.bin')
    for i in range(10,len(fpga_files)):
        filenum=str(i)
        fpga_layers.append(fpga_outputs_folder+'layer'+filenum+'.bin')
    fpga_files=fpga_layers

    

if len(fpga_files) != len(keras_files):
    print("Number of input files does not match")
num_files = min(len(layer_files), len(keras_files), len(fpga_files))


layers={}
keras_outputs={}
fpga_outputs={}


# for i, file in enumerate(layer_files):
for i in range(num_files):
    file = layer_files[i]
    name=os.path.basename(file).split('.')[0]
    layers[name]=file
    keras_outputs[name] = np.load(keras_files[i])[0]
    # if i!=0:
    fpga_dump = np.fromfile(fpga_files[i], dtype=np.float16)
    if len(fpga_dump)!=keras_outputs[name].size:
        fpga_dump = np.fromfile(fpga_files[i], dtype=np.float32)
    try:
        fpga_dump = remap(fpga_dump, keras_outputs[name].shape)
    except:
        print('Remap error layer:' + str(i))
    while(len(fpga_dump.shape)<4):
        fpga_dump = np.asarray([fpga_dump])
    fpga_outputs[name] = fpga_dump
    

prev_layer_name=""
for i, layer in enumerate(layers.items()):
    keras.backend.clear_session()
    layer_name = layer[0]
    layer_file = layer[1]

    if i==0:
        layer_input = keras_input
    else:
        if network_map:
            layer_input=[]
            for map_in in network_map[layer_name]:
                layer_input.append(fpga_outputs[map_in[:-4]])
        else:
            layer_input=[fpga_outputs[i-1]]
            

    with CustomObjectScope(custom_objects):
        keras_model = load_model(layer_file)
    try:
        layer_predict = keras_model.predict(layer_input)
        layer_predict.dump(debug_output_folder+layer_name+'.npy')
    except ValueError:
        if len(layer_input)==1:
            layer_input=layer_input[0]
            while layer_input.shape[0]==1:
                layer_input=layer_input[0]
            try:
                layer_predict = keras_model.predict(np.asarray([layer_input]))
                layer_predict.dump(debug_output_folder+layer_name+'.npy')
            except:
                print("Prediction/Shape Error Layer: "+ str(i))
    except:
        print("Prediction Error Layer: "+ str(i))
    
    prev_layer_name = layer_name

benchmark.write_benchmark_file(network_debug_folder)
print('Done.')
    			