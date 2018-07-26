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

# keras.backend.clear_session()
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras







network_debug_folder = "C:\\Alex\\Work\\fpga_perf\\debug\\mobilenet\\"
layers_folder = network_debug_folder + 'keras_networks\\'
keras_outputs_folder = network_debug_folder + 'keras_outputs\\'
fpga_outputs_folder = network_debug_folder + 'PLACE_FPGA_DUMPS_HERE\\'
if len(glob.glob(network_debug_folder+'*map.json'))>0:
    network_map_file = glob.glob(network_debug_folder+'*map.json')[0]
    network_map = json.load(open(network_map_file))
else:
    network_map_file = None
    network_map = None

layer_files = glob.glob(layers_folder+'/*')
keras_output_files = glob.glob(keras_outputs_folder+'/*')

fpga_output_files = glob.glob(fpga_outputs_folder+'/*')
fpga_regex = "layer_input.bin$"
r=re.compile(fpga_regex)
fpga_input_file = list(filter(lambda x: r.search(x), fpga_output_files))
fpga_output_files = list(filter(lambda x: not r.search(x), fpga_output_files))


layers={}
keras_outputs={}
fpga_outputs={}

for file in layer_files:
    name=os.path.basename(file).split('.')[0]
    layers[name]=layer_file

for file in keras_output_files:
    name = os.path.basename(file).split('.')[0]
    keras_outputs[name] = np.fromfile(file, dtype=np.float32)

for file in fpga_output_files:
    name = os.path.basename(file).split('.')[0]
    fpga_outputs[name] = np.fromfile(file, dtype=np.float32)

keras_input = np.fromfile(network_debug_folder+'input_image.npy', dtype=np.float32)




if network_map:
    use_map



