#!/usr/bin/python3.6
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

if __name__ != "__main__":
    raise ValueError(
        "This module must be run directly (cannot be used in import)")
import sys
if (sys.version_info.major < 3 or
        (sys.version_info.major == 3 and sys.version_info.minor < 6)):
    raise ValueError("python version 3.6+ is required")
import argparse
import os
import configparser
import logging
from importlib import import_module
from cnn_convertor import cnn_parser, fpga_layer, debug_keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
import keras
import tensorflow as tf
import pathlib
import numpy as np

# Handle parameters
parser = argparse.ArgumentParser(description="DNN to FPGA convertor")
parser.add_argument("INPUT_INI", type=str, help="Input ini file")
parser.add_argument("--debug", type=bool, default=0, help="Split Keras network for debugging")
parser.add_argument("--input_file", type=str, help="Input image for network debugging")
parser.add_argument("--integer_test", type=bool, default=0, help="Input image for network debugging")
parser.add_argument("--random_input", type=bool, default=0, help="Input image for network debugging")
parser.add_argument("--r_offs", type=float, default=0, help="R offset for debug")
parser.add_argument("--g_offs", type=float, default=0, help="G offset for debug")
parser.add_argument("--b_offs", type=float, default=0, help="B offset for debug")
parser.add_argument("--scale", type=float, default=1, help="scale for debug")
	
args = parser.parse_args()
debug=args.debug
integer_test = args.integer_test
# if debug:
    # if args.input_file is None:
    #     sys.exit("Input image required")

if debug:
    print("Debug mode")
    # parse config file
config = configparser.ConfigParser(strict=False,
                                   inline_comment_prefixes=('#', ';'))
config.read_dict({'INPUT': {'custom_layer': ''},
                  'OUTPUT': {'generate_source': 0,
                             'generate_doxy': 0,
                             'generate_dot': 0,
                             'quantization': 1},
                  'OPTIONAL': {'verbose': 0,
                               'graphviz_path': ''}
                  })
abspath = os.path.abspath(args.INPUT_INI)
absdir = os.path.dirname(abspath)
config.read(abspath)

try:
    network_name = config['INPUT']['name']
    network_def = config['INPUT']['definition']
    if integer_test:
        keras.backend.clear_session()
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        tfconfig.log_device_placement = True  # to log device placement (on which device the operation ran)
                                            # (nothing gets printed in Jupyter, only if you run it standalone)
        sess = tf.Session(config=tfconfig)
        keras.backend.tensorflow_backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
        network_name = network_def.split('.')[0]
        debug_network_folder_name = 'debug/' +network_name+'_integer_model/'
        custom_objects = {'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}
        model_load = load_model(absdir+'\\'+network_def, custom_objects=custom_objects)
        pathlib.Path(debug_network_folder_name).mkdir(parents=True, exist_ok=True)
        model_load.save(debug_network_folder_name+network_name+'_original_model.h5')
        original_weights = model_load.get_weights()
        int_weights=[]
        for w in original_weights:
            if np.min(w)<0:
                int_weights.append(np.random.randint(-2,2,size=w.shape))
            else:
                int_weights.append(np.random.randint(0,2,size=w.shape))
        model_load.set_weights(int_weights)
        integer_path = debug_network_folder_name+network_name+'_integer_model.h5'
        model_load.save(integer_path)


    if 'data' in config['INPUT']:
        network_data = config['INPUT']['data']
    else:
        network_data = network_def
    network_type = config['INPUT']['origin']
    custom_layer = config['INPUT']['custom_layer']
    output_folder = config['OUTPUT']['output_folder']
    output_gensource = config.getboolean('OUTPUT', 'generate_source')
    output_gendoc = config.getboolean('OUTPUT', 'generate_doxy')
    output_gengraph = config.getboolean('OUTPUT', 'generate_dot')
    output_quantization = config.getboolean('OUTPUT', 'quantization')
    verbose = config.getboolean('OPTIONAL', 'verbose')
    graphviz_path = config['OPTIONAL']['graphviz_path']
except:
    print("Error parsing config file.")
    sys.exit(-1)

# set log levels
root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
if verbose:
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.INFO)
else:
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.WARNING)

if integer_test==1:
    network_def = os.path.abspath(integer_path)
    network_data = os.path.abspath(integer_path)
else:
    network_def = os.path.abspath(os.path.join(absdir, network_def))
    network_data = os.path.abspath(os.path.join(absdir, network_data))
if custom_layer != '':
    custom_layer = os.path.abspath(os.path.join(absdir, custom_layer))
    sys.path.append(os.path.dirname(custom_layer))
    custom_module = import_module(os.path.basename(custom_layer.strip('.py')))
    custom_layer = custom_module.custom_layer
else:
    custom_layer = {}
if not os.path.exists(network_def) or not os.path.exists(network_data):
    logging.error("The input network specified does not exist.")
network_type = network_type.upper()
# strip double quotes
if output_folder[0] == '"':
    output_folder = output_folder[1:-1]
output_folder = os.path.abspath(os.path.join(absdir, output_folder))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# parse network

network = cnn_parser.parse_network(network_def, network_data, network_type,
                                   custom_layer)
fpga_net = fpga_layer.FPGANetwork(network, output_quantization)

if debug:
    debug_keras.layer_split(fpga_net, network_def, input_params=args)

fpga_net.output_network(output_folder, network_name, output_gensource,
                        output_gendoc, output_gengraph, graphviz_path)


