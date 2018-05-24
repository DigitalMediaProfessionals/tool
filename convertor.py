#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
 Copyright(c) 2017 by Digital Media Professionals Inc.
 All rights reserved.
------------------------------------------------------------
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
from cnn_convertor import cnn_parser, fpga_layer


# Handle parameters
parser = argparse.ArgumentParser(description="DNN to FPGA convertor")
parser.add_argument("INPUT_INI", type=str, help="Input ini file")
args = parser.parse_args()


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
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_folder = os.path.abspath(output_folder)

# parse network
network = cnn_parser.parse_network(network_def, network_data, network_type,
                                   custom_layer)
fpga_net = fpga_layer.FPGANetwork(network, output_quantization)
fpga_net.output_network(output_folder, network_name, output_gensource,
                        output_gendoc, output_gengraph, graphviz_path)