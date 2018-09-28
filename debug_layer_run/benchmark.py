import numpy as np
from bokeh.plotting import *
from bokeh.models import ColumnDataSource
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider
import glob
import os
import re
import pathlib
import argparse
import sys
import csv
# import bkserve
# from random import random


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def remap(arr, dim):
    if len(dim) == 1:
        return arr
    sub_arrs = []
    for step in range(0, dim[2], 8):
        step_end = dim[2] if step + 8 > dim[2] else step + 8
        sub_arr = arr[dim[0] * dim[1] * step : dim[0] * dim[1] * step_end]
        sub_arr.shape = (dim[1], dim[0], step_end - step)
        sub_arr = np.transpose(sub_arr, axes = (1, 0, 2))
        sub_arrs.append(sub_arr)
    return np.concatenate(tuple(sub_arrs), axis = 2)

def point_rel_difference(p,f):
#     Relative Percent Difference = (x−y) / (|x|+|y|)
    point_diffs = np.subtract(f, p)
    abs_point_sums = np.abs(f)+np.abs(p)
    reldiff = (point_diffs/abs_point_sums)
    reldiff[reldiff!=reldiff] = 0
    return reldiff

def get_layer_data(layer):
    # datasets=[keras_outputs, fpga_outputs, debug_outputs, keras16_outputs]    
    # dataset1 = np.asarray(datasets[data1])
    k_out=None
    f_out=None
    d_out=None
    k16_out=None
    # if (dataset1[layer].shape[0]==1)  or (len(dataset1[layer].shape)==1):
    #     x=dataset1[layer].size
    if len(keras_outputs[layer].shape)==3:
        if keras_output_length>layer:
            k_out = keras_outputs[layer][0][0]
        if fpga_output_length>layer:
            f_out = fpga_outputs[layer][0][0]
        if debug_output_length>layer:
            d_out = debug_outputs[layer][0][0]
        if keras16_output_length>layer:
            k16_out = keras16_outputs[layer][0][0]
        
    else:
        if keras_output_length>layer:
            k_out = keras_outputs[layer]
        if fpga_output_length>layer:
            f_out = fpga_outputs[layer]
        if debug_output_length>layer:
            d_out = debug_outputs[layer]
        if keras16_output_length>layer:
            k16_out = keras16_outputs[layer]
    # else:

        # k_out = np.rollaxis(keras_outputs[layer], 2)
        # f_out = np.rollaxis(fpga_outputs[layer], 2)
        # d_out = np.rollaxis(debug_outputs[layer], 2)
        # k16_out = np.rollaxis(keras16_outputs[layer], 2)

    return k_out, f_out, d_out, k16_out


# parser = argparse.ArgumentParser()
# # parser.add_argument("INPUT_FOLDER", type=str, help="Network folder")
# parser.add_argument("--save_layers", type=str, help="Saves all layers (can't be used with server)")


# args = parser.parse_args()
# # network_debug_folder = os.path.abspath(args.INPUT_FOLDER)+'\\'

# if not os.path.exists(network_debug_folder):
#     print("Folder does not exist")
#     sys.exit(0)






def write_benchmark_file(network_debug_folder):

    keras_folder = network_debug_folder + 'keras_outputs\\'
    keras16_folder = network_debug_folder + 'keras_outputs_float16\\'
    fpga_folder = network_debug_folder + 'PLACE_FPGA_DUMPS_HERE\\'
    debug_output_folder=network_debug_folder+'fpga_dump_debug_outputs\\'

    fpga_files = glob.glob(fpga_folder+'/*')
    keras_files = glob.glob(keras_folder+'/*')
    keras16_files = glob.glob(keras16_folder+'/*')
    debug_files = glob.glob(debug_output_folder+'/*')


    fpga_regex = "layer_input.bin$"
    r=re.compile(fpga_regex)
    fpga_files = list(filter(lambda x: not r.search(x), fpga_files))
    if len(fpga_files)>100:
        fpga_layers=[]
        for i in range(10):
            filenum=('0'+str(i))
            fpga_layers.append(fpga_folder+'layer'+filenum+'.bin')
        for i in range(10,len(fpga_files)):
            filenum=str(i)
            fpga_layers.append(fpga_folder+'layer'+filenum+'.bin')
        fpga_files=fpga_layers

    if len(fpga_files) != len(keras_files):
        print("Number of input files does not match")
    # num_files = min(len(fpga_files), len(keras_files), len(debug_files))
        # sys.exit()

    if len(debug_files) ==0:
        print("No debug data")
        # sys.exit()



    keras_outputs = []    
    for file in keras_files:
        keras_outputs.append(np.load(file)[0])
    # keras_outputs = keras_outputs[:num_files]

    keras16_outputs = []    
    for file in keras16_files:
        keras16_outputs.append(np.load(file)[0])


    fpga_outputs=[]
    for i in range(len(fpga_files)):
        fpga_dump = np.fromfile(fpga_files[i], dtype=np.float16)
        if len(fpga_dump)!=keras_outputs[i].size:
            fpga_dump = np.fromfile(fpga_files[i], dtype=np.float32)
        # print(i)
        fpga_dump = remap(fpga_dump, keras_outputs[i].shape)
        fmax = np.nanmax(fpga_dump[fpga_dump!=np.inf])
        fpga_dump[np.isinf(fpga_dump)] = fmax
        fpga_dump[np.isnan(fpga_dump)] = fmax
        fpga_outputs.append(fpga_dump)
    # fpga_outputs = fpga_outputs[:num_files]

    debug_outputs = []    
    for file in debug_files:
        debug_outputs.append(np.load(file)[0])



    keras_output_length = len(keras_outputs)
    keras16_output_length = len(keras16_outputs)
    fpga_output_length = len(fpga_outputs)
    debug_output_length = len(debug_outputs)

    fpga_output=[]
    keras_output=[]
    debug_output=[]
    keras16_output=[]
    fpga_keras_rmse=[]
    fpga_debug_rmse=[]
    fpga_keras16_rmse=[]
    
    file = open(network_debug_folder+"benchmark.csv", "w", newline='')
    wr = csv.writer(file)

    wr.writerow(['Layer', 'Max', 'Min', 'Mean', 'StDev', 'RMS Error'])
    for layer in range(fpga_output_length):
        

        fpga_data = fpga_outputs[layer]
        fpga_max=np.round(np.max(fpga_data),5)
        fpga_min=np.round(np.min(fpga_data),5)
        fpga_mean=np.round(np.mean(fpga_data),5)
        fpga_std=np.round(np.std(fpga_data),5)
        fpga_output=['Layer '+str(layer)+ ' FPGA', fpga_max, fpga_min, fpga_mean, fpga_std]
        wr.writerow(fpga_output)
        if keras_output_length>layer:
            keras_data = keras_outputs[layer]
            keras_max=np.round(np.max(keras_data),5)
            keras_min=np.round(np.min(keras_data),5)
            keras_mean=np.round(np.mean(keras_data),5)
            keras_std=np.round(np.std(keras_data),5)
            fpga_keras_rmse=rmse(fpga_data, keras_data)
            keras_output=['Layer '+str(layer)+ ' Keras', keras_max, keras_min, keras_mean, keras_std, fpga_keras_rmse]
            wr.writerow(keras_output)
            
        if debug_output_length>layer:
            debug_data = debug_outputs[layer]
            debug_max=np.round(np.max(fpga_data),5)
            debug_min=np.round(np.min(fpga_data),5)
            debug_mean=np.round(np.mean(fpga_data),5)
            debug_std=np.round(np.std(fpga_data),5)
            fpga_debug_rmse=rmse(fpga_data, debug_data)
            debug_output=['Layer '+str(layer)+ ' Debug', debug_max, debug_min, debug_mean, debug_std, fpga_debug_rmse]
            wr.writerow(debug_output)
            
        if keras16_output_length>layer:
            keras16_data = keras16_outputs[layer]
            keras16_max=np.round(np.max(keras16_data),5)
            keras16_min=np.round(np.min(keras16_data),5)
            keras16_mean=np.round(np.mean(keras16_data),5)
            keras16_std=np.round(np.std(keras16_data),5)
            fpga_keras16_rmse=rmse(keras16_data, keras16_data)
            keras16_output=['Layer '+str(layer)+ ' Keras16', keras16_max, keras16_min, keras16_mean, keras16_std, fpga_keras16_rmse]
            wr.writerow(keras16_output)

    file.close()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT_FOLDER", type=str, help="Input debug folder")
    args = parser.parse_args()
    network_debug_folder = os.path.abspath(args.INPUT_FOLDER)+'\\'
    write_benchmark_file(network_debug_folder)

    # network_debug_folder = "C:\\Alex\\Work\\dv-sdk\\debug\\pose_network_complete\\"