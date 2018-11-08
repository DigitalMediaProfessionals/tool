import numpy as np
from keras.models import load_model
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
        if fpga_min == -np.inf:
            fpga_min = "neg inf"
        fpga_mean=np.round(np.mean(fpga_data),5)
        if fpga_mean == -np.inf:
            fpga_mean ="neg inf" 
        fpga_std=np.round(np.std(fpga_data),5)
        fpga_output=['Layer '+str(layer)+ ' FPGA', fpga_max, fpga_min, fpga_mean, fpga_std]
        wr.writerow(fpga_output)
        if keras_output_length>layer:
            keras_data = keras_outputs[layer]
            keras_max=np.round(np.max(keras_data),5)
            keras_min=np.round(np.min(keras_data),5)
            if keras_min == -np.inf:
                keras_min ="neg inf"
            keras_mean=np.round(np.mean(keras_data),5)
            if keras_mean == -np.inf:
                keras_mean ="neg inf"  
            keras_std=np.round(np.std(keras_data),5)
            fpga_keras_rmse=rmse(fpga_data, keras_data)
            keras_output=['Layer '+str(layer)+ ' Keras', keras_max, keras_min, keras_mean, keras_std, fpga_keras_rmse]
            wr.writerow(keras_output)
            
        if debug_output_length>layer:
            debug_data = debug_outputs[layer]
            debug_max=np.round(np.max(fpga_data),5)
            debug_min=np.round(np.min(fpga_data),5)
            if debug_min == -np.inf:
                debug_min ="neg inf"
            debug_mean=np.round(np.mean(fpga_data),5)
            if debug_mean == -np.inf:
                debug_mean ="neg inf"
            debug_std=np.round(np.std(fpga_data),5)
            fpga_debug_rmse=rmse(fpga_data, debug_data)
            debug_output=['Layer '+str(layer)+ ' Debug', debug_max, debug_min, debug_mean, debug_std, fpga_debug_rmse]
            wr.writerow(debug_output)
            
        if keras16_output_length>layer:
            keras16_data = keras16_outputs[layer]
            keras16_max=np.round(np.max(keras16_data),5)
            keras16_min=np.round(np.min(keras16_data),5)
            if keras16_min == -np.inf:
                keras16_min = "neg inf"
            keras16_mean=np.round(np.mean(keras16_data),5)
            if keras16_mean == -np.inf:
                keras16_mean ="neg inf"            
            keras16_std=np.round(np.std(keras16_data),5)
            fpga_keras16_rmse=rmse(keras16_data, keras16_data)
            keras16_output=['Layer '+str(layer)+ ' Keras16', keras16_max, keras16_min, keras16_mean, keras16_std, fpga_keras16_rmse]
            wr.writerow(keras16_output)

    file.close()
    mobilenet_benchmark(fpga_data, keras_data, keras16_data)

    def final_layer_benchmark(fpga_data, keras_data, keras16_data):
        #mobilenet
        from output_categories import mobilenet_categories
        categories = mobilenet_categories()
        
        fpga_top5_index=fpga_data.argsort()[-5:][::-1]
        fpga_top5_score=[]
        for index in fpga_top5_index:
            fpga_top5_score.append(fpga_data[index])
        
        if keras_output_length>layer:
            keras_top5_index=keras_data.argsort()[-5:][::-1]
            keras_top5_score=[]
            for index in keras_top5_index:
                keras_top5_score.append(keras_data[index])
        
        
        if keras16_output_length>layer:
            keras16_top5_index=keras16_data.argsort()[-5:][::-1]
            keras16_top5_score=[]
            for index in keras16_top5_index:
                keras16_top5_score.append(keras16_data[index])


        file = open(network_debug_folder+"final_result_benchmark.csv", "w", newline='')
        wr = csv.writer(file)
        wr.writerow(['FPGA', 'Name','Index','Score','Keras 32bit', 'Name','Index','Score','%Error','Correct','Keras 16bit', 'Name','Index','Score','%Error','Correct'])

        for i in range(5):
            keras32_error = 100*(fpga_top5_score[i] - keras_top5_score[i])/fpga_top5_score[i]
            keras16_error = 100*(fpga_top5_score[i] - keras16_top5_score[i])/fpga_top5_score[i]

            wr.writerow([str(i+1), categories[fpga_top5_index[i]],str(fpga_top5_index[i]),str(fpga_top5_score[i]),'',
            categories[keras_top5_index[i]],str(keras_top5_index[i]),str(keras_top5_score[i]), str(keras32_error),'Yes' if keras_top5_index[i]==fpga_top5_index[i] else 'No','', 
            categories[keras16_top5_index[i]],str(keras16_top5_index[i]),str(keras16_top5_score[i]), str(keras16_error),'Yes' if keras16_top5_index[i]==fpga_top5_index[i] else 'No'])
        
        wr.writerow([''])
        wr.writerow(['Final Result'])
        wr.writerow(['', '','Keras 32bit','Keras 16bit'])
        wr.writerow(['', 'Result Correct','Yes' if keras_top5_index[0]==fpga_top5_index[0] else 'No','Yes' if keras16_top5_index[0]==fpga_top5_index[0] else 'No'])
        keras32_error = 100*(fpga_top5_score[0] - keras_top5_score[0])/fpga_top5_score[0]
        keras16_error = 100*(fpga_top5_score[0] - keras16_top5_score[0])/fpga_top5_score[0]
        wr.writerow(['', '%Error',keras32_error, keras16_error])
        

        wr.writerow([''])
        wr.writerow(['Top 3 Results Summary'])
        wr.writerow(['', '','Keras 32bit','Keras 16bit'])
        keras32_num_correct = len(np.intersect1d(keras_top5_index[:3],fpga_top5_index[:3]))
        keras16_num_correct = len(np.intersect1d(keras16_top5_index[:3],fpga_top5_index[:3]))
        keras32_order_correct = sum(keras_top5_index[:3]==fpga_top5_index[:3])
        keras16_order_correct = sum(keras_top5_index[:3]==fpga_top5_index[:3])
        wr.writerow(['', 'Num Correct',keras32_num_correct, keras16_num_correct])
        wr.writerow(['', 'Order Correct',keras32_order_correct, keras16_order_correct])

        wr.writerow([''])
        wr.writerow(['Top 5 Results Sumary'])
        wr.writerow(['', '','Keras 32bit','Keras 16bit'])
        keras32_num_correct = len(np.intersect1d(keras_top5_index,fpga_top5_index))
        keras16_num_correct = len(np.intersect1d(keras16_top5_index,fpga_top5_index))
        keras32_order_correct = sum(keras_top5_index==fpga_top5_index)
        keras16_order_correct = sum(keras_top5_index==fpga_top5_index)
        wr.writerow(['', 'Num Correct',keras32_num_correct, keras16_num_correct])
        wr.writerow(['', 'Order Correct',keras32_order_correct, keras16_order_correct])

        file.close()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT_FOLDER", type=str, help="Input debug folder")
    args = parser.parse_args()
    network_debug_folder = os.path.abspath(args.INPUT_FOLDER)+'\\'
    write_benchmark_file(network_debug_folder)

    # network_debug_folder = "C:\\Alex\\Work\\dv-sdk\\debug\\pose_network_complete\\"