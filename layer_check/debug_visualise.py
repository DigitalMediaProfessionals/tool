import glob
import re
import numpy as np
from visualisation import bkserve


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

fpga_folder = "C:/Alex/Work/debug_check/mobilenet/jaguar/fpga_dump"
keras_folder = "C:/Alex/Work/debug_check/posenet/jaguar/keras_outputs"

fpga_files = glob.glob(fpga_folder+'/*')
keras_files = glob.glob(keras_folder+'/*')
fpga_regex = "layer_input.bin$"
r=re.compile(fpga_regex)
fpga_files = list(filter(lambda x: not r.search(x), fpga_files))

if len(fpga_files) != len(keras_files):
    print("Number of input files does not match")

keras_outputs = []    
for i in range(len(keras_files)):
    keras_outputs.append(np.load(keras_files[i])[0])


fpga_outputs=[]
for i in range(len(fpga_files)):
    fpga_dump = np.fromfile(fpga_files[i], dtype=np.float16)
    if len(fpga_dump)!=keras_outputs[i].size:
        fpga_dump = np.fromfile(fpga_files[i], dtype=np.float32)
    fpga_dump = remap(fpga_dump, keras_outputs[i].shape)
    fpga_outputs.append(fpga_dump)



print('2545435vtrgfdghfx')
bkserve.bokehserver(keras_outputs, fpga_outputs)



