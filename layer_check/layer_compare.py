import glob
import re
import numpy as np
import matplotlib.pyplot as plt


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





fpga_folder = "C:/Alex/Work/debug_check/mobilenet/jaguar/fpga_dump"
keras_folder = "C:/Alex/Work/debug_check/mobilenet/jaguar/keras_outputs"

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



x=np.arange(100)

count=0
for layer in range(len(keras_outputs)):
    print(count)
    count+=1
    fig=plt.figure(facecolor='w', edgecolor='k')
    fig.suptitle('Layer '+str(layer))
    if len(keras_outputs[layer].shape)==1:
        x=np.arange(keras_outputs[layer].size)
        # fig=plt.figure(facecolor='w', edgecolor='k')
        ax=fig.add_subplot(111)
        ax.plot(x,keras_outputs[layer],c='b',label='K',fillstyle='none')
        ax.plot(x,fpga_outputs[layer],c='g',label='F')
        plt.xlabel("Channel " + str(layer))
        ax.legend(loc=2)

    elif keras_outputs[layer].shape[1]==1 and keras_outputs[layer].shape[0]==1:
        x=np.arange(keras_outputs[layer].size)
        # fig=plt.figure(facecolor='w', edgecolor='k')
        ax=fig.add_subplot(111)
        ax.plot(x,keras_outputs[layer][0][0],c='b',label='K',fillstyle='none')
        ax.plot(x,fpga_outputs[layer][0][0],c='g',label='F')
        plt.xlabel("Channel " + str(layer))
        ax.legend(loc=2)

    else:
        channels = keras_outputs[layer].shape[2]
        width = 8
        height = int(channels/8)+1
        k_out = np.rollaxis(keras_outputs[layer], 2)
        f_out = np.rollaxis(fpga_outputs[layer], 2)
       
        # fig=plt.figure(facecolor='w', edgecolor='k')
        # fig.clim(-1,2)
        for channel in range(channels):
            ax=fig.add_subplot(width, height, channel+1)
            
            point_diffs = point_rel_difference(k_out[channel], f_out[channel])
            im=ax.imshow(point_diffs, clim=[-1,1], )
            # plt.clim(-1,1)
            cbar_ax = fig.add_axes([0.1,0.9, 0.2, 0.03])
            plt.xlabel("Channel " + str(channel))
            plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        # cbar_ax = fig.add_axes([0.1,0.9, 0.2, 0.03])
        
    
    plt.savefig("figure_"+str(layer))
    plt.close()

# fig=plt.figure(facecolor='w', edgecolor='k')
# ax=fig.add_subplot(221)
# ax.plot(x,keras_outputs[27][0][0][250:350],c='b',label='K',fillstyle='none')
# ax.plot(x,fpga_outputs[27][0][0][250:350],c='g',label='F')
# ax.legend(loc=2)

# ax=fig.add_subplot(222)
# ax.plot(x,keras_outputs[28][0][0][250:350],c='b',label='K',fillstyle='none')
# ax.plot(x,fpga_outputs[28][0][0][250:350],c='g',label='F')
# ax.legend(loc=2)

# plt.draw()


# fig=plt.figure(facecolor='w', edgecolor='k')
# ax=fig.add_subplot(221)
# ax.plot(x,keras_outputs[27][0][0][250:350],c='b',label='K',fillstyle='none')
# ax.plot(x,fpga_outputs[27][0][0][250:350],c='g',label='F')
# ax.legend(loc=2)

# ax=fig.add_subplot(222)
# ax.plot(x,keras_outputs[28][0][0][250:350],c='b',label='K',fillstyle='none')
# ax.plot(x,fpga_outputs[28][0][0][250:350],c='g',label='F')
# ax.legend(loc=2)

# plt.draw()

# plt.show()