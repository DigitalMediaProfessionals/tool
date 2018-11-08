
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
import ntpath
import json
import numpy as np
import tensorflow as tf
from scipy import misc
import re
import cv2 
from keras.utils import plot_model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session


import os
import re
import pathlib
import argparse
import sys
import csv

from keras.engine import Layer, InputSpec
from keras import initializers
from keras import backend as K


class LRN2D(Layer):

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError(
                "LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)

        extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                                   input_sqr,
                                   extra_channels[:, half_n + ch:, :, :]],
                                  axis=1)
        scale = self.k

        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta

        return X / scale

    def get_config(self):
        config = {"name": self.name,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from output_categories import mobilenet_categories
categories = mobilenet_categories()

keras.backend.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
									# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

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

def convert_image(image_file, args):
	r_offs 		= args.r_offs
	g_offs		= args.g_offs
	b_offs		= args.b_offs
	scale 		= args.scale
	input_file 	= image_file
	data = cv2.imread(input_file)
	
	data=data+[r_offs, g_offs, b_offs]
	data=data*scale
	# data=np.rollaxis(data,2)
	data = np.asarray([data.astype(np.float32)])

	return data

try:
	custom_objects = {'relu6': keras.layers.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D, 'LRN': LRN2D}
except:
	custom_objects = {'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.layers.DepthwiseConv2D, 'LRN': LRN2D}
	

parser = argparse.ArgumentParser()
parser.add_argument("INPUT_FOLDER", type=str)
parser.add_argument("--r_offs", type=float, default=0, help="R offset for debug")
parser.add_argument("--g_offs", type=float, default=0, help="G offset for debug")
parser.add_argument("--b_offs", type=float, default=0, help="B offset for debug")
parser.add_argument("--scale", type=float, default=1, help="scale for debug")
parser.add_argument("--image_labels", type=float, default=0, help="scale for debug")
parser.add_argument("--channels_first", type=bool, default=0, help="Channels First")
parser.add_argument("--transpose", type=bool, default=1, help="transpose on/off")

args = parser.parse_args()
network_folder_name = os.path.abspath(args.INPUT_FOLDER)+'\\'


# layers_folder = network_debug_folder + 'keras_networks\\'
# keras_outputs_folder = network_debug_folder + 'keras_outputs\\'
# fpga_outputs_folder = network_debug_folder + 'PLACE_FPGA_DUMPS_HERE\\'
# debug_output_folder=network_debug_folder+'fpga_dump_debug_outputs\\'

# pathlib.Path(debug_output_folder).mkdir(parents=True, exist_ok=True)


# network_folder_name = "C:\\Alex\\Work\\dv-sdk\\debug\\batch_mobilenet_keras_notquantized\\"
fpga_main_folder=network_folder_name+"fpga_layers"
fpga_subfolders=glob.glob(fpga_main_folder+'/*')

images_folder=network_folder_name+"images"
keras_folder=network_folder_name+"keras_outputs\\"
pathlib.Path(keras_folder).mkdir(parents=True, exist_ok=True)
# keras_networks_folder=network_folder_name+"keras_networks\\"


# fpga_files = glob.glob(fpga_folder+'/*')
fpga_files_dict = {}
for subfolder in fpga_subfolders:
	fpga_files_dict[os.path.basename(subfolder)]=glob.glob(subfolder+'/*')
image_files = glob.glob(images_folder+'/*')
keras_files = glob.glob(keras_folder+'/*')
# keras16_files = glob.glob(keras16_folder+'/*')
# keras_networks = glob.glob(keras_networks_folder+'/*')

# fpga_files = sorted(fpga_files)
image_files = sorted(image_files)
keras_files = sorted(keras_files)
# keras16_files = sorted(keras16_files)
# keras_networks = sorted(keras_networks)


if len(image_files)!=len(keras_files):
	model_file=glob.glob(network_folder_name+'*.h5')[0]
	with CustomObjectScope(custom_objects):
		keras_model = load_model(model_file)
	for i in range(len(image_files)):
		print(i)
		# fpganame=ntpath.basename(fpga_files[i])
		imagename=ntpath.basename(image_files[i])
		# if fpganame.find(imagename) == -1:
		# 	print("filename error")
		# 	print(imagename)
		# 	print(fpganame)
		# 	break

		converted_image = convert_image(image_files[i], args)
		
		image_predict = keras_model.predict(converted_image)
		image_predict.dump(keras_folder+imagename[:-4]+'.npy')

# if len(keras_networks)>0:
# 	for i in range(len(image_files)):
# 		print(i)
# 		imagename=ntpath.basename(image_files[i])
# 		data = convert_image(image_files[i]).astype(np.float16)
# 		for network_file in keras_networks:
# 			print(network_file)
# 			keras.backend.clear_session()
# 			with CustomObjectScope(custom_objects):
# 				keras_model=load_model(network_file)
# 			weights16=[]
# 			for weight in keras_model.get_weights():
# 				weights16.append(weight.astype(np.float16))
# 			keras_model.set_weights(weights16)
# 			data=keras_model.predict(data)
# 			keras.backend.clear_session()
# 			data=data.astype(np.float16)
# 		data.dump(keras16_folder+imagename[:-4]+'.npy')

keras_files = glob.glob(keras_folder+'/*')
# keras16_files = glob.glob(keras16_folder+'/*')
keras_files = sorted(keras_files)

# subfolder_summary={}
if args.image_labels==1:
	labels=[]
	for imagefile in image_files:
		labels.append(int(re.search(r'\d*(?=(.jpg|.png))',imagefile)[0]))
keras_data={}
fpga_data={}

for subfolder in list(fpga_files_dict.keys()):
	# subfolder_summary[subfolder]=dict.fromkeys([1, 2, 3, 4])
	keras_data[subfolder]=[]
	fpga_data[subfolder]=[]
	if len(fpga_files_dict[subfolder])!=len(keras_files):
		next
	else:
		fpga_files = sorted(fpga_files_dict[subfolder])
		file = open(network_folder_name+subfolder+"_batch_accuracy.csv", "w", newline='')
		wr = csv.writer(file)

		titles=['No.','Filename', 'CPU Result','FPGA Result', 'FPGA-CPU Match','CPU Confidence', 'FPGA Confidence', 'Result Error%', 'Abs Result Error%','Top 3 Similarity','Top 3 Match Order']
		if args.image_labels==1:
			titles.append('Label')
			titles.append('Keras Prediction')
			titles.append('FPGA Prediction')
			titles.append('Keras Correct')
			titles.append('FPGA Correct')

		wr.writerow(titles)

		for i in range(len(keras_files)):
			# fpganame=ntpath.basename(fpga_files[i])
			kerasname=ntpath.basename(keras_files[i])
			
			keras_data[subfolder].append(np.squeeze(np.load(keras_files[i])))
			fpga_data[subfolder].append(np.fromfile(fpga_files[i], dtype=np.float32))

			fpga_top3_index=fpga_data[subfolder][-1].argsort()[-3:][::-1]
			fpga_top3_score=[]
			for index in fpga_top3_index:
				fpga_top3_score.append(fpga_data[subfolder][-1][index])
			
			keras_top3_index=keras_data[subfolder][-1].argsort()[-3:][::-1]
			keras_top3_score=[]
			for index in keras_top3_index:
				keras_top3_score.append(keras_data[subfolder][-1][index])
			
			titles=['No.','Filename', 'CPU Result','FPGA Result', 'FPGA-CPU Match','CPU Confidence', 'FPGA Confidence', 'Result Error%', 'Abs Result Error%','Top 3 Similarity','Top 3 Match Order']

			filename=kerasname[:-4]
			cpu_result=keras_top3_index[0]
			fpga_result=fpga_top3_index[0]
			fpga_cpu_match = str(1) if keras_top3_index[0]==fpga_top3_index[0] else str(0)
			cpu_confidence=keras_top3_score[0]
			fpga_confidence=fpga_top3_score[0]
			result_error=100*(fpga_top3_score[0] - keras_top3_score[0])/fpga_top3_score[0] if keras_top3_index[0]==fpga_top3_index[0] else 'N/A'
			abs_result_error=np.abs(100*(fpga_top3_score[0] - keras_top3_score[0])/fpga_top3_score[0]) if keras_top3_index[0]==fpga_top3_index[0] else 'N/A'
			top3_similarity=str(len(np.intersect1d(keras_top3_index,fpga_top3_index)))
			top3_match=str(sum(keras_top3_index==fpga_top3_index))

			excel_row = [str(i), filename,cpu_result,fpga_result, fpga_cpu_match, cpu_confidence, fpga_confidence, 
			result_error, abs_result_error, top3_similarity, top3_match]


			if args.image_labels==1:
				label=labels[i]
				# if fpganame.find(kerasname[:-4]) == -1:
				# 	print("filename error")
				# 	print(kerasname)
				# 	print(fpganame)
				# 	break
				excel_row.append(categories[label])
				excel_row.append(categories[keras_top3_index[0]])
				excel_row.append(categories[fpga_top3_index[0]])
				excel_row.append(str(1) if keras_top3_index[0]==label else str(0))
				excel_row.append(str(1) if fpga_top3_index[0]==label else str(0))

			
			wr.writerow(excel_row)


		file.close()




print("done")





	