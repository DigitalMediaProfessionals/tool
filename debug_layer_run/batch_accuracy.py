
import keras
from keras import layers
from keras import models
from keras.layers import Input, DepthwiseConv2D, Conv2D
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope, deserialize_keras_object
from keras.layers import deserialize as layer_from_config
import matplotlib
import matplotlib.pyplot as plt
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

result_data={}

# threshold_dict = dict([('indices',[]),('match',[])])


result_data['cpu_result']={}
result_data['cpu_result']['total']=dict([('filename',[]),('cpu_data',[]), ('cpu_result',[]),
			 ('cpu_confidence',[]), ('cpu_label_correct',[]), ('fpga_label_correct',[])
			 ])

thresholds = dict([
		(0.50,dict([('indices',[]),('cpu_label_correct',[]) ])), 
		(0.60,dict([('indices',[]),('cpu_label_correct',[]) ])),
		(0.70,dict([('indices',[]),('cpu_label_correct',[]) ])),
		(0.75,dict([('indices',[]),('cpu_label_correct',[]) ])),
		(0.80,dict([('indices',[]),('cpu_label_correct',[]) ])),
		(0.85,dict([('indices',[]),('cpu_label_correct',[]) ])),
		(0.90,dict([('indices',[]),('cpu_label_correct',[]) ])),
		(0.95,dict([('indices',[]),('cpu_label_correct',[]) ]))])
result_data['cpu_result']['thresholds']=thresholds

for i in range(len(keras_files)):
	cpu_result_total=result_data['cpu_result']['total']
			
	cpu_result_total['cpu_data'].append(np.squeeze(np.load(keras_files[i])))

	cpu_top3_index=cpu_result_total['cpu_data'][-1].argsort()[-3:][::-1]
	cpu_top3_score=[]
	for index in cpu_top3_index:
		cpu_top3_score.append(cpu_result_total['cpu_data'][-1][index])

	cpu_result_total['filename'].append(ntpath.basename(keras_files[i])[:-4])
	cpu_result_total['cpu_result'].append(cpu_top3_index[0])
	cpu_result_total['cpu_confidence'].append(cpu_top3_score[0])

	if args.image_labels==1:
		label=labels[i]
		cpu_label_correct = 1 if cpu_top3_index[0]==label else 0
		cpu_result_total['cpu_label_correct'].append(cpu_label_correct)
				
	for threshold in list(result_data['cpu_result']['thresholds'].keys()):
		if cpu_top3_score[0]>threshold:
			result_data['cpu_result']['thresholds'][threshold]['indices'].append(i)
			if args.image_labels==1:
				result_data['cpu_result']['thresholds'][threshold]['cpu_label_correct'].append(cpu_label_correct)



for subfolder in list(fpga_files_dict.keys()):
	# subfolder_summary[subfolder]=dict.fromkeys([1, 2, 3, 4])
	total_dict=dict([('filename',[]),('cpu_data',[]), ('fpga_data',[]), ('cpu_result',[]), ('fpga_result',[]), ('fpga_cpu_match',[]),
			 ('cpu_confidence',[]), ('fpga_confidence',[]), ('result_error',[]), ('abs_result_error',[]), ('top3_similarity',[]), ('top3_match',[]),
			 ('cpu_label_correct',[]), ('fpga_label_correct',[])
			 ])

	thresholds = dict([
		(0.50,dict([('indices',[]),('match',[]),('cpu_label_correct',[]), ('fpga_label_correct',[])])), 
		(0.60,dict([('indices',[]),('match',[]),('cpu_label_correct',[]), ('fpga_label_correct',[])])),
		(0.70,dict([('indices',[]),('match',[]),('cpu_label_correct',[]), ('fpga_label_correct',[])])),
		(0.75,dict([('indices',[]),('match',[]),('cpu_label_correct',[]), ('fpga_label_correct',[])])),
		(0.80,dict([('indices',[]),('match',[]),('cpu_label_correct',[]), ('fpga_label_correct',[])])),
		(0.85,dict([('indices',[]),('match',[]),('cpu_label_correct',[]), ('fpga_label_correct',[])])),
		(0.90,dict([('indices',[]),('match',[]),('cpu_label_correct',[]), ('fpga_label_correct',[])])),
		(0.95,dict([('indices',[]),('match',[]),('cpu_label_correct',[]), ('fpga_label_correct',[])]))])


	result_data[subfolder]=dict([('total',total_dict),('thresholds',thresholds)])

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

			# result_data[subfolder]['total'].append(dict.fromkeys(['filename','cpu_data', 'fpga_data', 'cpu_result', 'fpga_result', 'fpga_cpu_match',
			#  'cpu_confidence', 'fpga_confidence', 'result_error', 'abs_result_error', 'top3_similarity', 'top3_match']))

			subfolder_image_total=result_data[subfolder]['total']
			
			subfolder_image_total['cpu_data'].append(np.squeeze(np.load(keras_files[i])))
			subfolder_image_total['fpga_data'].append(np.fromfile(fpga_files[i], dtype=np.float32))

			cpu_top3_index=subfolder_image_total['cpu_data'][-1].argsort()[-3:][::-1]
			cpu_top3_score=[]
			for index in cpu_top3_index:
				cpu_top3_score.append(subfolder_image_total['cpu_data'][-1][index])

			fpga_top3_index=subfolder_image_total['fpga_data'][-1].argsort()[-3:][::-1]
			fpga_top3_score=[]
			for index in fpga_top3_index:
				fpga_top3_score.append(subfolder_image_total['fpga_data'][-1][index])
			
			fpga_cpu_match = 1 if cpu_top3_index[0]==fpga_top3_index[0] else 0

			titles=['No.','Filename', 'CPU Result','FPGA Result', 'FPGA-CPU Match','CPU Confidence', 'FPGA Confidence', 'Result Error%', 'Abs Result Error%','Top 3 Similarity','Top 3 Match Order']

			subfolder_image_total['filename'].append(ntpath.basename(keras_files[i])[:-4])
			subfolder_image_total['cpu_result'].append(cpu_top3_index[0])
			subfolder_image_total['fpga_result'].append(fpga_top3_index[0])
			subfolder_image_total['fpga_cpu_match'].append(fpga_cpu_match)
			subfolder_image_total['cpu_confidence'].append(cpu_top3_score[0])
			subfolder_image_total['fpga_confidence'].append(fpga_top3_score[0])
			subfolder_image_total['result_error'].append(100*(fpga_top3_score[0] - cpu_top3_score[0])/fpga_top3_score[0] if cpu_top3_index[0]==fpga_top3_index[0] else 'N/A')
			subfolder_image_total['abs_result_error'].append(np.abs(100*(fpga_top3_score[0] - cpu_top3_score[0])/fpga_top3_score[0]) if cpu_top3_index[0]==fpga_top3_index[0] else 'N/A')
			subfolder_image_total['top3_similarity'].append(len(np.intersect1d(cpu_top3_index,fpga_top3_index)))
			subfolder_image_total['top3_match'].append(sum(cpu_top3_index==fpga_top3_index))





			excel_row = [i, subfolder_image_total['filename'][-1],subfolder_image_total['cpu_result'][-1],subfolder_image_total['fpga_result'][-1], subfolder_image_total['fpga_cpu_match'][-1], 
			subfolder_image_total['cpu_confidence'][-1], subfolder_image_total['fpga_confidence'][-1], subfolder_image_total['result_error'][-1], 
			subfolder_image_total['abs_result_error'][-1], subfolder_image_total['top3_similarity'][-1], subfolder_image_total['top3_match'][-1]]


			if args.image_labels==1:
				label=labels[i]
				# if fpganame.find(kerasname[:-4]) == -1:
				# 	print("filename error")
				# 	print(kerasname)
				# 	print(fpganame)
				# 	break
				cpu_label_correct = 1 if cpu_top3_index[0]==label else 0
				fpga_label_correct = 1 if fpga_top3_index[0]==label else 0

				subfolder_image_total['cpu_label_correct'].append(cpu_label_correct)
				subfolder_image_total['fpga_label_correct'].append(fpga_label_correct)

				excel_row.append(categories[label])
				excel_row.append(categories[cpu_top3_index[0]])
				excel_row.append(categories[fpga_top3_index[0]])
				excel_row.append(cpu_label_correct)
				excel_row.append(fpga_label_correct)


			for threshold in list(result_data[subfolder]['thresholds'].keys()):
				if cpu_top3_score[0]>threshold:
					result_data[subfolder]['thresholds'][threshold]['indices'].append(i)
					result_data[subfolder]['thresholds'][threshold]['match'].append(fpga_cpu_match)
					if args.image_labels==1:
						result_data[subfolder]['thresholds'][threshold]['cpu_label_correct'].append(cpu_label_correct)
						result_data[subfolder]['thresholds'][threshold]['fpga_label_correct'].append(fpga_label_correct)

			
			wr.writerow(excel_row)

		# sklearn.metrics.accuracy_score(labels,result_data[subfolder]['total']['fpga_result'])
		# 0.483
		# import sklearn.metrics
		file.close()





result_data['cpu_result']['summary']={}
result_data['cpu_result']['summary']['threshold_names']=['Total']

result_data['cpu_result']['summary']['num_images']=[len(result_data['cpu_result']['total']['cpu_label_correct'])]
result_data['cpu_result']['summary']['num_correct_label']=[sum(result_data['cpu_result']['total']['cpu_label_correct'])]
result_data['cpu_result']['summary']['percent_correct']=[100*result_data['cpu_result']['summary']['num_correct_label'][-1]/result_data['cpu_result']['summary']['num_images'][-1]]


for t in list(result_data['cpu_result']['thresholds'].keys()):
	result_data['cpu_result']['summary']['threshold_names'].append(t)
	result_data['cpu_result']['summary']['num_images'].append(len(result_data['cpu_result']['thresholds'][t]['cpu_label_correct']))
	result_data['cpu_result']['summary']['num_correct_label'].append(sum(result_data['cpu_result']['thresholds'][t]['cpu_label_correct']))
	result_data['cpu_result']['summary']['percent_correct'].append(100*result_data['cpu_result']['summary']['num_correct_label'][-1]/result_data['cpu_result']['summary']['num_images'][-1])

result_keys = list(result_data.keys())
result_keys.remove('cpu_result')
for result_key in result_keys:
	result_data[result_key]['summary']={}
	# result_data[result_key]['summary']['num_images']=[len(result_data[result_key]['total']['fpga_label_correct'])]
	result_data[result_key]['summary']['num_correct_label']=[sum(result_data[result_key]['total']['fpga_label_correct'])]
	result_data[result_key]['summary']['percent_correct']=[100*result_data[result_key]['summary']['num_correct_label'][-1]/result_data['cpu_result']['summary']['num_images'][0]]

	for i, t in enumerate(list(result_data[result_key]['thresholds'].keys())):
		# result_data[result_key]['summary']['num_images'].append(len(result_data[result_key]['thresholds'][t]['fpga_label_correct']))
		result_data[result_key]['summary']['num_correct_label'].append(sum(result_data[result_key]['thresholds'][t]['fpga_label_correct']))
		result_data[result_key]['summary']['percent_correct'].append(100*result_data[result_key]['summary']['num_correct_label'][-1]/result_data['cpu_result']['summary']['num_images'][i+1])


file = open(network_folder_name+"accuracy_summary.csv", "w", newline='')
wr = csv.writer(file)

row =['Confidence', 'Total', 'CPU Correct', 'CPU %']
for result_key in result_keys:
	row.append(result_key+' Correct')
	row.append(result_key+' %')
wr.writerow(row)




for i, t in enumerate(list(result_data['cpu_result']['summary']['threshold_names'])):
	row=[t]
	row.append(result_data['cpu_result']['summary']['num_images'][i])
	row.append(result_data['cpu_result']['summary']['num_correct_label'][i])
	row.append(result_data['cpu_result']['summary']['percent_correct'][i])

	for result_key in result_keys:
		row.append(result_data[result_key]['summary']['num_correct_label'][i])
		row.append(result_data[result_key]['summary']['percent_correct'][i])
	wr.writerow(row)
	
file.close()






fig = plt.figure()
ax = plt.axes()
colormap = matplotlib.cm.Set1.colors
x_range = range(len(result_data['cpu_result']['summary']['threshold_names']))
plt.xticks(x_range, result_data['cpu_result']['summary']['threshold_names'])
plt.ylim(0, 100);
plt.title(args.INPUT_FOLDER+" Accuracy")
plt.xlabel("Confidence Threshold")
plt.plot(x_range, result_data['cpu_result']['summary']['percent_correct'], color = colormap[0], label='cpu_result', marker="o")
result_keys = list(result_data.keys())
result_keys.remove('cpu_result')
for i, result_key in enumerate(result_keys):
	plt.plot(x_range, result_data[result_key]['summary']['percent_correct'], color = colormap[i+1], label=result_key,  marker=".")
plt.legend();

fig.savefig(network_folder_name+"AccuracySummary.png")



print("done")





	