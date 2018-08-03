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
# import bkserve
# from random import random
from bokeh.layouts import column, row
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.models import HoverTool, GlyphRenderer, CustomJSHover
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker, Label

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

def get_layer_data(layer, data1, data2):
    datasets=[keras_outputs, fpga_outputs, debug_outputs]
    dataset1 = datasets[data1]
    
    if len(dataset1[layer].shape)==1:
        x=dataset1[layer].size
        return x, datasets[0][layer], datasets[1][layer],datasets[2][layer], 0, 0

    elif dataset1[layer].shape[1]==1 and dataset1[layer].shape[0]==1:
        x=dataset1[layer].size
        return x, datasets[0][layer][0][0], datasets[1][layer][0][0],datasets[2][layer][0][0], 0, 0

    else:
        dataset2 = datasets[data2]
        min_val=[]
        min_val.append([np.min (sublist) for sublist in datasets[0][layer]])
        min_val.append([np.min (sublist) for sublist in datasets[1][layer]])
        min_val.append([np.min (sublist) for sublist in datasets[2][layer]])
        min_val=np.min(min_val)
        max_val=[]
        max_val.append([np.max (sublist) for sublist in datasets[0][layer]])
        max_val.append([np.max (sublist) for sublist in datasets[1][layer]])
        max_val.append([np.max (sublist) for sublist in datasets[2][layer]])
        max_val=np.min(max_val)
        channels = dataset1[layer].shape[2]
        d1_out = np.rollaxis(dataset1[layer], 2)
        d2_out = np.rollaxis(dataset2[layer], 2)
        return 0, d1_out, d2_out,None, min_val, max_val

def get_image(channel, view, dataset1, dataset2):
    if view==0:
        image=dataset1[channel]
    if view==1:
        image = dataset2[channel]
    if view==2:
        image = dataset1[channel]-dataset2[channel]
    if view==3:
        image = point_rel_difference(dataset1[channel] , dataset2[channel])
    return image

def make_plot(layer, view, data1, data2):
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    length, dataset1, dataset2,dataset3, min_val, max_val = get_layer_data(layer, data1, data2)
    plotgrid=[]

    if length == 0:
        channels = dataset1.shape[0]
        width = 8
        height = int(channels/width)+1

        row=[]
        imgs=[]
        for channel in range(channels):
            img=get_image(channel, view, dataset1, dataset2)
            
            img=img[::-1] #image shows upside down by default
            if view==3: 
                img=np.pad(img, pad_width=1, mode='constant', constant_values=1)
                view_label='Normalised Difference'
            else:
                img=np.pad(img, pad_width=1, mode='constant', constant_values=np.max(img))
                if view==0:
                    view_label=data1select.value
                if view==1:
                    view_label=data2select.value
                if view==2:
                    view_label='{0}, {1} Difference'.format(data1select.value, data2select.value)
                if view==3:
                    view_label='{0}, {1} Normalised Difference'.format(data1select.value, data2select.value)

            if channel%8==0:
                if channel!=0:
                    row=np.hstack(row)
                    imgs.append(row)
                    row=[]
            row.append(img)

        row=np.hstack(row)
        imgs.append(row)
        if imgs[0].shape!=imgs[-1].shape:
            imgs[-1] = np.pad(imgs[-1], pad_width=[(0,0),(0,imgs[0].shape[1]-imgs[-1].shape[1])], mode='constant', constant_values=0)
        imgs = np.vstack(imgs)

        p = figure(title=view_label + ' View. Layer: '+ os.path.basename(keras_files[layer]),x_range=(0, imgs.shape[1]), y_range=(0, imgs.shape[0]),toolbar_location="left", plot_width=width*200, plot_height=height*150)
        p.title.text_font_size = "20px"
        if view==3:
            color_mapper = LinearColorMapper(palette="Viridis256", low=-1, high=1)
        else:
            color_mapper = LinearColorMapper(palette="Viridis256", low=min_val, high=max_val)
        p.image(image=[imgs], x=0, y=0, dw=imgs.shape[1], dh=imgs.shape[0], color_mapper=color_mapper)

        hover = HoverTool(tooltips = [("x", "$x{int}"), ("y", "$y{int}"), ("value", "@image")])
        p.add_tools(hover)
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                 label_standoff=5, border_line_color=None, location=(0,0), width=10, height=250)
        p.add_layout(color_bar, 'right')

    else:
        channels=0
        x_vals = np.arange(length)
        p = figure(tools=TOOLS,title="Layer "+str(layer) + 'Graph View. Layer: '+ os.path.basename(keras_files[layer]), plot_width=1200, plot_height=800)
        if debug_present:
            source=ColumnDataSource(data=dict(x=x_vals, keras=dataset1, fpga=dataset2, debug=dataset3))
            d_line = p.line('x', 'debug', source=source,legend=dict(value="Debug"), line_color="green", line_width=1)
            hover = HoverTool(tooltips = [("x", "$x{(0)}"), ("keras", "@keras"), ("fpga", "@fpga"), ("debug", "@debug")])
        else:
            source=ColumnDataSource(data=dict(x=x_vals, keras=dataset1, fpga=dataset2, debug=dataset3))
            hover = HoverTool(tooltips = [("x", "$x{(0)}"), ("keras", "@keras"), ("fpga", "@fpga")])
        k_line = p.line('x', 'keras', source=source, legend=dict(value="Keras"), line_color="red", line_width=1)
        f_line = p.line('x', 'fpga', source=source,legend=dict(value="FPGA"), line_color="blue", line_width=1)
        
        p.legend.click_policy="hide"
        
        print(hover.renderers)
        p.add_tools(hover)

    return p, channels

def update_plot(attrname, old, new):
    root = curdoc().roots[1]
    curdoc().remove_root(root)
    layer=layer_dict[layerselect.value]
    view=view_dict[viewselect.value]
    data1=data_dict[data1select.value]
    data2=data_dict[data2select.value]
    p, channels=make_plot(layer,view, data1, data2)

    curdoc().add_root(column(p))

parser = argparse.ArgumentParser()
parser.add_argument("INPUT_FOLDER", type=str, help="Network folder")
parser.add_argument("--save_layers", type=str, help="Saves all layers (can't be used with server)")


args = parser.parse_args()
network_debug_folder = os.path.abspath(args.INPUT_FOLDER)+'\\'

if not os.path.exists(network_debug_folder):
    print("crud")
    sys.exit(0)

# network_debug_folder = "C:\\Alex\\Work\\fpga_perf\\debug\\mobilenet\\"


keras_folder = network_debug_folder + 'keras_outputs\\'
fpga_folder = network_debug_folder + 'PLACE_FPGA_DUMPS_HERE\\'
debug_output_folder=network_debug_folder+'fpga_dump_debug_outputs\\'
visual_output_folder=network_debug_folder+'visualisation_outputs\\'

pathlib.Path(visual_output_folder).mkdir(parents=True, exist_ok=True)



# fpga_folder = "C:/Alex/Work/debug_check/posenet/fpga_dump"
# keras_folder = "C:/Alex/Work/debug_check/posenet/keras_outputs"

fpga_files = glob.glob(fpga_folder+'/*')
keras_files = glob.glob(keras_folder+'/*')
debug_files = glob.glob(debug_output_folder+'/*')

fpga_regex = "layer_input.bin$"
r=re.compile(fpga_regex)
fpga_files = list(filter(lambda x: not r.search(x), fpga_files))

if len(fpga_files) != len(keras_files):
    print("Number of input files does not match")
    sys.exit()

if len(debug_files) ==0:
    print("No debug data")
    sys.exit()



keras_outputs = []    
for file in keras_files:
    keras_outputs.append(np.load(file)[0])

fpga_outputs=[]
for i in range(len(fpga_files)):
    fpga_dump = np.fromfile(fpga_files[i], dtype=np.float16)
    if len(fpga_dump)!=keras_outputs[i].size:
        fpga_dump = np.fromfile(fpga_files[i], dtype=np.float32)
    # print(i)
    fpga_dump = remap(fpga_dump, keras_outputs[i].shape)
    fpga_outputs.append(fpga_dump)

debug_outputs = []    
for file in debug_files:
    debug_outputs.append(np.load(file)[0])

keras_length = np.arange(len(keras_outputs))

layer_name=[]
layer_dict={}
for l in keras_length:
    l_name=str(l).zfill(2)+': '+os.path.basename(keras_files[l])
    layer_name.append(l_name)
    layer_dict[l_name]=l
layer=layer_name[0]

view_name=[]
view_dict={
    'Data1'         : 0,
    'Data2'          : 1,
    'Difference'    : 2,
    'Normalised Difference':3,
}

data_dict={}
data_dict['Keras']=0
data_dict['FPGA']=1
if len(debug_outputs)!=0:
    debug_present=1
    data_dict['Debug']=2
else:
    debug_present=0
    debug_outputs=keras_outputs

layerselect = Select(title="Layer:", value=layer, options=layer_name)
viewselect = Select(title="View:", value='Data1', options=list(view_dict.keys()))
data1select=Select(title="Dataset 1:", value='Keras', options=list(data_dict.keys()))
data2select=Select(title="Dataset 2:", value='FPGA', options=list(data_dict.keys()))

layerselect.on_change('value', update_plot)
viewselect.on_change('value', update_plot)
data1select.on_change('value', update_plot)
data2select.on_change('value', update_plot)


if args.save_layers==1:
    for l in range(len(keras_outputs)):
        html_name="layer"+str(l).zfill(3)+"_norm_difference"
        output_file(visual_output_folder+html_name+".html", title=html_name)
        p, channels=make_plot(l, 0, 0, 1)
        save(column(p))
        html_name="layer"+str(l).zfill(3)+"_norm_difference"
        output_file(visual_output_folder+html_name+".html", title=html_name)
        p, channels=make_plot(l, 1, 0, 1)
        save(column(p))
        html_name="layer"+str(l).zfill(3)+"_norm_difference"
        output_file(visual_output_folder+html_name+".html", title=html_name)
        p, channels=make_plot(l, 2, 0, 1)
        save(column(p))
        html_name="layer"+str(l).zfill(3)+"_norm_difference"
        output_file(visual_output_folder+html_name+".html", title=html_name)
        p, channels=make_plot(l, 3, 0, 1)
        save(column(p))
    


p, channels=make_plot(0, 0, 0, 1)

curdoc().add_root(column(row(layerselect, viewselect, data1select, data2select)))
curdoc().add_root(column(p))
show(column(row( data1select, data2select, layerselect, viewselect), p))