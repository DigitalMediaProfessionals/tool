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
from bokeh.palettes import RdYlBu3, viridis
from bokeh.plotting import figure, curdoc
from bokeh.models import HoverTool, GlyphRenderer
from bokeh.models import LinearColorMapper, CategoricalColorMapper, ColorBar, BasicTicker, Label

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
    datasets=[keras_outputs, fpga_outputs, debug_outputs, keras16_outputs]    
    # dataset1 = np.asarray(datasets[data1])
    dataset1 = datasets[data1]
    if (dataset1[layer].shape[0]==1)  or (len(dataset1[layer].shape)==1):
        x=dataset1[layer].size
        if len(dataset1[layer].shape)==3:
            k_out=None
            f_out=None
            d_out=None
            k16_out=None
            if keras_output_length>layer:
                k_out = datasets[0][layer][0][0]
            if fpga_output_length>layer:
                f_out = datasets[1][layer][0][0]
            if debug_output_length>layer:
                d_out = datasets[2][layer][0][0]
            if keras16_output_length>layer:
                k16_out = datasets[3][layer][0][0]
            
        else:
            k_out=None
            f_out=None
            d_out=None
            k16_out=None
            if keras_output_length>layer:
                k_out = datasets[0][layer]
            if fpga_output_length>layer:
                f_out = datasets[1][layer]
            if debug_output_length>layer:
                d_out = datasets[2][layer]
            if keras16_output_length>layer:
                k16_out = datasets[3][layer]
        return x, k_out, f_out, d_out, 0, 0, k16_out

    else:
        min_val=[]
        max_val=[]
        if keras_output_length>layer:
            min_val.append([np.min (sublist) for sublist in keras_outputs[layer]])
            max_val.append([np.max (sublist) for sublist in keras_outputs[layer]])
        if fpga_output_length>layer:
            min_val.append([np.min (sublist) for sublist in fpga_outputs[layer]])
            max_val.append([np.max (sublist) for sublist in fpga_outputs[layer]])
        if debug_output_length>layer:
            min_val.append([np.min (sublist) for sublist in debug_outputs[layer]])
            max_val.append([np.max (sublist) for sublist in debug_outputs[layer]])

        min_val=np.min(min_val)
        max_val=np.max(max_val)

        # dataset2 = np.asarray(datasets[data2])
        dataset2 = datasets[data2]
        channels = dataset1[layer].shape[2]
        d1_out = np.rollaxis(dataset1[layer], 2)
        d2_out = np.rollaxis(dataset2[layer], 2)
        return 0, d1_out, d2_out,None, min_val, max_val, None

def get_image(channel, view, dataset1, dataset2):
    if view==0:
        image=dataset1[channel]
    if view==1:
        image = dataset2[channel]
    if view==2:
        image = dataset1[channel]-dataset2[channel]
    if view==3:
        image = point_rel_difference(dataset1[channel] , dataset2[channel])
    if view==4:
        image = dataset1[channel]-dataset2[channel]
    return image

def make_data_dict(layer):
    data_dict={}
    if keras_output_length>layer:
        data_dict['Keras']=0
    if fpga_output_length>layer:
        data_dict['FPGA']=1
    if debug_output_length>layer:
        data_dict['Debug']=2
    if keras16_output_length>layer:
        data_dict['Keras16']=3
    return data_dict


def make_plot(layer, view, data1, data2):
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    length, dataset1, dataset2, dataset3, min_val, max_val, dataset4 = get_layer_data(layer, data1, data2)
    plotgrid=[]

    if length == 0:
        channels = dataset1.shape[0]
        width = 8
        height = int(channels/width)+1

        row=[]
        imgs=[]
        for channel in range(channels):
            img=get_image(channel, view, dataset1, dataset2)
            
            # img=img[::-1] #image shows upside down by default
            if view==3: 
                img=np.pad(img, pad_width=1, mode='constant', constant_values=1)
                view_label='Activation Difference'
            else:
                img=np.pad(img, pad_width=1, mode='constant', constant_values=np.max(img))
                if view==0:
                    view_label=data1select.value
                if view==1:
                    view_label=data2select.value
                if view==2:
                    view_label='{0}, {1} Difference'.format(data1select.value, data2select.value)
                if view==4:
                    view_label='{0}, {1} Difference - Limited ColorRange'.format(data1select.value, data2select.value)


            if channel%8==0:
                if channel!=0:
                    row=np.hstack(row)
                    imgs.append(row)
                    row=[]
            row.append(img)

        row=np.hstack(row)
        imgs.append(row)
        if view==4:
            data1std = np.std(dataset1)    

        if imgs[0].shape!=imgs[-1].shape:
            imgs[-1] = np.pad(imgs[-1], pad_width=[(0,0),(0,imgs[0].shape[1]-imgs[-1].shape[1])], mode='constant', constant_values=0)
        imgs = np.vstack(imgs)
        imgs=imgs[::-1]
        # imgs=imgs[:1000]
        ratio = imgs.shape[0]/imgs.shape[1]
        width=width*200
        height = int(width*ratio)

        if height>16000:
            height=16000
            width=int(height/ratio)
        p = figure(title=view_label + ' View. Layer: '+ os.path.basename(keras_files[layer]),x_range=(0, imgs.shape[1]), y_range=(0, imgs.shape[0]),toolbar_location="left", plot_width=width, plot_height=height)
        p.title.text_font_size = "20px"
        if view==2:
            # color_mapper = LinearColorMapper(palette="Viridis256", low=-0.4, high=0.6)
            color_mapper = LinearColorMapper(palette="Viridis256", low=np.min(imgs), high=np.max(imgs))
        elif view==3:
            color_mapper = LinearColorMapper(palette="Viridis256", low=-1, high=1)
        elif view==4:
            color_mapper = LinearColorMapper(palette="Viridis256", low=-data1std/2, high=data1std/2)
        else:
            # color_mapper = LinearColorMapper(palette="Viridis256", low=min_val, high=max_val)
            color_mapper = LinearColorMapper(palette="Viridis256", low=np.min(imgs), high=np.max(imgs))
        p.image(image=[imgs], x=0, y=0, dw=imgs.shape[1], dh=imgs.shape[0], color_mapper=color_mapper)

        hover = HoverTool(tooltips = [("value", "@image")])
        # hover = HoverTool(tooltips = [("x", "$x{int}"), ("y", "$y{int}"), ("value", "@image")])
        p.add_tools(hover)
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                 label_standoff=5, border_line_color=None, location=(0,0), width=10, height=250)
        p.add_layout(color_bar, 'right')

    else:
        channels=0
        x_vals = np.arange(length)
        p = figure(tools=TOOLS,title="Layer "+str(layer) + 'Graph View. Layer: '+ os.path.basename(keras_files[layer]),toolbar_location="left", plot_width=1200, plot_height=800)
        source_dict={}
        source_dict['x']=x_vals
        hover_tooltips = [("x", "$x{(0)}")]
        if dataset1 is not None and keras_output_length>layer:
            source_dict['keras'] = dataset1
            hover_tooltips.append(("keras", "@keras"))
        if dataset2 is not None and fpga_output_length>layer:
            source_dict['fpga'] = dataset2
            hover_tooltips.append(("fpga", "@fpga"))
        if dataset3 is not None and debug_output_length>layer:
            source_dict['debug'] = dataset3
            hover_tooltips.append(("debug", "@debug"))
        if dataset4 is not None and keras_output_length>layer:
            source_dict['keras16'] = dataset4
            hover_tooltips.append(("keras16", "@keras16"))
        source=ColumnDataSource(data=source_dict)
        hover = HoverTool(tooltips = hover_tooltips)
        # source=ColumnDataSource(data=dict(x=x_vals, keras=dataset1, fpga=dataset2, debug=dataset3))
        # hover = HoverTool(tooltips = [("x", "$x{(0)}"), ("keras", "@keras"), ("fpga", "@fpga")])
        # k_line = p.line('x', 'keras', source=source, legend=dict(value="Keras"), line_color="red", line_width=1)
        if keras_output_length>layer:
            k_line = p.line('x', 'keras', source=source, legend=dict(value="Keras"), line_color="red", line_width=1)
        if fpga_output_length>layer:
            f_line = p.line('x', 'fpga', source=source,legend=dict(value="FPGA"), line_color="blue", line_width=1)
        if debug_output_length>layer:
            d_line = p.line('x', 'debug', source=source,legend=dict(value="Debug"), line_color="green", line_width=1)
        if keras16_output_length>layer:
            k16_line = p.line('x', 'keras16', source=source, legend=dict(value="Keras16"), line_color="red", line_width=1)
        
        p.legend.click_policy="hide"
        
        print(hover.renderers)
        p.add_tools(hover)

    return p, channels

def update_plot(attrname, old, new):
    root = curdoc().roots[1]
    curdoc().remove_root(root)
    layer=layer_dict[layerselect.value]
    data_dict=make_data_dict(layer)
    view=view_dict[viewselect.value]
    data1select.options=list(data_dict.keys())
    data2select.options=list(data_dict.keys())
    if data1select.value not in data_dict:
        data1 = list(data_dict.values())[0]
    else:
        data1=data_dict[data1select.value]
    if data2select.value not in data_dict:
        data2 = list(data_dict.values())[0]
    else:
        data2=data_dict[data2select.value]

    p, channels=make_plot(layer,view, data1, data2)

    curdoc().add_root(column(p))

parser = argparse.ArgumentParser()
parser.add_argument("INPUT_FOLDER", type=str, help="Network folder")
parser.add_argument("--save_layers", type=str, help="Saves all layers (can't be used with server)")


args = parser.parse_args()
network_debug_folder = os.path.abspath(args.INPUT_FOLDER)+'\\'

if not os.path.exists(network_debug_folder):
    print("Folder does not exist")
    sys.exit(0)

# network_debug_folder = "C:\\Alex\\Work\\fpga_perf\\debug\\MConv_Stage1_L1_5_model\\"


keras_folder = network_debug_folder + 'keras_outputs\\'
keras16_folder = network_debug_folder + 'keras_outputs_float16\\'
fpga_folder = network_debug_folder + 'PLACE_FPGA_DUMPS_HERE\\'
debug_output_folder=network_debug_folder+'fpga_dump_debug_outputs\\'
visual_output_folder=network_debug_folder+'visualisation_outputs\\'

pathlib.Path(visual_output_folder).mkdir(parents=True, exist_ok=True)



# fpga_folder = "C:/Alex/Work/debug_check/posenet/fpga_dump"
# keras_folder = "C:/Alex/Work/debug_check/posenet/keras_outputs"

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
# debug_outputs=debug_outputs[:num_files]



keras_output_length = len(keras_outputs)
keras16_output_length = len(keras16_outputs)
fpga_output_length = len(fpga_outputs)
debug_output_length = len(debug_outputs)


layer_name=[]
layer_dict={}
for l in range(keras_output_length):
    l_name=str(l).zfill(2)+': '+os.path.basename(keras_files[l])
    layer_name.append(l_name)
    layer_dict[l_name]=l


view_name=[]
view_dict={
    'Data1'         : 0,
    'Data2'          : 1,
    'Difference'    : 2,
    'Activation Difference':3,
    'Difference - Limited ColorRange':4,
}
layer=0
data_dict=make_data_dict(layer)

layerselect = Select(title="Layer:", value=layer_name[layer], options=layer_name)
viewselect = Select(title="View:", value='Data1', options=list(view_dict.keys()))
data1select=Select(title="Dataset 1:", value=list(data_dict.keys())[0], options=list(data_dict.keys()))
if len(data_dict.keys())>1:
    data2select=Select(title="Dataset 2:", value=list(data_dict.keys())[1], options=list(data_dict.keys()))
else:
    data2select=Select(title="Dataset 2:", value=list(data_dict.keys())[0], options=list(data_dict.keys()))
layerselect.on_change('value', update_plot)
viewselect.on_change('value', update_plot)
data1select.on_change('value', update_plot)
data2select.on_change('value', update_plot)


if args.save_layers==1:
    for l in range(len(keras_outputs)):
        html_name="layer"+str(l).zfill(3)+"_data1"
        output_file(visual_output_folder+html_name+".html", title=html_name)
        p, channels=make_plot(l, 0, 0, 1)
        save(column(p))
        html_name="layer"+str(l).zfill(3)+"_data2"
        output_file(visual_output_folder+html_name+".html", title=html_name)
        p, channels=make_plot(l, 1, 0, 1)
        save(column(p))
        html_name="layer"+str(l).zfill(3)+"_difference"
        output_file(visual_output_folder+html_name+".html", title=html_name)
        p, channels=make_plot(l, 2, 0, 1)
        save(column(p))
        html_name="layer"+str(l).zfill(3)+"_act_difference"
        output_file(visual_output_folder+html_name+".html", title=html_name)
        p, channels=make_plot(l, 3, 0, 1)
        save(column(p))
        html_name="layer"+str(l).zfill(3)+"_color_limit_difference"
        output_file(visual_output_folder+html_name+".html", title=html_name)
        p, channels=make_plot(l, 4, 0, 1)
        save(column(p))
    


p, channels=make_plot(layer, 0, 0, 1)

curdoc().add_root(column(row(layerselect, viewselect, data1select, data2select)))
curdoc().add_root(column(p))
show(column(row( data1select, data2select, layerselect, viewselect), p))
print("done")