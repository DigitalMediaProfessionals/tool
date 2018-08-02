import numpy as np
from bokeh.plotting import *
from bokeh.models import ColumnDataSource
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider
import glob
import os
import re
# import bkserve
# from random import random
from bokeh.layouts import column, row
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.models import HoverTool, GlyphRenderer, CustomJSHover
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker

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
    if len(keras_outputs[layer].shape)==1:
        x=keras_outputs[layer].size
        return x, keras_outputs[layer], fpga_outputs[layer]

    elif keras_outputs[layer].shape[1]==1 and keras_outputs[layer].shape[0]==1:
        x=keras_outputs[layer].size
        return x, keras_outputs[layer][0][0], fpga_outputs[layer][0][0]

    else:
        channels = keras_outputs[layer].shape[2]
        k_out = np.rollaxis(keras_outputs[layer], 2)
        f_out = np.rollaxis(fpga_outputs[layer], 2)
        return 0, k_out, f_out

def get_image(channel, view, keras_data, fpga_data):
    if view==0:
        image=keras_data[channel]
    if view==1:
        image = fpga_data[channel]
    if view==2:
        image = keras_data[channel]-fpga_data[channel]
    if view==3:
        image = point_rel_difference(keras_data[channel] , fpga_data[channel])
    return image

def make_plot(layer, view):
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    length, keras_data, fpga_data = get_layer_data(layer)
    
    plotgrid=[]


    if length == 0:
        channels = keras_data.shape[0]
        # chan_min = channel*40
        # chan_max = min((channel*40 + 40),channels)
        width = 8
        height = int(channels/width)+1
        # color_mapper = LinearColorMapper(palette="Viridis256", low=-1, high=1)
        
        # plots=[]
        row=[]
        imgs=[]
        for channel in range(channels):
            img=get_image(channel, view, keras_data, fpga_data)
            
            img=img[::-1] #image shows upside down by default
            if view==3: 
                img=np.pad(img, pad_width=1, mode='constant', constant_values=1)
                view_label='Normalised Difference'
            else:
                img=np.pad(img, pad_width=1, mode='constant', constant_values=np.max(img))
                if view==0:
                    view_label='Keras'
                if view==1:
                    view_label='FPGA'
                if view==2:
                    view_label='Difference'

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

        p = figure(title='Layer '+str(layer) +', '+view_label + 'View. Keras Filename: '+ os.path.basename(keras_files[layer]),x_range=(0, imgs.shape[1]), y_range=(0, imgs.shape[0]),toolbar_location="left", plot_width=width*200, plot_height=height*150)
        if view==3:
            color_mapper = LinearColorMapper(palette="Viridis256", low=-1, high=1)
        else:
            color_mapper = LinearColorMapper(palette="Viridis256", low=np.min(imgs), high=np.max(imgs))
        p.image(image=[imgs], x=0, y=0, dw=imgs.shape[1], dh=imgs.shape[0], color_mapper=color_mapper)

        lat_custom = CustomJSHover(code="""
            var projections = require("core/util/projections");
            var x = special_vars.x
            var y = 5
            var coords = projections.wgs84_mercator.inverse([x, y])
            return "" + coords[1]
            """)

        # hover = HoverTool(tooltips = [("x", "$x{int}"), ("y", "$y{int}"), ("value", "@image")])
        hover = HoverTool(tooltips=[( 'lat','@y{custom}' ), ( 'xx','$x' )], formatters=dict(y=lat_custom))
        print(hover.renderers)
        # p.add_tools(hover)

        code="""
            var x = special_vars.x
            return '' + x
        """

        p.add_tools(HoverTool(
            tooltips=[
                ( 'x',   "$x"            ),
                ("y",   "$y")
            ],

            formatters={

                'y' : CustomJSHover(code=''' return value + " CUSTOM STUFF" '''),   # use 'printf' formatter for 'adj close' field
                                        # use default 'numeral' formatter for other fields
            },

            # # display a tooltip whenever the cursor is vertically in line with a glyph
            # mode='vline'
        ))




        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                 label_standoff=5, border_line_color=None, location=(0,0), width=10, height=250)
        p.add_layout(color_bar, 'right')

            # p = figure(title=str(channel),x_range=(0, keras_data.shape[1]), y_range=(0, keras_data.shape[2]),toolbar_location="left")
            # p.image(image=[img], x=0, y=0, dw=keras_data.shape[1], dh=keras_data.shape[2], color_mapper=color_mapper)
            # hover = HoverTool()
            # hover.tooltips = [("x", "$x"), ("value", "@image")]
            # p.tools.append(hover)
            # color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
            #          label_standoff=3, border_line_color=None, location=(0,0), width=5)
            # p.add_layout(color_bar, 'right')
            # plots.append(p)


        # print("channelsdone")
        # for l in range(0,len(plots),width):
        #     plotgrid.append(plots[l:l+width])
        
        # print("plotgriddonw")
        # color_mapper = LinearColorMapper(palette="Viridis256", low=-1, high=1)
        # p=gridplot(plotgrid, plot_width=220, plot_height=190, color_mapper=color_mapper, color_bar=color_bar, title=str(layer), toolbar_location="left", webgl=True)
        # print("p done")
    else:
        channels=0
        x_vals = np.arange(length)
        source=ColumnDataSource(data=dict(x=x_vals, keras=keras_data, fpga=fpga_data))
        p = figure(tools=TOOLS,title="Layer "+str(layer) + 'Graph View. Keras Filename: '+ os.path.basename(keras_files[layer]), plot_width=1200, plot_height=800)
        k_line = p.line('x', 'keras', source=source, legend=dict(value="Keras"), line_color="red", line_width=1)
        f_line = p.line('x', 'fpga', source=source,legend=dict(value="FPGA"), line_color="blue", line_width=1)
        hover = HoverTool(tooltips = [("x", "$x{(0)}"), ("keras", "@keras"), ("fpga", "@fpga")])
        print(hover.renderers)
        p.add_tools(hover)
        # plotgrid.append([p])
        # p=gridplot(plotgrid)

    return p, channels

def update_plot(attrname, old, new):
    root = curdoc().roots[1]
    curdoc().remove_root(root)
    layer=layer_dict[layerselect.value]
    view=view_dict[viewselect.value]

    # viewselect.value=viewselect.value
    p, channels=make_plot(layer,view)
    # channel_select_range = make_channel_range(channels)
    # channelselect.options = channel_select_range
    curdoc().add_root(column(p))
    



# def make_channel_range(channels):
#     channel_select_range=[]
#     for c in range(0,channels,40):
#         if (c+39)>channels:
#             channel_select_range.append(str(c)+'-'+str(channels))
#         else:
#             channel_select_range.append(str(c)+'-'+str(c+39))
#     return channel_select_range


# output_file("test.html")

parser = argparse.ArgumentParser()
parser.add_argument("INPUT_FOLDER", type=str, help="Input ini file")

	
args = parser.parse_args()
network_debug_folder = os.path.abspath(args.INPUT_FOLDER)+'\\'


fpga_folder = "C:/Alex/Work/debug_check/mobilenet/jaguar/fpga_dump"
keras_folder = "C:/Alex/Work/debug_check/mobilenet/jaguar/keras_outputs"

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
if len(fpga_files) != len(debug_files):
    print("Number of input files does not match")

keras_outputs = []    
for file in keras_files:
    keras_outputs.append(np.load(file)[0])

debug_files = []    
for file in debug_files:
    debug_files.append(np.load(file)[0])

fpga_outputs=[]
for i in range(len(fpga_files)):
    fpga_dump = np.fromfile(fpga_files[i], dtype=np.float16)
    if len(fpga_dump)!=keras_outputs[i].size:
        fpga_dump = np.fromfile(fpga_files[i], dtype=np.float32)
    # print(i)
    fpga_dump = remap(fpga_dump, keras_outputs[i].shape)
    fpga_outputs.append(fpga_dump)

keras_length = np.arange(len(keras_outputs))

layer_name=[]
layer_dict={}
for l in keras_length:
    l_name=str(l).zfill(2)+': '+os.path.basename(keras_files[l])
    layer_name.append(l_name)
    layer_dict[l_name]=l
layer=layer_name[0]
layerselect = Select(title="Layer:", value=layer, options=layer_name)
p, channels=make_plot(0, 0)

view_name=[]
view_dict={
    'Keras'         : 0,
    'FPGA'          : 1,
    'Difference'    : 2,
    'Normalised Difference':3,
}


viewselect = Select(title="View:", value='Keras', options=list(view_dict.keys()))

layerselect.on_change('value', update_plot)
viewselect.on_change('value', update_plot)

curdoc().add_root(column(row(layerselect, viewselect)))
curdoc().add_root(column(p))
show(column(row(layerselect, viewselect), p))