from random import random
from bokeh.layouts import column
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

import numpy as np
from bokeh.plotting import *
from bokeh.models import ColumnDataSource
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider

print('dfgsd')
# create a plot and style its properties

def my_radio_handler(new):
    print(new)
    s1.title = "sdfsad"


def make_dataset(keras_data, fpga_data):
    data={}
    data['x'] = np.arange(32)
    for i in range(len(keras_data[0][0][0])):
        data['keras_'+str(i)]=keras_data[0][0][i]
        data['fpga_'+str(i)]=fpga_data[0][0][i]
    return ColumnDataSource(data)


def get_layer_data(layer):
        if len(keras_outputs[layer].shape)==1:
            x=np.arange(keras_outputs[layer].size)
            return x, keras_outputs[layer], fpga_outputs[layer]

        elif keras_outputs[layer].shape[1]==1 and keras_outputs[layer].shape[0]==1:
            x=np.arange(keras_outputs[layer].size)
            return x, keras_outputs[layer][0][0], fpga_outputs[layer][0][0]

        else:
            channels = keras_outputs[layer].shape[2]
            k_out = np.rollaxis(keras_outputs[layer], 2)
            f_out = np.rollaxis(fpga_outputs[layer], 2)
            return 0, k_out, f_out


def make_plot(source, layer):
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
    x=np.arange(32)
    s1 = figure(tools=TOOLS, width=350, plot_height=350, title="Channel 0")
    s1.circle('x','keras_0',source=source, legend="keras", line_color="red", size=3, alpha=0.5)
    s1.circle('x','fpga_0',source=source, legend="fpga", line_color="blue", size=3, alpha=0.5)
    s1.line('x','keras_0',source=source, line_color="red", line_width=3)
    s1.line('x','fpga_0',source=source, line_color="blue", line_width=3)

    s2 = figure(tools=TOOLS, width=350, plot_height=350, title="Channel 1", x_range=s1.x_range, y_range=s1.y_range)
    s2.line('x','keras_1',source=source, legend="keras", line_color="red", line_width=3)
    s2.line('x','fpga_1',source=source, legend="fpga", line_color="blue", line_width=3)
    s2.circle('x','keras_1',source=source, color="red", size=3, alpha=0.5)
    s2.circle('x','fpga_1',source=source, color="blue", size=3, alpha=0.5)


    # plots=[]
    # for key,entry in source:
    #     plot=figure(tools=TOOLS, width=350, plot_height=350, title=key)
    #     plot.circle(x,entry,source=source, line_color="red", size=3, alpha=0.5)
    #     plots.append(plot)
    # put the results in a row
    # p = gridplot([plots])


    # create some widgets
    slider = Slider(start=0, end=10, value=1, step=.1, title="Slider")
    button_group = RadioButtonGroup(labels=["Option 1", "Option 2", "Option 3"], active=0)
    select = Select(title="Option:", value="foo", options=["foo", "bar", "baz", "quux"])
    button_1 = Button(label="Button 1")
    button_2 = Button(label="Button 2")
    radio_group = RadioButtonGroup(
        labels=["Option 1", "Option 2", "Option 3"], active=0)
    radio_group.on_click(my_radio_handler)

    # put the results in a row
    p = gridplot([[s1, s2, widgetbox(button_1, slider, button_group, select, button_2, radio_group, width=300)]])

    # show the results

    return p


def bokehserver(keras_data, fpga_data):
    source = make_dataset(keras_data, fpga_data)
    
    p = make_plot(source, 0)
    show(p)
    print('regersdavfsav')
    # show the results
    curdoc().add_root(column(p))












# p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
# p.border_fill_color = 'black'
# p.background_fill_color = 'black'
# p.outline_line_color = None
# p.grid.grid_line_color = None








# # add a text renderer to our plot (no data yet)
# r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="20pt",
#            text_baseline="middle", text_align="center")

# i = 0

# ds = r.data_source

# # create a callback that will add a number in a random location
# def callback(keras_data, fpga_data):
#     global i

#     # BEST PRACTICE --- update .data in one step with a new dict
#     new_data = dict()
#     new_data['x'] = ds.data['x'] + [random()*70 + 15]
#     new_data['y'] = ds.data['y'] + [random()*70 + 15]
#     new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i%3]]
#     new_data['text'] = ds.data['text'] + [str(i)]
#     ds.data = new_data

#     i = i + 1

# # add a button widget and configure with the call back
# button = Button(label="Press Me")
# button.on_click(callback)

# # put the button and plot in a layout and add to the document
# curdoc().add_root(column(button, p))

