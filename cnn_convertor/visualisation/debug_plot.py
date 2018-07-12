import numpy as np
from bokeh.plotting import *
from bokeh.models import ColumnDataSource
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider

# import visualisation.bkserve


def my_radio_handler(new):
    print('Radio button option ' + str(new) + ' selected.')
    s1.title = "sdfsad"


def interactive_plot(keras_data, fpga_data):
    
    data = make_dataset(keras_data, fpga_data)
    x= np.arange(len(keras_data[0][0][0]))
    
    data = {}
    data['x']=x
    for i in range(len(keras_data[0][0][0])):
        data['keras_'+str(i)]=keras_data[0][0][i]
        data['fpga_'+str(i)]=fpga_data[0][0][i]

    source = ColumnDataSource(data)
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
    
    
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
    show(p)
