# CNN Model Conversion Tool

This python script can parse Caffe and Keras CNN models and convert them to DV700 compatible formats.

## Python Environment Setup
Python with version >= 3.6 is required to run this script.
It also needs the following modules: `numpy`, `protobuf`, `h5py`, `jinja2` and `opencv-python`. Run the following command to install them:

```
python -m pip install numpy protobuf h5py==2.7.1 opencv-python jinja2
```

or on Ubuntu 18.04:
```
sudo apt-get install python3-numpy python3-protobuf python3-h5py python3-opencv python3-jinja2
```
