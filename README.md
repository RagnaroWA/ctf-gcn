# Graph Convolutional Network in Cyclops Tensor Framework

Graph convolutional network in ctf, numpy and scipy

![badge](https://img.shields.io/badge/CTF%20Python-GCN-green.svg?logo=python&style=for-the-badge)

## Building

* `pip install -r requirements.txt` to install the required python package first.
* To install Cyclops Tensor Framework which is a parallel (distributed-memory) numerical library for multidimensional arrays (tensors) in C++ and Python, please refer to [Cyclops Tensor Framework](https://github.com/cyclops-community/ctf).

## Usage

* Enter `python train.py -h` to see args parameters and default values.
* Example: `python train.py --save_best=True --epochs=100 --package=ctf`
