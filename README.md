# Graph Convolutional Network in Cyclops Tensor Framework

Graph convolutional network in ctf, numpy and scipy

## Building

* `pip install -r requirements.txt` to install the required python package first.
* To install Cyclops Tensor Framework (a parallel (distributed-memory) numerical library for multidimensional arrays (tensors) in C++ and Python), please refer to [Cyclops Tensor Framework](https://github.com/cyclops-community/ctf).

## Usage

* Enter `python train.py -h` to see args parameters and default values.
* Example: `python train.py --save_best=True --epochs=100 --package=ctf`
