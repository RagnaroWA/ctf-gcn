# Graph Convolutional Network in Cyclops Tensor Framework

![badge](https://img.shields.io/badge/CTF%20PYTHON-GCN-green.svg?&logoColor=3272A5&color=86b200&style=for-the-badge&logo=python)

Graph convolutional network in ctf, numpy and scipy

## Building

* `pip install -r requirements.txt` to install the required python package first.
* To install Cyclops Tensor Framework (CTF) which is a parallel (distributed-memory) numerical library for multidimensional arrays (tensors) in C++ and Python, please refer to [Cyclops Tensor Framework Building and Testing](https://github.com/cyclops-community/ctf/wiki/Building-and-testing).

## Usage

* Enter `python train.py -h` to see args parameters and default values.
* Example: `python train.py --save_best=True --epochs=100 --package=scipy`
* To run with CTF in parallel: `mpirun -n 8 python train.py --package=ctf --epochs=100`
