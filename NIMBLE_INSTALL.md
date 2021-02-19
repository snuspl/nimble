# Nimble Installation Guide

NOTE: This version of Nimble supports PyTorch v1.4.1 with CUDA 10.2. Please check other branches if you seek for newer versions.

## Prerequisites
* anaconda3
* cuda 10.2
* cudnn 8.0.2 for cuda 10.2
* Environment variables
```bash
# create new conda environment
conda create -n nimble python=3.7 -y

# environment variables, we need this setting for every installation and experiment
conda activate nimble
export CUDA_HOME=<YOUR_CUDA_10.2_PATH>
export CUDA_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export CUDNN_LIB_DIR=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include/
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
```
* Installing dependencies
```bash
# ensure prerequisites for pytorch build
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing -y
conda install -c pytorch magma-cuda102 -y

# install onnx
conda install -c conda-forge onnx -y

# downgrade protobuf ([why?](https://github.com/onnx/onnx/issues/2434))
conda install -c conda-forge protobuf=3.9 -y

# ensure prerequisites for caffe2 build
pip install future
```

## Build Nimble
```bash
# clone nimble and run setup.py
git clone --recursive https://github.com/snuspl/nimble
export NIMBLE_HOME=/path/to/nimble
cd $NIMBLE_HOME
BUILD_TEST=0 USE_DISTRIBUTED=0 USE_NCCL=0 USE_NUMA=0 USE_MPI=0 python setup.py install
```
