# Nimble Evaluation Guide

## Installing TensorRT
```bash
# pycuda
pip install 'pycuda>=2019.1.1' --no-cache-dir

# addtional environment variables, we need this setting for every experiment using TensorRT
export TENSORRT_HOME=<YOUR_TENSORRT_PATH>
export LD_LIBRARY_PATH=$TENSORRT_HOME/lib:$LD_LIBRARY_PATH

# install TensorRT for python
cd $TENSORRT_HOME/python
pip install tensorrt-7.1.3.4-cp37-none-linux_x86_64.whl
cd $TENSORRT_HOME/graphsurgeon
pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
```

## Preparing dependencies for experiments
```bash
pip install pandas

cd $NIMBLE_HOME/experiment/torchvision
python setup.py install
cd $NIMBLE_HOME/experiment/pretrained-models
python setup.py install
cd $NIMBLE_HOME/experiment/timm
python setup.py install
```

## Running experiments
```bash
cd $NIMBLE_HOME/experiment
python run_inference.py resnet50 --mode nimble
```
```
        mean (ms)  stdev (ms)
nimble   1.887833    0.004812
```
```
python run_training.py resnet50 --mode nimble --use_optimizer
```
```
        mean (ms)  stdev (ms)
nimble   17.79115    0.036144
```
