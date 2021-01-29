# Copyright (c) 2020 Software Platform Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Software Platform Lab nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import json
import tempfile
import time

import numpy as np
import pandas as pd
import torch

import pycuda.autoinit
import pycuda.driver as cuda_driver
import onnx
from onnx_to_trt import allocate_buffers, get_engine

NUM_CLASSES_IMAGENET = 1000
NUM_WARM_UPS = 50
NUM_ITERS = 100


with open('models.json', 'r') as f:
    repo_to_models = json.load(f)

model_to_repo = {}
for repo, models in repo_to_models.items():
    for model in models:
        model_to_repo[model] = repo


def get_image_size(model_name):
    image_size_dict = {
        'inception_v3': 299,
        'nasnetalarge': 331,
        'efficientnet_b5': 456,
    }
    return image_size_dict.get(model_name, 224)


def get_model(model_name, pretrained=True, num_classes=NUM_CLASSES_IMAGENET):
    if model_to_repo[model_name] == 'torchvision':
        import torchvision
        if model_name in ['googlenet', 'inception_v3']:
            model = getattr(torchvision.models, model_name)(
                pretrained=pretrained, transform_input=False, aux_logits=True, num_classes=num_classes)
        else:
            model = getattr(torchvision.models, model_name)(pretrained=pretrained, num_classes=num_classes)
    elif model_to_repo[model_name] == 'pretrainedmodels':
        import pretrainedmodels
        model = pretrainedmodels.__dict__[model_name](
            num_classes=num_classes, pretrained='imagenet' if pretrained else pretrained)
    elif model_to_repo[model_name] == 'timm':
        import timm
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_to_repo[model_name] == 'darts':
        import darts
        model = getattr(darts, model_name)()
    return model.cuda().eval()


def get_inference_wrapper(model, dummy_input, mode):
    if mode == 'pytorch':
        wrapper = PyTorchInferenceWrapper(model, dummy_input)
    elif mode == 'trace':
        wrapper = TracingJITInferenceWrapper(model, dummy_input)
    elif mode == 'c2':
        wrapper = C2InferenceWrapper(model, dummy_input, net_type='async_scheduling')
    elif mode == 'trt':
        wrapper = ONNXTRTInferenceWrapper(model, dummy_input)
    elif mode == 'nimble':
        wrapper = NimbleInferenceWrapper(model, dummy_input, use_multi_stream=False)
    elif mode == 'nimble-multi':
        wrapper = NimbleInferenceWrapper(model, dummy_input, use_multi_stream=True)
    else:
        raise RuntimeError('Unsupported mode %s' % mode)
    return wrapper


def get_training_wrapper(model, dummy_input, dummy_label, mode, use_optimizer):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) if use_optimizer else None

    if mode == 'pytorch':
        wrapper = PyTorchTrainingWrapper(model, dummy_input, dummy_label, criterion, optimizer)
    elif mode == 'trace':
        wrapper = TracingJITTrainingWrapper(model, dummy_input, dummy_label, criterion, optimizer)
    elif mode == 'nimble':
        wrapper = NimbleTrainingWrapper(model, dummy_input, dummy_label, criterion, optimizer, use_multi_stream=False)
    elif mode == 'nimble-multi':
        wrapper = NimbleTrainingWrapper(model, dummy_input, dummy_label, criterion, optimizer, use_multi_stream=True)
    else:
        raise RuntimeError('Unsupported mode %s' % mode)
    return wrapper


def evaluate(wrapper):
    torch.cuda.empty_cache()
    for _ in range(NUM_WARM_UPS):
        wrapper.launch()
    torch.cuda.synchronize()

    results = []
    for _ in range(NUM_ITERS):
        wrapper.measure()
        results.append(wrapper.elapsed)
    return results


def eval_result_to_df(mode, result):
    df = pd.DataFrame({mode: result})
    df.index.name = 'iteration'
    return df


class InferenceWrapperBase(object):
    def close(self):
        pass


class BlockingInferenceWrapperBase(InferenceWrapperBase):
    def __init__(self):
        super(BlockingInferenceWrapperBase, self).__init__()

    def measure(self):
        start_time = time.time()
        self.launch()
        end_time = time.time()
        self.elapsed = (end_time - start_time) * 1000


class EventSynchronizedInferenceWrapperBase(InferenceWrapperBase):
    def __init__(self):
        super(EventSynchronizedInferenceWrapperBase, self).__init__()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def measure(self):
        self.start_event.record()
        self.launch()
        self.end_event.record()
        self.end_event.synchronize()
        self.elapsed = self.start_event.elapsed_time(self.end_event)


class PyTorchInferenceWrapperBase(EventSynchronizedInferenceWrapperBase):
    def __init__(self, model, dummy_input):
        super(PyTorchInferenceWrapperBase, self).__init__()
        self.model = model
        self.dummy_input = dummy_input

    def launch(self):
        with torch.no_grad():
            self.output = self.model(self.dummy_input)

    def get_output(self):
        return self.output.cpu().numpy()


class PyTorchInferenceWrapper(PyTorchInferenceWrapperBase):
    def __init__(self, model, dummy_input):
        super(PyTorchInferenceWrapper, self).__init__(model, dummy_input)


class TracingJITInferenceWrapper(PyTorchInferenceWrapperBase):
    def __init__(self, model, dummy_input):
        dummy_input = dummy_input.cuda()
        jit_model = torch.jit.trace(model, dummy_input)
        super(TracingJITInferenceWrapper, self).__init__(jit_model, dummy_input)


class C2InferenceWrapper(BlockingInferenceWrapperBase):
    def __init__(self, model, dummy_input, net_type):
        super(C2InferenceWrapper, self).__init__()
        from caffe2.python import core as c2_core
        import caffe2.python.onnx.backend as onnx_c2_backend
        import onnx
        self.c2_core = c2_core

        self.onnx_model_file = tempfile.NamedTemporaryFile()
        torch.onnx.export(model,
                          dummy_input,
                          self.onnx_model_file.name,
                          keep_initializers_as_inputs=True)  # see https://github.com/onnx/onnx/issues/2417
        onnx_model = onnx.load(self.onnx_model_file.name)
        onnx.checker.check_model(onnx_model)
        self.backend = onnx_c2_backend.prepare(onnx_model, device="CUDA:0")
        self.backend.predict_net.type = net_type
        for op in self.backend.predict_net.op:
            op.engine = 'CUDNN'

        # for initialization (e.g., create net)
        self.backend.run(dummy_input.cpu().numpy())

        with self.c2_core.DeviceScope(self.backend.predict_net.device_option):
            self.backend.workspace.FeedBlob(self.backend.uninitialized[0], dummy_input.cpu().numpy())

    def close(self):
        self.onnx_model_file.close()

    def launch(self):
        with self.c2_core.DeviceScope(self.backend.predict_net.device_option):
            self.backend.workspace.RunNet(self.backend.predict_net.name)

    def get_output(self):
        with self.c2_core.DeviceScope(self.backend.predict_net.device_option):
            return self.backend.workspace.FetchBlob(self.backend.predict_net.external_output[0])


class ONNXTRTInferenceWrapper(InferenceWrapperBase):
    def __init__(self, model, dummy_input):
        super(ONNXTRTInferenceWrapper, self).__init__()
        self.cuda_driver = cuda_driver

        self.onnx_model_file = tempfile.NamedTemporaryFile()
        torch.onnx.export(model,
                          dummy_input,
                          self.onnx_model_file.name)
        onnx_model = onnx.load(self.onnx_model_file.name)
        onnx.checker.check_model(onnx_model)

        self.input_shape = dummy_input.cpu().numpy().shape
        self.engine = get_engine(self.onnx_model_file.name, self.input_shape,
            max_batch_size=self.input_shape[0])
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

        self.output_shape = self.engine.get_binding_shape(1)
        self.start_event = self.cuda_driver.Event()
        self.end_event = self.cuda_driver.Event()

        self.inputs[0].host = dummy_input.cpu().numpy()
        self.cuda_driver.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        self.stream.synchronize()

    def launch(self):
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

    def measure(self):
        self.start_event.record(self.stream)
        self.launch()
        self.end_event.record(self.stream)
        self.end_event.synchronize()
        self.elapsed = self.end_event.time_since(self.start_event)

    def get_output(self):
        for out in self.outputs:
            self.cuda_driver.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()
        self.outputs[0].host = self.outputs[0].host.reshape(self.output_shape)
        return self.outputs[0].host


class NimbleInferenceWrapper(EventSynchronizedInferenceWrapperBase):
    def __init__(self, model, dummy_input, use_multi_stream):
        super(NimbleInferenceWrapper, self).__init__()
        self.nimble_model = torch.cuda.Nimble(model)
        self.nimble_model.prepare(dummy_input, use_multi_stream=use_multi_stream)
        self.nimble_model.forward_graph.inputs[0].copy_(dummy_input)

    def launch(self):
        self.nimble_model.launch()

    def get_output(self):
        return self.nimble_model.forward_graph.outputs[0].cpu().numpy()


class TrainingWrapperBase(object):
    def __init__(self, module, criterion, optimizer):
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def measure(self):
        self.start_event.record()
        self.launch()
        self.end_event.record()
        self.end_event.synchronize()
        self.elapsed = self.start_event.elapsed_time(self.end_event)

    def close(self):
        pass


class PyTorchTrainingWrapperBase(TrainingWrapperBase):
    def __init__(self, model, dummy_input, dummy_label, criterion, optimizer):
        super(PyTorchTrainingWrapperBase, self).__init__(model, criterion, optimizer)
        self.dummy_input = dummy_input
        self.dummy_label = dummy_label

    def launch(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        dummy_output = self.module(self.dummy_input)
        loss = self.criterion(dummy_output, self.dummy_label)
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()


class PyTorchTrainingWrapper(PyTorchTrainingWrapperBase):
    def __init__(self, model, dummy_input, dummy_label, criterion, optimizer):
        super(PyTorchTrainingWrapper, self).__init__(model, dummy_input, dummy_label, criterion, optimizer)


class TracingJITTrainingWrapper(PyTorchTrainingWrapperBase):
    def __init__(self, model, dummy_input, dummy_label, criterion, optimizer):
        jit_model = torch.jit.trace(model, dummy_input)
        super(TracingJITTrainingWrapper, self).__init__(jit_model, dummy_input, dummy_label, criterion, optimizer)


class NimbleTrainingWrapper(TrainingWrapperBase):
    def __init__(self, model, dummy_input, dummy_label, criterion, optimizer, use_multi_stream):
        nimble_model = torch.cuda.Nimble(model)
        nimble_model.prepare(dummy_input, training=True, use_multi_stream=use_multi_stream)
        super(NimbleTrainingWrapper, self).__init__(nimble_model, criterion, optimizer)
        self.dummy_label = dummy_label

    def launch(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        self.module.launch()
        dummy_output = self.module.forward_graph.outputs[0]
        loss = self.criterion(dummy_output, self.dummy_label)
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
