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

import torch
import copy
import types


def _capture_stream(is_origin=True):
    torch.cuda.init()
    return torch.cuda.Stream(_cdata=torch._C._cuda_getCaptureStream(is_origin))


class Graph(torch._C._CudaGraphBase):
    def __new__(cls, **kwargs):
        return super(Graph, cls).__new__(cls, **kwargs)

    def launch(self):
        super(Graph, self).launch()


class Nimble(object):
    def __init__(self, original_module):
        self.original_module = original_module
        self.prepared = False
        self.dummy_input_to_autograd_fn = torch.tensor([]).requires_grad_(True)

    def __call__(self, *args):
        self.copy_inputs(*args)
        self.launch()
        if self.use_tuple_as_output:
            return self.forward_graph.outputs
        else:
            return self.forward_graph.outputs[0]

    def copy_inputs(self, *args):
        placeholders = self.forward_graph.inputs
        for ph, arg in zip(placeholders, args):
            ph.copy_(arg, non_blocking=True)

    def launch(self):
        assert self.prepared
        if self.training:
            self.nimble_autograd_fn.apply(self.dummy_input_to_autograd_fn)
        else:
            self.forward_graph.launch()

    def detach_outputs(self, outputs):
        if isinstance(outputs, torch.Tensor):
            use_tuple_as_output = False
            detached_outputs = (outputs.detach(),)
        elif isinstance(outputs, (tuple, list)):
            use_tuple_as_output = True
            if len(outputs) == 0:
                raise ValueError("The output must be a tensor or a tuple/list of tensors with at least one element, but got an empty tuple/list")
            detached_outputs = []
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    detached_outputs.append(output.detach())
                else:
                    raise ValueError("The output must be a tensor or a tuple/list of tensors, but got a tuple/list element of type %s" % type(output))
            detached_outputs = tuple(detached_outputs)
        else:
            raise ValueError("The output must be a tensor or a tuple/list of tensors")
        return use_tuple_as_output, detached_outputs

    def build_inference_graph(self, module, dummy_inputs, use_multi_stream=True, relaxed=False):
        stream = _capture_stream()
        forward_graph = Graph()
        forward_graph.inputs = tuple([dummy_input.to(device=stream.device, copy=True) for dummy_input in dummy_inputs])

        # prepare forward graph
        with torch.no_grad(), torch.cuda.stream(stream):
            torch._C._cuda_beginStreamPrecapture(stream._cdata, use_multi_stream)
            module(*forward_graph.inputs)
            torch._C._cuda_endStreamPrecapture(stream._cdata)

            torch._C._cuda_beginStreamCapture(stream._cdata, use_multi_stream, relaxed)
            dummy_outputs = module(*forward_graph.inputs)
            torch._C._cuda_endStreamCapture(stream._cdata, forward_graph)

        use_tuple_as_output, forward_graph.outputs = self.detach_outputs(dummy_outputs)
        return forward_graph, use_tuple_as_output

    def build_training_graph(self, module, dummy_inputs, use_multi_stream=True, relaxed=False):
        stream = _capture_stream()
        forward_graph = Graph()
        forward_graph.inputs = tuple([dummy_input.to(device=stream.device, copy=True) for dummy_input in dummy_inputs])

        # helper classes
        class FakeLoss(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *outputs):
                FakeLoss.grad_outputs = tuple([torch.zeros_like(output) for output in outputs])
                return torch.Tensor([0.]) # arbitrary value

            @staticmethod
            def backward(ctx, grad_output):
                return FakeLoss.grad_outputs

        # execute a single interation of training for cuDNN benchmarking
        with torch.random.fork_rng():
            outputs = module(*dummy_inputs)
            if isinstance(outputs, torch.Tensor):
                fakeloss = FakeLoss.apply(outputs)
            else:
                fakeloss = FakeLoss.apply(outputs)
            fakeloss.backward()

        # prepare forward graph
        with torch.enable_grad(), torch.random.fork_rng(), torch.cuda.stream(stream):
            torch._C._cuda_beginStreamPrecapture(stream._cdata, use_multi_stream)
            module(*forward_graph.inputs)
            torch._C._cuda_endStreamPrecapture(stream._cdata)

            torch._C._cuda_beginStreamCapture(stream._cdata, use_multi_stream, relaxed)
            dummy_outputs = module(*forward_graph.inputs)
            torch._C._cuda_endStreamCapture(stream._cdata, forward_graph)

        # check outputs
        use_tuple_as_output, forward_graph.outputs = self.detach_outputs(dummy_outputs)

        if use_tuple_as_output:
            fakeloss = FakeLoss.apply(*dummy_outputs)
        else:
            fakeloss = FakeLoss.apply(dummy_outputs)

        backward_graph = Graph()
        backward_graph.inputs = FakeLoss.grad_outputs

        # prepare backward graph
        with torch.random.fork_rng(), torch.cuda.stream(stream):
            torch._C._cuda_beginStreamPrecapture(stream._cdata, False)
            fakeloss.backward(retain_graph=True)
            torch._C._cuda_endStreamPrecapture(stream._cdata)

            torch._C._cuda_beginStreamCapture(stream._cdata, False, relaxed)
            fakeloss.backward()
            torch._C._cuda_endStreamCapture(stream._cdata, backward_graph)

        # Set dummy tensor as output because we don't require further flow of gradients.
        backward_graph.outputs = (torch.tensor([]),)

        class NimbleFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                forward_graph.launch()
                if use_tuple_as_output:
                    return forward_graph.outputs
                else:
                    return forward_graph.outputs[0]

            @staticmethod
            def backward(ctx, *grad_outputs):
                placeholders = backward_graph.inputs
                for ph, grad_output in zip(placeholders, grad_outputs):
                    ph.copy_(grad_output, non_blocking=True)
                backward_graph.launch()
                return backward_graph.outputs[0]

        return forward_graph, use_tuple_as_output, backward_graph, NimbleFunction()

    """
    Args:
        dummy_inputs: CUDA Tensor or tuple/list of CUDA Tensors for inputs of the module. Should not require gradients.
        training: Prepare trainable Nimble module or not.
        use_multi_stream: Use multiple CUDA streams or not.
        relaxed: Set stream capture mode as `cudaStreamCaptureModeRelaxed`.
        Refer to https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 for more details.
    """
    def prepare(self, dummy_inputs, training=False, use_multi_stream=True, relaxed=False):
        # check input types
        if isinstance(dummy_inputs, torch.Tensor):
            assert not dummy_inputs.requires_grad
            assert dummy_inputs.is_cuda
            dummy_inputs = (dummy_inputs,)
        elif isinstance(dummy_inputs, (tuple, list)):
            if len(dummy_inputs) == 0:
                raise ValueError("example_inputs must be a tensor or a tuple/list of tensors with at least one element, but got an empty tuple/list")
            for dummy_input in dummy_inputs:
                if not isinstance(dummy_input, torch.Tensor):
                    raise ValueError("example_inputs must be a tensor or a tuple/list of tensors, but got a tuple/list element of type %s" % type(dummy_input))
                else:
                    assert not dummy_input.requires_grad
                    assert dummy_input.is_cuda
            dummy_inputs = tuple(dummy_inputs)
        else:
            raise ValueError("example_inputs must be a tensor or a tuple/list of tensors")

        # store original state and training flag
        init_state = copy.deepcopy(self.original_module.state_dict())
        if training:
            backup_grads = {}
            for name, param in self.original_module.named_parameters():
                if param.grad is None:
                    # manually allocate grad tensors
                    param.grad = torch.zeros_like(param.data)
                backup_grads[name] = param.grad.clone().detach()
        original_training = self.original_module.training
        self.original_module.train(training)

        # graph rewriting: conv kernel selection, basic operator fusion, inserting instructions for multi-stream execution
        rewritten_module = rewrite_graph(self.original_module, dummy_inputs, training, use_multi_stream)

        # Well-written torch.nn.Module should have every tensor
        # required for computing `forward` as either parameter
        # or buffer (except input arguments of forward).
        # We maintain references to these tensors to make sure that
        # Nimble works even when the original Module is deleted,
        # following the behavior of TorchScript.
        self.parameters = list(rewritten_module.parameters())
        self.buffers = list(rewritten_module.buffers())

        if training:
            self.forward_graph, self.use_tuple_as_output, self.backward_graph, self.nimble_autograd_fn = self.build_training_graph(rewritten_module, dummy_inputs, use_multi_stream, relaxed)
        else:
            self.forward_graph, self.use_tuple_as_output = self.build_inference_graph(rewritten_module, dummy_inputs, use_multi_stream, relaxed)

        # revert changes
        self.original_module.load_state_dict(init_state)
        self.original_module.train(original_training)
        if training:
            for name, param in self.original_module.named_parameters():
                param.grad.copy_(backup_grads[name])
                param.grad.detach_()
        del self.original_module

        # set flags
        self.prepared = True
        self.training = training
        return


class torch_set_cudnn_enabled(object):
    def __init__(self, enabled):
        self.prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        torch.backends.cudnn.enabled = self.prev
        return False


def rewrite_graph(model, dummy_inputs, training=False, use_multi_stream=False):
    if training:
        jit_model = torch.jit.trace(model, dummy_inputs).cuda().train(True)
        torch._C._jit_clear_optimized_graph(jit_model._c)

        prev_autostream_mode = torch._C._cuda_getAutoStreamMode()
        torch._C._cuda_setAutoStreamMode(use_multi_stream)
        jit_model(*dummy_inputs)
        torch._C._cuda_setAutoStreamMode(prev_autostream_mode)

    else:
        # conv selection (only for inference)
        run_conv_selection(model, dummy_inputs)

        with torch.no_grad(), torch_set_cudnn_enabled(False):
            jit_model = torch.jit.trace(model, dummy_inputs).cuda().train(False)
            # basic operator fusions
            torch._C._jit_replace_graph_with_optimized_graph(jit_model._c, *dummy_inputs)
            torch._C._jit_pass_fold_convbn_for_traced_module(jit_model._c)
            torch._C._jit_pass_fold_conv_cat_bn_for_traced_module(jit_model._c)
            torch._C._jit_pass_prepare_elementwise_op_fusion(jit_model._c)

            prev_autostream_mode = torch._C._cuda_getAutoStreamMode()
            torch._C._cuda_setAutoStreamMode(use_multi_stream)
            jit_model(*dummy_inputs)
            torch._C._cuda_setAutoStreamMode(prev_autostream_mode)

    return jit_model


def tag_conv(module, x):
    def _dfs_traverse(module):
        for _, submodule in module.named_children():
            if isinstance(submodule, torch.nn.Conv2d):
                def tag_forward(self, input):
                    self.input = input
                    return self._conv_forward(input, self.weight)
                submodule.forward = types.MethodType(tag_forward, submodule)
            else:
                _dfs_traverse(submodule)

    _dfs_traverse(module)
    with torch.no_grad():
        module(*x)


class MeasurableConv(object):
    def __init__(self, original_conv, iter_num=20):
        super(MeasurableConv, self).__init__()
        self.original_conv = original_conv
        self.iter_num = iter_num

    def prepare(self, dummy_input):
        # Build temporary Graph module that runs the conv `iter_num` times.
        # We need this for measuring the time spent on different conv implementations correctly
        # without the overhead from GPU task scheduling.
        stream = _capture_stream()
        dummy_input = dummy_input.to(device=stream.device, copy=True)
        self.forward_graph = Graph()
        self.forward_graph.inputs = (dummy_input,)

        with torch.no_grad(), torch.cuda.stream(stream):
                torch._C._cuda_beginStreamPrecapture(stream._cdata, False)
                for _ in range(self.iter_num):
                    self.original_conv(dummy_input)
                torch._C._cuda_endStreamPrecapture(stream._cdata)

                torch._C._cuda_beginStreamCapture(stream._cdata, False, False)
                for _ in range(self.iter_num):
                    output = self.original_conv(dummy_input)
                torch._C._cuda_endStreamCapture(stream._cdata, self.forward_graph)

        self.forward_graph.outputs = (output.detach(),)

    def launch(self):
        self.forward_graph.launch() # don't care about input


def benchmark_conv(module, x, warmup=10, num_iter=10):
    # Ensure that PyTorch already selected proper conv algorithm when torch.backends.cudnn.benchmark==True
    with torch_set_cudnn_enabled(True), torch.no_grad():
        module(x)

    def _measure(module):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            module.launch()
        torch.cuda.synchronize()

        latencies = []
        for _ in range(num_iter):
            start_event.record()
            module.launch()
            end_event.record()
            end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
        return sum(latencies) / num_iter

    # cuDNN conv
    with torch_set_cudnn_enabled(True):
        cudnn_conv = MeasurableConv(module)
        cudnn_conv.prepare(x)
        cudnn_time = _measure(cudnn_conv)

    # PyTorch conv
    with torch_set_cudnn_enabled(False):
        pytorch_conv = MeasurableConv(module)
        pytorch_conv.prepare(x)
        pytorch_time = _measure(pytorch_conv)

    return {'cudnn': cudnn_time, 'pytorch': pytorch_time}


def select_conv(conv, x):
    def _cudnn_forward(self, input):
        with torch_set_cudnn_enabled(True):
            return self.conv2d_forward(input, self.weight)

    def _no_cudnn_forward(self, input):
        with torch_set_cudnn_enabled(False):
            return self.conv2d_forward(input, self.weight)

    # for dilated convolutions, use cuDNN
    if conv.dilation != (1, 1):
        return _cudnn_forward

    benchmark_result = benchmark_conv(conv, x)
    use_cudnn = benchmark_result['cudnn'] < benchmark_result['pytorch']
    return _cudnn_forward if use_cudnn else _no_cudnn_forward


def run_conv_selection(module, x):
    tag_conv(module, x)

    def _dfs_traverse(module):
        for name, submodule in module.named_children():
            if isinstance(submodule, torch.nn.Conv2d) and hasattr(submodule, 'input'):
                selected_conv = select_conv(submodule, submodule.input)
                submodule.forward = types.MethodType(selected_conv, submodule)
            else:
                _dfs_traverse(submodule)

    _dfs_traverse(module)
