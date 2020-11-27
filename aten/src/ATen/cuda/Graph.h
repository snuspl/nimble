/*
Copyright (c) 2020 Software Platform Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the Software Platform Lab nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <memory>
#include <vector>

namespace at {
namespace cuda {

struct TORCH_CUDA_API Graph {
  Graph();

  // no copy
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  // no move
  Graph(Graph&& other) = delete;
  Graph& operator=(Graph&& other) = delete;

  ~Graph();

  void launch();
  CUDAEvent launch(CUDAEvent&& event);

  std::unique_ptr<cudaGraph_t> cuda_graph;
  std::unique_ptr<cudaGraphExec_t> cuda_graph_exec;
  std::unique_ptr<std::vector<c10::DataPtr>> captured_cuda_pointers;
  std::unique_ptr<std::vector<c10::DataPtr>> captured_host_pointers;

  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> outputs;

  /**
   * Should be changed to CUDA device later to launch the graph.
   * This initialization (with CPU device) is inevitable since at::Device does
   * not have default constructor.
   */
  at::Device device = at::Device(at::DeviceType::CPU);

  std::unordered_set<at::cuda::CUDAStream> launched_streams;
};

} // namespace cuda
} // namespace at
