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

#include <ATen/cuda/Graph.h>
#include <c10/cuda/CUDAGuard.h>

namespace at {
namespace cuda {

Graph::Graph() {}

Graph::~Graph() {
  // wait for all the streams where the graph launch took place
  for (auto& stream : launched_streams) {
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  if (cuda_graph_exec) {
    C10_CUDA_CHECK(cudaGraphExecDestroy(*cuda_graph_exec));
  }
  if (cuda_graph) {
    C10_CUDA_CHECK(cudaGraphDestroy(*cuda_graph));
  }
}

void Graph::launch() {
  auto stream = cuda::getCurrentCUDAStream(device.index());
  C10_CUDA_CHECK(cudaGraphLaunch(*cuda_graph_exec, stream));
  launched_streams.insert(stream);
}

CUDAEvent Graph::launch(CUDAEvent&& event) {
  launch();
  event.record();
  return std::move(event);
}

} // namespace cuda
} // namespace at
