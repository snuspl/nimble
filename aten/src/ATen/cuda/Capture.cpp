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

#include <ATen/cuda/Capture.h>
#include <ATen/cuda/AutoStream.h>

namespace at {
namespace cuda {

static constexpr int kBufferSize = 1024;

void beginCUDAStreamPrecapture(CUDAStream stream, bool multi_stream) {
  TORCH_INTERNAL_ASSERT(
      !stream.is_capturing(),
      "Expected stream.is_capturing false, got true. ",
      "Why dedicated origin stream is already being used for capture? ",
      "Make sure that previous capture was safely finished.");
  if (multi_stream) {
    autostream::AutoStreamMode::set_enabled(true);
    autostream::record_init_event();
  }
}

void endCUDAStreamPrecapture(CUDAStream stream) {
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  TORCH_INTERNAL_ASSERT(
      !stream.is_capturing(),
      "Expected stream.is_capturing false, got true. ",
      "Why dedicated origin stream is already being used for capture? ",
      "Make sure that previous capture was safely finished.");
  if (autostream::AutoStreamMode::is_enabled()) {
    autostream::AutoStreamMode::set_enabled(false);
    autostream::wait_all_streams();
  }
}

void beginCUDAStreamCapture(CUDAStream stream, bool multi_stream, bool relaxed) {
  cudaStreamCaptureMode mode;
  if (relaxed) {
    mode = cudaStreamCaptureModeRelaxed;
  } else {
    mode = cudaStreamCaptureModeGlobal;
  }
  TORCH_INTERNAL_ASSERT(
      !stream.is_capturing(),
      "Expected stream.is_capturing false, got true. ",
      "Why dedicated origin stream is already being used for capture? ",
      "Make sure that previous capture was safely finished.");
  C10_CUDA_CHECK(cudaStreamBeginCapture(stream, mode));

  if (multi_stream) {
    autostream::AutoStreamMode::set_enabled(true);
    autostream::record_init_event();
  }
}

void endCUDAStreamCapture(CUDAStream stream, Graph& graph) {
  TORCH_INTERNAL_ASSERT(
      stream.is_capturing(),
      "Expected stream.is_capturing true, got false. ",
      "Please check errors occured between the cons- and des- truction of this guard.");

  if (autostream::AutoStreamMode::is_enabled()) {
    autostream::AutoStreamMode::set_enabled(false);
    autostream::wait_all_streams();
    autostream::clear_capture_streams();
  }

  graph.cuda_graph.reset(new cudaGraph_t());
  C10_CUDA_CHECK(cudaStreamEndCapture(stream, graph.cuda_graph.get()));
  TORCH_INTERNAL_ASSERT(
      graph.cuda_graph.get() != NULL,
      "The stream capture returned a NULL graph, due to an unknown violation of the rules of stream capture");

  graph.cuda_graph_exec.reset(new cudaGraphExec_t());
  cudaGraphNode_t pErrorNode;
  char pLogBuffer[kBufferSize];
  cudaError_t err = cudaGraphInstantiate(
      graph.cuda_graph_exec.get(),
      *graph.cuda_graph,
      &pErrorNode,
      pLogBuffer,
      kBufferSize);

  // TODO: print more information about pErrorNode using
  // cudaGraphNodeGetType, cudaGraphKernelNodeGetParams, etc.
  if (err != cudaSuccess) {
    auto error_unused = cudaGetLastError();
    TORCH_INTERNAL_ASSERT(
        false,
        "Error occured while instantiating CUDA graph at node ",
        pErrorNode,
        ". Diagnostic message: ",
        pLogBuffer);
  }

  graph.device = stream.device();
  graph.captured_cuda_pointers =
      c10::cuda::CUDACachingAllocator::collectCapturedDevicePointers();
  graph.captured_host_pointers = collectCapturedHostPointers();
}

} // namespace cuda
} // namespace at
