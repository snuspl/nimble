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

#include <ATen/cuda/AutoStream.h>
#include <c10/core/Event.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <memory>
#include <map>


namespace at {
namespace cuda {
namespace autostream {

static bool AutoStream_enabled = false;

bool AutoStreamMode::is_enabled() {
  return AutoStream_enabled;
}

void AutoStreamMode::set_enabled(bool enabled) {
  AutoStream_enabled = enabled;
}

static std::unique_ptr<c10::Event> init_event;
static std::vector<c10::cuda::CUDAStream> capture_streams;
static std::unordered_map<int, std::shared_ptr<c10::Event>> capture_events;
static std::unordered_map<int, int> node_to_streams;

void generate_streams(int stream_num) {
  AT_ASSERT(capture_streams.empty());
  capture_streams.push_back(c10::cuda::getCaptureStreamFromPool(true));
  for (int i = 1; i < stream_num; i++) {
    capture_streams.push_back(c10::cuda::getCaptureStreamFromPool(false));
  }
}

void record_init_event() {
  init_event.reset(new c10::Event(c10::DeviceType::CUDA));
  init_event->record(c10::cuda::getCaptureStreamFromPool(true));
}

void set_stream(int stream_idx, c10::List<int64_t> parent_ids) {
  auto current_stream = at::cuda::getCurrentCUDAStream();
  if (!current_stream.is_capture_stream()) {
    return;
  }
  auto cuda_stream = capture_streams.at(stream_idx);
  auto c10_stream = cuda_stream.unwrap();
  if (parent_ids.empty()) {
    c10_stream.wait(*init_event);
  } else {
    for (int i = 0; i < parent_ids.size(); i++) {
      if (node_to_streams.find(parent_ids[i]) == node_to_streams.end() ||
          node_to_streams[parent_ids[i]] != stream_idx) {
        auto event = capture_events.at(parent_ids[i]);
        AT_ASSERT(event->was_marked_for_recording());
        c10_stream.wait(*event);
      }
    }
  }
  at::cuda::setCurrentCUDAStream(cuda_stream);
}

void record_event(int node_id, int stream_idx) {
  auto current_stream = at::cuda::getCurrentCUDAStream();
  if (!current_stream.is_capture_stream()) {
    return;
  }
  const auto stream = capture_streams.at(stream_idx);
  AT_ASSERT(capture_events.find(node_id) == capture_events.end());
  auto event = std::make_shared<c10::Event>(c10::DeviceType::CUDA);
  event->record(stream);
  capture_events[node_id] = event;
  node_to_streams[node_id] = stream_idx;
}

void wait_all_streams() {
  auto origin_stream = c10::cuda::getCaptureStreamFromPool(true);
  for (auto& capture_stream : capture_streams) {
    c10::Event event(c10::DeviceType::CUDA);
    event.record(capture_stream);
    origin_stream.unwrap().wait(event);
  }
  at::cuda::setCurrentCUDAStream(origin_stream);

  init_event.reset();
  capture_events.clear();
  node_to_streams.clear();
}

void clear_capture_streams() {
  capture_streams.clear();
}

} // namespace autostream
} // namespace cuda
} // namespace at
