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

#include <torch/csrc/jit/passes/autostream.h>
#include <torch/csrc/jit/passes/utils/DAGDecomposition.h>

#include <ATen/cuda/AutoStream.h>

#include <tuple>
#include <vector>

namespace torch {
namespace jit {

namespace {

static int node_counter = 0;
int generate_node_id() {
  return node_counter++;
}

void reset_node_id() {
  node_counter = 0;
}

bool skip_node(Node* node) {
  return node->kind() == aten::dim;
}

int register_node(
    Node* opnode,
    std::vector<Node*>& opnodes,
    std::unordered_map<int, c10::List<int64_t>>& node_to_parent_ids,
    DAGDecomposition::Graph& nn_dag) {
  if (skip_node(opnode))
    return -1;
  if (opnode->node_id != -1)
    return opnode->node_id;

  int new_id = generate_node_id();
  opnode->node_id = new_id;
  nn_dag.push_back({});
  opnodes.push_back(opnode);

  const auto& output_vals = opnode->outputs();
  for (const auto& output_val : output_vals) {
    if (output_val && output_val->hasUses()) {
      const auto& output_uses = output_val->uses();
      for (const auto& output_use : output_uses) {
        if (output_use.user && output_use.user->kind() != prim::Return) {
          int child_id = register_node(
              output_use.user, opnodes, node_to_parent_ids, nn_dag);
          if (child_id != -1) {
            nn_dag.at(new_id).push_back(child_id);
            if (node_to_parent_ids.find(child_id) == node_to_parent_ids.end()) {
              node_to_parent_ids[child_id] = c10::List<int64_t>({new_id});
            } else {
              node_to_parent_ids[child_id].push_back(new_id);
            }
          }
        }
      }
    }
  }
  return new_id;
}

DAGDecomposition::Graph build_stream_dag(
    const DAGDecomposition::Graph& nn_dag,
    const std::vector<std::array<int, 2>>& stream_chains) {
  const auto& transitive_closure =
      DAGDecomposition::get_transitive_closure(nn_dag);
  DAGDecomposition::Graph stream_dag;
  for (int i = 0; i < stream_chains.size(); i++) {
    std::vector<int> ensuing_streams;
    auto chain_end = stream_chains.at(i).at(1);
    for (int j = 0; j < stream_chains.size(); j++) {
      auto chain_begin = stream_chains.at(j).at(0);
      if (transitive_closure.at(chain_end).at(chain_begin)) {
        ensuing_streams.push_back(j);
      }
    }
    stream_dag.push_back(ensuing_streams);
  }
  return stream_dag;
}

std::tuple<std::vector<int>, std::vector<int>> map_streams(
    const DAGDecomposition::Graph& nn_dag) {
  DAGDecomposition::Graph nn_meg = DAGDecomposition::get_MEG(nn_dag);
  const auto& bigraph = DAGDecomposition::meg_to_bigraph(nn_meg);
  const auto& matching = DAGDecomposition::maximum_matching(bigraph);
  const auto& result = DAGDecomposition::get_mapping(matching);
  std::vector<int> node_to_chain = std::get<0>(result);

  const auto& stream_chains = std::get<1>(result);
  const auto& stream_dag = build_stream_dag(nn_meg, stream_chains);
  const auto& stream_bigraph = DAGDecomposition::dag_to_bigraph(stream_dag);
  const auto& rematching = DAGDecomposition::maximum_matching(stream_bigraph);
  const auto& remapping = DAGDecomposition::get_mapping(rematching);

  std::vector<int> chain_to_stream = std::get<0>(remapping);
  int stream_num = std::get<2>(remapping);
  at::cuda::autostream::generate_streams(stream_num);

  return std::make_tuple(node_to_chain, chain_to_stream);
}

void insert_autostream_hooks(
    std::shared_ptr<Graph> graph,
    const std::vector<Node*>& opnodes,
    std::unordered_map<int, c10::List<int64_t>>& node_to_parent_ids,
    const DAGDecomposition::Graph& nn_dag,
    const std::vector<int>& node_to_chain,
    const std::vector<int>& chain_to_stream) {
  for (int i = 0; i < opnodes.size(); i++) {
    Node* node = opnodes[i];
    int node_id = node->node_id;
    AT_ASSERT(node_id == i);
    int stream_idx = chain_to_stream.at(node_to_chain.at(node_id));
    Value* stream_idx_val = graph->insertConstant(stream_idx);

    Value* parent_ids_val = graph->insertConstant(node_to_parent_ids[node_id]);

    auto stream_node = graph->create(
        NodeKind::fromQualString("prim::SetStream"),
        {stream_idx_val, parent_ids_val},
        0);
    stream_node->insertBefore(node);
    stream_idx_val->node()->moveBefore(stream_node);
    parent_ids_val->node()->moveBefore(stream_node);

    Value* node_id_val = graph->insertConstant(node_id);
    auto event_node = graph->create(
        NodeKind::fromQualString("prim::RecordEvent"),
        {node_id_val, stream_idx_val},
        0);
    event_node->insertAfter(node);
    node_id_val->node()->moveBefore(event_node);
  }
}
} // namespace

TORCH_API void AutoStream(const std::shared_ptr<Graph> graph) {
  auto inputs = graph->inputs();

  std::unordered_set<Node*> root_nodes;
  int input_num = inputs.size();
  for (int i = 1; i < input_num; i++) {
    const auto& input_uses = inputs[i]->uses();
    for (const auto& input_use : input_uses)
      if (input_use.user)
        root_nodes.insert(input_use.user);
  }

  DAGDecomposition::Graph nn_dag;
  std::vector<Node*> opnodes;
  std::unordered_map<int, c10::List<int64_t>> node_to_parent_ids;
  for (const auto& root_node : root_nodes)
    register_node(root_node, opnodes, node_to_parent_ids, nn_dag);

  const auto& mapping_results = map_streams(nn_dag);
  std::vector<int> node_to_chain = std::get<0>(mapping_results);
  std::vector<int> chain_to_stream = std::get<1>(mapping_results);

  insert_autostream_hooks(
      graph,
      opnodes,
      node_to_parent_ids,
      nn_dag,
      node_to_chain,
      chain_to_stream);

  reset_node_id();
}

} // namespace jit
} // namespace torch
