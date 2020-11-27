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

#include <ATen/cuda/DAGDecomposition.h>

#include <utility>
#include <algorithm>
#include <iostream>

namespace DAGDecomposition {

void print_dag(const Graph& dag) {
  int num_nodes = dag.size();
  for (int i = 0; i < num_nodes; i++) {
    std::cout << i << " -> ";
    const auto& child_nodes = dag.at(i);
    for (auto child_id : child_nodes) {
      std::cout << child_id << ", ";
    }
    std::cout << std::endl;
  }
}

void dfs(int start, const Graph& graph, std::vector<bool>& visited) {
  visited.at(start) = true;
  const std::vector<int>& adjacent_vertices = graph.at(start);
  for (auto vertex : adjacent_vertices) {
    if (!visited.at(vertex)) {
      dfs(vertex, graph, visited);
    }
  }
}

std::vector<std::vector<bool>> get_transitive_closure(const Graph& dag) {
  std::vector<std::vector<bool>> transitive_closure;
  int num_nodes = dag.size();
  for (int i = 0; i < num_nodes; i++) {
    std::vector<bool> reachable(num_nodes, false);
    DAGDecomposition::dfs(i, dag, reachable);
    reachable.at(i) = false;
    transitive_closure.push_back(std::move(reachable));
  }
  return transitive_closure;
}

Graph get_MEG(const Graph& dag) {
  const auto& transitive_closure = get_transitive_closure(dag);
  int num_nodes = dag.size();
  DAGDecomposition::Graph meg = dag;
  for (int i = 0; i < num_nodes; i++) {
    auto& meg_child_nodes = meg.at(i);
    const auto& child_nodes = dag.at(i);
    for (const auto child : child_nodes) {
      if (std::find(meg_child_nodes.begin(), meg_child_nodes.end(), child) == meg_child_nodes.end()) {
        continue;
      }
      for (const auto another_child : child_nodes) {
        if (transitive_closure.at(child).at(another_child)) {
          auto it = std::find(meg_child_nodes.begin(), meg_child_nodes.end(), another_child);
          if (it != meg_child_nodes.end()) {
            meg_child_nodes.erase(it);
          }
        }
      }
    }
  }
  return meg;
}

Bigraph meg_to_bigraph(const Graph& meg) {
  Bigraph bigraph;
  int num_vertices = meg.size();
  for (int i = 0; i < num_vertices; i++) {
    std::vector<bool> adjacency(num_vertices, false);
    for (auto child : meg.at(i)) {
      adjacency.at(child) = true;
    }
    bigraph.push_back(std::move(adjacency));
  }
  return bigraph;
}

Bigraph dag_to_bigraph(const Graph& dag) {
  Bigraph closure;
  int num_vertices = dag.size();
  for (int i = 0; i < num_vertices; i++) {
    std::vector<bool> reachable(num_vertices, false);
    dfs(i, dag, reachable);
    reachable.at(i) = false;
    closure.push_back(std::move(reachable));
  }
  return closure;
}

bool find_matching(int start, const Bigraph& graph, std::vector<bool>& seen, std::vector<int>& match_status) {
  int num_b = graph.at(0).size();
  for (int i = 0; i < num_b; i++) {
    if (graph.at(start).at(i) && !seen.at(i)) {
      seen.at(i) = true;
      int curr_match = match_status.at(i);
      if (match_status.at(i) == -1 || find_matching(curr_match, graph, seen, match_status)) {
        match_status.at(i) = start;
        return true;
      }
    }
  }
  return false;
}

std::vector<int> maximum_matching(const Bigraph& graph) {
  int num_b = graph.at(0).size();
  std::vector<int> match_result(num_b, -1);
  int num_a = graph.size();
  for (int i = 0; i < num_a; i++) {
    std::vector<bool> seen(num_b, false);
    find_matching(i, graph, seen, match_result);
  }
  return match_result;
}

std::tuple<std::vector<int>, std::vector<std::array<int, 2>>, int> get_mapping(const std::vector<int>& matching_BtoA) {
  int num_vertices = matching_BtoA.size();
  std::vector<std::array<int, 2>> chains;
  for(int i = 0; i < num_vertices; i++) {
    auto it = std::find(matching_BtoA.begin(), matching_BtoA.end(), i);
    if (it == matching_BtoA.end()) {
      chains.push_back({i, i});
    }
  }

  int group_num = 0;
  std::vector<int> mapping(num_vertices, -1);
  for (auto& chain : chains) {
    int group_id = group_num++;
    int curr = chain.at(1);
    while (true) {
      mapping.at(curr) = group_id;
      if (matching_BtoA.at(curr) == -1) {
        chain.at(0) = curr;
        break;
      } else {
        curr = matching_BtoA.at(curr);
      }
    }
  }
  return std::make_tuple(mapping, chains, group_num);
}
} // namespace DAGDecomposition
