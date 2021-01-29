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

#include <torch/csrc/jit/passes/prepare_elementwise_op_fusion.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <utility>
#include <vector>


namespace torch {
namespace jit {

namespace {

const std::string operation_string = "__OPERATION__";

const std::string pattern_template_for_unary_op = R"IR(
graph(%a):
    %r = __OPERATION__(%a)
    return (%r))IR";
const std::string pattern_template_for_binary_op = R"IR(
graph(%a, %b):
    %r = __OPERATION__(%a, %b)
    return (%r))IR";
const std::string pattern_template_for_ternary_op = R"IR(
graph(%a, %b, %c):
    %r = __OPERATION__(%a, %b, %c)
    return (%r))IR";

const std::vector<std::pair<std::string, std::string>> unary_op_to_rewriting_op = {
    {"aten::relu_", "aten::relu"},
    {"aten::tanh_", "aten::tanh"}
};
const std::vector<std::pair<std::string, std::string>> binary_op_to_rewriting_op = {
    {"aten::mul_", "aten::mul"},
    {"aten::div_", "aten::div"}
};
const std::vector<std::pair<std::string, std::string>> ternary_op_to_rewriting_op = {
    {"aten::add_", "aten::add"},
    {"aten::sub_", "aten::sub"},
    {"aten::clamp_", "aten::clamp"},
    {"aten::hardtanh_", "aten::clamp"},
    {"aten::hardtanh", "aten::clamp"}
};

// special handling for sigmoid
const std::string pattern_string_for_sigmoid_inplace = R"IR(
graph(%a):
    %r = aten::sigmoid(%a, %a)
    return (%r))IR";
const std::string pattern_string_for_sigmoid = R"IR(
graph(%a):
    %r = aten::sigmoid(%a)
    return (%r))IR";

std::string replaceString(std::string target,
                          const std::string& find,
                          const std::string& replace) {
  size_t pos = 0;
  while ((pos = target.find(find, pos)) != std::string::npos) {
    target.replace(pos, find.length(), replace);
    pos += replace.length();
  }
  return target;
}

SubgraphRewriter getRewriter(const std::string& pattern_template,
                             const std::pair<std::string, std::string>& p) {
  auto original_pattern = replaceString(
      pattern_template,
      operation_string,
      p.first);
  auto rewriting_pattern = replaceString(
      pattern_template,
      operation_string,
      p.second);
  SubgraphRewriter subgraph_rewriter;
  subgraph_rewriter.RegisterRewritePattern(original_pattern, rewriting_pattern);
  return subgraph_rewriter;
}

} // namespace


void PrepareElementwiseOpFusion(const Module& module) {
  for (const std::pair<std::string, std::string>& p : unary_op_to_rewriting_op) {
    SubgraphRewriter rewriter = getRewriter(pattern_template_for_unary_op, p);
    rewriter.runOnModule(module);
  }
  for (const std::pair<std::string, std::string>& p : binary_op_to_rewriting_op) {
    SubgraphRewriter rewriter = getRewriter(pattern_template_for_binary_op, p);
    rewriter.runOnModule(module);
  }
  for (const std::pair<std::string, std::string>& p : ternary_op_to_rewriting_op) {
    SubgraphRewriter rewriter = getRewriter(pattern_template_for_ternary_op, p);
    rewriter.runOnModule(module);
  }

  // special handling for sigmoid
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(pattern_string_for_sigmoid_inplace, pattern_string_for_sigmoid);
  rewriter.runOnModule(module);
}

} // namespace jit
} // namespace torch
