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

#include <torch/csrc/jit/passes/fold_conv_bn.h>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/subgraph_matcher.h>

#include <ATen/core/grad_mode.h>

#include <algorithm>

namespace torch {
namespace jit {

namespace {

std::string conv_bn_pattern_string = R"IR(
graph(%x, %conv_submodule, %conv_b, %stride:int[], %padding:int[], %dilation:int[],
      %transposed:bool, %output_padding:int[], %groups:int,
      %benchmark:bool, %deterministic:bool, %cudnn_enabled:bool,
      %bn_submodule, %training:bool, %momentum:float, %eps:float):
    %conv_w = prim::GetAttr[name="weight"](%conv_submodule)
    %conv_out = aten::_convolution(%x, %conv_w, %conv_b, %stride, %padding, %dilation, %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled)
    %bn_w = prim::GetAttr[name="weight"](%bn_submodule)
    %bn_b = prim::GetAttr[name="bias"](%bn_submodule)
    %running_mean = prim::GetAttr[name="running_mean"](%bn_submodule)
    %running_var = prim::GetAttr[name="running_var"](%bn_submodule)
    %bn_out = aten::batch_norm(%conv_out, %bn_w, %bn_b, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled)
    return (%bn_out))IR";

bool hitGraphInput(Value* value) {
  Graph* graph = value->owningGraph();
  const auto& inputs = graph->inputs();
  return std::find(inputs.begin(), inputs.end(), value) != inputs.end();
}

std::vector<std::string> getModuleAccessPath(Value* instance, Value* self) {
  std::vector<std::string> path;
  Value* iter = instance;
  while (!hitGraphInput(iter) && iter->node()->kind() == prim::GetAttr) {
    Node* get_attr = iter->node();
    path.push_back(get_attr->s(attr::name));
    iter = get_attr->inputs()[0];
  }
  TORCH_INTERNAL_ASSERT(iter == self);
  std::reverse(path.begin(), path.end());
  return path;
}

script::Module getChildModuleByPath(
    const script::Module& module,
    const std::vector<std::string>& path) {
  script::Module m = module;
  for (const auto& p : path) {
    m = m.attr(p).toModule();
  }
  return m;
}

struct ConvBNParameters {
  at::Tensor conv_w;
  at::Tensor conv_b;
  at::Tensor bn_rm;
  at::Tensor bn_rv;
  double bn_eps = 0.0;
  at::Tensor bn_w;
  at::Tensor bn_b;
};

std::tuple<at::Tensor, at::Tensor> computeUpdatedConvWeightAndBias(
    const ConvBNParameters& p) {
  at::NoGradGuard no_grad;
  at::Tensor bn_var_rsqrt = at::rsqrt(p.bn_rv + p.bn_eps);
  at::Tensor new_w = p.conv_w * (p.bn_w * bn_var_rsqrt).reshape({-1, 1, 1, 1});
  at::Tensor new_b = (p.conv_b - p.bn_rm) * bn_var_rsqrt * p.bn_w + p.bn_b;
  return std::make_tuple(new_w, new_b);
}

bool hastensor(script::Module& m, const char* name) {
  return m.hasattr(name) && m.attr(name).isTensor();
}

void tryExtractingConvBNParameters(
    script::Module& conv,
    script::Module& bn,
    Value* bn_eps,
    ConvBNParameters& r,
    bool has_bias) {
  TORCH_INTERNAL_ASSERT(hastensor(conv, "weight"));
  TORCH_INTERNAL_ASSERT(conv.hasattr("bias") == has_bias);
  TORCH_INTERNAL_ASSERT(hastensor(bn, "weight"));
  TORCH_INTERNAL_ASSERT(hastensor(bn, "bias"));
  TORCH_INTERNAL_ASSERT(hastensor(bn, "running_mean"));
  TORCH_INTERNAL_ASSERT(hastensor(bn, "running_var"));

  r.bn_rm = bn.attr("running_mean").toTensor();
  r.bn_rv = bn.attr("running_var").toTensor();
  r.bn_eps = toIValue(bn_eps).value().toDouble();
  r.bn_w = bn.attr("weight").toTensor();
  r.bn_b = bn.attr("bias").toTensor();

  r.conv_w = conv.attr("weight").toTensor();
  if (has_bias) {
    r.conv_b = conv.attr("bias").toTensor();
  } else {
    r.conv_b = at::zeros_like(r.bn_rm);
  }
}

} // namespace


void FoldConvBatchNorm2dForTracedModule(const script::Module& module) {
  std::unique_ptr<Graph> conv_bn_pattern_graph = at::guts::make_unique<Graph>();
  std::unordered_map<std::string, Value*> conv_bn_vmap;
  script::parseIR(conv_bn_pattern_string, conv_bn_pattern_graph.get(), conv_bn_vmap);

  Value* pattern_conv_out = conv_bn_vmap.at("conv_out");
  Value* pattern_conv_b = conv_bn_vmap.at("conv_b");
  Value* pattern_bn_out = conv_bn_vmap.at("bn_out");
  Value* pattern_conv_submodule = conv_bn_vmap.at("conv_submodule");
  Value* pattern_bn_submodule = conv_bn_vmap.at("bn_submodule");
  Value* pattern_training = conv_bn_vmap.at("training");
  Value* pattern_eps = conv_bn_vmap.at("eps");
  Node* pattern_conv = pattern_conv_out->node();
  Node* pattern_bn = pattern_bn_out->node();

  script::Method top_method = module.get_method("forward");
  auto top_graph = top_method.graph();
  Value* top_module = top_graph->inputs()[0];

  std::unordered_map<Value*, Value*> matched_bn_value_to_rewriting_value;
  std::vector<Value*> matched_bn_values;
  std::unordered_set<Node*> matched_bn_nodes;

  const auto& matches = findPatternMatches(*conv_bn_pattern_graph, *top_graph);

  for (const Match& match : matches) {
    // check training == False
    auto training = toIValue(match.values_map.at(pattern_training));
    TORCH_INTERNAL_ASSERT(training);
    TORCH_INTERNAL_ASSERT(!training.value().toBool());

    auto conv_b = match.values_map.at(pattern_conv_b);
    bool has_bias = true;
    auto conv_b_value = toIValue(conv_b);
    if (conv_b_value) {
      TORCH_INTERNAL_ASSERT(conv_b_value.value().isNone());
      has_bias = false;
    }

    Node* matched_conv = match.nodes_map.at(pattern_conv);
    Node* matched_bn = match.nodes_map.at(pattern_bn);
    Node* matched_conv_submodule =
        match.values_map.at(pattern_conv_submodule)->node();
    Node* matched_bn_submodule =
        match.values_map.at(pattern_bn_submodule)->node();

    TORCH_INTERNAL_ASSERT(matched_conv_submodule->kind() == prim::GetAttr);
    TORCH_INTERNAL_ASSERT(matched_bn_submodule->kind() == prim::GetAttr);

    auto conv_module_path = getModuleAccessPath(matched_conv_submodule->output(), top_module);
    auto bn_module_path = getModuleAccessPath(matched_bn_submodule->output(), top_module);
    script::Module conv_submodule = getChildModuleByPath(module, conv_module_path);
    script::Module bn_submodule = getChildModuleByPath(module, bn_module_path);

    ConvBNParameters params;
    tryExtractingConvBNParameters(conv_submodule, bn_submodule, match.values_map.at(pattern_eps), params, has_bias);

    matched_bn_values.push_back(matched_bn->output());
    matched_bn_nodes.insert(matched_bn);

    auto new_w_b = computeUpdatedConvWeightAndBias(params);
    conv_submodule.setattr("weight", std::get<0>(new_w_b));
    if (has_bias) {
      conv_submodule.setattr("bias", std::get<1>(new_w_b));
      matched_bn_value_to_rewriting_value[matched_bn->output()] = matched_conv->output();
    } else {
      // Instead of using "conv with bias", we add an extra node for bias addition.
      conv_submodule.register_parameter("_bias_from_folded_bn", std::get<1>(new_w_b).unsqueeze(0).unsqueeze(2).unsqueeze(3), false);
      auto conv_module_t = matched_conv_submodule->output()->type()->expect<ClassType>();
      conv_module_t->addAttribute("_bias_from_folded_bn", TensorType::get(), false);

      auto get_bias = top_graph->createGetAttr(matched_conv_submodule->output(), "_bias_from_folded_bn")->insertAfter(matched_conv);
      auto one = top_graph->insertConstant(1);
      auto add_bias = top_graph->create(aten::add, {matched_conv->output(), get_bias->output(), one})->insertAfter(get_bias);
      one->node()->moveBefore(add_bias);
      matched_bn_value_to_rewriting_value[matched_bn->output()] = add_bias->output();
    }
  }

  for (auto v : matched_bn_values) {
    v->replaceAllUsesWith(matched_bn_value_to_rewriting_value.at(v));
  }

  for (auto n : matched_bn_nodes) {
    n->removeAllInputs();
  }
  for (auto n : matched_bn_nodes) {
    n->destroy();
  }
}

namespace {
struct ConvCatBNParameters {
  std::vector<at::Tensor> conv_w;
  std::vector<at::Tensor> conv_b;

  at::Tensor bn_rm;
  at::Tensor bn_rv;
  double bn_eps = 0.0;
  at::Tensor bn_w;
  at::Tensor bn_b;
};

void tryExtractingConvCatBNParameters(
    std::vector<script::Module>& conv_modules,
    std::vector<bool>& has_bias,
    script::Module& bn,
    Value* bn_eps,
    ConvCatBNParameters& r) {

  int conv_num = conv_modules.size();
  for (int i = 0; i < conv_num; i++) {
    const auto& conv = conv_modules[i];
    TORCH_INTERNAL_ASSERT(hastensor(conv_modules[i], "weight"));
    TORCH_INTERNAL_ASSERT(conv.hasattr("bias") == has_bias[i]);

    auto conv_w = conv.attr("weight").toTensor();
    r.conv_w.push_back(conv_w);
    if (has_bias[i]) {
      r.conv_b.push_back(conv.attr("bias").toTensor());
    } else {
      at::Tensor zero_bias = at::zeros({conv_w.size(0)}, conv_w.options());
      r.conv_b.push_back(zero_bias);
    }
  }

  TORCH_INTERNAL_ASSERT(hastensor(bn, "weight"));
  TORCH_INTERNAL_ASSERT(hastensor(bn, "bias"));
  TORCH_INTERNAL_ASSERT(hastensor(bn, "running_mean"));
  TORCH_INTERNAL_ASSERT(hastensor(bn, "running_var"));

  r.bn_rm = bn.attr("running_mean").toTensor();
  r.bn_rv = bn.attr("running_var").toTensor();
  r.bn_eps = toIValue(bn_eps).value().toDouble();
  r.bn_w = bn.attr("weight").toTensor();
  r.bn_b = bn.attr("bias").toTensor();
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> computeUpdatedConvWeightAndBias(
    const ConvCatBNParameters& p) {
  at::NoGradGuard no_grad;
  at::Tensor bn_var_rsqrt = at::rsqrt(p.bn_rv + p.bn_eps);

  std::vector<int64_t> channel_sizes;
  for (const auto& conv : p.conv_w) {
    channel_sizes.push_back(conv.size(0));
  }

  auto bn_coeffs = (p.bn_w * bn_var_rsqrt).split_with_sizes(channel_sizes, 0);
  auto bn_running_means = p.bn_rm.split_with_sizes(channel_sizes, 0);
  auto bn_biases = p.bn_b.split_with_sizes(channel_sizes, 0);

  std::vector<at::Tensor> new_weights;
  std::vector<at::Tensor> new_biases;
  int conv_num = p.conv_w.size();
  for (int i = 0; i < conv_num; i++) {
    at::Tensor new_w = p.conv_w[i] * (bn_coeffs[i].reshape({-1, 1, 1, 1}));
    new_weights.push_back(new_w);
    at::Tensor new_b = (p.conv_b[i] - bn_running_means[i]) * bn_coeffs[i] + bn_biases[i];
    new_biases.push_back(new_b);
  }

  return std::make_tuple(new_weights, new_biases);
}
} // namespace


void FoldConvCatBatchNorm2dForTracedModule(const script::Module& module) {
  script::Method top_method = module.get_method("forward");
  auto graph = top_method.graph();

  std::unordered_map<Value*, Value*> matched_bn_value_to_rewriting_value;
  std::vector<Value*> matched_bn_values;
  std::unordered_set<Node*> matched_bn_nodes;
  std::unordered_set<Node*> single_tensor_cat_nodes;

  for (auto it = graph->nodes().rbegin(); it != graph->nodes().rend(); it++) {
    Node* cat = *it;
    if (cat->kind() != aten::cat)
      continue;
    if (!cat->is_constant(attr::dim))
      continue;

    // NOTE: here we only handle the concatenation along channel dimension
    int64_t cat_dim = cat->get<int64_t>(attr::dim).value();
    if (cat_dim != 1)
      continue;

    Node* list_construct = cat->namedInput(attr::tensors)->node();
    if (list_construct->kind() != prim::ListConstruct)
      continue;
    if (list_construct->output()->uses().size() > 1)
      continue;
    if (!cat->output()->hasUses() || cat->output()->uses().size() > 1)
      continue;

    // NOTE: here we assume the graph haven't gone through the non-diff optimiation pass
    Node* batchnorm = cat->output()->uses()[0].user;
    if (batchnorm->kind() != aten::batch_norm)
      continue;

    // NOTE: here we only handle the case where all operands of the concatenation are convolution outputs.
    bool conv_cat = true;
    auto cat_inputs = list_construct->inputs();
    for (auto input_it = cat_inputs.begin(); input_it != cat_inputs.end(); input_it++) {
      Value* input_val = *input_it;
      if (input_val->node()->kind() != aten::_convolution || input_val->uses().size() > 1) {
        conv_cat = false;
        break;
      }
    }
    if (!conv_cat)
      continue;

    Value* top_module = graph->inputs()[0];
    std::vector<script::Module> conv_submodules;
    std::vector<bool> has_bias;
    for (auto input_it = cat_inputs.begin(); input_it != cat_inputs.end(); input_it++) {
      Value* input_val = *input_it;
      Node* conv = input_val->node();
      Value* conv_w = conv->inputs()[1];
      Value* conv_module_val = conv_w->node()->input();
      auto conv_module_path = getModuleAccessPath(conv_module_val, top_module);
      conv_submodules.push_back(getChildModuleByPath(module, conv_module_path));

      Value* conv_b = conv->inputs()[2];
      auto conv_b_val = toIValue(conv_b);
      if (conv_b_val) {
        TORCH_INTERNAL_ASSERT(conv_b_val.value().isNone());
        has_bias.push_back(false);
      } else {
        has_bias.push_back(true);
      }
    }

    Value* bn_w = batchnorm->inputs()[1];
    Value* bn_module_val = bn_w->node()->input();
    auto bn_module_path = getModuleAccessPath(bn_module_val, top_module);
    script::Module bn_submodule = getChildModuleByPath(module, bn_module_path);

    const int eps_index = 7;
    Value* bn_eps = batchnorm->inputs()[eps_index];

    ConvCatBNParameters params;
    tryExtractingConvCatBNParameters(conv_submodules, has_bias, bn_submodule, bn_eps, params);
    matched_bn_values.push_back(batchnorm->output());
    matched_bn_nodes.insert(batchnorm);

    auto new_w_b = computeUpdatedConvWeightAndBias(params);
    int conv_num = conv_submodules.size();
    for (int i = 0; i < conv_num; i++) {
      conv_submodules[i].setattr("weight", std::get<0>(new_w_b)[i]);
      if (has_bias[i]) {
        conv_submodules[i].setattr("bias", std::get<1>(new_w_b)[i]);
      } else {
        Node* conv = cat_inputs[i]->node();
        Value* conv_w = conv->inputs()[1];
        Value* conv_module_val = conv_w->node()->input();

        conv_submodules[i].register_parameter("_bias_from_folded_bn", std::get<1>(new_w_b)[i].unsqueeze(0).unsqueeze(2).unsqueeze(3), false);
        auto conv_module_t = conv_module_val->type()->expect<ClassType>();
        conv_module_t->addAttribute("_bias_from_folded_bn", TensorType::get(), false);

        auto get_bias = graph->createGetAttr(conv_module_val, "_bias_from_folded_bn")->insertAfter(conv);
        auto one = graph->insertConstant(1);
        auto add_bias = graph->create(aten::add, {conv->output(), get_bias->output(), one})->insertAfter(get_bias);
        one->node()->moveBefore(add_bias);

        list_construct->replaceInput(i, add_bias->output());
      }
    }

    if (list_construct->inputs().size() == 1) {
      Node* conv_bias = list_construct->input()->node();
      matched_bn_value_to_rewriting_value[batchnorm->output()] = conv_bias->output();
      single_tensor_cat_nodes.insert(cat);
      single_tensor_cat_nodes.insert(list_construct);
    } else {
      matched_bn_value_to_rewriting_value[batchnorm->output()] = cat->output();
    }
  }

  for (auto v : matched_bn_values) {
    v->replaceAllUsesWith(matched_bn_value_to_rewriting_value.at(v));
  }

  for (auto n : matched_bn_nodes) {
    n->removeAllInputs();
  }
  for (auto n : single_tensor_cat_nodes) {
    n->removeAllInputs();
  }
  for (auto n : matched_bn_nodes) {
    n->destroy();
  }
  for (auto n : single_tensor_cat_nodes) {
    n->destroy();
  }
}

} // namespace jit
} // namespace torch
