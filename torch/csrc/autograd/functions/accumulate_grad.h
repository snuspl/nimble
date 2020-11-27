#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

struct TORCH_API AccumulateGrad : public Node {
  explicit AccumulateGrad(Variable variable_);

  variable_list apply(variable_list&& grads) override;
  
  c10::optional<c10::Stream> stream(const c10::DeviceType device_type) override;

  Variable variable;

  c10::optional<c10::Stream> captured_stream = c10::nullopt;
};

}} // namespace torch::autograd
