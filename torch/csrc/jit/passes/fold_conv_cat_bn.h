#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

TORCH_API void FoldConvCatBatchNorm2dForTracedModule(const Module& module);

} // namespace jit
} // namespace torch
