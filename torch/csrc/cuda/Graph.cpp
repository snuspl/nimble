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

#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/cuda/Event.h>
#include <torch/csrc/cuda/Graph.h>
#include <torch/csrc/cuda/Module.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>

#include <structmember.h>
#include <cuda_runtime_api.h>

PyObject *THCPGraphClass = nullptr;

static PyObject *THCPGraph_pynew(
  PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THCPGraph* self = (THCPGraph *)ptr.get();
  new (&self->cdata) at::cuda::Graph();

  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THCPGraph_dealloc(THCPGraph *self) {
  self->cdata.~Graph();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *THCPGraph_launch(THCPGraph *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  auto& graph = self->cdata;
  with_no_gil([&] { graph.launch(); });
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject *pack_tensors(std::vector<at::Tensor> tensors) {
  if (tensors.empty())
    return PyTuple_New(0);

  auto num_tensors = tensors.size();
  THPObjectPtr packed(PyTuple_New(num_tensors));
  if (!packed)
    return nullptr;

  for (int i = 0; i < num_tensors; i++) {
    auto& tensor = tensors[i];
    THPObjectPtr value;
    if (!tensor.defined()) {
      Py_INCREF(Py_None);
      value = Py_None;
    } else {
      value = THPVariable_Wrap(tensor);
    }
    PyTuple_SET_ITEM(packed.get(), i, value.release());
  }

  return packed.release();
}

static std::vector<at::Tensor> unpack_tensors(PyObject *tensors) {
  std::vector<at::Tensor> unpacked;
  auto num_tensors = PyTuple_GET_SIZE(tensors);
  unpacked.reserve(num_tensors);

  for (int i = 0; i < num_tensors; i++) {
    PyObject *tensor = PyTuple_GET_ITEM(tensors, i);
    if (!THPVariable_Check(tensor)) {
      throw torch::TypeError("expected a torch.Tensor, but got %s", Py_TYPE(tensor)->tp_name);
    }
    unpacked.push_back(THPVariable_Unpack(tensor));
  }

  return unpacked;
}

static PyObject *THCPGraph_get_inputs(THCPGraph *self, void *unused) {
  HANDLE_TH_ERRORS
  return pack_tensors(self->cdata.inputs);
  END_HANDLE_TH_ERRORS
}

static PyObject *THCPGraph_set_inputs(THCPGraph *self, PyObject *arg, void *unused) {
  HANDLE_TH_ERRORS
  THPUtils_assert(!self->inputs_initialized,
    "The graph's inputs are already initialized");
  auto tensors = unpack_tensors(arg);
  self->cdata.inputs.clear();
  self->cdata.inputs = std::move(tensors);
  self->inputs_initialized = true;
  return 0;
  END_HANDLE_TH_ERRORS
}

static PyObject *THCPGraph_get_outputs(THCPGraph *self, void *unused) {
  HANDLE_TH_ERRORS
  return pack_tensors(self->cdata.outputs);
  END_HANDLE_TH_ERRORS
}

static PyObject *THCPGraph_set_outputs(THCPGraph *self, PyObject *arg, void *unused) {
  HANDLE_TH_ERRORS
  THPUtils_assert(!self->outputs_initialized,
    "The graph's outputs are already initialized");
  auto tensors = unpack_tensors(arg);
  self->cdata.outputs.clear();
  self->cdata.outputs = std::move(tensors);
  self->outputs_initialized = true;
  return 0;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THCPGraph_properties[] = {
  {"inputs", (getter)THCPGraph_get_inputs, (setter)THCPGraph_set_inputs, nullptr, nullptr},
  {"outputs", (getter)THCPGraph_get_outputs, (setter)THCPGraph_set_outputs, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THCPGraph_methods[] = {
  {"launch", (PyCFunction)THCPGraph_launch, METH_NOARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyTypeObject THCPGraphType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._CudaGraphBase",             /* tp_name */
  sizeof(THCPGraph),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THCPGraph_dealloc,         /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THCPGraph_methods,                     /* tp_methods */
  0,                                     /* tp_members */
  THCPGraph_properties,                  /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THCPGraph_pynew,                      /* tp_new */
};


void THCPGraph_init(PyObject *module)
{
  THCPGraphClass = (PyObject*)&THCPGraphType;
  if (PyType_Ready(&THCPGraphType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THCPGraphType);
  if (PyModule_AddObject(
      module, "_CudaGraphBase", (PyObject *)&THCPGraphType) < 0) {
    throw python_error();
  }
}
