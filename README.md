# Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

Nimble is a deep learning execution engine that accelerates model inference and training by running GPU tasks (i.e., GPU kernels and memory operations) in parallel with minimal scheduling overhead.
Given a PyTorch DL model, Nimble automatically generates a GPU task schedule, which employs an optimal parallelization strategy for the model.
The schedule is wrapped in a `Nimble` object and can be seamlessly applied to PyTorch programs.
Nimble improves the speed of inference and training by up to 22.34× and 3.61× compared to PyTorch, respectively. Moreover, Nimble outperforms TensorRT by up to 2.81×.

* Speedup in Inference (ImageNet models)

<div align="center">
  <img src="https://github.com/snuspl/nimble/blob/main/figures/inference.png">
  <br/>
  Inference performance comparison on an NVIDIA V100 GPU.
</div>

* Speedup in Training (CIFAR-10 models)

| Batch 32 |  Batch 64 | Batch 128 |
|:---:|:---:|:---:|
| <img src="https://github.com/snuspl/nimble/blob/main/figures/batch_32.png"> | <img src="https://github.com/snuspl/nimble/blob/main/figures/batch_64.png"> | <img src="https://github.com/snuspl/nimble/blob/main/figures/batch_128.png"> |

<p align="middle">
  Training performance comparison on an NVIDIA V100 GPU.
</p>


## Install Nimble

Please refer to [instructions](NIMBLE_INSTALL.md) to install Nimble from source.

## Use Nimble

Nimble supports both inference and training of neural networks.

### Model Inference

```python
import torch
import torchvision

# Instantiate a PyTorch Module and move it to a GPU
model = torchvision.models.resnet50()
model = model.cuda()
model.eval()

# Prepare a dummy input
input_shape = [1, 3, 224, 224]
dummy_input = torch.randn(*input_shape).cuda()

# Create a Nimble object
nimble_model = torch.cuda.Nimble(model)
nimble_model.prepare(dummy_input, training=False)

# Execute the object
rand_input = torch.rand(*input_shape).cuda()
output = nimble_model(rand_input)
```

### Model Training

```python
import torch
import torchvision

BATCH = 32

# Instantiate a PyTorch Module and move it to a GPU
model = torchvision.models.mobilenet_v2(num_classes=10)
model = model.cuda()
model.train()

# Define a loss function and an optimizer
loss_fn = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Prepare a dummy input
input_shape = [BATCH, 3, 32, 32]
dummy_input = torch.randn(*input_shape).cuda()

# Create a Nimble object
nimble_model = torch.cuda.Nimble(model)
nimble_model.prepare(dummy_input, training=True)

# Execute the forward pass
rand_input = torch.rand(*input_shape).cuda()
output = nimble_model(rand_input)

# Compute loss
label = torch.zeros(BATCH, dtype=torch.long).cuda()
loss = loss_fn(output, label)

# Execute the backward pass
loss.backward()

# Perform an optimization step
optimizer.step()
```

## Reproduce Evaluation Results

Please refer to [evaluation instructions](NIMBLE_EVAL.md) to reproduce the evaluation results.

## Publication

Woosuk Kwon*, Gyeong-In Yu*, Eunji Jeong, and Byung-Gon Chun (* equal contribution), [Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning](https://papers.nips.cc/paper/2020/file/5f0ad4db43d8723d18169b2e4817a160-Paper.pdf), 34th Conference on Neural Information Processing Systems (NeurIPS), December 2020.
Spotlight (385/9467=4.1%)

## Citation

```bibtex
@inproceedings{kwon2020nimble,
  title={Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning},
  author={Kwon, Woosuk and Yu, Gyeong-In and Jeong, Eunji and Chun, Byung-Gon},
  booktitle={NeurIPS},
  year={2020}
}
```

## Troubleshooting
Create an issue for questions and bug reports.

## Contribution
Nimble adopts the Apache project model. We aim to create an open-source project that is contributed by the open-source community.
We maintain a mailing list (nimble-discuss@googlegroups.com) for general discussions about development.

## License
[BSD 3-clause license](LICENSE)
