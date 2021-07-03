import torch

def neuron(input):
  weights = torch.Tensor([0.5, 0.5, 0.5])
  b = torch.Tensor([0.5])
  return torch.add(torch.matmul(input, weights), b)