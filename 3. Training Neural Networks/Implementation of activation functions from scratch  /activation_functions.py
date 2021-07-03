import torch

def m_sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def m_tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def m_relu(x):
   return torch.mul(x , (x > 0))


def m_softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x))