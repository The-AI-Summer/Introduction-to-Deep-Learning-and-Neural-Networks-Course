import torch
import torch.nn as nn

seed = 172
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def fnn(input):
    model = nn.Sequential(nn.Linear(10, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 2)
                          )
    return model(input)