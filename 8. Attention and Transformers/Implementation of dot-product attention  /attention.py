import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 172
torch.manual_seed(seed)

class Attention(nn.Module):

    def __init__(self, y_dim: int, h_dim: int):
        super().__init__()
        self.y_dim = y_dim
        self.h_dim = h_dim

        self.W = nn.Parameter(torch.FloatTensor(
            self.y_dim, self.h_dim))

    def forward(self,
                y: torch.Tensor,
                h: torch.Tensor):

        score = torch.matmul(torch.matmul(y, self.W), h.T)
        z = F.softmax(score, dim=0)
        return torch.matmul(z, h)