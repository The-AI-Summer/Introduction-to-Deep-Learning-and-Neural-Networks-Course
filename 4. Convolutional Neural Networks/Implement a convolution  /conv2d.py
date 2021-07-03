import torch

def conv2d(image, kernel):
    H, W = list(image.size())
    M, N = list(kernel.size())

    out= torch.zeros(H-M+1, W-N+1, dtype=torch.float32)
    for i in range(H-M+1):
        for j in range(W-N+1):
            out[i,j]= torch.sum(image[i:i+M,j:j+N]*kernel)
    return out