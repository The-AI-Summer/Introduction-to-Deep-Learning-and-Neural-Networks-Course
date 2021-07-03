import torch
import torch.nn as nn

seed = 172
torch.manual_seed(seed)

# reconstructed and input are tensors of the same size
# mu and logvar are vectors of the same size

def elbo(reconstructed, input, mu, logvar):

    bce_loss = nn.BCELoss(reduction='sum')
    BCE = bce_loss(reconstructed, input)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD