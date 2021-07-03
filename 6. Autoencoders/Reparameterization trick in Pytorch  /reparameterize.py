import torch

seed = 172
torch.manual_seed(seed)

def reparameterize(mu, log_var):
    """
        Args:
            `mu`: mean from the encoder's latent space
            `log_var`: log variance from the encoder's latent space

        Returns:
            the reparameterized latent vector z
      """
    var = torch.exp(log_var)  # standard deviation
    eps = torch.randn_like(var)  # `randn_like` as we need the same size
    sample = mu + (eps * var)  # sampling as if coming from the input space
    return sample