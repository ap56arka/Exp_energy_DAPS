import torch


class LinearSchedule:
    """A linear noise schedule (PyTorch version)"""

    def __init__(self, num_diffusion_steps, beta_start, beta_end, device="cpu"):
        self.num_diffusion_steps = num_diffusion_steps
        
        # Linear beta schedule
        self.beta = torch.linspace(beta_start, beta_end, num_diffusion_steps, device=device)
        
        # Add beta_0 = 0 at the beginning (to match your JAX version)
        #self.beta = torch.cat([torch.zeros(1, device=device), self.beta])
        
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def __len__(self):
        return self.num_diffusion_steps


