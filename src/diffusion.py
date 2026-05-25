import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class DDPMSampler:
    def __init__(self, model, num_steps=1000, device='cpu'):
        self.model = model
        self.device = device
        self.num_steps = num_steps

        self.betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_1_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def forward_loss(self, x, num_steps, loss_criterion):
        x = x.to(self.device)  # (B, 3, 32, 32)
        B = x.size(0)

        t_indices = torch.randint(0, num_steps, (B,), device=self.device)
        t = t_indices.float() / num_steps  # Normalize to [0, 1]
        
        epsilon = torch.randn_like(x)

        sqrt_alpha_bars = self.sqrt_alpha_bars[t_indices]
        sqrt_1_minus_alpha_bars = self.sqrt_1_minus_alpha_bars[t_indices]

        sqrt_alpha_bars = sqrt_alpha_bars.view(B, 1, 1, 1)
        sqrt_1_minus_alpha_bars = sqrt_1_minus_alpha_bars.view(B, 1, 1, 1)
        
        x_t = sqrt_alpha_bars * x + sqrt_1_minus_alpha_bars * epsilon

        epsilon_pred = self.model(x_t, t) 
        loss = loss_criterion(epsilon_pred, epsilon)

        return loss


    def sample(self, x_t, t_steps=50, verbose=False):
        timesteps = torch.linspace(self.num_steps - 1, 0, t_steps, dtype=torch.long, device=self.device)
        iterator = tqdm(timesteps, desc="DDPM") if verbose else timesteps  

        with torch.no_grad():
            for idx, step in enumerate(iterator):
                t = step/self.num_steps
                t_tensor = torch.full((x_t.size(0), ), t, device=self.device, dtype=torch.float32)

                # Predict noise
                epsilon_pred = self.model(x_t, t_tensor)

                # Denoise

                beta_t = self.betas[step]
                alpha_t = self.alphas[step]
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_1_minus_alpha_bar_t = self.sqrt_1_minus_alpha_bars[step]

                x_t = (1/sqrt_alpha_t) * (x_t - (beta_t/sqrt_1_minus_alpha_bar_t) * epsilon_pred)

                # Stochastic noise 
                if step > 0:
                    sigma = torch.sqrt(beta_t)
                    z = torch.randn_like(x_t)
                    x_t = x_t + sigma * z
            
        return x_t.clamp(-1,1)
        
    def benchmark(self, x):
        pass