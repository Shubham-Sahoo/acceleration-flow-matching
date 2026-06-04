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

    def forward_loss(self, x, num_steps, loss_criterion=None):
        x = x.to(self.device)  # (B, 3, 8, 8)
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

        loss = None

        if loss_criterion is not None:
            loss = loss_criterion(epsilon_pred, epsilon)

        return loss, epsilon_pred


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


class DDPMSpatialLatent:
    def __init__(self, model, vae, num_steps=1000, device='cpu'):
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.vae = vae

        self.betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_1_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def forward_loss(self, x, class_labels, num_steps, loss_criterion):
        x = x.to(self.device)  # (B, 3, 28, 28)
        B = x.size(0)
        class_labels = class_labels.to(self.device)

        self.vae.eval()

        with torch.no_grad():
            z_0 = self.vae.encode_only(x)  # (B, 16, 8, 8)

        t_indices = torch.randint(0, num_steps, (B,), device=self.device)
        t = t_indices.float() / num_steps  # Normalize to [0, 1]
        
        epsilon = torch.randn_like(z_0)

        sqrt_alpha_bars = self.sqrt_alpha_bars[t_indices]
        sqrt_1_minus_alpha_bars = self.sqrt_1_minus_alpha_bars[t_indices]

        sqrt_alpha_bars = sqrt_alpha_bars.view(B, 1, 1, 1)
        sqrt_1_minus_alpha_bars = sqrt_1_minus_alpha_bars.view(B, 1, 1, 1)
        
        z_t = sqrt_alpha_bars * z_0 + sqrt_1_minus_alpha_bars * epsilon

        epsilon_pred = self.model(z_t, class_labels, t) 
        loss = None
        
        if loss_criterion is not None:
            loss = loss_criterion(epsilon_pred, epsilon)

        return loss, epsilon_pred


    def sample(self, z_t, class_labels, t_steps=50, verbose=False):
        timesteps = torch.linspace(self.num_steps - 1, 0, t_steps, dtype=torch.long, device=self.device)
        iterator = tqdm(timesteps, desc="DDPM") if verbose else timesteps  

        with torch.no_grad():
            for idx, step in enumerate(iterator):
                t = step/self.num_steps
                t_tensor = torch.full((z_t.size(0), ), t, device=self.device, dtype=torch.float32)

                # Predict noise
                epsilon_pred = self.model(z_t, class_labels, t_tensor)

                # Denoise

                beta_t = self.betas[step]
                alpha_t = self.alphas[step]
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_1_minus_alpha_bar_t = self.sqrt_1_minus_alpha_bars[step]

                z_t = (1/sqrt_alpha_t) * (z_t - (beta_t/sqrt_1_minus_alpha_bar_t) * epsilon_pred)

                # Stochastic noise 
                if step > 0:
                    sigma = torch.sqrt(beta_t)
                    z_noise = torch.randn_like(z_t)
                    z_t = z_t + sigma * z_noise
        
        x_t = self.vae.decode_only(z_t)

        return x_t.clamp(-1,1)
        
    def benchmark(self, x):
        pass




class DDIMSampler:
    def __init__(self, model, num_steps=1000, device='cpu'):
        self.model = model
        self.device = device
        self.num_steps = num_steps

        self.betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_1_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

        self.history = {
            'timestep': [],           
            'epsilon_pred_norm': [],  
            'x_0_hat_norm': [],      
            'sigma': [],             
            'direction_norm': [],    
            'x_t_norm': [],          
            'noise_added': [],       
        }
        
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


    def sample(self, x_t, t_steps=50, eta = 0.0, verbose=False):
        timesteps = torch.linspace(self.num_steps - 1, 0, t_steps, dtype=torch.long, device=self.device)
        iterator = tqdm(timesteps, desc="DDPM") if verbose else timesteps  

        with torch.no_grad():
            for idx, step in enumerate(iterator):
                t = step/self.num_steps
                t_tensor = torch.full((x_t.size(0), ), t, device=self.device, dtype=torch.float32)

                # Predict noise
                epsilon_pred = self.model(x_t, t_tensor)
                
                # Alphas
                alpha_bar_t = self.alpha_bars[step]
                sqrt_alpha_bar_t = self.sqrt_alpha_bars[step]
                sqrt_1_minus_alpha_bar_t = self.sqrt_1_minus_alpha_bars[step]
                sqrt_alpha_bar_t_prev = self.sqrt_alpha_bars[step-1] if step>0 else torch.tensor(1.0, device=self.device)
                sqrt_1_minus_alpha_bar_t_prev = torch.sqrt(1-self.alpha_bars[step-1]) if step>0 else torch.tensor(0.0, device=self.device)
                alpha_bar_t_prev = self.alpha_bars[step-1] if step>0 else torch.tensor(1.0, device=self.device)

                # Model's best guess

                x_hat_0 = (1.0/sqrt_alpha_bar_t) * (x_t - sqrt_1_minus_alpha_bar_t * epsilon_pred) 

                # Stochastic noise 
                if step > 0 and eta > 0:
                    sigma_2 = eta * (sqrt_1_minus_alpha_bar_t_prev/sqrt_1_minus_alpha_bar_t)*(torch.sqrt(1 - alpha_bar_t/alpha_bar_t_prev))
                    sigma = torch.sqrt(sigma_2)
                    
                else:
                    sigma = 0.0
                    sigma_2 = 0.0
                

                # Denoise
                noise_direction = torch.sqrt(torch.clamp(1 - alpha_bar_t_prev - sigma_2, min=1e-8))
                
                x_t = sqrt_alpha_bar_t_prev * x_hat_0 + noise_direction * epsilon_pred

                z = torch.randn_like(x_t)
                x_t = x_t + sigma * z
            
        return x_t.clamp(-1,1)
        
    def benchmark(self, x):
        pass