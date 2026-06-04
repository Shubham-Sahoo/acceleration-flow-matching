import torch
from tqdm import tqdm

class FlowMatchingSpatialLatent:
    def __init__(self, model, vae, device='cpu'):
        self.model = model
        self.device = device
        self.vae = vae

    def forward_loss(self, x, class_labels, loss_criterion):
        x = x.to(self.device)  # (B, 3, 28, 28)
        B = x.size(0)
        class_labels = class_labels.to(self.device)

        self.vae.eval()

        with torch.no_grad():
            z_1 = self.vae.encode_only(x)  # (B, 16, 8, 8)

        t = torch.rand(B, device=self.device)
        t_new = t.view(B, 1, 1, 1)

        z_0 = torch.randn_like(z_1)
        
        z_t = (1-t_new) * z_0 + t_new * z_1
        
        v_target = z_1 - z_0

        v_pred = self.model(z_t, class_labels, t) 
        loss = None
        
        if loss_criterion is not None:
            loss = loss_criterion(v_pred, v_target)

        return loss, v_pred


    def sample(self, z_0, class_labels, t_steps=50, verbose=False):
        
        dt = 1.0/t_steps

        timesteps = torch.linspace(0, 1 - dt, t_steps, device=self.device)
        iterator = tqdm(timesteps, desc="Flow Matching") if verbose else timesteps  

        z_t = z_0.clone()

        with torch.no_grad():
            for idx, step in enumerate(iterator):
                t = step
                t_tensor = torch.full((z_t.size(0), ), t, device=self.device, dtype=torch.float32)

                # Predict noise
                v_pred = self.model(z_t, class_labels, t_tensor)

                # Denoise
                z_t = z_t + v_pred * dt
        
        x_t = self.vae.decode_only(z_t)

        return x_t.clamp(-1,1)
        
    def benchmark(self, x):
        pass