import torch
import torch.nn as nn


class SpatialVae(nn.Module):
    def __init__(self, in_channels=3, latent_channels=16, scale_factor=4):
        super().__init__()
        self.encoder = nn.Sequential()
        self.latent_channels = latent_channels
        self.scale_factor = scale_factor  # 32×32 → 8×8


        # Encode
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )


        # Z-latent space variables

        self.fc_mu = nn.Conv2d(128, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(128, latent_channels, kernel_size=1)

        # Decode
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):

        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        z = mu + std * eps

        return z

    def decode(self, z):
        x_recon = self.decoder(z)

        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)

        return y, mu, logvar, z
    

    def encode_only(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode_only(self, z):
        x = self.decode(z)

        return x
    

if __name__ == "__main__":
    vae = SpatialVae(in_channels=3, latent_channels=16, scale_factor=4)
    
    x = torch.randn(2, 3, 32, 32)
    x_recon, mu, logvar, z = vae(x)
    
    print(f"Input shape: {x.shape}")           # (2, 3, 32, 32)
    print(f"Latent shape: {z.shape}")          # (2, 16, 8, 8) ✓
    print(f"Reconstructed shape: {x_recon.shape}")  # (2, 3, 32, 32)
    
    # Model size
    params = sum(p.numel() for p in vae.parameters())
    print(f"VAE parameters: {params/1e6:.2f}M")