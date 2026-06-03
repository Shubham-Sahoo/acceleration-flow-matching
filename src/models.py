import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import time
import warnings
from einops import rearrange, reduce, repeat
warnings.filterwarnings('ignore')


# class PatchEmbedding(nn.Module):
#     def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = (img_size // patch_size) ** 2

#         self.proj = nn.Conv2d(
#             in_channels, embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size,
#         )

#     def forward(self, x):
#         # x -> (B,3,32,32)

#         x = self.proj(x)  # (B, embed_dim, 8, 8)
#         x = x.flatten(2)  # (B, embed_dim, 64)
#         x = x.permute(0, 2, 1) # (B, 64, embed_dim)

#         return x


# class BaseViT(nn.Module):
#     """Unified ViT backend for all models."""

#     def __init__(self, img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3):
#         super().__init__()

#         self.patch_embed = PatchEmbedding(img_size=32, patch_size=4, embed_dim=192)
#         self.num_patches = (img_size//patch_size)**2
#         self.num_patches_dim = (img_size//patch_size)
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, embed_dim))
#         self.time_embed = nn.Sequential(
#             nn.Linear(1, 128),
#             nn.GELU(),
#             nn.Linear(128, embed_dim)
#         )

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, 
#             nhead=num_heads,
#             dim_feedforward=embed_dim*4,
#             dropout=0.1,
#             batch_first=True,
#             activation='gelu'
#             )

#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

#         self.output_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, patch_size * patch_size * 3)
#         )

#         self.patch_size = patch_size
#         self.embed_dim = embed_dim

#         nn.init.normal_(self.pos_embed, std=0.02)
#         nn.init.normal_(self.cls_token, std=0.02)

#     def forward(self, x, t):
        
#         B = x.size(0)

#         x = self.patch_embed(x)

#         if t.dim() == 1:
#             t = t.unsqueeze(1)
        
#         t_emb = self.time_embed(t)
#         x = x + t_emb.unsqueeze(1)
#         cls = self.cls_token.expand(B, -1, -1)
#         x = torch.cat([cls, x], dim=1)
#         x = x + self.pos_embed
#         x = self.transformer(x)
#         x = x[:, 1: , :]
#         x = self.output_proj(x)
#         x = x.reshape(B, self.num_patches_dim, self.num_patches_dim, self.patch_size, self.patch_size, 3)
#         x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
#         x = x.reshape(B, 3, self.num_patches_dim * self.patch_size, self.num_patches_dim * self.patch_size)
#         return x


class PatchEmbedding(nn.Module):
    def __init__(self, latent_channels=16, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(
            latent_channels, embed_dim,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        # x -> (B,16,8,8)

        x = self.proj(x)  # (B, embed_dim, 8, 8)
        x = rearrange(x, 'b e h w -> b (h w) e')  # (B, 64, embed_dim)

        return x


class LatentViT(nn.Module):
    """Unified ViT backend for all models."""

    def __init__(self, latent_channels=16, embed_dim=192, depth=12, num_heads=3):
        super().__init__()

        self.patch_embed = PatchEmbedding(latent_channels=latent_channels, embed_dim=embed_dim)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))
        self.class_embed = nn.Embedding(10, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.class_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            batch_first=True,
            activation='gelu',
            norm_first=True
            )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim) 
            )

        self.output_proj = nn.Linear(embed_dim, latent_channels)

        self.embed_dim = embed_dim

        nn.init.normal_(self.pos_embed, std=0.02)
        # nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z, class_label, t):
        """
            z : (B, C, H, W)
            t : (B, 1, 1)
        """

        B, C, H, W = z.shape

        x = self.patch_embed(z)

        if t.dim() == 1:
            t = t.unsqueeze(1)

        c_emb = self.class_embed(class_label)       # (B, embed_dim)
        
        c_emb = self.class_proj(c_emb)
        t_emb = self.time_embed(t)
        x = x + t_emb.unsqueeze(1) + c_emb.unsqueeze(1)
        # cls = self.cls_token.expand(B, -1, -1)
        # x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        # x = x[:, 1: , :]
        x = self.output_proj(x)
        
        x = rearrange(x, 'b (h w) l -> b l h w', h=H, w=W)
        return x



# model = LatentViT(latent_channels=16, embed_dim=192, depth=12, num_heads=3)
# model = model.to('cpu')

# z = torch.randn((8, 16, 8, 8))
# t = torch.randn((8, 1))

# e = model.forward(z,t)
# print(e.shape)

# print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# print("✓ Model ready")
# torch.save(model.state_dict(), 'simplevit_baseline.pth')