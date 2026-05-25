import numpy as np
import matplotlib.pyplot as plt

def visualize_samples(samples, nrow=4, title="Generated Samples"):
    """Visualize generated samples"""
    import torchvision.utils as vutils
    
    # Normalize to [0, 1] for visualization
    samples = (samples + 1) / 2  # From [-1, 1] to [0, 1]
    samples = samples.clamp(0, 1)
    
    grid = vutils.make_grid(samples, nrow=nrow, normalize=False)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


