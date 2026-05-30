from torch.utils.data import DataLoader
from torchvision import datasets, transforms



transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)


