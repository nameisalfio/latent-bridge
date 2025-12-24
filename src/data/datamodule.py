import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig

def get_dataloader(cfg: DictConfig, train: bool = True) -> DataLoader:
    """
    Standardized DataLoader provider for CIFAR-10.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    dataset = datasets.CIFAR10(
        root=cfg.dataset.data_dir, 
        train=train, 
        download=True, 
        transform=transform
    )
    
    return DataLoader(
        dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=train,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory
    )