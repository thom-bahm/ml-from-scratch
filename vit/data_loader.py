"""
Data loading utilities for MNIST dataset.
Handles downloading, preprocessing, and creating DataLoaders.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(config):
    """
    Create train and test DataLoaders for MNIST.
    
    Args:
        config: Configuration object with data settings
    
    Returns:
        train_loader, test_loader
    """
    
    # Training transforms with data augmentation
    if config.use_augmentation:
        train_transform = transforms.Compose([
            # Random crop with padding for slight translation invariance
            transforms.RandomCrop(config.img_size, padding=config.random_crop_padding),
            # Random rotation (small angles for MNIST)
            transforms.RandomRotation(10),
            # Convert to tensor and normalize
            transforms.ToTensor(),
            # Normalize to zero mean and unit variance
            # MNIST mean=0.1307, std=0.3081
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    print(f"Dataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    from config import config
    
    print("Testing MNIST data loading...")
    train_loader, test_loader = get_mnist_loaders(config)
    
    # Get a batch and print shapes
    images, labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Images shape: {images.shape}")  # Should be (batch_size, 1, 28, 28)
    print(f"  Labels shape: {labels.shape}")  # Should be (batch_size,)
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Sample labels: {labels[:10].tolist()}")
