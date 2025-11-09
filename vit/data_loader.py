"""
Data loading utilities for multiple datasets.
Handles downloading, preprocessing, and creating DataLoaders.
Supports: MNIST, CIFAR-10, CIFAR-100, ImageNet.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data_loaders(config):
    """
    Get appropriate data loaders based on dataset in config.
    
    Args:
        config: Configuration object with dataset attribute
    
    Returns:
        train_loader, test_loader
    """
    dataset_name = config.dataset.lower()
    
    if dataset_name == 'mnist':
        return get_mnist_loaders(config)
    elif dataset_name == 'cifar10':
        return get_cifar10_loaders(config)
    elif dataset_name == 'cifar100':
        return get_cifar100_loaders(config)
    elif dataset_name == 'imagenet':
        return get_imagenet_loaders(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


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


def get_cifar10_loaders(config):
    """Create train and test DataLoaders for CIFAR-10."""
    
    # Training transforms with augmentation
    if config.use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(config.img_size, padding=config.random_crop_padding),
            transforms.RandomHorizontalFlip() if getattr(config, 'random_horizontal_flip', False) else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    # Test transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True if config.device == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True if config.device == 'cuda' else False
    )
    
    print(f"CIFAR-10 loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader


def get_cifar100_loaders(config):
    """Create train and test DataLoaders for CIFAR-100."""
    
    # Training transforms (same as CIFAR-10 but different normalization)
    if config.use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(config.img_size, padding=config.random_crop_padding),
            transforms.RandomHorizontalFlip() if getattr(config, 'random_horizontal_flip', False) else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True if config.device == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True if config.device == 'cuda' else False
    )
    
    print(f"CIFAR-100 loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader


def get_imagenet_loaders(config):
    """
    Create train and test DataLoaders for ImageNet.
    Note: Requires ImageNet dataset to be downloaded and organized.
    Expected structure: ./data/imagenet/train/ and ./data/imagenet/val/
    """
    
    # Training transforms with strong augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    try:
        train_dataset = torchvision.datasets.ImageFolder(
            root='./data/imagenet/train', transform=train_transform
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root='./data/imagenet/val', transform=test_transform
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "ImageNet dataset not found. Please download and organize ImageNet dataset in:\n"
            "  ./data/imagenet/train/\n"
            "  ./data/imagenet/val/"
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    
    print(f"ImageNet loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader


if __name__ == "__main__":
    # Test data loading for different datasets
    from config import get_config
    
    datasets = ['mnist', 'cifar10', 'cifar100']
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Testing {dataset_name.upper()} data loading...")
        print(f"{'='*70}")
        
        cfg = get_config(dataset_name)
        train_loader, test_loader = get_data_loaders(cfg)
        
        # Get a batch and print shapes
        images, labels = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Sample labels: {labels[:10].tolist()}")
