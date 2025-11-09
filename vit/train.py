"""
Training script for Vision Transformer on MNIST.
Includes training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import os
import time
from pathlib import Path

from model import VisionTransformer
from data_loader import get_data_loaders
from config import get_config, CONFIG_REGISTRY


def get_lr_scheduler(optimizer, config, num_batches):
    """
    Create learning rate scheduler with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration object
        num_batches: Number of batches per epoch
    
    Returns:
        scheduler, warmup_scheduler
    """
    if config.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs - config.warmup_epochs,
            eta_min=1e-6
        )
    elif config.lr_schedule == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """
    Train for one epoch.
    
    Args:
        model: ViT model
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        config: Configuration object
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        if batch_idx % config.print_freq == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, test_loader, criterion, device):
    """
    Validate the model on test set.
    
    Args:
        model: ViT model
        test_loader: Test DataLoader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Validating')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, train_acc, val_acc, config, is_best=False):
    """
    Save model checkpoint with organized directory structure.
    Creates a folder with dataset name and config name.
    """
    # Create organized checkpoint directory
    # Format: checkpoints/{dataset}_{config_name}/
    checkpoint_base = Path(config.checkpoint_dir)
    run_dir = checkpoint_base / f"{config.dataset}_{config.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'config': config.to_dict()  # Save config as dict
    }
    
    # Save regular checkpoint
    checkpoint_path = run_dir / f'checkpoint_epoch_{epoch+1}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = run_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")
    
    # Save config separately for easy reference
    config_path = run_dir / 'config.txt'
    with open(config_path, 'w') as f:
        f.write(str(config))


def train_vit(config):
    """
    Main training function.
    
    Args:
        config: Configuration object
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n" + str(config) + "\n")
    
    # Create data loaders
    print("Loading data...")
    train_loader, test_loader = get_data_loaders(config)
    
    # Create model
    print("\nCreating model...")
    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_size=config.mlp_size,
        dropout=config.dropout
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))
    
    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(config.epochs):
        # Warmup learning rate for first few epochs
        if epoch < config.warmup_epochs:
            lr = config.learning_rate * (epoch + 1) / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Update learning rate (after warmup)
        if scheduler is not None and epoch >= config.warmup_epochs:
            scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        if (epoch + 1) % config.save_freq == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, train_acc, val_acc, config, is_best)
        
        print("-" * 70)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Vision Transformer')
    parser.add_argument(
        '--config', 
        type=str, 
        default='mnist',
        choices=list(CONFIG_REGISTRY.keys()),
        help=f'Configuration to use. Options: {list(CONFIG_REGISTRY.keys())}'
    )
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--device', type=str, default=None, help='Override device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.config)
    
    # Override config with command line arguments if provided
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        print(f"Overriding batch size: {config.batch_size}")
    if args.epochs is not None:
        config.epochs = args.epochs
        print(f"Overriding epochs: {config.epochs}")
    if args.lr is not None:
        config.learning_rate = args.lr
        print(f"Overriding learning rate: {config.learning_rate}")
    if args.device is not None:
        config.device = args.device
        print(f"Overriding device: {config.device}")
    
    print(f"\nStarting training with config: {args.config}")
    train_vit(config)
