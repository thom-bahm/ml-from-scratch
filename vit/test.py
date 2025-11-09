"""
Testing/evaluation script for trained Vision Transformer models.
Includes inference, metrics, and visualization utilities.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from model import VisionTransformer
from data_loader import get_mnist_loaders
from config import config


def load_model(checkpoint_path, config, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration object
        device: Device to load model on
    
    Returns:
        model: Loaded model
        checkpoint: Checkpoint dictionary with training info
    """
    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_size=config.mlp_size,
        dropout=0.0  # No dropout for inference
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']+1} epochs")
    print(f"Train accuracy: {checkpoint['train_acc']:.2f}%")
    print(f"Val accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model, checkpoint


def evaluate_model(model, test_loader, device):
    """
    Evaluate model and compute detailed metrics.
    
    Args:
        model: Trained ViT model
        test_loader: Test DataLoader
        device: Device to evaluate on
    
    Returns:
        accuracy, per_class_accuracy, confusion_matrix
    """
    model.eval()
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    confusion_matrix = torch.zeros(10, 10)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Overall accuracy
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
                
                # Confusion matrix
                confusion_matrix[label, predicted[i]] += 1
    
    accuracy = 100. * correct / total
    per_class_accuracy = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                         for i in range(10)]
    
    return accuracy, per_class_accuracy, confusion_matrix


def visualize_predictions(model, test_loader, device, num_samples=10):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained ViT model
        test_loader: Test DataLoader
        device: Device
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get a batch
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].cpu().squeeze().numpy()
        pred = predicted[i].item()
        true = labels[i].item()
        prob = probabilities[i, pred].item()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        color = 'green' if pred == true else 'red'
        axes[i].set_title(f'Pred: {pred} ({prob*100:.1f}%)\nTrue: {true}', 
                         color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: predictions_visualization.png")
    plt.close()


def plot_confusion_matrix(confusion_matrix, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix.numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, f'{int(cm[i, j])}',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def test_model(checkpoint_path='./checkpoints/best_model.pth'):
    """
    Main testing function.
    
    Args:
        checkpoint_path: Path to model checkpoint
    """
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading test data...")
    _, test_loader = get_mnist_loaders(config)
    
    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(checkpoint_path, config, device)
    
    # Evaluate
    print("\nEvaluating model on test set...")
    accuracy, per_class_acc, confusion_matrix = evaluate_model(model, test_loader, device)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Test Results:")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {accuracy:.2f}%\n")
    
    print("Per-class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"  Digit {i}: {acc:.2f}%")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_predictions(model, test_loader, device, num_samples=10)
    plot_confusion_matrix(confusion_matrix)
    
    print(f"\n{'='*70}")
    print("Testing complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    
    # Allow custom checkpoint path as command line argument
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else './checkpoints/best_model.pth'
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Train a model first using: python train.py")
        sys.exit(1)
    
    test_model(checkpoint_path)
