# Vision Transformer (ViT) - Quick Start Guide

This guide will help you quickly train and test Vision Transformer models on different datasets.

## Table of Contents
- [Installation](#installation)
- [Available Configurations](#available-configurations)
- [Training](#training)
- [Testing](#testing)
- [Examples](#examples)

---

## Installation

Ensure you have the required dependencies:
```bash
pip install torch torchvision tqdm matplotlib numpy
```

---

## Available Configurations

We provide pre-configured setups for multiple datasets:

| Config Name | Dataset | Image Size | Channels | Classes | Parameters | Training Time* |
|-------------|---------|------------|----------|---------|------------|----------------|
| `mnist` | MNIST | 28×28 | 1 | 10 | ~2M | 5-8 min |
| `cifar10` | CIFAR-10 | 32×32 | 3 | 10 | ~8M | 2-3 hrs |
| `cifar100` | CIFAR-100 | 32×32 | 3 | 100 | ~9M | 4-5 hrs |
| `imagenet` | ImageNet | 224×224 | 3 | 1000 | ~86M | 7-10 days |

*Estimated on 48GB GPU

View detailed config information:
```bash
python config.py
```

---

## Training

### Basic Training

Train with default configuration (MNIST):
```bash
python train.py
```

Train on a specific dataset:
```bash
python train.py --config cifar10
```

### Advanced Training Options

Override hyperparameters:
```bash
# Custom batch size and epochs
python train.py --config cifar10 --batch-size 64 --epochs 100

# Custom learning rate
python train.py --config mnist --lr 1e-3

# Use CPU instead of GPU
python train.py --config mnist --device cpu
```

### Full Command Options
```bash
python train.py --help
```

**Available arguments:**
- `--config {mnist,cifar10,cifar100,imagenet}` - Dataset configuration
- `--batch-size INT` - Override batch size
- `--epochs INT` - Override number of epochs
- `--lr FLOAT` - Override learning rate
- `--device {cuda,cpu}` - Override device

---

## Testing

### Basic Testing

Test the best model from training:
```bash
# Auto-detects config from checkpoint
python test.py --checkpoint ./checkpoints/mnist_vit_tiny_mnist/best_model.pth
```

### Test Specific Models

```bash
# Test CIFAR-10 model
python test.py --checkpoint ./checkpoints/cifar10_vit_small_cifar10/best_model.pth

# Test a specific epoch checkpoint
python test.py --checkpoint ./checkpoints/mnist_vit_tiny_mnist/checkpoint_epoch_20.pth

# Override config for testing
python test.py --checkpoint ./path/to/model.pth --config mnist
```

### Testing Output

The test script will:
- Load the model and display training info
- Evaluate on the test set
- Compute per-class accuracy
- Generate visualizations:
  - `predictions_visualization.png` - Sample predictions
  - `confusion_matrix.png` - Confusion matrix

---

## Examples

### Example 1: Quick MNIST Test (5 minutes)
```bash
# Train for just 10 epochs to test the pipeline
python train.py --config mnist --epochs 10

# Test the trained model
python test.py --checkpoint ./checkpoints/mnist_vit_tiny_mnist/best_model.pth
```

### Example 2: Full MNIST Training
```bash
# Train with default 30 epochs
python train.py --config mnist

# Expected accuracy: 98.5-99.2%
python test.py --checkpoint ./checkpoints/mnist_vit_tiny_mnist/best_model.pth
```

### Example 3: CIFAR-10 Training
```bash
# Full training (200 epochs, ~2-3 hours)
python train.py --config cifar10

# Test the model
python test.py --checkpoint ./checkpoints/cifar10_vit_small_cifar10/best_model.pth
```

### Example 4: CIFAR-10 with Custom Settings
```bash
# Smaller batch size, fewer epochs for faster experimentation
python train.py --config cifar10 --batch-size 64 --epochs 50 --lr 1e-4

# Test
python test.py --checkpoint ./checkpoints/cifar10_vit_small_cifar10/best_model.pth
```

### Example 5: CIFAR-100 Training
```bash
# More challenging dataset with 100 classes
python train.py --config cifar100

# Expected accuracy: ~70-75%
python test.py --checkpoint ./checkpoints/cifar100_vit_small_cifar100/best_model.pth
```

### Example 6: ImageNet Training (Requires ImageNet Dataset)
```bash
# Note: You must download and organize ImageNet first
# Expected location: ./data/imagenet/train/ and ./data/imagenet/val/

python train.py --config imagenet

# This will take 7-10 days on a 48GB GPU!
python test.py --checkpoint ./checkpoints/imagenet_vit_base_imagenet/best_model.pth
```

---

## Checkpoint Organization

Checkpoints are automatically organized by dataset and config:

```
checkpoints/
├── mnist_vit_tiny_mnist/
│   ├── best_model.pth           # Best validation accuracy
│   ├── checkpoint_epoch_5.pth   # Regular checkpoints
│   ├── checkpoint_epoch_10.pth
│   └── config.txt               # Config used for training
├── cifar10_vit_small_cifar10/
│   ├── best_model.pth
│   └── config.txt
└── imagenet_vit_base_imagenet/
    ├── best_model.pth
    └── config.txt
```

---

## Monitoring Training

During training, you'll see:
- Real-time progress bars with loss and accuracy
- Per-epoch summaries
- Validation metrics after each epoch
- Checkpoint saving notifications

Example output:
```
Epoch 1/30 [Train]: 100%|████████| 235/235 [00:12<00:00, loss=2.3012, acc=10.23%]

Epoch 1/30 Summary:
  Train Loss: 2.3012 | Train Acc: 10.23%
  Val Loss: 2.2156 | Val Acc: 15.67%
  Learning Rate: 0.000060

Checkpoint saved: ./checkpoints/mnist_vit_tiny_mnist/checkpoint_epoch_1.pth
```

---

## Tips for Success

### For MNIST:
- ✅ Very fast to train and test
- ✅ Great for debugging and learning
- ✅ Should reach 98%+ accuracy easily

### For CIFAR-10:
- ⏰ Takes a few hours to train properly
- 🎯 Aim for 85-90% accuracy with ViT-Small
- 💡 Use data augmentation (enabled by default)

### For CIFAR-100:
- ⏰ Takes longer (300 epochs recommended)
- 🎯 More challenging - expect 70-75% accuracy
- 💡 Consider mixup augmentation (enabled in config)

### For ImageNet:
- ⚠️ Requires downloading ImageNet separately
- ⏰ Very long training time (days)
- 💾 Large storage requirements
- 🎯 ViT-Base should reach 75-80% top-1 accuracy

---

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python train.py --config cifar10 --batch-size 64
```

### Training Too Slow
```bash
# Reduce number of data loader workers
# Edit config.py and change num_workers
```

### Want Faster Experimentation
```bash
# Train for fewer epochs
python train.py --config mnist --epochs 5
```

### Can't Find Checkpoint
```bash
# List available checkpoints
ls -R checkpoints/
```

---

## Next Steps

After training your models:
1. ✅ Check the generated visualizations
2. 📊 Compare different configurations
3. 🔬 Experiment with hyperparameters
4. 📈 Try different datasets
5. 🚀 Scale up to larger models

---

## Additional Resources

- View model architecture: `python model.py`
- Test data loading: `python data_loader.py`
- Inspect configurations: `python config.py`

For more information, see the original [ViT paper](https://arxiv.org/abs/2010.11929).
