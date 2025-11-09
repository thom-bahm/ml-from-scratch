"""
Configuration file for Vision Transformer training on MNIST.
This uses a ViT-Tiny architecture optimized for the small MNIST images.
"""

class ViTConfig:
    """ViT-Tiny configuration for MNIST (28x28 grayscale images)"""
    
    # Model Architecture
    img_size = 28          # MNIST native size (no resizing needed)
    patch_size = 7         # 7x7 patches -> 4x4 = 16 patches total
                          # Alternative: patch_size=4 for 49 patches (more detail, slower)
    in_channels = 1        # Grayscale images
    num_classes = 10       # Digits 0-9
    embed_dim = 192        # Embedding dimension (much smaller than ViT-Base's 768)
    num_layers = 6         # Number of transformer blocks (vs 12 in ViT-Base)
    num_heads = 3          # Number of attention heads (must divide embed_dim)
    mlp_size = 768         # MLP hidden dimension (4x embed_dim is standard)
    dropout = 0.1          # Dropout rate
    
    # Training Hyperparameters
    batch_size = 256       # Can go higher with 48GB GPU (try 512)
    learning_rate = 3e-4   # Standard for ViT with AdamW
    weight_decay = 0.01    # L2 regularization
    epochs = 30            # MNIST learns quickly
    
    # Optimizer settings
    optimizer = 'adamw'
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # Learning rate scheduler
    warmup_epochs = 5      # Warmup learning rate for first N epochs
    lr_schedule = 'cosine' # 'cosine' or 'step'
    
    # Data augmentation (light for MNIST)
    use_augmentation = True
    random_crop_padding = 4  # Padding for random crop
    
    # System
    device = 'cuda'        # 'cuda' or 'cpu'
    num_workers = 4        # DataLoader workers
    seed = 42              # Random seed for reproducibility
    
    # Logging and checkpoints
    print_freq = 50        # Print training stats every N batches
    save_freq = 5          # Save checkpoint every N epochs
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    
    def __repr__(self):
        """Pretty print configuration"""
        config_str = "ViT-Tiny Configuration for MNIST:\n"
        config_str += "=" * 50 + "\n"
        config_str += f"Architecture:\n"
        config_str += f"  Image size: {self.img_size}x{self.img_size}\n"
        config_str += f"  Patch size: {self.patch_size}x{self.patch_size}\n"
        config_str += f"  Num patches: {(self.img_size // self.patch_size) ** 2}\n"
        config_str += f"  Embedding dim: {self.embed_dim}\n"
        config_str += f"  Num layers: {self.num_layers}\n"
        config_str += f"  Num heads: {self.num_heads}\n"
        config_str += f"  MLP size: {self.mlp_size}\n"
        config_str += f"\nTraining:\n"
        config_str += f"  Batch size: {self.batch_size}\n"
        config_str += f"  Learning rate: {self.learning_rate}\n"
        config_str += f"  Epochs: {self.epochs}\n"
        config_str += f"  Device: {self.device}\n"
        config_str += "=" * 50
        return config_str


class ViTConfigAlternative:
    """Alternative ViT configuration with smaller patches for more detail"""
    
    # Model Architecture
    img_size = 28
    patch_size = 4         # 4x4 patches -> 7x7 = 49 patches (more patches, more detail)
    in_channels = 1
    num_classes = 10
    embed_dim = 192
    num_layers = 8         # Slightly deeper since we have more patches
    num_heads = 3
    mlp_size = 768
    dropout = 0.1
    
    # Training Hyperparameters (same as default)
    batch_size = 256
    learning_rate = 3e-4
    weight_decay = 0.01
    epochs = 30
    
    optimizer = 'adamw'
    betas = (0.9, 0.999)
    eps = 1e-8
    
    warmup_epochs = 5
    lr_schedule = 'cosine'
    
    use_augmentation = True
    random_crop_padding = 4
    
    device = 'cuda'
    num_workers = 4
    seed = 42
    
    print_freq = 50
    save_freq = 5
    checkpoint_dir = './checkpoints'
    log_dir = './logs'


# Default config to use
config = ViTConfig()

if __name__ == "__main__":
    print(config)
    print("\n\nAlternative configuration (more patches):")
    print(f"Patch size: {ViTConfigAlternative.patch_size}")
    print(f"Num patches: {(ViTConfigAlternative.img_size // ViTConfigAlternative.patch_size) ** 2}")