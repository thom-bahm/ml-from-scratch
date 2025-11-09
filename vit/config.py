"""
Configuration file for Vision Transformer training.
Supports multiple datasets: MNIST, CIFAR-10, CIFAR-100, ImageNet.
"""

class ViTConfig:
    """Base ViT configuration class"""
    
    def __init__(self):
        """Initialize with dataset name for checkpoint organization"""
        self.name = "vit_base"  # Config identifier
        self.dataset = "mnist"   # Dataset name for organization
    
    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        """Pretty print configuration"""
        config_str = f"{self.name.upper()} Configuration:\n"
        config_str += "=" * 70 + "\n"
        config_str += f"Dataset: {self.dataset.upper()}\n"
        config_str += "=" * 70 + "\n"
        config_str += f"Architecture:\n"
        config_str += f"  Image size: {self.img_size}x{self.img_size}\n"
        config_str += f"  Patch size: {self.patch_size}x{self.patch_size}\n"
        config_str += f"  Num patches: {(self.img_size // self.patch_size) ** 2}\n"
        config_str += f"  Channels: {self.in_channels}\n"
        config_str += f"  Num classes: {self.num_classes}\n"
        config_str += f"  Embedding dim: {self.embed_dim}\n"
        config_str += f"  Num layers: {self.num_layers}\n"
        config_str += f"  Num heads: {self.num_heads}\n"
        config_str += f"  MLP size: {self.mlp_size}\n"
        config_str += f"  Dropout: {self.dropout}\n"
        config_str += f"\nTraining:\n"
        config_str += f"  Batch size: {self.batch_size}\n"
        config_str += f"  Learning rate: {self.learning_rate}\n"
        config_str += f"  Weight decay: {self.weight_decay}\n"
        config_str += f"  Epochs: {self.epochs}\n"
        config_str += f"  Warmup epochs: {self.warmup_epochs}\n"
        config_str += f"  LR schedule: {self.lr_schedule}\n"
        config_str += f"  Augmentation: {self.use_augmentation}\n"
        config_str += f"\nSystem:\n"
        config_str += f"  Device: {self.device}\n"
        config_str += f"  Num workers: {self.num_workers}\n"
        config_str += f"  Seed: {self.seed}\n"
        config_str += "=" * 70
        return config_str


class ViTMNIST(ViTConfig):
    """ViT-Tiny configuration for MNIST (28x28 grayscale images)"""
    
    def __init__(self):
        super().__init__()
        self.name = "vit_tiny_mnist"
        self.dataset = "mnist"
        
        # Model Architecture
        self.img_size = 28          # MNIST native size
        self.patch_size = 7         # 7x7 patches -> 4x4 = 16 patches total
        self.in_channels = 1        # Grayscale images
        self.num_classes = 10       # Digits 0-9
        self.embed_dim = 192        # Small embedding for tiny dataset
        self.num_layers = 6         # Shallow network
        self.num_heads = 3          # Few attention heads
        self.mlp_size = 768         # 4x embed_dim
        self.dropout = 0.1
        
        # Training Hyperparameters
        self.batch_size = 256
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.epochs = 30
        
        # Optimizer settings
        self.optimizer = 'adamw'
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        
        # Learning rate scheduler
        self.warmup_epochs = 5
        self.lr_schedule = 'cosine'
        
        # Data augmentation (light for MNIST)
        self.use_augmentation = True
        self.random_crop_padding = 4
        
        # System
        self.device = 'cuda'
        self.num_workers = 4
        self.seed = 42
        
        # Logging and checkpoints
        self.print_freq = 50
        self.save_freq = 5
        self.checkpoint_dir = './checkpoints'
        self.log_dir = './logs'
    
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


class ViTCIFAR10(ViTConfig):
    """ViT-Small configuration for CIFAR-10 (32x32 RGB images)"""
    
    def __init__(self):
        super().__init__()
        self.name = "vit_small_cifar10"
        self.dataset = "cifar10"
        
        # Model Architecture
        self.img_size = 32          # CIFAR-10 native size
        self.patch_size = 4         # 4x4 patches -> 8x8 = 64 patches
        self.in_channels = 3        # RGB images
        self.num_classes = 10       # 10 classes
        self.embed_dim = 384        # Medium size for CIFAR
        self.num_layers = 8         # Moderate depth
        self.num_heads = 6          # More heads for RGB
        self.mlp_size = 1536        # 4x embed_dim
        self.dropout = 0.1
        
        # Training Hyperparameters
        self.batch_size = 128       # CIFAR is more complex
        self.learning_rate = 3e-4
        self.weight_decay = 0.05    # More regularization
        self.epochs = 200           # CIFAR needs more training
        
        # Optimizer settings
        self.optimizer = 'adamw'
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        
        # Learning rate scheduler
        self.warmup_epochs = 10
        self.lr_schedule = 'cosine'
        
        # Data augmentation (stronger for CIFAR)
        self.use_augmentation = True
        self.random_crop_padding = 4
        self.random_horizontal_flip = True
        self.cutout = True
        self.cutout_length = 16
        
        # System
        self.device = 'cuda'
        self.num_workers = 8
        self.seed = 42
        
        # Logging and checkpoints
        self.print_freq = 100
        self.save_freq = 10
        self.checkpoint_dir = './checkpoints'
        self.log_dir = './logs'


class ViTCIFAR100(ViTConfig):
    """ViT-Small configuration for CIFAR-100 (32x32 RGB, 100 classes)"""
    
    def __init__(self):
        super().__init__()
        self.name = "vit_small_cifar100"
        self.dataset = "cifar100"
        
        # Model Architecture
        self.img_size = 32
        self.patch_size = 4
        self.in_channels = 3
        self.num_classes = 100      # 100 classes
        self.embed_dim = 384
        self.num_layers = 10        # Deeper for more classes
        self.num_heads = 6
        self.mlp_size = 1536
        self.dropout = 0.1
        
        # Training Hyperparameters
        self.batch_size = 128
        self.learning_rate = 3e-4
        self.weight_decay = 0.05
        self.epochs = 300           # More epochs for 100 classes
        
        # Optimizer settings
        self.optimizer = 'adamw'
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        
        # Learning rate scheduler
        self.warmup_epochs = 15
        self.lr_schedule = 'cosine'
        
        # Data augmentation
        self.use_augmentation = True
        self.random_crop_padding = 4
        self.random_horizontal_flip = True
        self.cutout = True
        self.cutout_length = 16
        self.mixup = True
        self.mixup_alpha = 0.2
        
        # System
        self.device = 'cuda'
        self.num_workers = 8
        self.seed = 42
        
        # Logging and checkpoints
        self.print_freq = 100
        self.save_freq = 20
        self.checkpoint_dir = './checkpoints'
        self.log_dir = './logs'


class ViTImageNet(ViTConfig):
    """ViT-Base configuration for ImageNet (224x224 RGB, 1000 classes)"""
    
    def __init__(self):
        super().__init__()
        self.name = "vit_base_imagenet"
        self.dataset = "imagenet"
        
        # Model Architecture - Standard ViT-Base/16
        self.img_size = 224         # ImageNet standard size
        self.patch_size = 16        # 16x16 patches -> 14x14 = 196 patches
        self.in_channels = 3
        self.num_classes = 1000     # ImageNet classes
        self.embed_dim = 768        # Full ViT-Base size
        self.num_layers = 12        # Standard depth
        self.num_heads = 12         # Standard heads
        self.mlp_size = 3072        # 4x embed_dim
        self.dropout = 0.1
        
        # Training Hyperparameters (following ViT paper)
        self.batch_size = 512       # Large batch with 48GB GPU
        self.learning_rate = 3e-3   # Higher LR for large batch
        self.weight_decay = 0.3     # Strong regularization
        self.epochs = 300           # ImageNet standard
        
        # Optimizer settings
        self.optimizer = 'adamw'
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        
        # Learning rate scheduler
        self.warmup_epochs = 30     # Longer warmup for ImageNet
        self.lr_schedule = 'cosine'
        
        # Data augmentation (ImageNet standard)
        self.use_augmentation = True
        self.random_crop_padding = 32
        self.random_horizontal_flip = True
        self.color_jitter = True
        self.auto_augment = True
        self.random_erasing = True
        self.mixup = True
        self.mixup_alpha = 0.8
        self.cutmix = True
        self.cutmix_alpha = 1.0
        
        # System
        self.device = 'cuda'
        self.num_workers = 16       # More workers for ImageNet
        self.seed = 42
        
        # Logging and checkpoints
        self.print_freq = 200
        self.save_freq = 25
        self.checkpoint_dir = './checkpoints'
        self.log_dir = './logs'


# Configuration registry - maps names to config classes
CONFIG_REGISTRY = {
    'mnist': ViTMNIST,
    'cifar10': ViTCIFAR10,
    'cifar100': ViTCIFAR100,
    'imagenet': ViTImageNet,
}


def get_config(config_name='mnist'):
    """
    Get configuration by name.
    
    Args:
        config_name: Name of configuration ('mnist', 'cifar10', 'cifar100', 'imagenet')
    
    Returns:
        Configuration instance
    """
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIG_REGISTRY.keys())}")
    
    return CONFIG_REGISTRY[config_name]()


# Default config for backward compatibility
config = ViTMNIST()


if __name__ == "__main__":
    print("Available configurations:\n")
    
    for name in CONFIG_REGISTRY.keys():
        cfg = get_config(name)
        print(cfg)
        print("\n" + "="*70 + "\n")