# Image parameters
IMAGE_SIZE = 224  # Standard ViT size
PATCH_SIZE = 16   # Each patch is 16x16
IN_CHANNELS = 3   # RGB images

# Model architecture parameters
EMBEDDING_DIM = 768      # Hidden size (D in the paper)
NUM_HEADS = 12           # Multi-head attention heads
NUM_LAYERS = 12          # Number of transformer blocks
MLP_SIZE = 3072          # MLP hidden dimension (usually 4*EMBEDDING_DIM)
DROPOUT = 0.1            # Dropout rate

# Training parameters
BATCH_SIZE = 64          # Adjust based on your GPU
LEARNING_RATE = 3e-4
NUM_EPOCHS = 100
WEIGHT_DECAY = 0.1

# Dataset
NUM_CLASSES = 1000       # Adjust to your dataset