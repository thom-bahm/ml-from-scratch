import torch
from torch import nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # TODO: Create patch projection using Conv2d
        # Hint: Use kernel_size=patch_size, stride=patch_size to extract patches
        # This effectively divides the image into non-overlapping patches
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # TODO: Create learnable CLS token
        # Hint: Shape should be (1, 1, embed_dim) so it can be prepended to batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # TODO: Create positional embeddings
        # Hint: Need one for each patch + one for CLS token
        # Shape: (1, n_patches + 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
    
    def forward(self, x):
        # TODO: Extract patches
        # Input: (B, C, H, W)
        B = x.shape[0]
        
        # TODO: Project patches
        # After conv: (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        
        # TODO: Flatten spatial dimensions
        # Hint: Use .flatten(2) to flatten H and W, then transpose
        # Target: (B, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # TODO: Expand CLS token for batch
        # Hint: Use .expand() to repeat for batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # TODO: Prepend CLS token to sequence
        # Hint: Use torch.cat on dimension 1
        x = torch.cat([cls_tokens, x], dim=1)
        
        # TODO: Add positional embeddings
        x = x + self.pos_embed
        
        return x  # Output: (B, n_patches + 1, embed_dim)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        # TODO: Create Q, K, V projection layers
        # TODO: Create output projection layer
        # TODO: Initialize dropout
        pass
    
    def forward(self, x):
        # Input: (B, num_patches+1, embed_dim)
        # TODO: Project to Q, K, V
        # TODO: Reshape for multi-head: (B, num_heads, num_patches+1, head_dim)
        # TODO: Compute attention scores
        # TODO: Apply softmax and dropout
        # TODO: Apply attention to values
        # TODO: Concatenate heads and project
        pass

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_size, dropout=0.1):
        # TODO: First linear layer
        # TODO: GELU activation
        # TODO: Dropout
        # TODO: Second linear layer
        # TODO: Dropout
        pass
    
    def forward(self, x):
        # TODO: Pass through layers
        pass

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_size, dropout=0.1):
        # TODO: Create layer norms
        # TODO: Create attention module
        # TODO: Create MLP module
        pass
    
    def forward(self, x):
        # TODO: Attention block with residual
        # TODO: MLP block with residual
        pass


class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_size=3072,
                 dropout=0.1):
        # TODO: Patch embedding layer
        # TODO: Stack of transformer blocks (use nn.ModuleList or nn.Sequential)
        # TODO: Final layer norm
        # TODO: Classification head (linear layer)
        pass
    
    def forward(self, x):
        # Input: (B, 3, 224, 224)
        # TODO: Get patch embeddings
        # TODO: Pass through transformer blocks
        # TODO: Apply final layer norm
        # TODO: Extract CLS token (first token)
        # TODO: Pass through classification head
        # Output: (B, num_classes)
        pass