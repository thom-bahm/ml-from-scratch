import torch
from torch import nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # PATCH EMBEDDING PROJECTION - Two mathematically equivalent approaches:
        
        # APPROACH 1: Conv2d (RECOMMENDED - What we use)
        # Uses a convolutional layer with kernel_size=patch_size and stride=patch_size
        # This simultaneously extracts patches AND projects them to embed_dim
        # Advantages:
        #   - More efficient (single optimized operation)
        #   - Leverages highly optimized conv implementations
        #   - Cleaner code (one line instead of extract + project)
        #   - Standard in all major ViT implementations
        # Internally has weights of shape: (embed_dim, in_channels, patch_size, patch_size)
        # This is the 'E' matrix from the paper, just in 4D form
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # APPROACH 2: Explicit Linear Projection (As described in the paper)
        # This is how the paper describes it conceptually:
        # 1. Flatten each patch: (patch_size, patch_size, in_channels) -> (patch_size² * in_channels,)
        # 2. Linear projection: (patch_size² * in_channels) -> (embed_dim)
        # This creates the explicit 'E' matrix: ℝ^(P²·C × D)
        #
        # # Uncomment to use explicit approach:
        # patch_dim = in_channels * patch_size * patch_size  # P² * C = 3 * 16 * 16 = 768
        # self.proj = nn.Linear(patch_dim, embed_dim)
        #
        # Why we don't use this:
        #   - Requires manual patch extraction (unfold or reshaping)
        #   - Slower than Conv2d for this specific operation
        #   - Same number of parameters, same mathematical operation
        #   - Just a different representation of the same transformation
        
        # TODO: Create learnable CLS token
        # Hint: Shape should be (1, 1, embed_dim) so it can be prepended to batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # TODO: Create positional embeddings
        # Hint: Need one for each patch + one for CLS token
        # Shape: (1, n_patches + 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
    
    def forward(self, x):
        # Input: (B, C, H, W) - Batch of images
        # Example: (B, 3, 224, 224)
        B = x.shape[0]
        
        # APPROACH 1: Using Conv2d (Current implementation)
        # The conv layer extracts patches and projects them in one step
        x = self.proj(x)  # (B, embed_dim, H/P, W/P) = (B, 768, 14, 14)
        
        # Flatten spatial dimensions and transpose to get sequence format
        # flatten(2) flattens dimensions 2 and 3 (H/P and W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim) = (B, 196, 768)
        
        # APPROACH 2: Explicit patch extraction and linear projection (Commented out)
        # This is conceptually clearer but less efficient:
        #
        # # Step 1: Extract patches using unfold
        # # unfold extracts sliding windows from the image
        # x = x.unfold(2, self.patch_size, self.patch_size)  # Extract along height
        # x = x.unfold(3, self.patch_size, self.patch_size)  # Extract along width
        # # Result: (B, C, H/P, W/P, P, P)
        #
        # # Step 2: Reshape to flatten each patch
        # x = x.contiguous().view(B, self.in_channels, self.n_patches, -1)
        # x = x.permute(0, 2, 1, 3)  # (B, n_patches, C, P*P)
        # x = x.reshape(B, self.n_patches, -1)  # (B, n_patches, C*P*P) = (B, 196, 768)
        #
        # # Step 3: Apply linear projection (the 'E' matrix)
        # x = self.proj(x)  # (B, n_patches, embed_dim) = (B, 196, 768)
        #
        # Both approaches produce the same output shape and are mathematically equivalent
        
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
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # TODO: Create Q, K, V projection layers
        # Hint: All project from embed_dim to embed_dim
        self.qkv = nn.Linear(embed_dim, embed_dim * 3) # using a single layer to produce Q,K,V for efficiency
        
        # TODO: Create output projection layer
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # TODO: Initialize dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        # Input: (B, num_patches+1, embed_dim)
        B, N, C = x.shape
        
        # TODO: Project to Q, K, V
        # Hint: Use the qkv layer to get all three at once
        # Output shape: (B, N, 3 * embed_dim)
        qkv = self.qkv(x)
        
        # TODO: Reshape for multi-head: (B, num_heads, num_patches+1, head_dim)
        # Hint: Reshape to (B, N, 3, num_heads, head_dim) then permute
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        
        # Split into Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # TODO: Compute attention scores
        # Hint: Q @ K^T, then scale by sqrt(head_dim)
        # Formula: attention = softmax(Q @ K^T / sqrt(d_k)) @ V
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        # TODO: Apply softmax and dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # TODO: Apply attention to values
        # Hint: attn @ V
        x = attn @ v  # (B, num_heads, N, head_dim)
        
        # TODO: Concatenate heads and project
        # Hint: Transpose and reshape to (B, N, embed_dim)
        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_size, dropout=0.1):
        super().__init__()
        # TODO: First linear layer
        self.fc1 = nn.Linear(embed_dim, mlp_size)
        # TODO: GELU activation
        self.gelu = nn.GELU()
        # TODO: Dropout
        self.dropout1 = nn.Dropout(dropout)
        # TODO: Second linear layer
        self.fc2 = nn.Linear(mlp_size, embed_dim)
        # TODO: Dropout
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # TODO: Pass through layers
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_size, dropout=0.1):
        # TODO: Create layer norms
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # TODO: Create attention module
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        # TODO: Create MLP module
        self.mlp = MLP(embed_dim, mlp_size, dropout)
    
    def forward(self, x):
        # TODO: Attention block with residual
        x = x + self.attn(self.norm1(x))
        # TODO: MLP block with residual
        x = x + self.mlp(self.norm2(x))
        return x


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
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # TODO: Stack of transformer blocks (use nn.ModuleList or nn.Sequential)
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_size, dropout)
            for _ in range(num_layers)
        ])
        # TODO: Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        # TODO: Classification head (linear layer)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # Input: (B, 3, 224, 224)
        # TODO: Get patch embeddings
        x = self.patch_embed(x)
        # TODO: Pass through transformer blocks
        x = self.transformer_blocks(x)
        # TODO: Apply final layer norm
        x = self.norm(x)
        # TODO: Extract CLS token (first token)
        cls_token = x[:, 0]
        # TODO: Pass through classification head
        x = self.head(cls_token)
        # Output: (B, num_classes)
        return x
    

    def test_vit_implementation():
        """
        Comprehensive test suite to verify the Vision Transformer implementation.
        Tests each component individually and then the full model.
        """
        print("=" * 70)
        print("Testing Vision Transformer Implementation")
        print("=" * 70)
        
        # Configuration for testing (smaller than standard ViT-Base for faster testing)
        batch_size = 4
        img_size = 224
        patch_size = 16
        in_channels = 3
        num_classes = 10  # Using 10 classes for testing (e.g., CIFAR-10)
        embed_dim = 192
        num_layers = 4
        num_heads = 3
        mlp_size = 768
        
        # Calculate expected dimensions
        n_patches = (img_size // patch_size) ** 2  # 14 * 14 = 196
        seq_length = n_patches + 1  # +1 for CLS token = 197
        
        print(f"\nTest Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Number of patches: {n_patches}")
        print(f"  Sequence length (with CLS): {seq_length}")
        print(f"  Embedding dimension: {embed_dim}")
        print(f"  Number of transformer layers: {num_layers}")
        print(f"  Number of attention heads: {num_heads}")
        print(f"  MLP hidden size: {mlp_size}")
        
        # Create dummy input
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        print(f"\n{'='*70}")
        print(f"Input shape: {x.shape}")
        
        # =========================================================================
        # Test 1: PatchEmbedding
        # =========================================================================
        print(f"\n{'='*70}")
        print("Test 1: PatchEmbedding Layer")
        print(f"{'='*70}")
        
        patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        patch_output = patch_embed(x)
        
        # Assertions for PatchEmbedding
        expected_shape = (batch_size, seq_length, embed_dim)
        assert patch_output.shape == expected_shape, \
            f"PatchEmbedding output shape mismatch! Expected {expected_shape}, got {patch_output.shape}"
        print(f"✓ Output shape correct: {patch_output.shape}")
        
        # Check that CLS token is prepended (first token should be different from patches)
        assert patch_embed.cls_token.shape == (1, 1, embed_dim), \
            "CLS token shape incorrect"
        print(f"✓ CLS token shape correct: {patch_embed.cls_token.shape}")
        
        # Check positional embeddings
        assert patch_embed.pos_embed.shape == (1, seq_length, embed_dim), \
            "Positional embedding shape incorrect"
        print(f"✓ Positional embedding shape correct: {patch_embed.pos_embed.shape}")
        
        # =========================================================================
        # Test 2: MultiHeadSelfAttention
        # =========================================================================
        print(f"\n{'='*70}")
        print("Test 2: MultiHeadSelfAttention")
        print(f"{'='*70}")
        
        mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout=0.0)
        attn_output = mhsa(patch_output)
        
        # Assertions for MHSA
        assert attn_output.shape == patch_output.shape, \
            f"MHSA should preserve shape! Expected {patch_output.shape}, got {attn_output.shape}"
        print(f"✓ Output shape preserved: {attn_output.shape}")
        
        # Check that head_dim is correct
        expected_head_dim = embed_dim // num_heads
        assert mhsa.head_dim == expected_head_dim, \
            f"Head dimension incorrect! Expected {expected_head_dim}, got {mhsa.head_dim}"
        print(f"✓ Head dimension correct: {mhsa.head_dim}")
        
        # Check that output is different from input (attention actually does something)
        assert not torch.allclose(attn_output, patch_output), \
            "MHSA output should be different from input!"
        print(f"✓ Attention transforms the input (not identity)")
        
        # =========================================================================
        # Test 3: MLP
        # =========================================================================
        print(f"\n{'='*70}")
        print("Test 3: MLP (Feed-Forward Network)")
        print(f"{'='*70}")
        
        mlp = MLP(embed_dim, mlp_size, dropout=0.0)
        mlp_output = mlp(patch_output)
        
        # Assertions for MLP
        assert mlp_output.shape == patch_output.shape, \
            f"MLP should preserve shape! Expected {patch_output.shape}, got {mlp_output.shape}"
        print(f"✓ Output shape preserved: {mlp_output.shape}")
        
        # Check that MLP expands then contracts
        assert mlp.fc1.out_features == mlp_size, \
            f"First FC layer should expand to {mlp_size}"
        assert mlp.fc2.out_features == embed_dim, \
            f"Second FC layer should contract back to {embed_dim}"
        print(f"✓ MLP correctly expands ({embed_dim} → {mlp_size}) and contracts ({mlp_size} → {embed_dim})")
        
        # =========================================================================
        # Test 4: TransformerBlock
        # =========================================================================
        print(f"\n{'='*70}")
        print("Test 4: TransformerBlock")
        print(f"{'='*70}")
        
        transformer_block = TransformerBlock(embed_dim, num_heads, mlp_size, dropout=0.0)
        block_output = transformer_block(patch_output)
        
        # Assertions for TransformerBlock
        assert block_output.shape == patch_output.shape, \
            f"TransformerBlock should preserve shape! Expected {patch_output.shape}, got {block_output.shape}"
        print(f"✓ Output shape preserved: {block_output.shape}")
        
        # Check residual connections work (output should be significantly different due to residuals)
        assert not torch.allclose(block_output, patch_output, rtol=1e-3), \
            "TransformerBlock should transform the input!"
        print(f"✓ Residual connections working (output differs from input)")
        
        # =========================================================================
        # Test 5: Full VisionTransformer
        # =========================================================================
        print(f"\n{'='*70}")
        print("Test 5: Complete VisionTransformer Model")
        print(f"{'='*70}")
        
        vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_size=mlp_size,
            dropout=0.0
        )
        
        # Set to eval mode to disable dropout for consistent testing
        vit.eval()
        
        with torch.no_grad():
            output = vit(x)
        
        # Assertions for full ViT
        expected_output_shape = (batch_size, num_classes)
        assert output.shape == expected_output_shape, \
            f"ViT output shape mismatch! Expected {expected_output_shape}, got {output.shape}"
        print(f"✓ Output shape correct: {output.shape}")
        
        # Check that outputs are reasonable (not NaN or Inf)
        assert not torch.isnan(output).any(), "Output contains NaN values!"
        assert not torch.isinf(output).any(), "Output contains Inf values!"
        print(f"✓ Output values are valid (no NaN or Inf)")
        
        # Check that different inputs produce different outputs
        x2 = torch.randn(batch_size, in_channels, img_size, img_size)
        with torch.no_grad():
            output2 = vit(x2)
        assert not torch.allclose(output, output2, rtol=1e-3), \
            "Different inputs should produce different outputs!"
        print(f"✓ Model produces different outputs for different inputs")
        
        # =========================================================================
        # Test 6: Parameter Count
        # =========================================================================
        print(f"\n{'='*70}")
        print("Test 6: Model Parameters")
        print(f"{'='*70}")
        
        total_params = sum(p.numel() for p in vit.parameters())
        trainable_params = sum(p.numel() for p in vit.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
        
        # For reference, ViT-Base has ~86M parameters
        # Our smaller test model should have fewer
        assert total_params > 0, "Model should have parameters!"
        assert trainable_params == total_params, "All parameters should be trainable!"
        
        # =========================================================================
        # Test 7: Gradient Flow
        # =========================================================================
        print(f"\n{'='*70}")
        print("Test 7: Gradient Flow (Backpropagation)")
        print(f"{'='*70}")
        
        vit.train()
        x_grad = torch.randn(batch_size, in_channels, img_size, img_size, requires_grad=True)
        output_grad = vit(x_grad)
        
        # Create dummy target and loss
        target = torch.randint(0, num_classes, (batch_size,))
        loss = nn.CrossEntropyLoss()(output_grad, target)
        loss.backward()
        
        # Check that gradients exist
        assert x_grad.grad is not None, "Gradients should flow back to input!"
        print(f"✓ Gradients flow back to input")
        
        # Check that model parameters have gradients
        has_grad = [p.grad is not None for p in vit.parameters() if p.requires_grad]
        assert all(has_grad), "All parameters should have gradients after backward!"
        print(f"✓ All parameters received gradients")
        
        # Check gradient magnitudes are reasonable
        grad_norms = [p.grad.norm().item() for p in vit.parameters() if p.grad is not None]
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        assert avg_grad_norm > 0, "Gradients should be non-zero!"
        assert avg_grad_norm < 1000, "Gradients seem unusually large (potential instability)!"
        print(f"✓ Average gradient norm: {avg_grad_norm:.4f} (reasonable)")
        
        # =========================================================================
        # Summary
        # =========================================================================
        print(f"\n{'='*70}")
        print("✓ ALL TESTS PASSED!")
        print(f"{'='*70}")
        print("\nYour Vision Transformer implementation is correct and working!")
        print("\nNext steps:")
        print("  1. Train on a dataset (CIFAR-10, ImageNet, etc.)")
        print("  2. Experiment with different hyperparameters")
        print("  3. Try different model sizes (ViT-Tiny, ViT-Small, ViT-Base)")
        print("  4. Add advanced features (dropout, stochastic depth, etc.)")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    # Run the comprehensive test suite
    VisionTransformer.test_vit_implementation()