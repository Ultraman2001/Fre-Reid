"""
Quick test: verify FreqAwareFusionBlock forward/backward pass
and compare params with the old Block approach.
"""
import sys
sys.path.insert(0, '/home/user/PL/mamba-reid/TransReID-main')

import torch
import torch.nn as nn

# Test 1: Import and instantiate FreqAwareFusionBlock
print("=" * 60)
print("Test 1: Import FreqAwareFusionBlock")
print("=" * 60)
try:
    from model.backbones.mambavision.mamba_vision_reid import (
        FreqAwareFusionBlock, RepDW, Block, MambaVisionMixer
    )
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Forward pass
print("\n" + "=" * 60)
print("Test 2: FreqAwareFusionBlock forward pass")
print("=" * 60)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

dim = 1024  # 2C (out_dim=512, Gated Concat)
H, W = 16, 8
B = 2

for ratio_name, mamba_ratio in [("SFM_12", 0.25), ("SFM_23", 0.50), ("SFM_34", 0.75)]:
    print(f"\n--- {ratio_name} (mamba_ratio={mamba_ratio}) ---")
    block = FreqAwareFusionBlock(
        dim=dim, mamba_ratio=mamba_ratio, pool_factor=2, drop_path=0.1
    ).to(device)
    
    x = torch.randn(B, H * W, dim).to(device)
    out = block(x, H=H, W=W)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch! {out.shape} != {x.shape}"
    print(f"  ✅ Shape correct")
    
    # Count params
    params = sum(p.numel() for p in block.parameters())
    print(f"  Params: {params:,} ({params/1e6:.2f}M)")

# Test 3: Compare with old Block
print("\n" + "=" * 60)
print("Test 3: Parameter comparison (old Block vs new FreqAwareFusionBlock)")
print("=" * 60)

old_block = Block(
    dim=dim, counter=0, transformer_blocks=[], num_heads=8,
    mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
    drop_path=0.1, layer_scale=1e-5, use_sasf=False,
).to(device)
old_params = sum(p.numel() for p in old_block.parameters())
print(f"Old Block params:              {old_params:,} ({old_params/1e6:.2f}M)")

for ratio_name, mamba_ratio in [("SFM_12", 0.25), ("SFM_23", 0.50), ("SFM_34", 0.75)]:
    new_block = FreqAwareFusionBlock(
        dim=dim, mamba_ratio=mamba_ratio, pool_factor=2, drop_path=0.1
    ).to(device)
    new_params = sum(p.numel() for p in new_block.parameters())
    reduction = (1 - new_params / old_params) * 100
    print(f"New FreqAware ({ratio_name}): {new_params:,} ({new_params/1e6:.2f}M) | "
          f"{'↓' if reduction > 0 else '↑'}{abs(reduction):.1f}%")

# Test 4: Backward pass (gradient flow)
print("\n" + "=" * 60)
print("Test 4: Backward pass (gradient flow)")
print("=" * 60)
block = FreqAwareFusionBlock(
    dim=dim, mamba_ratio=0.5, pool_factor=2, drop_path=0.1
).to(device)
x = torch.randn(B, H * W, dim, device=device, requires_grad=True)
out = block(x, H=H, W=W)
loss = out.sum()
loss.backward()
print(f"  x.grad norm: {x.grad.norm().item():.4f}")
print(f"  ✅ Backward pass successful, gradients flowing")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
