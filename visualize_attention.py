"""
Diagnostic Script for Dual-Branch Analysis
Visualizes:
1. Main Branch (Stage 3) output heatmap
2. Fine Branch output heatmap
3. Fused output heatmap
4. Fusion Gate weight distribution

Usage:
    python diagnose_branches.py --config_file configs/Market/your_config.yml \
                                --weights path/to/checkpoint.pth \
                                --image path/to/test_image.jpg
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from model import make_model


def load_model(cfg, weights_path, device='cuda'):
    """Load trained model with dual-branch enabled"""
    model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    
    state_dict = torch.load(weights_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, size=(256, 128)):
    """Preprocess image for model input"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size[1], size[0]))  # (W, H)
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img / 255.0 - mean) / std
    
    # To tensor
    tensor = torch.from_numpy(img_norm).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    return tensor, img


def extract_branch_features(model, img_tensor, device='cuda'):
    """Return the final backbone feature map for a single image."""
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model.base(img_tensor)

    if isinstance(output, dict):
        feature_map = output.get('backbone_map')
        if feature_map is None:
            fused_maps = output.get('fused_maps')
            feature_map = fused_maps[-1] if fused_maps else None
    elif isinstance(output, (list, tuple)):
        feature_map = output[-1]
    else:
        feature_map = output

    if feature_map is None:
        raise RuntimeError('Unable to extract backbone feature map from model output.')

    return {'backbone': feature_map}


def feature_to_heatmap(feature_map, original_size=(256, 128)):
    """Convert feature map to heatmap"""
    # feature_map: (B, C, H, W)
    # Take mean across channels
    heatmap = feature_map[0].mean(dim=0).cpu().numpy()  # (H, W)
    
    # Normalize to 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Resize to original image size
    heatmap = cv2.resize(heatmap, (original_size[1], original_size[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    return heatmap, heatmap_colored


def visualize_branches(image_path, model, device='cuda', save_dir='diagnose_output'):
    """Generate and save heatmap visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Preprocess
    img_tensor, original_img = preprocess_image(image_path)
    
    # Extract features
    features = extract_branch_features(model, img_tensor, device)
    
    backbone_feat = features.get('backbone')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    if backbone_feat is not None:
        heatmap, heatmap_colored = feature_to_heatmap(backbone_feat)
        overlay = (0.4 * original_img + 0.6 * heatmap_colored).astype(np.uint8)
        axes[1].imshow(overlay)
        axes[1].set_title('Backbone Feature Heatmap')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Backbone output unavailable', ha='center', va='center')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save
    image_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(save_dir, f'{image_name}_diagnosis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    
    plt.close()
    
    return features


def diagnose_gate_collapse(model):
    """Check if fusion gate has collapsed"""
    backbone = model.base
    
    if not hasattr(backbone, 'fusion') or not hasattr(backbone.fusion, 'gate'):
        print("No fusion gate found.")
        return
    
    gate = backbone.fusion.gate
    
    print("\n=== Fusion Gate Analysis ===")
    for name, param in gate.named_parameters():
        print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
        if param.data.std() < 0.01:
            print(f"  ⚠️ WARNING: {name} has very low variance - possible collapse!")


def test_branch_performance(cfg, model, device='cuda'):
    """Test mAP using only the backbone feature map."""
    from datasets import make_dataloader
    from utils.metrics import R1_mAP_eval

    _, _, val_loader, num_query, _, _, _ = make_dataloader(cfg)

    def _select_backbone_map(output):
        if isinstance(output, dict):
            feature_map = output.get('backbone_map')
            if feature_map is None:
                fused_maps = output.get('fused_maps')
                feature_map = fused_maps[-1] if fused_maps else None
        elif isinstance(output, (list, tuple)):
            feature_map = output[-1]
        else:
            feature_map = output
        if feature_map is None:
            raise RuntimeError('Backbone feature map not found while evaluating performance.')
        return feature_map

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm='yes')
    evaluator.reset()

    model.eval()
    for img, pid, camid, camids, target_view, _ in val_loader:
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)

            feature_map = _select_backbone_map(model.base(img, cam_label=camids))

            feat = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
            feat = model.bottleneck(feat) if hasattr(model, 'bottleneck') else feat

            evaluator.update((feat, pid, camid))

    cmc, mAP, *_ = evaluator.compute()
    print(f"  backbone_only: mAP={mAP:.1%}, Rank-1={cmc[0]:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--image', type=str, default=None, help='Path to test image')
    parser.add_argument('--output_dir', type=str, default='diagnose_output')
    parser.add_argument('--test_branches', action='store_true', help='Evaluate mAP using the backbone output')
    args = parser.parse_args()
    
    # Load config
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"Loading model from: {args.weights}")
    model = load_model(cfg, args.weights, device)
    
    # Gate analysis
    diagnose_gate_collapse(model)
    
    # Visualization
    if args.image:
        print(f"\nGenerating heatmaps for: {args.image}")
        visualize_branches(args.image, model, device, args.output_dir)
    
    # Branch performance test
    if args.test_branches:
        test_branch_performance(cfg, model, device)
    
    print("\nDiagnosis complete!")
