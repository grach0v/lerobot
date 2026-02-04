#!/usr/bin/env python3
"""
Test the simplified Fusion implementation against original A2.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/home/denis-office/lerobot_a2/lerobot_grach0v/src/lerobot_policy_a2/src")

def test_fusion_simple():
    """Test FusionSimple produces correct output."""
    print("="*60)
    print("TESTING SIMPLIFIED FUSION")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create synthetic test data
    num_cam = 3
    H, W = 480, 640

    # Simple gradient images
    color = np.zeros((num_cam, H, W, 3), dtype=np.uint8)
    for c in range(num_cam):
        for i in range(H):
            for j in range(W):
                color[c, i, j] = [
                    int(255 * j / W),
                    int(100 + 50 * c),
                    int(255 * (1 - j / W))
                ]

    # Simple depth (all 1.0)
    depth = np.ones((num_cam, H, W), dtype=np.float32)

    # Identity poses (looking down Z axis)
    poses = np.zeros((num_cam, 3, 4), dtype=np.float32)
    for i in range(num_cam):
        poses[i, :3, :3] = np.eye(3)
        poses[i, 2, 3] = 1.0  # Camera at z=1

    # Standard intrinsics
    K = np.array([
        [616, 0, 320],
        [0, 616, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    Ks = np.stack([K] * num_cam)

    obs = {
        'color': color,
        'depth': depth,
        'pose': poses,
        'K': Ks,
    }

    # Test points
    points = np.array([
        [0.0, 0.0, 0.5],
        [0.1, 0.0, 0.5],
        [-0.1, 0.0, 0.5],
    ], dtype=np.float32)

    text = "grasp a red object"

    # Test our simplified Fusion
    from lerobot_policy_a2.feature_field_simple import FusionSimple

    print("\nTesting FusionSimple...")
    fusion = FusionSimple(num_cam=num_cam, feat_backbone="clip", device=device)
    fusion.update(obs, text, last_text_feat=False, visualize=False)

    pts_torch = torch.from_numpy(points).float().to(device)
    outputs = fusion.eval(pts_torch, return_names=["clip_feats", "clip_sims"])

    print(f"clip_feats shape: {outputs['clip_feats'].shape}")
    print(f"clip_feats stats: min={outputs['clip_feats'].min():.4f}, max={outputs['clip_feats'].max():.4f}")
    print(f"clip_sims shape: {outputs['clip_sims'].shape}")
    print(f"clip_sims stats: min={outputs['clip_sims'].min():.4f}, max={outputs['clip_sims'].max():.4f}")
    print(f"valid_mask: {outputs['valid_mask']}")

    # Check if features are non-zero
    feat_norm = outputs['clip_feats'].norm(dim=1)
    print(f"Feature norms: {feat_norm}")

    if feat_norm.min() < 0.01:
        print("WARNING: Some features are nearly zero!")
        return False

    print("\nFusionSimple test PASSED!")
    return True


def compare_with_original():
    """Compare FusionSimple with original A2 Fusion."""
    print("\n" + "="*60)
    print("COMPARING WITH ORIGINAL A2")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test data
    num_cam = 3
    H, W = 480, 640

    # Random images for more realistic test
    np.random.seed(42)
    color = np.random.randint(0, 255, (num_cam, H, W, 3), dtype=np.uint8)
    depth = np.random.uniform(0.5, 2.0, (num_cam, H, W)).astype(np.float32)

    # Camera poses - looking at origin from different angles
    poses = np.zeros((num_cam, 3, 4), dtype=np.float32)
    for i in range(num_cam):
        # Simple rotation around Y axis
        angle = i * 2 * np.pi / num_cam
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        t = np.array([0, 0, 1.0])
        poses[i, :3, :3] = R
        poses[i, :3, 3] = t

    K = np.array([
        [616, 0, 320],
        [0, 616, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    Ks = np.stack([K] * num_cam)

    obs = {
        'color': color,
        'depth': depth,
        'pose': poses,
        'K': Ks,
    }

    points = np.random.uniform(-0.2, 0.2, (100, 3)).astype(np.float32)
    text = "grasp a red object"

    # Our implementation
    print("\nRunning FusionSimple...")
    from lerobot_policy_a2.feature_field_simple import FusionSimple

    fusion_ours = FusionSimple(num_cam=num_cam, feat_backbone="clip", device=device)
    fusion_ours.update(obs, text, last_text_feat=False, visualize=False)

    pts_torch = torch.from_numpy(points).float().to(device)
    outputs_ours = fusion_ours.eval(pts_torch, return_names=["clip_feats", "clip_sims"])

    print(f"Our clip_feats: shape={outputs_ours['clip_feats'].shape}, norm={outputs_ours['clip_feats'].norm(dim=1).mean():.4f}")
    print(f"Our clip_sims: shape={outputs_ours['clip_sims'].shape}, mean={outputs_ours['clip_sims'][:, 0].mean():.4f}")

    # Try original A2
    try:
        sys.path.insert(0, "/home/denis-office/lerobot_a2/A2_new")
        from models.feature_fusion import Fusion

        print("\nRunning Original A2 Fusion...")
        fusion_orig = Fusion(num_cam=num_cam, feat_backbone="clip", device=device)
        fusion_orig.update(obs, text, last_text_feat=False, visualize=False)

        outputs_orig = fusion_orig.eval(pts_torch, return_names=["clip_feats", "clip_sims"])

        print(f"Original clip_feats: shape={outputs_orig['clip_feats'].shape}, norm={outputs_orig['clip_feats'].norm(dim=1).mean():.4f}")
        print(f"Original clip_sims: shape={outputs_orig['clip_sims'].shape}, mean={outputs_orig['clip_sims'][:, 0].mean():.4f}")

        # Compare
        feat_diff = torch.abs(outputs_ours['clip_feats'] - outputs_orig['clip_feats']).mean().item()
        sim_diff = torch.abs(outputs_ours['clip_sims'] - outputs_orig['clip_sims']).mean().item()

        print(f"\nFeature mean abs diff: {feat_diff:.6f}")
        print(f"Similarity mean abs diff: {sim_diff:.6f}")

        if feat_diff < 0.01 and sim_diff < 0.01:
            print("\nCOMPARISON PASSED - implementations match!")
            return True
        else:
            print("\nCOMPARISON FAILED - implementations differ!")
            return False

    except Exception as e:
        print(f"\nCould not run original A2: {e}")
        print("Skipping comparison with original")
        return True


if __name__ == "__main__":
    test_fusion_simple()
    compare_with_original()
