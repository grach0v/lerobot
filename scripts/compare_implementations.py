#!/usr/bin/env python3
"""
Compare feature extraction between original A2 and our implementation.

Run this script to identify exactly where the implementations differ.
"""

import sys
import os
import numpy as np
import torch

# Add paths
sys.path.insert(0, "/home/denis-office/lerobot_a2/Action-Prior-Alignment")
sys.path.insert(0, "/home/denis-office/lerobot_a2/lerobot_grach0v/src/lerobot_policy_a2/src")

def compare_camera_transforms():
    """Compare camera transform computation between implementations."""
    print("\n" + "="*60)
    print("COMPARING CAMERA TRANSFORMS")
    print("="*60)

    import pybullet as pb
    pb.connect(pb.DIRECT)

    # Use the same camera config
    from lerobot.envs.a2.a2_cameras import RealSenseL515

    position = RealSenseL515.front_position
    rotation_quat = RealSenseL515.front_rotation

    print(f"Camera position: {position}")
    print(f"Camera rotation (quat): {rotation_quat}")

    # Original A2 method
    position_arr = np.array(position).reshape(3, 1)
    rotation_mat = np.array(pb.getMatrixFromQuaternion(rotation_quat)).reshape(3, 3)

    transform_orig = np.eye(4)
    transform_orig[:3, :] = np.hstack((rotation_mat, position_arr))
    world_to_cam_orig = np.linalg.inv(transform_orig)

    print(f"\nOriginal A2 world_to_camera (4x4):")
    print(world_to_cam_orig)

    # Our method
    transform_ours = np.eye(4)
    transform_ours[:3, :3] = rotation_mat
    transform_ours[:3, 3:] = position_arr
    world_to_cam_ours = np.linalg.inv(transform_ours)

    print(f"\nOur world_to_camera (4x4):")
    print(world_to_cam_ours)

    diff = np.abs(world_to_cam_orig - world_to_cam_ours).max()
    print(f"\nMax difference: {diff}")

    pb.disconnect()
    return diff < 1e-10


def compare_point_projection():
    """Compare point projection between implementations."""
    print("\n" + "="*60)
    print("COMPARING POINT PROJECTION")
    print("="*60)

    import pybullet as pb
    pb.connect(pb.DIRECT)

    # Test points
    points = np.array([
        [0.4, -0.1, 0.02],
        [0.5, 0.0, 0.03],
        [0.6, 0.1, 0.02],
    ], dtype=np.float32)

    # Camera config
    from lerobot.envs.a2.a2_cameras import RealSenseL515

    position = np.array(RealSenseL515.front_position).reshape(3, 1)
    rotation = np.array(pb.getMatrixFromQuaternion(RealSenseL515.front_rotation)).reshape(3, 3)
    K = RealSenseL515.intrinsics
    H, W = RealSenseL515.image_size

    # Compute camera transform
    transform = np.eye(4)
    transform[:3, :] = np.hstack((rotation, position))
    world_to_cam = np.linalg.inv(transform)

    # Original A2 projection method (using their function)
    from models.feature_fusion import project_points_coords

    pts_tensor = torch.from_numpy(points).float()
    Rt_tensor = torch.from_numpy(world_to_cam[:3, :]).float().unsqueeze(0)  # (1, 3, 4)
    K_tensor = torch.from_numpy(K).float().unsqueeze(0)  # (1, 3, 3)

    pts_2d_orig, valid_orig, depth_orig = project_points_coords(pts_tensor, Rt_tensor, K_tensor)
    print(f"\nOriginal A2 2D coords: {pts_2d_orig[0].numpy()}")
    print(f"Original A2 valid mask: {valid_orig[0].numpy()}")
    print(f"Original A2 depths: {depth_orig[0, :, 0].numpy()}")

    # Our projection method
    from lerobot_policy_a2.feature_field import project_points_to_camera

    coords_2d_ours, valid_ours = project_points_to_camera(
        points, K, world_to_cam[:3, :], (H, W)
    )
    print(f"\nOur 2D coords: {coords_2d_ours}")
    print(f"Our valid mask: {valid_ours}")

    # Compare
    diff_coords = np.abs(pts_2d_orig[0].numpy() - coords_2d_ours).max()
    diff_valid = (valid_orig[0].numpy() != valid_ours).sum()
    print(f"\nMax coord difference: {diff_coords}")
    print(f"Valid mask differences: {diff_valid}")

    pb.disconnect()
    return diff_coords < 1.0 and diff_valid == 0


def compare_dense_features():
    """Compare dense CLIP feature extraction."""
    print("\n" + "="*60)
    print("COMPARING DENSE CLIP FEATURES")
    print("="*60)

    # Create synthetic test image
    H, W = 480, 640  # Standard A2 size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            img[i, j] = [int(255 * j / W), 100, int(255 * (1 - j / W))]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Original A2 method
    print("\nExtracting features with Original A2...")
    from models.feature_fusion import Fusion
    from env.constants import WORKSPACE_LIMITS

    fusion = Fusion(num_cam=1, feat_backbone="clip", device=device)

    # Prepare observation
    obs = {
        'color': img[np.newaxis, ...],  # (1, H, W, 3)
        'depth': np.ones((1, H, W), dtype=np.float32),  # Dummy depth
        'pose': np.eye(4)[np.newaxis, :3, :],  # (1, 3, 4) identity
        'K': np.array([[[616, 0, 320], [0, 616, 240], [0, 0, 1]]], dtype=np.float32),  # (1, 3, 3)
    }

    fusion.update(obs, "grasp a red object", last_text_feat=False, visualize=False)
    clip_feats_orig = fusion.curr_obs_torch["clip_feats"]  # (1, fH, fW, D)
    clip_sims_orig = fusion.curr_obs_torch["clip_sims"]  # (1, fH, fW, 3)

    print(f"Original clip_feats shape: {clip_feats_orig.shape}")
    print(f"Original clip_feats stats: min={clip_feats_orig.min():.4f}, max={clip_feats_orig.max():.4f}, mean={clip_feats_orig.mean():.4f}")
    print(f"Original clip_sims shape: {clip_sims_orig.shape}")
    print(f"Original clip_sims stats: min={clip_sims_orig.min():.4f}, max={clip_sims_orig.max():.4f}")

    # Our method
    print("\nExtracting features with Our implementation...")
    from lerobot_policy_a2.feature_field import FeatureField

    ff = FeatureField(device=device)

    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to(device)

    dense_features_ours = ff.extract_dense_features(img_tensor)  # (D, fH, fW)
    text_features = ff.compute_text_features("grasp a red object")

    # Compute similarity like original
    D, fH, fW = dense_features_ours.shape
    feat_spatial = dense_features_ours.permute(1, 2, 0)  # (fH, fW, D)
    feat_norm = feat_spatial / (feat_spatial.norm(dim=-1, keepdim=True) + 1e-6)
    sims_ours = feat_norm @ text_features.unsqueeze(1)  # (fH, fW, 1)

    print(f"Our dense_features shape: {dense_features_ours.shape}")
    print(f"Our dense_features stats: min={dense_features_ours.min():.4f}, max={dense_features_ours.max():.4f}, mean={dense_features_ours.mean():.4f}")
    print(f"Our similarity shape: {sims_ours.shape}")
    print(f"Our similarity stats: min={sims_ours.min():.4f}, max={sims_ours.max():.4f}")

    # Compare shapes
    orig_shape = (clip_feats_orig.shape[1], clip_feats_orig.shape[2], clip_feats_orig.shape[3])
    ours_shape = (fH, fW, D)
    print(f"\nShape comparison: Original {orig_shape} vs Ours {ours_shape}")

    # Compare feature values at same location
    if orig_shape[:2] == ours_shape[:2]:
        # Compare at center pixel
        cy, cx = fH // 2, fW // 2
        feat_orig_center = clip_feats_orig[0, cy, cx, :].cpu().numpy()
        feat_ours_center = dense_features_ours[:, cy, cx].cpu().numpy()

        # Compute cosine similarity
        cosine_sim = np.dot(feat_orig_center, feat_ours_center) / (
            np.linalg.norm(feat_orig_center) * np.linalg.norm(feat_ours_center) + 1e-6
        )
        print(f"Feature cosine similarity at center: {cosine_sim:.4f}")

        return cosine_sim > 0.9
    else:
        print("WARNING: Feature shapes don't match!")
        return False


def compare_interpolation():
    """Compare feature interpolation."""
    print("\n" + "="*60)
    print("COMPARING FEATURE INTERPOLATION")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test feature map
    fH, fW, D = 72, 96, 768
    feature_map = torch.randn(D, fH, fW, device=device)

    # Test coordinates (in image space 480x640)
    H, W = 480, 640
    coords = np.array([
        [100.5, 200.3],
        [320.0, 240.0],
        [500.7, 400.2],
    ], dtype=np.float32)

    # Original A2 interpolation
    from models.feature_fusion import interpolate_feats

    # Original expects (b, C, fH, fW) and (b, n, 2) in image coords
    feat_orig = feature_map.unsqueeze(0)  # (1, D, fH, fW)
    coords_orig = torch.from_numpy(coords).unsqueeze(0).to(device)  # (1, 3, 2)

    result_orig = interpolate_feats(
        feat_orig,
        coords_orig,
        h=H,
        w=W,
        padding_mode="zeros",
        align_corners=True,
        inter_mode="bilinear"
    )  # (1, 3, D)

    print(f"Original interpolated shape: {result_orig.shape}")
    print(f"Original interpolated[0, 0, :5]: {result_orig[0, 0, :5]}")

    # Our interpolation
    from lerobot_policy_a2.feature_field import interpolate_features

    coords_ours = torch.from_numpy(coords).to(device)
    result_ours = interpolate_features(feature_map, coords_ours, (H, W))  # (3, D)

    print(f"Our interpolated shape: {result_ours.shape}")
    print(f"Our interpolated[0, :5]: {result_ours[0, :5]}")

    # Compare
    diff = torch.abs(result_orig[0] - result_ours).max().item()
    print(f"\nMax difference: {diff}")

    return diff < 0.01


def compare_network_forward():
    """Compare network forward pass."""
    print("\n" + "="*60)
    print("COMPARING NETWORK FORWARD PASS")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the same checkpoint
    checkpoint_path = "/home/denis-office/lerobot_a2/A2_new/a2_pretrained/checkpoints/sl_checkpoint_199.pth"

    # Create test inputs
    batch_size = 1
    num_points = 500
    num_actions = 10

    pts_pos = torch.randn(batch_size, num_points, 3, device=device)
    pts_feat = torch.randn(batch_size, num_points, 768, device=device)
    pts_sim = torch.rand(batch_size, num_points, 1, device=device)  # [0, 1]
    actions = torch.randn(batch_size, num_actions, 7, device=device)

    # Original A2 network
    print("\nLoading Original A2 network...")
    from models.networks import CLIPActionFusion, Policy

    class Args:
        width = 768
        layers = 1
        heads = 8
        hidden_size = 384
        use_rope = True
        no_feat_rope = False
        no_rgb_feat = False
        device = device

    args = Args()

    vilg_fusion_orig = CLIPActionFusion(action_dim=7, args=args).to(device)
    policy_orig = Policy(args.width, args.hidden_size).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Filter keys for vilg_fusion and policy
    vilg_keys = {k.replace("vilg_fusion.", ""): v for k, v in checkpoint.items() if k.startswith("vilg_fusion.")}
    policy_keys = {k.replace("policy.", ""): v for k, v in checkpoint.items() if k.startswith("policy.")}

    vilg_fusion_orig.load_state_dict(vilg_keys, strict=False)
    policy_orig.load_state_dict(policy_keys, strict=False)
    vilg_fusion_orig.eval()
    policy_orig.eval()

    # Run forward pass
    with torch.no_grad():
        cross_feat_orig, attn_orig = vilg_fusion_orig(
            pts_pos=pts_pos,
            pts_feat=pts_feat,
            pts_sim=pts_sim,
            actions=actions,
            mode="grasp"
        )
        logits_orig = policy_orig(cross_feat_orig)

    print(f"Original cross_feat shape: {cross_feat_orig.shape}")
    print(f"Original logits shape: {logits_orig.shape}")
    print(f"Original logits: {logits_orig.squeeze()[:5]}")

    # Our network
    print("\nLoading Our network...")
    from lerobot_policy_a2.modeling_a2 import A2Policy

    # Create policy with same config
    policy_ours = A2Policy(device=device)
    policy_ours._load_checkpoint(checkpoint_path)

    with torch.no_grad():
        cross_feat_ours, attn_ours = policy_ours.vilg_fusion(
            pts_pos=pts_pos,
            pts_feat=pts_feat,
            pts_sim=pts_sim,
            actions=actions,
            mode="grasp"
        )
        logits_ours = policy_ours.policy(cross_feat_ours)

    print(f"Our cross_feat shape: {cross_feat_ours.shape}")
    print(f"Our logits shape: {logits_ours.shape}")
    print(f"Our logits: {logits_ours.squeeze()[:5]}")

    # Compare
    diff_feat = torch.abs(cross_feat_orig - cross_feat_ours).max().item()
    diff_logits = torch.abs(logits_orig - logits_ours).max().item()
    print(f"\nMax cross_feat difference: {diff_feat}")
    print(f"Max logits difference: {diff_logits}")

    return diff_logits < 0.01


def main():
    print("="*60)
    print("A2 IMPLEMENTATION COMPARISON")
    print("="*60)

    results = {}

    # 1. Compare camera transforms
    results["camera_transforms"] = compare_camera_transforms()

    # 2. Compare point projection
    results["point_projection"] = compare_point_projection()

    # 3. Compare dense features
    results["dense_features"] = compare_dense_features()

    # 4. Compare interpolation
    results["interpolation"] = compare_interpolation()

    # 5. Compare network forward
    try:
        results["network_forward"] = compare_network_forward()
    except Exception as e:
        print(f"Network comparison failed: {e}")
        results["network_forward"] = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    main()
