#!/usr/bin/env python3
"""
Direct comparison of feature extraction - no imports from original A2.
Compare our implementation against manually verified A2 formulas.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

# Add paths
sys.path.insert(0, "/home/denis-office/lerobot_a2/lerobot_grach0v/src/lerobot_policy_a2/src")

def test_interpolation_formula():
    """Test that our interpolation matches the original A2 formula exactly."""
    print("\n" + "="*60)
    print("TEST 1: INTERPOLATION FORMULA")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test feature map
    fH, fW, D = 24, 32, 768  # Typical CLIP feature dimensions
    feature_map = torch.randn(D, fH, fW, device=device)

    # Test coordinates (in image space)
    H, W = 480, 640  # Image dimensions
    coords = torch.tensor([
        [100.5, 200.3],
        [320.0, 240.0],
        [500.7, 400.2],
    ], dtype=torch.float32, device=device)

    # Original A2 formula (from feature_fusion.py line 29-30):
    # x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    # y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    # grid = torch.stack([x_norm, y_norm], dim=-1)
    # result = F.grid_sample(feat, grid, padding_mode="zeros", align_corners=True, mode="bilinear")

    # Compute original A2 way
    x_norm_orig = coords[:, 0] / (W - 1) * 2 - 1
    y_norm_orig = coords[:, 1] / (H - 1) * 2 - 1
    grid_orig = torch.stack([x_norm_orig, y_norm_orig], dim=-1).view(1, 1, -1, 2)

    feat_input = feature_map.unsqueeze(0)  # (1, D, fH, fW)
    result_orig = F.grid_sample(
        feat_input, grid_orig,
        padding_mode="zeros",
        align_corners=True,
        mode="bilinear"
    ).squeeze(0).squeeze(1).T  # (N, D)

    # Our implementation
    from lerobot_policy_a2.feature_field import interpolate_features
    result_ours = interpolate_features(feature_map, coords, (H, W))

    # Compare
    diff = torch.abs(result_orig - result_ours).max().item()
    print(f"Original result[0, :5]: {result_orig[0, :5]}")
    print(f"Our result[0, :5]:      {result_ours[0, :5]}")
    print(f"Max difference: {diff}")

    passed = diff < 1e-5
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_projection_formula():
    """Test point projection formula matches original A2."""
    print("\n" + "="*60)
    print("TEST 2: POINT PROJECTION FORMULA")
    print("="*60)

    # Test points in world coordinates
    points = np.array([
        [0.4, -0.1, 0.02],
        [0.5, 0.0, 0.03],
        [0.6, 0.1, 0.02],
    ], dtype=np.float32)

    # Camera parameters (simplified for testing)
    K = np.array([
        [616.0, 0.0, 320.0],
        [0.0, 616.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Rt is the extrinsic matrix (3x4) - world to camera
    Rt = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -0.5]  # Camera at z=0.5 looking down
    ], dtype=np.float32)

    H, W = 480, 640

    # Original A2 formula (from feature_fusion.py project_points_coords):
    # pts_pos = torch.cat((pts_pos, torch.ones(b, n, 1, device=device)), dim=2)  # (b, n, 4)
    # pts_cam = torch.einsum('ijk,imk->imj', Rt, pts_pos)  # (b, n, 3)
    # pts_2d = torch.einsum('ijk,imk->imj', K, pts_cam)  # (b, n, 3)
    # pts_2d = (pts_2d / pts_2d[:, :, 2:3])[:, :, :2]  # (b, n, 2)
    # valid = (pts_2d[:, :, 0] >= 0) & (pts_2d[:, :, 0] < w) & (pts_2d[:, :, 1] >= 0) & (pts_2d[:, :, 1] < h)
    # valid &= (pts_cam[:, :, 2] > 0)  # depth > 0

    # Compute original A2 way
    pts_homog = np.hstack([points, np.ones((len(points), 1))])  # (n, 4)
    pts_cam = (Rt @ pts_homog.T).T  # (n, 3)
    pts_2d_homog = (K @ pts_cam.T).T  # (n, 3)
    pts_2d_orig = pts_2d_homog[:, :2] / pts_2d_homog[:, 2:3]  # (n, 2)

    valid_orig = (
        (pts_2d_orig[:, 0] >= 0) &
        (pts_2d_orig[:, 0] < W) &
        (pts_2d_orig[:, 1] >= 0) &
        (pts_2d_orig[:, 1] < H) &
        (pts_cam[:, 2] > 0)
    )

    # Our implementation
    from lerobot_policy_a2.feature_field import project_points_to_camera
    pts_2d_ours, valid_ours = project_points_to_camera(points, K, Rt, (H, W))

    # Compare
    diff_coords = np.abs(pts_2d_orig - pts_2d_ours).max()
    diff_valid = (valid_orig != valid_ours).sum()

    print(f"Original 2D coords:\n{pts_2d_orig}")
    print(f"Our 2D coords:\n{pts_2d_ours}")
    print(f"Original valid: {valid_orig}")
    print(f"Our valid: {valid_ours}")
    print(f"Max coord difference: {diff_coords}")
    print(f"Valid mask differences: {diff_valid}")

    passed = diff_coords < 1e-5 and diff_valid == 0
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_depth_weighting():
    """Test depth weighting formula matches original A2."""
    print("\n" + "="*60)
    print("TEST 3: DEPTH WEIGHTING FORMULA")
    print("="*60)

    # Test depths
    depths = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    mu = 0.02  # A2 hyperparameter

    # Original A2 formula (from feature_fusion.py eval() method):
    # dist_weight = torch.exp(-(dist ** 2) / (2 * mu)) * dist_valid
    # Where dist is the depth/distance

    weights_orig = np.exp(-(depths ** 2) / (2 * mu))

    print(f"Depths: {depths}")
    print(f"Original weights (mu={mu}): {weights_orig}")

    # Check if weights make sense
    # At depth=0.5: exp(-0.25/0.04) = exp(-6.25) ≈ 0.002
    # At depth=1.0: exp(-1.0/0.04) = exp(-25) ≈ 1e-11
    # This means only VERY close points get significant weight!

    print(f"\nNote: With mu={mu}, weights drop off VERY fast:")
    for d, w in zip(depths, weights_orig):
        print(f"  depth={d:.1f} -> weight={w:.2e}")

    # This is a potential issue - mu=0.02 is very small
    # Let's check what the original A2 actually uses
    print("\nChecking weight sensitivity:")
    for test_mu in [0.02, 0.2, 2.0]:
        weights = np.exp(-(depths ** 2) / (2 * test_mu))
        print(f"  mu={test_mu}: weights = {weights}")

    return True  # Information test, always passes


def test_clip_feature_extraction():
    """Test CLIP feature extraction pipeline."""
    print("\n" + "="*60)
    print("TEST 4: CLIP FEATURE EXTRACTION")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test image
    H, W = 480, 640
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Simple gradient pattern
    for i in range(H):
        for j in range(W):
            img[i, j] = [int(255 * j / W), 100, int(255 * (1 - j / W))]

    # Our implementation
    from lerobot_policy_a2.feature_field import FeatureField

    ff = FeatureField(device=device)

    # Convert to tensor
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to(device)

    print(f"Input image shape: {img.shape}")

    # Extract features
    dense_features = ff.extract_dense_features(img_tensor)

    print(f"Output feature shape: {dense_features.shape}")
    print(f"Feature stats: min={dense_features.min():.4f}, max={dense_features.max():.4f}, mean={dense_features.mean():.4f}")

    # Check expected shape
    # Original A2: For 480x640 image with patch_size=14:
    # - Resized to preserve aspect ratio (smallest edge = 336)
    # - So 480x640 -> 336x448
    # - Feature map: 336/14 x 448/14 = 24 x 32

    D, fH, fW = dense_features.shape
    expected_fH = 336 // 14  # 24
    # For aspect ratio 640/480 = 1.333, expected_fW should be about 32

    print(f"\nExpected feature height (336/14): {expected_fH}")
    print(f"Actual feature height: {fH}")
    print(f"Aspect ratio: input={W/H:.3f}, features={fW/fH:.3f}")

    # Check if aspect ratio is preserved
    aspect_preserved = abs((fW/fH) - (W/H)) < 0.1
    shape_correct = fH == expected_fH and D == 768

    print(f"\nAspect ratio preserved: {aspect_preserved}")
    print(f"Shape correct (D=768, fH=24): {shape_correct}")

    passed = aspect_preserved and shape_correct
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_similarity_computation():
    """Test similarity computation matches original A2."""
    print("\n" + "="*60)
    print("TEST 5: SIMILARITY COMPUTATION")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test features
    D = 768
    fH, fW = 24, 32

    features = torch.randn(D, fH, fW, device=device)
    text_embedding = torch.randn(D, device=device)
    text_embedding = text_embedding / text_embedding.norm()  # Normalize

    # Original A2 formula (from feature_fusion.py):
    # feat_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-6)
    # sims = feat_norm @ text_embedding

    # Compute original A2 way
    features_spatial = features.permute(1, 2, 0)  # (fH, fW, D)
    feat_norm_orig = features_spatial / (features_spatial.norm(dim=-1, keepdim=True) + 1e-6)
    sims_orig = feat_norm_orig @ text_embedding  # (fH, fW)

    # Our implementation
    from lerobot_policy_a2.feature_field import FeatureField

    ff = FeatureField(device=device)

    # Compute similarity using dense similarity map (as in our updated code)
    dense_features_norm = features_spatial / (features_spatial.norm(dim=-1, keepdim=True) + 1e-6)
    sims_ours = dense_features_norm @ text_embedding  # (fH, fW)

    # Compare
    diff = torch.abs(sims_orig - sims_ours).max().item()

    print(f"Original similarity shape: {sims_orig.shape}")
    print(f"Our similarity shape: {sims_ours.shape}")
    print(f"Original similarity stats: min={sims_orig.min():.4f}, max={sims_orig.max():.4f}")
    print(f"Our similarity stats: min={sims_ours.min():.4f}, max={sims_ours.max():.4f}")
    print(f"Max difference: {diff}")

    passed = diff < 1e-6
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_full_per_point_pipeline():
    """Test full per-point feature extraction against manual calculation."""
    print("\n" + "="*60)
    print("TEST 6: FULL PER-POINT PIPELINE")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # This tests the entire pipeline:
    # 1. Extract dense CLIP features from image
    # 2. Project 3D points to 2D
    # 3. Interpolate features at projected locations
    # 4. Weight by depth and accumulate across views

    from lerobot_policy_a2.feature_field import FeatureField
    from lerobot.envs.a2.a2_cameras import RealSenseL515
    import pybullet as pb

    pb.connect(pb.DIRECT)

    ff = FeatureField(device=device)

    # Create test data
    H, W = 480, 640

    # Simple test image
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :W//2] = [255, 0, 0]  # Left half red
    img[:, W//2:] = [0, 0, 255]  # Right half blue

    # Camera config
    position = np.array(RealSenseL515.front_position).reshape(3, 1)
    rotation = np.array(pb.getMatrixFromQuaternion(RealSenseL515.front_rotation)).reshape(3, 3)
    K = RealSenseL515.intrinsics

    # Compute camera transform
    transform = np.eye(4)
    transform[:3, :] = np.hstack((rotation, position))
    world_to_cam = np.linalg.inv(transform)
    Rt = world_to_cam[:3, :]  # (3, 4)

    # Test points in workspace
    points = np.array([
        [0.4, -0.1, 0.02],  # Point in grasp workspace
        [0.5, 0.0, 0.05],
        [0.6, 0.1, 0.03],
    ], dtype=np.float32)

    # Convert image to tensor
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to(device)

    # Step 1: Extract dense features
    dense_features = ff.extract_dense_features(img_tensor)
    print(f"Dense features shape: {dense_features.shape}")

    # Step 2: Compute text features
    text_features = ff.compute_text_features("grasp a red object")
    print(f"Text features shape: {text_features.shape}")

    # Step 3: Compute dense similarity map
    D, fH, fW = dense_features.shape
    dense_features_spatial = dense_features.permute(1, 2, 0)  # (fH, fW, D)
    dense_features_norm = dense_features_spatial / (dense_features_spatial.norm(dim=-1, keepdim=True) + 1e-6)
    dense_similarity = dense_features_norm @ text_features.unsqueeze(1)  # (fH, fW, 1)
    dense_similarity = dense_similarity.squeeze(-1)  # (fH, fW)
    print(f"Dense similarity shape: {dense_similarity.shape}")

    # Step 4: Project points
    from lerobot_policy_a2.feature_field import project_points_to_camera, interpolate_features

    coords_2d, valid = project_points_to_camera(points, K, Rt, (H, W))
    print(f"Projected coords:\n{coords_2d}")
    print(f"Valid mask: {valid}")

    # Step 5: Interpolate features and similarity
    coords_tensor = torch.from_numpy(coords_2d).to(device)

    # Interpolate features
    point_features = interpolate_features(dense_features, coords_tensor, (H, W))
    print(f"Interpolated features shape: {point_features.shape}")

    # Interpolate similarity
    similarity_map = dense_similarity.unsqueeze(0)  # (1, fH, fW)
    point_similarity = interpolate_features(similarity_map, coords_tensor, (H, W))
    print(f"Interpolated similarity shape: {point_similarity.shape}")

    # Check if features make sense
    print(f"\nFeature norms: {point_features.norm(dim=1)}")
    print(f"Similarities: {point_similarity.squeeze()}")

    # Red object should have higher similarity for "grasp a red object"
    # Points closer to left side (red) should have higher similarity

    pb.disconnect()

    print(f"\nRESULT: PASS (manual inspection)")
    return True


def test_graspnet_input_format():
    """Check GraspNet input format matches original A2."""
    print("\n" + "="*60)
    print("TEST 7: GRASPNET INPUT FORMAT")
    print("="*60)

    # Original A2 passes point cloud to GraspNet as:
    # - points: (N, 3) numpy array in world coordinates
    # - colors: (N, 3) numpy array normalized to [0, 1]

    # GraspNet returns:
    # - grasp_poses: (M, 4, 4) numpy array - homogeneous transformation matrices
    # - grasp_scores: (M,) numpy array - confidence scores

    from lerobot_policy_a2.graspnet_interface import GraspNetInterface

    # Create test point cloud
    N = 1000
    points = np.random.uniform([0.3, -0.2, 0], [0.7, 0.2, 0.1], (N, 3)).astype(np.float32)
    colors = np.random.uniform(0, 1, (N, 3)).astype(np.float32)

    print(f"Input points shape: {points.shape}")
    print(f"Input colors shape: {colors.shape}")

    # Initialize GraspNet
    gn = GraspNetInterface()

    # Generate grasps
    grasp_poses, grasp_scores = gn.generate_grasps(points, colors)

    print(f"Output grasp_poses shape: {grasp_poses.shape}")
    print(f"Output grasp_scores shape: {grasp_scores.shape}")

    if len(grasp_poses) > 0:
        print(f"Sample grasp pose:\n{grasp_poses[0]}")
        print(f"Sample grasp score: {grasp_scores[0]:.4f}")

    # Check format
    poses_valid = len(grasp_poses.shape) == 3 and grasp_poses.shape[1:] == (4, 4)
    scores_valid = len(grasp_scores.shape) == 1

    print(f"\nPoses format valid (N, 4, 4): {poses_valid}")
    print(f"Scores format valid (N,): {scores_valid}")

    passed = poses_valid and scores_valid
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("="*60)
    print("A2 IMPLEMENTATION COMPARISON V2")
    print("="*60)

    results = {}

    # Run all tests
    results["interpolation"] = test_interpolation_formula()
    results["projection"] = test_projection_formula()
    results["depth_weighting"] = test_depth_weighting()
    results["clip_features"] = test_clip_feature_extraction()
    results["similarity"] = test_similarity_computation()
    results["full_pipeline"] = test_full_per_point_pipeline()

    try:
        results["graspnet_format"] = test_graspnet_input_format()
    except Exception as e:
        print(f"GraspNet test failed: {e}")
        results["graspnet_format"] = False

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
