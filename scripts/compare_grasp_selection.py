#!/usr/bin/env python3
"""
Compare grasp selection between original A2 and our implementation on the same scene.
"""

import sys
import os
import numpy as np
import torch
import json

# Add paths
sys.path.insert(0, "/home/denis-office/lerobot_a2/lerobot_grach0v/src/lerobot_policy_a2/src")

def load_test_case(path):
    """Load test case from A2 text format."""
    with open(path) as f:
        lines = f.readlines()

    lang_goal = lines[0].strip()
    target_indices = list(map(int, lines[1].strip().split()))

    objects = []
    for i, line in enumerate(lines[2:]):
        parts = line.strip().split()
        urdf_path = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        rx, ry, rz = float(parts[4]), float(parts[5]), float(parts[6])
        objects.append({
            "urdf_path": urdf_path,
            "position": [x, y, z],
            "euler": [rx, ry, rz],
            "target": (i in target_indices)
        })

    return {
        "lang_goal": lang_goal,
        "target_indices": target_indices,
        "objects": objects
    }


def compare_grasp_scoring():
    """Compare grasp scoring between implementations."""
    print("\n" + "="*60)
    print("COMPARING GRASP SCORING")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a test case
    test_case_path = "/home/denis-office/lerobot_a2/A2_new/testing_cases/grasp_testing_cases/seen/case00-round.txt"

    if not os.path.exists(test_case_path):
        print(f"Test case not found: {test_case_path}")
        return False

    test_case = load_test_case(test_case_path)

    print(f"Test case: {test_case_path}")
    print(f"Language goal: {test_case['lang_goal']}")

    # Setup environment with this test case
    from lerobot.envs.a2.a2_sim import A2Sim
    import pybullet as pb

    env = A2Sim(task="grasp", gui=False)

    # Load scene from test case
    objects = test_case["objects"]
    print(f"Objects in scene: {len(objects)}")

    # Reset and load objects
    env.reset_arm()
    env.remove_all_objects()

    # Load objects from test case
    for i, obj in enumerate(objects):
        obj_path = obj["urdf_path"]
        # Convert to absolute path
        if not os.path.isabs(obj_path):
            obj_path = os.path.join("/home/denis-office/lerobot_a2/A2_new", obj_path)
        pos = obj["position"]
        euler = obj["euler"]
        # Convert Euler angles to quaternion
        orn = pb.getQuaternionFromEuler(euler)
        try:
            env.load_single_urdf(
                urdf_filepath=obj_path,
                position=pos,
                orientation=orn,
                target=obj.get("target", False)
            )
        except Exception as e:
            print(f"  Failed to load {obj_path}: {e}")

    # Get observation
    env._run_sim(50)  # Let objects settle

    # Get point cloud and images
    cameras = ["front", "left", "right"]
    rgb_images = env.get_camera_images(cameras, render_size=(480, 640))
    depth_images = {}
    for cam_name in cameras:
        try:
            _, depth = env._render_camera(env.CAMERA_NAME_TO_IDX[cam_name], (480, 640), return_depth=True)
            depth_images[cam_name] = depth
        except:
            pass

    # Get point cloud
    pcd = env.get_multi_view_pointcloud(cameras, downsample=True, voxel_size=0.01)
    points = np.asarray(pcd.points).astype(np.float32)
    colors = np.asarray(pcd.colors).astype(np.float32)

    print(f"Point cloud: {len(points)} points")
    print(f"RGB images: {list(rgb_images.keys())}")
    print(f"Depth images: {list(depth_images.keys())}")

    # Generate grasp candidates
    from lerobot_policy_a2.graspnet_interface import GraspNetInterface
    gn = GraspNetInterface()

    grasp_poses, grasp_scores = gn.generate_grasps(points, colors)
    print(f"Generated {len(grasp_poses)} grasp candidates")

    if len(grasp_poses) == 0:
        print("No grasps generated!")
        return False

    # Get camera configs
    from lerobot.envs.a2.a2_cameras import RealSenseL515
    import pybullet as pb

    camera_configs = []
    for cam_name in cameras:
        config = {
            "name": cam_name,
            "position": getattr(RealSenseL515, f"{cam_name}_position"),
            "rotation": getattr(RealSenseL515, f"{cam_name}_rotation"),
            "intrinsics": RealSenseL515.intrinsics,
        }
        camera_configs.append(config)

    # Score grasps with OUR implementation
    print("\n--- OUR IMPLEMENTATION ---")
    from lerobot_policy_a2.feature_field import FeatureField
    from lerobot_policy_a2.modeling_a2 import A2Policy

    # Load our policy
    checkpoint_path = "/home/denis-office/lerobot_a2/A2_new/logs/a2_pretrained/checkpoints/sl_checkpoint_199.pth"
    policy = A2Policy(device=device)
    policy._load_checkpoint(checkpoint_path)

    # Get grasp positions
    grasp_positions = grasp_poses[:, :3, 3]

    # Compute features
    ff = FeatureField(device=device)
    lang_goal = test_case.get("lang_goal", "grasp a round object")

    pts_feat, pts_sim = ff.compute_per_point_features(
        points, rgb_images, camera_configs, lang_goal, depth_images
    )

    print(f"pts_feat shape: {pts_feat.shape}")
    print(f"pts_feat stats: min={pts_feat.min():.4f}, max={pts_feat.max():.4f}, mean={pts_feat.mean():.4f}")
    print(f"pts_sim shape: {pts_sim.shape}")
    print(f"pts_sim stats: min={pts_sim.min():.4f}, max={pts_sim.max():.4f}, mean={pts_sim.mean():.4f}")

    # Sample N points for the network (like original A2)
    N = min(500, len(points))
    indices = np.random.choice(len(points), N, replace=False)
    pts_pos_sample = torch.from_numpy(points[indices]).float().to(device).unsqueeze(0)
    pts_feat_sample = pts_feat[:, indices, :]
    pts_sim_sample = pts_sim[:, indices, :]

    # Prepare grasps as actions
    from lerobot_policy_a2.graspnet_interface import grasp_pose_to_action
    actions = []
    for pose in grasp_poses[:50]:  # Score top 50
        action = grasp_pose_to_action(pose)
        actions.append(action)
    actions = torch.from_numpy(np.array(actions)).float().to(device).unsqueeze(0)  # (1, 50, 7)

    # Score with network
    with torch.no_grad():
        cross_feat, attn = policy.vilg_fusion(
            pts_pos=pts_pos_sample,
            pts_feat=pts_feat_sample,
            pts_sim=pts_sim_sample,
            actions=actions,
            mode="grasp"
        )
        scores_ours = policy.policy(cross_feat).squeeze()

    print(f"Our scores: {scores_ours[:10]}")
    best_grasp_ours = scores_ours.argmax().item()
    print(f"Our best grasp: {best_grasp_ours} with score {scores_ours[best_grasp_ours]:.4f}")

    # Run original A2 for comparison
    print("\n--- ORIGINAL A2 ---")
    try:
        # Add original A2 to path
        sys.path.insert(0, "/home/denis-office/lerobot_a2/A2_new")
        from models.feature_fusion import Fusion
        from models.networks import CLIPActionFusion, Policy

        # Setup original fusion
        fusion = Fusion(num_cam=3, feat_backbone="clip", device=device)

        # Prepare observation for original A2
        obs = {
            'color': np.stack([rgb_images[cam][:,:,:3] for cam in cameras]),  # (3, H, W, 3)
            'depth': np.stack([depth_images[cam] for cam in cameras]),  # (3, H, W)
        }

        # Need poses and K
        poses = []
        Ks = []
        for config in camera_configs:
            pos = np.array(config["position"]).reshape(3, 1)
            rot = np.array(pb.getMatrixFromQuaternion(config["rotation"])).reshape(3, 3)
            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3:] = pos
            world_to_cam = np.linalg.inv(transform)
            poses.append(world_to_cam[:3, :].astype(np.float32))
            Ks.append(config["intrinsics"].astype(np.float32))

        obs['pose'] = np.stack(poses)  # (3, 3, 4)
        obs['K'] = np.stack(Ks)  # (3, 3, 3)

        fusion.update(obs, lang_goal, last_text_feat=False, visualize=False)

        # Get features for sampled points
        pts_torch = torch.from_numpy(points[indices]).float().to(device)
        outputs = fusion.eval(pts_torch, return_names=["clip_feats", "clip_sims"])

        pts_feat_orig = outputs["clip_feats"].unsqueeze(0)  # (1, N, D)
        pts_sim_orig = outputs["clip_sims"][:, 0:1].unsqueeze(0)  # (1, N, 1)

        print(f"Original pts_feat shape: {pts_feat_orig.shape}")
        print(f"Original pts_feat stats: min={pts_feat_orig.min():.4f}, max={pts_feat_orig.max():.4f}, mean={pts_feat_orig.mean():.4f}")
        print(f"Original pts_sim shape: {pts_sim_orig.shape}")
        print(f"Original pts_sim stats: min={pts_sim_orig.min():.4f}, max={pts_sim_orig.max():.4f}, mean={pts_sim_orig.mean():.4f}")

        # Compare features
        feat_diff = torch.abs(pts_feat_sample - pts_feat_orig).mean().item()
        sim_diff = torch.abs(pts_sim_sample - pts_sim_orig).mean().item()
        print(f"\nFeature mean absolute difference: {feat_diff:.6f}")
        print(f"Similarity mean absolute difference: {sim_diff:.6f}")

        # Score grasps with original network
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
        vilg_keys = {k.replace("vilg_fusion.", ""): v for k, v in checkpoint.items() if k.startswith("vilg_fusion.")}
        policy_keys = {k.replace("policy.", ""): v for k, v in checkpoint.items() if k.startswith("policy.")}
        vilg_fusion_orig.load_state_dict(vilg_keys, strict=False)
        policy_orig.load_state_dict(policy_keys, strict=False)
        vilg_fusion_orig.eval()
        policy_orig.eval()

        with torch.no_grad():
            cross_feat_orig, attn_orig = vilg_fusion_orig(
                pts_pos=pts_pos_sample,
                pts_feat=pts_feat_orig,
                pts_sim=pts_sim_orig,
                actions=actions,
                mode="grasp"
            )
            scores_orig = policy_orig(cross_feat_orig).squeeze()

        print(f"Original scores: {scores_orig[:10]}")
        best_grasp_orig = scores_orig.argmax().item()
        print(f"Original best grasp: {best_grasp_orig} with score {scores_orig[best_grasp_orig]:.4f}")

        # Compare scores
        score_diff = torch.abs(scores_ours - scores_orig).mean().item()
        print(f"\nScore mean absolute difference: {score_diff:.6f}")
        print(f"Same best grasp: {best_grasp_ours == best_grasp_orig}")

    except Exception as e:
        import traceback
        print(f"Original A2 comparison failed: {e}")
        traceback.print_exc()

    env.close()
    return True


if __name__ == "__main__":
    compare_grasp_scoring()
