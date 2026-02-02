#!/usr/bin/env python
"""Test script for A2 environment.

This script verifies that the A2 simulation environment works correctly:
- Environment creation and reset
- Object spawning
- Camera rendering
- Robot control basics
- Point cloud generation

Usage:
    uv run python scripts/test_a2_env.py
    uv run python scripts/test_a2_env.py --gui  # With visualization
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def test_environment_creation(gui: bool = False, object_set: str = "train") -> bool:
    """Test basic environment creation."""
    print("\n" + "=" * 60)
    print("TEST: Environment Creation")
    print("=" * 60)

    try:
        from lerobot.envs.a2.a2_sim import Environment

        env = Environment(gui=gui, object_set=object_set)
        print(f"  [PASS] Created Environment with object_set='{object_set}'")

        env.reset(workspace="raw")
        print("  [PASS] Environment reset successful")

        # Check workspace bounds
        print(f"  Workspace bounds: {env.bounds}")

        env.close()
        print("  [PASS] Environment closed successfully")
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_object_spawning(gui: bool = False) -> bool:
    """Test object spawning and language goal generation."""
    print("\n" + "=" * 60)
    print("TEST: Object Spawning")
    print("=" * 60)

    try:
        from lerobot.envs.a2.a2_sim import Environment

        env = Environment(gui=gui, object_set="train")
        env.reset(workspace="raw")

        # Generate language goal (this sets up target objects)
        lang_goal = env.generate_lang_goal()
        print(f"  Language goal: '{lang_goal}'")
        print(f"  Target object list: {env.target_obj_lst}")

        # Add objects
        env.add_objects(num_obj=5, workspace_limits=env.bounds)
        num_objects = len(env.obj_ids["rigid"])
        print(f"  [PASS] Spawned {num_objects} objects")

        # Check target objects
        num_targets = len(env.target_obj_ids)
        print(f"  Target objects in scene: {num_targets}")

        if num_targets == 0:
            print("  [WARN] No target objects found - this may be expected")

        # Get object info
        for obj_id in env.obj_ids["rigid"][:3]:
            pos, rot, dim = env.obj_info(obj_id)
            label = env.obj_labels.get(obj_id, ["unknown"])[0]
            print(f"    Object {obj_id}: {label} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        env.close()
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_camera_rendering(gui: bool = False) -> bool:
    """Test camera rendering functionality."""
    print("\n" + "=" * 60)
    print("TEST: Camera Rendering")
    print("=" * 60)

    try:
        from lerobot.envs.a2.a2_sim import Environment

        env = Environment(gui=gui, object_set="train")
        env.reset(workspace="raw")
        env.add_objects(num_obj=5, workspace_limits=env.bounds)

        # Test each camera
        cam_names = ["front", "left", "right", "top", "side_left", "side_right", "overview"]

        for i, config in enumerate(env.agent_cams):
            cam_name = cam_names[i] if i < len(cam_names) else f"cam_{i}"
            color, depth, segm = env.render_camera(config)

            print(f"  Camera '{cam_name}':")
            print(f"    Color: {color.shape}, dtype={color.dtype}")
            print(f"    Depth: {depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]")
            print(f"    Segmentation: {segm.shape}, unique IDs={len(np.unique(segm))}")

            # Verify image content
            if color.max() == 0:
                print(f"    [WARN] Color image is all black")
            if depth.max() == 0:
                print(f"    [WARN] Depth image is all zero")

        print("  [PASS] All cameras rendered successfully")

        # Test fast render
        fast_img = env.render_camera_fast(env.agent_cams[0], (480, 640))
        print(f"  Fast render: {fast_img.shape}")
        print("  [PASS] Fast render works")

        env.close()
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_point_cloud(gui: bool = False) -> bool:
    """Test point cloud generation."""
    print("\n" + "=" * 60)
    print("TEST: Point Cloud Generation")
    print("=" * 60)

    try:
        from lerobot.envs.a2.a2_sim import Environment

        env = Environment(gui=gui, object_set="train")
        env.reset(workspace="raw")
        env.add_objects(num_obj=5, workspace_limits=env.bounds)

        # Get point cloud
        pcd_array = env.get_pointcloud_array(cameras=[0, 1, 2], max_points=10000)
        print(f"  Point cloud shape: {pcd_array.shape}")
        print(f"  Point cloud dtype: {pcd_array.dtype}")

        if pcd_array.shape[1] == 6:
            print("  Point cloud format: [x, y, z, r, g, b]")
            xyz = pcd_array[:, :3]
            rgb = pcd_array[:, 3:6]
            print(f"  XYZ range: x=[{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
            print(f"             y=[{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
            print(f"             z=[{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
            print(f"  RGB range: [{rgb.min():.1f}, {rgb.max():.1f}]")
        else:
            print(f"  [WARN] Unexpected point cloud shape: {pcd_array.shape}")

        # Test multi-view point cloud
        pcd, colors, depths = env.get_multi_view_pointcloud(cameras=[0, 1, 2])
        print(f"  Multi-view point cloud: {len(pcd.points)} points")
        print(f"  Color images: {list(colors.keys())}")
        print(f"  Depth images: {list(depths.keys())}")

        print("  [PASS] Point cloud generation works")

        env.close()
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_robot_control(gui: bool = False) -> bool:
    """Test basic robot control."""
    print("\n" + "=" * 60)
    print("TEST: Robot Control")
    print("=" * 60)

    try:
        from lerobot.envs.a2.a2_sim import Environment

        env = Environment(gui=gui, object_set="train")
        env.reset(workspace="raw")

        # Get initial robot state
        state = env.get_robot_state()
        print(f"  Initial EE position: ({state['ee_pos'][0]:.3f}, {state['ee_pos'][1]:.3f}, {state['ee_pos'][2]:.3f})")
        print(f"  Initial joints: {np.round(state['joints'], 3)}")
        print(f"  Gripper angle: {state['gripper_angle']:.3f}")

        # Test gripper
        print("  Testing gripper...")
        env.close_gripper()
        state = env.get_robot_state()
        print(f"    After close: gripper_angle={state['gripper_angle']:.3f}")

        env.open_gripper()
        state = env.get_robot_state()
        print(f"    After open: gripper_angle={state['gripper_angle']:.3f}")
        print("  [PASS] Gripper control works")

        # Test go home
        print("  Testing go_home...")
        env.go_home()
        state = env.get_robot_state()
        print(f"    Home EE position: ({state['ee_pos'][0]:.3f}, {state['ee_pos'][1]:.3f}, {state['ee_pos'][2]:.3f})")
        print("  [PASS] go_home works")

        env.close()
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_grasp_execution(gui: bool = False) -> bool:
    """Test grasp primitive execution."""
    print("\n" + "=" * 60)
    print("TEST: Grasp Execution")
    print("=" * 60)

    try:
        from lerobot.envs.a2.a2_sim import Environment

        env = Environment(gui=gui, object_set="train")
        env.reset(workspace="raw")

        # Generate language goal and add objects
        lang_goal = env.generate_lang_goal()
        env.add_objects(num_obj=5, workspace_limits=env.bounds)

        print(f"  Language goal: '{lang_goal}'")
        print(f"  Target objects: {env.target_obj_ids}")

        if not env.target_obj_ids:
            print("  [SKIP] No target objects, skipping grasp test")
            env.close()
            return True

        # Get position of first target object
        target_id = env.target_obj_ids[0]
        pos, rot, dim = env.obj_info(target_id)
        print(f"  Target object {target_id} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        # Create grasp pose (top-down grasp)
        grasp_pos = np.array([pos[0], pos[1], pos[2] + 0.02])
        grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Top-down orientation
        grasp_pose = np.concatenate([grasp_pos, grasp_quat])

        print(f"  Attempting grasp at ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")

        # Execute grasp
        t0 = time.time()
        success, grasped_id, min_dist = env.grasp(grasp_pose)
        elapsed = time.time() - t0

        print(f"  Grasp result: success={success}, grasped_id={grasped_id}, time={elapsed:.2f}s")

        if success:
            print("  [PASS] Grasp executed successfully")
        else:
            print("  [WARN] Grasp failed (may be expected depending on object position)")

        env.close()
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gymnasium_wrapper(gui: bool = False) -> bool:
    """Test the gymnasium-compatible A2Env wrapper."""
    print("\n" + "=" * 60)
    print("TEST: Gymnasium Wrapper (A2Env)")
    print("=" * 60)

    try:
        from lerobot.envs.a2.a2 import A2Env

        env = A2Env(
            task="grasp",
            object_set="train",
            num_objects=5,
            camera_name="front,overview",
            gui=gui,
        )

        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Reset
        obs, info = env.reset(seed=42)
        print(f"  Reset successful, info keys: {list(info.keys())}")

        # Check observation structure
        print(f"  Observation keys: {list(obs.keys())}")
        if "pixels" in obs:
            print(f"    Pixel keys: {list(obs['pixels'].keys())}")
            for k, v in obs["pixels"].items():
                print(f"      {k}: shape={v.shape}, dtype={v.dtype}")

        if "robot_state" in obs:
            print(f"    Robot state keys: {list(obs['robot_state'].keys())}")

        # Take a random action
        action = env.action_space.sample()
        print(f"  Sample action shape: {action.shape}")

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step result: reward={reward}, terminated={terminated}, truncated={truncated}")

        print("  [PASS] Gymnasium wrapper works")

        env.close()
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests(gui: bool = False) -> dict:
    """Run all tests and return results."""
    tests = [
        ("Environment Creation", lambda: test_environment_creation(gui)),
        ("Object Spawning", lambda: test_object_spawning(gui)),
        ("Camera Rendering", lambda: test_camera_rendering(gui)),
        ("Point Cloud", lambda: test_point_cloud(gui)),
        ("Robot Control", lambda: test_robot_control(gui)),
        ("Grasp Execution", lambda: test_grasp_execution(gui)),
        ("Gymnasium Wrapper", lambda: test_gymnasium_wrapper(gui)),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  [FAIL] Unhandled exception: {e}")
            results[name] = False

    return results


def main():
    parser = argparse.ArgumentParser(description="Test A2 Environment")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--test", type=str, default=None, help="Run specific test")
    args = parser.parse_args()

    print("=" * 60)
    print("A2 ENVIRONMENT TEST SUITE")
    print("=" * 60)

    if args.test:
        # Run specific test
        test_map = {
            "env": lambda: test_environment_creation(args.gui),
            "objects": lambda: test_object_spawning(args.gui),
            "camera": lambda: test_camera_rendering(args.gui),
            "pointcloud": lambda: test_point_cloud(args.gui),
            "robot": lambda: test_robot_control(args.gui),
            "grasp": lambda: test_grasp_execution(args.gui),
            "gym": lambda: test_gymnasium_wrapper(args.gui),
        }
        if args.test in test_map:
            result = test_map[args.test]()
            sys.exit(0 if result else 1)
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {list(test_map.keys())}")
            sys.exit(1)

    # Run all tests
    results = run_all_tests(args.gui)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print("-" * 60)
    print(f"  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed!")
        sys.exit(0)
    else:
        print(f"\n  {total - passed} test(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
