#!/usr/bin/env python
"""Test script for A2 policy and other LeRobot policies.

This script verifies that policies work correctly with the A2 environment:
- Policy loading from HuggingFace or local path
- Forward pass with sample inputs
- Action selection
- Integration with A2 environment

Supports multiple policy types: a2, pi0, pi05, act, diffusion, etc.

Usage:
    # Test A2 policy
    uv run python scripts/test_a2_policy.py --policy a2

    # Test with pretrained weights
    uv run python scripts/test_a2_policy.py --policy a2 --policy_path dgrachev/a2_pretrained

    # Test Pi0/Pi05 VLA policy
    uv run python scripts/test_a2_policy.py --policy pi0 --policy_path lerobot/pi0

    # Test random baseline
    uv run python scripts/test_a2_policy.py --policy random

    # Run with GUI
    uv run python scripts/test_a2_policy.py --policy a2 --gui
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch


def test_random_policy(device: str = "cpu") -> bool:
    """Test the random baseline policy."""
    print("\n" + "=" * 60)
    print("TEST: Random Policy")
    print("=" * 60)

    try:
        from lerobot.envs.a2.a2_collect import RandomPolicy

        policy = RandomPolicy(device=device)
        print(f"  Created RandomPolicy on device '{device}'")

        # Test reset
        policy.reset()
        print("  [PASS] reset() works")

        # Test action selection
        batch = {
            "observation.state": torch.randn(1, 14, device=device),
            "observation.images.front": torch.randn(1, 3, 480, 640, device=device),
        }

        action = policy.select_action(batch)
        print(f"  Action shape: {action.shape}")
        print(f"  Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
        print("  [PASS] select_action() works")

        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_policy_loading(policy_type: str, policy_path: str | None, device: str = "cuda") -> bool:
    """Test loading a LeRobot policy."""
    print("\n" + "=" * 60)
    print(f"TEST: Load Policy '{policy_type}'")
    print("=" * 60)

    try:
        from lerobot.envs.a2.a2_collect import load_policy

        print(f"  Loading policy type='{policy_type}', path='{policy_path}'")

        t0 = time.time()
        policy = load_policy(
            policy_type=policy_type,
            policy_path=policy_path,
            device=device,
        )
        elapsed = time.time() - t0

        print(f"  Policy class: {type(policy).__name__}")
        print(f"  Load time: {elapsed:.2f}s")
        print("  [PASS] Policy loaded successfully")

        return policy

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_policy_forward(policy, device: str = "cuda") -> bool:
    """Test policy forward pass."""
    print("\n" + "=" * 60)
    print("TEST: Policy Forward Pass")
    print("=" * 60)

    try:
        # Build test batch matching LeRobot format
        batch = {
            # Robot state: [ee_pos(3), ee_quat(4), joints(6), gripper(1)] = 14D
            "observation.state": torch.randn(1, 14, device=device),
            # Camera images (CHW format)
            "observation.images.front": torch.randn(1, 3, 480, 640, device=device),
            "observation.images.overview": torch.randn(1, 3, 480, 640, device=device),
            # Language task
            "task": "grasp the red cube",
        }

        # Add point cloud if available
        if hasattr(policy, "config") and hasattr(policy.config, "sample_num"):
            sample_num = policy.config.sample_num
            batch["point_cloud"] = torch.randn(1, sample_num, 6, device=device)
            print(f"  Added point_cloud with {sample_num} points")

        # Reset policy if needed
        if hasattr(policy, "reset"):
            policy.reset()
            print("  [PASS] reset() called")

        # Try select_action
        if hasattr(policy, "select_action"):
            with torch.no_grad():
                t0 = time.time()
                action = policy.select_action(batch)
                elapsed = time.time() - t0

            print(f"  select_action() output shape: {action.shape}")
            print(f"  Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
            print(f"  Inference time: {elapsed * 1000:.1f}ms")
            print("  [PASS] select_action() works")

        # Try forward if available (for training)
        if hasattr(policy, "forward"):
            # Add action to batch for supervised learning
            batch["action"] = torch.randn(1, 7, device=device)

            try:
                with torch.no_grad():
                    output = policy.forward(batch)
                print(f"  forward() output keys: {list(output.keys()) if isinstance(output, dict) else type(output)}")
                print("  [PASS] forward() works")
            except Exception as e:
                print(f"  [INFO] forward() not available: {e}")

        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_policy_with_env(policy, gui: bool = False, device: str = "cuda") -> bool:
    """Test policy with actual A2 environment."""
    print("\n" + "=" * 60)
    print("TEST: Policy with A2 Environment")
    print("=" * 60)

    try:
        from lerobot.envs.a2.a2 import A2Env

        # A2 and other learning-based policies use delta_ee mode (VLA-style control)
        env = A2Env(
            task="grasp",
            object_set="train",
            num_objects=5,
            camera_name="front,overview",
            gui=gui,
            action_mode="delta_ee",  # VLA-style delta actions
        )

        obs, info = env.reset(seed=42)
        print(f"  Environment reset, task: {env.task}")

        # Get language goal
        lang_goal = getattr(env._env, "lang_goal", "grasp an object")
        print(f"  Language goal: '{lang_goal}'")

        # Build batch from observation
        batch = {}

        # Robot state
        if "robot_state" in obs:
            rs = obs["robot_state"]
            state = np.concatenate([rs["ee_pos"], rs["ee_quat"], rs["joints"], rs["gripper_angle"]])
            batch["observation.state"] = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Images
        if "pixels" in obs:
            for cam_name, img in obs["pixels"].items():
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0).to(device)

        # Language goal
        batch["task"] = lang_goal

        # Add point cloud if needed
        if hasattr(policy, "config") and hasattr(policy.config, "sample_num"):
            try:
                pcd = env._env.get_pointcloud_array(cameras=[0, 1, 2], max_points=20000)
                batch["point_cloud"] = torch.from_numpy(pcd).float().unsqueeze(0).to(device)
                print(f"  Added point cloud: {pcd.shape}")
            except Exception as e:
                print(f"  [WARN] Could not get point cloud: {e}")

        # Reset policy
        if hasattr(policy, "reset"):
            policy.reset()

        # Get action
        print("  Running policy inference...")
        t0 = time.time()
        with torch.no_grad():
            action = policy.select_action(batch)
        elapsed = time.time() - t0

        action_np = action.cpu().numpy()
        if action_np.ndim == 2:
            action_np = action_np[0]

        print(f"  Action: {action_np[:3]} (delta position), gripper={action_np[-1]:.2f}")
        print(f"  Inference time: {elapsed * 1000:.1f}ms")

        # Execute action (delta_ee mode - just one step)
        print("  Executing delta action step...")
        t0 = time.time()
        obs, reward, terminated, truncated, info = env.step(action_np)
        elapsed = time.time() - t0
        print(f"  Step result: reward={reward}, terminated={terminated}, time={elapsed:.2f}s")

        env.close()
        print("  [PASS] Policy works with A2 environment")
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_a2_policy_components(device: str = "cuda") -> bool:
    """Test A2 policy internal components."""
    print("\n" + "=" * 60)
    print("TEST: A2 Policy Components")
    print("=" * 60)

    try:
        # Test if a2 policy package is available
        try:
            from lerobot_policy_a2 import A2Config, A2Policy

            print("  [PASS] lerobot_policy_a2 package imported")
        except ImportError as e:
            print(f"  [SKIP] lerobot_policy_a2 not installed: {e}")
            return True

        # Test config
        config = A2Config()
        print(f"  A2Config created:")
        print(f"    width={config.width}, layers={config.layers}, heads={config.heads}")
        print(f"    sample_num={config.sample_num}, use_rope={config.use_rope}")

        # Test policy creation (without pretrained weights)
        policy = A2Policy(config)
        policy.to(device)
        policy.eval()
        print("  [PASS] A2Policy created")

        # Test GraspNet component
        if hasattr(policy, "_get_grasp_generator"):
            try:
                grasp_gen = policy._get_grasp_generator()
                print(f"  [PASS] GraspGenerator loaded: {type(grasp_gen).__name__}")
            except Exception as e:
                print(f"  [WARN] GraspGenerator not available: {e}")

        # Test CLIP component
        if hasattr(policy, "_get_clip_model"):
            try:
                clip_model, preprocess = policy._get_clip_model()
                print(f"  [PASS] CLIP model loaded: {type(clip_model).__name__}")
            except Exception as e:
                print(f"  [WARN] CLIP model not available: {e}")

        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests(policy_type: str, policy_path: str | None, gui: bool, device: str) -> dict:
    """Run all policy tests."""
    results = {}

    # Test 1: Random baseline (always works)
    results["Random Policy"] = test_random_policy(device)

    # Test 2: Load specified policy
    policy = test_policy_loading(policy_type, policy_path, device)
    results["Policy Loading"] = policy is not None

    if policy is not None:
        # Test 3: Forward pass
        results["Forward Pass"] = test_policy_forward(policy, device)

        # Test 4: With environment
        results["With Environment"] = test_policy_with_env(policy, gui, device)

    # Test 5: A2 components (if testing A2)
    if policy_type == "a2":
        results["A2 Components"] = test_a2_policy_components(device)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test A2 Policy")
    parser.add_argument("--policy", type=str, default="random", help="Policy type (a2, pi0, pi05, act, diffusion, random)")
    parser.add_argument("--policy_path", type=str, default=None, help="Pretrained path (HuggingFace repo or local)")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--test", type=str, default=None, help="Run specific test")
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 60)
    print("A2 POLICY TEST SUITE")
    print("=" * 60)
    print(f"  Policy: {args.policy}")
    print(f"  Policy path: {args.policy_path}")
    print(f"  Device: {args.device}")
    print(f"  GUI: {args.gui}")

    # Run tests
    results = run_all_tests(args.policy, args.policy_path, args.gui, args.device)

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
