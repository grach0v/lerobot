#!/usr/bin/env python
"""Comprehensive benchmark script for LeRobot policies on A2 tasks.

This script runs benchmark evaluations for grasp, place, and pick-and-place tasks
using different policies. It supports A2-style pose policies and VLA-style
delta action policies (like Pi0/Pi05).

Supported policies:
- a2: A2 grasp/place pose selection policy
- pi0, pi05: VLA policies using delta_ee actions
- act, diffusion: Other LeRobot policies
- random: Random baseline

Usage:
    # Quick smoke test
    uv run python scripts/benchmark_a2.py --policy random --task grasp --num_episodes 3

    # Benchmark A2 policy on grasp task
    uv run python scripts/benchmark_a2.py --policy a2 --task grasp --num_episodes 15

    # Benchmark Pi05 VLA policy
    uv run python scripts/benchmark_a2.py --policy pi05 --policy_path lerobot/pi05 --task grasp

    # Full benchmark with all tasks
    uv run python scripts/benchmark_a2.py --policy a2 --task all --num_episodes 15

    # Benchmark on unseen objects
    uv run python scripts/benchmark_a2.py --policy a2 --task grasp --object_set test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Import shared utilities from a2_collect to avoid code duplication
from lerobot.envs.a2.a2_collect import (
    cleanup_gpu_memory,
    load_policy,
    _get_observation_data,
    _predict_grasp_pose,
    _obs_to_batch,
)


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark."""

    task: str
    object_set: str
    policy_type: str
    num_episodes: int
    num_successes: int = 0
    total_steps: int = 0
    total_time: float = 0.0
    episode_results: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.num_successes / self.num_episodes if self.num_episodes > 0 else 0.0

    @property
    def avg_steps(self) -> float:
        return self.total_steps / self.num_episodes if self.num_episodes > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.num_episodes if self.num_episodes > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "object_set": self.object_set,
            "policy_type": self.policy_type,
            "num_episodes": self.num_episodes,
            "num_successes": self.num_successes,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "avg_time_per_episode": self.avg_time,
            "total_time": self.total_time,
        }


def create_policy(policy_type: str, policy_path: str | None, device: str, direct_grounding: bool = True) -> Any:
    """Create a policy instance using shared load_policy from a2_collect."""
    policy = load_policy(policy_type=policy_type, policy_path=policy_path, device=device)

    # Set direct_grounding mode for A2 policy
    if hasattr(policy, "config") and hasattr(policy.config, "direct_grounding"):
        policy.config.direct_grounding = direct_grounding
        print(f"  Direct grounding mode: {direct_grounding}")

    return policy


def get_action_mode(policy_type: str) -> str:
    """Determine action mode based on policy type."""
    # A2 policy uses pose mode: outputs full 7D grasp/place poses, env executes full trajectory
    # VLA policies (pi0, pi05, act, diffusion) use delta_ee: outputs delta actions each step
    pose_policies = {"a2", "oracle", "scripted"}

    if policy_type in pose_policies:
        return "pose"
    else:
        # VLA-style policies use delta_ee
        return "delta_ee"


def _predict_grasp_pose_for_benchmark(policy, env, lang_goal: str, device: str) -> np.ndarray | None:
    """Get grasp pose from A2 policy.

    Uses shared _predict_grasp_pose and _get_observation_data from a2_collect.
    """
    try:
        # Get observation data using shared function
        color_images, depth_images, pcd = _get_observation_data(env)

        # Reset policy before prediction
        if hasattr(policy, "reset"):
            policy.reset()

        # Use shared prediction function
        action, info = _predict_grasp_pose(
            policy, color_images, depth_images, pcd, lang_goal, device, env=env
        )
        return action

    except Exception as e:
        print(f"  Warning: Grasp prediction failed: {e}")
        import traceback
        traceback.print_exc()

    return None


def run_pose_mode_episode(
    env,
    policy,
    lang_goal: str,
    max_attempts: int,
    device: str,
) -> dict:
    """Run single episode with A2-style pose mode (full grasp execution)."""
    episode_result = {
        "success": False,
        "steps": 0,
        "reward": 0.0,
        "time": 0.0,
    }

    t0 = time.time()

    for attempt in range(max_attempts):
        # Check if target objects still in workspace
        if hasattr(env._env, "target_obj_ids") and env._env.target_obj_ids:
            bounds = env._env.bounds
            all_out = True
            for obj_id in env._env.target_obj_ids:
                pos, _, _ = env._env.obj_info(obj_id)
                if bounds[0][0] <= pos[0] <= bounds[0][1] and bounds[1][0] <= pos[1] <= bounds[1][1]:
                    all_out = False
                    break
            if all_out:
                break

        # Get grasp pose from policy
        grasp_pose = _predict_grasp_pose_for_benchmark(policy, env, lang_goal, device)

        if grasp_pose is None:
            print(f"  Attempt {attempt + 1}: No grasp pose generated")
            continue

        print(f"  Attempt {attempt + 1}: Grasp at ({grasp_pose[0]:.3f}, {grasp_pose[1]:.3f}, {grasp_pose[2]:.3f})")

        # Execute grasp using environment's grasp primitive
        grasp_success, grasped_id, _ = env._env.grasp(grasp_pose)

        episode_result["steps"] += 1

        if grasp_success and grasped_id in env._env.target_obj_ids:
            episode_result["success"] = True
            episode_result["reward"] = 1.0
            print(f"    SUCCESS: Grasped target object {grasped_id}")
            break
        elif grasp_success:
            print(f"    Grasped wrong object {grasped_id}, retrying...")
            env._env.place_out_of_workspace()
        else:
            print(f"    Grasp failed, retrying...")

        env._env.go_home()

    episode_result["time"] = time.time() - t0
    return episode_result


def run_delta_mode_episode(
    env,
    policy,
    lang_goal: str,
    max_steps: int,
    device: str,
) -> dict:
    """Run single episode with VLA-style delta action mode."""
    episode_result = {
        "success": False,
        "steps": 0,
        "reward": 0.0,
        "time": 0.0,
    }

    t0 = time.time()
    obs = env._get_observation()

    # Reset policy
    if hasattr(policy, "reset"):
        policy.reset()

    for step in range(max_steps):
        # Build batch
        batch = _obs_to_batch(obs, lang_goal, device, env=None)  # No pointcloud for delta-ee mode

        # Get action
        with torch.no_grad():
            action = policy.select_action(batch)

        action_np = action.cpu().numpy()
        if action_np.ndim == 2:
            action_np = action_np[0]

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action_np)

        episode_result["steps"] += 1
        episode_result["reward"] += reward

        if info.get("is_success", False):
            episode_result["success"] = True
            break

        if terminated or truncated:
            break

    episode_result["time"] = time.time() - t0
    return episode_result


def run_benchmark(
    policy,
    policy_type: str,
    task: str,
    object_set: str,
    num_episodes: int,
    max_attempts: int,
    seed: int,
    device: str,
    gui: bool = False,
    verbose: bool = True,
    test_case_dir: str | None = None,
    test_case: str | None = None,
    env_reset_interval: int = 10,
) -> BenchmarkMetrics:
    """Run benchmark for a specific task.

    Args:
        test_case_dir: Directory with predefined test cases (original A2 format)
        test_case: Specific test case file to run (if None, runs all in directory)
        env_reset_interval: Recreate environment every N episodes to prevent VRAM leaks (0 to disable)
    """
    from lerobot.envs.a2.a2 import A2Env

    # Determine action mode
    action_mode = get_action_mode(policy_type)

    # Get list of test case files if using predefined cases
    test_case_files = []
    if test_case_dir is not None:
        if test_case is not None:
            # Single test case
            test_case_files = [os.path.join(test_case_dir, test_case)]
        else:
            # All test cases in directory
            import glob
            test_case_files = sorted(glob.glob(os.path.join(test_case_dir, "*.txt")))
            if not test_case_files:
                print(f"Warning: No test case files found in {test_case_dir}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {task.upper()}")
        print(f"{'='*60}")
        print(f"  Policy: {policy_type} (action_mode={action_mode})")
        print(f"  Object set: {object_set}")
        if test_case_dir:
            print(f"  Test cases: {len(test_case_files)} files from {test_case_dir}")
        print(f"  Episodes per case: {num_episodes}")
        print(f"  Max attempts/steps: {max_attempts}")

    # Create environment
    cameras = "front,overview,gripper" if action_mode == "delta_ee" else "front,overview"
    env = A2Env(
        task=task if task != "pickplace" else "pick_and_place",
        object_set=object_set,
        num_objects=8,
        camera_name=cameras,
        gui=gui,
        action_mode=action_mode,
        episode_length=500 if action_mode == "delta_ee" else max_attempts,
    )

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Determine total episodes
    if test_case_files:
        total_episodes = len(test_case_files) * num_episodes
    else:
        total_episodes = num_episodes

    # Initialize metrics
    metrics = BenchmarkMetrics(
        task=task,
        object_set=object_set,
        policy_type=policy_type,
        num_episodes=total_episodes,
    )

    # Helper to create environment
    def create_env():
        return A2Env(
            task=task if task != "pickplace" else "pick_and_place",
            object_set=object_set,
            num_objects=8,
            camera_name=cameras,
            gui=gui,
            action_mode=action_mode,
            episode_length=500 if action_mode == "delta_ee" else max_attempts,
        )

    # Run episodes
    episode_idx = 0
    if test_case_files:
        # Run with predefined test cases
        for case_file in test_case_files:
            case_name = os.path.basename(case_file)
            if verbose:
                print(f"\n  Test case: {case_name}")

            for ep in range(num_episodes):
                episode_idx += 1

                # Periodically recreate environment to prevent VRAM/EGL memory leak
                if (
                    env_reset_interval > 0
                    and episode_idx > 1
                    and (episode_idx - 1) % env_reset_interval == 0
                ):
                    if verbose:
                        print("  [Memory cleanup] Recreating environment to release VRAM...")
                    env.close()
                    cleanup_gpu_memory()
                    env = create_env()

                # Reset environment fully first (this sets up robot, workspace, etc.)
                env.reset(seed=seed + episode_idx)

                # Clear any randomly generated objects
                for obj_id in list(env._env.obj_ids.get("rigid", [])):
                    try:
                        import pybullet as pb
                        pb.removeBody(obj_id)
                    except Exception:
                        pass
                env._env.obj_ids["rigid"] = []
                env._env.target_obj_ids = []
                env._env.obj_labels = {}
                env._env.obj_dirs = {}

                # Load objects from test case file
                body_ids, success, lang_goal = env._env.add_objects_from_file(case_file)

                if not success:
                    print(f"    Failed to load test case {case_file}")
                    continue

                # Let objects settle
                import pybullet as pb
                for _ in range(100):
                    pb.stepSimulation()

                if verbose:
                    print(f"\n    Episode {ep + 1}/{num_episodes} (case {case_name}): {lang_goal[:50]}...")

                # Run episode
                if action_mode == "pose":
                    result = run_pose_mode_episode(env, policy, lang_goal, max_attempts, device)
                else:
                    result = run_delta_mode_episode(env, policy, lang_goal, max_attempts * 50, device)

                # Update metrics
                if result["success"]:
                    metrics.num_successes += 1
                metrics.total_steps += result["steps"]
                metrics.total_time += result["time"]
                metrics.episode_results.append(result)

                if verbose:
                    status = "SUCCESS" if result["success"] else "FAIL"
                    print(f"      [{status}] steps={result['steps']}, reward={result['reward']:.2f}, time={result['time']:.2f}s")

                # Cleanup GPU memory after each episode
                cleanup_gpu_memory()

            # Print per-case summary
            case_successes = sum(1 for r in metrics.episode_results[-num_episodes:] if r["success"])
            if verbose:
                print(f"    Case result: {case_successes}/{num_episodes} ({case_successes/num_episodes*100:.0f}%)")
    else:
        # Run with random scene generation
        for episode in range(num_episodes):
            # Periodically recreate environment to prevent VRAM/EGL memory leak
            if (
                env_reset_interval > 0
                and episode > 0
                and episode % env_reset_interval == 0
            ):
                if verbose:
                    print("  [Memory cleanup] Recreating environment to release VRAM...")
                env.close()
                cleanup_gpu_memory()
                env = create_env()

            obs, info = env.reset(seed=seed + episode)

            # Get language goal
            lang_goal = getattr(env._env, "lang_goal", None) or "grasp an object"

            if verbose:
                print(f"\n  Episode {episode + 1}/{num_episodes}: {lang_goal[:50]}...")

            # Run episode
            if action_mode == "pose":
                result = run_pose_mode_episode(env, policy, lang_goal, max_attempts, device)
            else:
                result = run_delta_mode_episode(env, policy, lang_goal, max_attempts * 50, device)

            # Update metrics
            if result["success"]:
                metrics.num_successes += 1
            metrics.total_steps += result["steps"]
            metrics.total_time += result["time"]
            metrics.episode_results.append(result)

            if verbose:
                status = "SUCCESS" if result["success"] else "FAIL"
                print(f"    [{status}] steps={result['steps']}, reward={result['reward']:.2f}, time={result['time']:.2f}s")

            # Cleanup GPU memory after each episode
            cleanup_gpu_memory()

    env.close()

    if verbose:
        print(f"\n  Results: {metrics.num_successes}/{metrics.num_episodes} success ({metrics.success_rate*100:.1f}%)")
        print(f"  Avg steps: {metrics.avg_steps:.1f}, Avg time: {metrics.avg_time:.2f}s")

    return metrics


def run_all_tasks(
    policy,
    policy_type: str,
    object_set: str,
    num_episodes: int,
    max_attempts: int,
    seed: int,
    device: str,
    gui: bool = False,
    env_reset_interval: int = 10,
) -> dict[str, BenchmarkMetrics]:
    """Run benchmarks for all tasks."""
    tasks = ["grasp", "place", "pickplace"]
    results = {}

    for task in tasks:
        try:
            metrics = run_benchmark(
                policy=policy,
                policy_type=policy_type,
                task=task,
                object_set=object_set,
                num_episodes=num_episodes,
                max_attempts=max_attempts,
                seed=seed,
                device=device,
                gui=gui,
                env_reset_interval=env_reset_interval,
            )
            results[task] = metrics
        except Exception as e:
            print(f"  [ERROR] Task '{task}' failed: {e}")
            import traceback

            traceback.print_exc()

    return results


def save_results(results: dict[str, BenchmarkMetrics], output_dir: str, policy_type: str):
    """Save benchmark results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{policy_type}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    data = {
        "timestamp": timestamp,
        "policy_type": policy_type,
        "tasks": {task: metrics.to_dict() for task, metrics in results.items()},
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def print_summary(results: dict[str, BenchmarkMetrics]):
    """Print benchmark summary table."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"{'Task':<15} {'Success Rate':>15} {'Avg Steps':>12} {'Avg Time':>12}")
    print("-" * 60)

    for task, metrics in results.items():
        print(f"{task:<15} {metrics.success_rate*100:>14.1f}% {metrics.avg_steps:>12.1f} {metrics.avg_time:>11.2f}s")

    print("-" * 60)

    # Overall
    total_episodes = sum(m.num_episodes for m in results.values())
    total_successes = sum(m.num_successes for m in results.values())
    overall_rate = total_successes / total_episodes if total_episodes > 0 else 0

    print(f"{'Overall':<15} {overall_rate*100:>14.1f}% ({total_successes}/{total_episodes} episodes)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LeRobot Policies on A2 Tasks")

    # Policy arguments
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        help="Policy type (a2, pi0, pi05, act, diffusion, random)",
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default=None,
        help="Pretrained path (HuggingFace repo or local path)",
    )

    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        default="grasp",
        choices=["grasp", "place", "pickplace", "all"],
        help="Task to benchmark",
    )
    parser.add_argument(
        "--object_set",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Object set (train=seen, test=unseen)",
    )

    # Benchmark arguments
    parser.add_argument("--num_episodes", type=int, default=15, help="Number of episodes per task")
    parser.add_argument("--max_attempts", type=int, default=8, help="Max attempts per episode (pose mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--env_reset_interval",
        type=int,
        default=10,
        help="Recreate environment every N episodes to prevent VRAM leaks (0 to disable)",
    )

    # Test case arguments (for matching original A2 benchmark)
    parser.add_argument(
        "--test_case_dir",
        type=str,
        default=None,
        help="Directory with predefined test cases (e.g., /path/to/A2_new/testing_cases/grasp_testing_cases/seen)",
    )
    parser.add_argument(
        "--test_case",
        type=str,
        default=None,
        help="Specific test case file to run (e.g., case00-round.txt)",
    )

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")

    # Other arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument(
        "--direct_grounding",
        action="store_true",
        default=True,
        help="Use direct CLIP-based grounding (default: True). Set --no-direct_grounding to use learned networks.",
    )
    parser.add_argument(
        "--no-direct_grounding",
        dest="direct_grounding",
        action="store_false",
        help="Use learned ViLG networks instead of direct CLIP grounding",
    )

    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 60)
    print("A2 BENCHMARK")
    print("=" * 60)
    print(f"  Policy: {args.policy}")
    print(f"  Policy path: {args.policy_path}")
    print(f"  Task: {args.task}")
    print(f"  Object set: {args.object_set}")
    print(f"  Episodes: {args.num_episodes}")
    if args.test_case_dir:
        print(f"  Test case dir: {args.test_case_dir}")
        if args.test_case:
            print(f"  Test case: {args.test_case}")
    print(f"  Device: {args.device}")

    # Load policy
    print("\nLoading policy...")
    t0 = time.time()
    policy = create_policy(args.policy, args.policy_path, args.device, args.direct_grounding)
    print(f"  Policy loaded in {time.time() - t0:.2f}s")
    print(f"  Policy class: {type(policy).__name__}")

    # Run benchmarks
    if args.task == "all":
        results = run_all_tasks(
            policy=policy,
            policy_type=args.policy,
            object_set=args.object_set,
            num_episodes=args.num_episodes,
            max_attempts=args.max_attempts,
            seed=args.seed,
            device=args.device,
            gui=args.gui,
            env_reset_interval=args.env_reset_interval,
        )
    else:
        metrics = run_benchmark(
            policy=policy,
            policy_type=args.policy,
            task=args.task,
            object_set=args.object_set,
            num_episodes=args.num_episodes,
            max_attempts=args.max_attempts,
            seed=args.seed,
            device=args.device,
            gui=args.gui,
            verbose=not args.quiet,
            test_case_dir=args.test_case_dir,
            test_case=args.test_case,
            env_reset_interval=args.env_reset_interval,
        )
        results = {args.task: metrics}

    # Print summary
    print_summary(results)

    # Save results
    if args.save:
        save_results(results, args.output_dir, args.policy)

    # Return exit code based on success rate
    overall_success = sum(m.num_successes for m in results.values())
    overall_total = sum(m.num_episodes for m in results.values())
    success_rate = overall_success / overall_total if overall_total > 0 else 0

    if success_rate >= 0.5:
        print("\n  Benchmark PASSED (success rate >= 50%)")
        sys.exit(0)
    elif args.policy == "random" and success_rate >= 0:
        print("\n  Random baseline completed (no pass/fail criteria)")
        sys.exit(0)
    else:
        print("\n  Benchmark FAILED (success rate < 50%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
