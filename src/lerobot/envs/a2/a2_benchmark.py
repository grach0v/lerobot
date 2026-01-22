#!/usr/bin/env python
"""A2 Benchmark Evaluation Module.

This module provides a generalized benchmark evaluation framework that works
with any policy implementing the GraspPlacePolicy interface.

Usage:
    # Evaluate grasp task
    python -m lerobot.envs.a2_benchmark --task grasp --policy a2 --model_path model.pth

    # Evaluate pick-and-place on unseen objects
    python -m lerobot.envs.a2_benchmark --task pickplace --object_set test --policy a2
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np
import torch


class GraspPlacePolicy(Protocol):
    """Protocol for grasp/place policies.

    Any policy can be used with the benchmark as long as it implements
    these methods.
    """

    def select_grasp_action(
        self,
        color_images: dict[str, np.ndarray],
        depth_images: dict[str, np.ndarray],
        point_cloud: np.ndarray,
        lang_goal: str,
        candidate_poses: Optional[np.ndarray] = None,
        **kwargs
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select grasp action. Returns (7D pose, info dict)."""
        ...

    def select_place_action(
        self,
        color_images: dict[str, np.ndarray],
        depth_images: dict[str, np.ndarray],
        point_cloud: np.ndarray,
        lang_goal: str,
        candidate_poses: Optional[np.ndarray] = None,
        grasped_object_id: Optional[int] = None,
        **kwargs
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select place action. Returns (7D pose, info dict)."""
        ...


@dataclass
class BenchmarkResult:
    """Results from a benchmark evaluation."""
    task: str
    object_set: str
    test_case: str
    num_episodes: int
    success_rate: float
    avg_steps: float
    avg_reward: float
    success_steps: float  # Average steps for successful episodes
    episode_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "object_set": self.object_set,
            "test_case": self.test_case,
            "num_episodes": self.num_episodes,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "avg_reward": self.avg_reward,
            "success_steps": self.success_steps,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    task: str = "grasp"  # grasp, place, pickplace
    object_set: str = "train"  # train, test
    testing_case_dir: str = ""  # Auto-detected from task/object_set if empty
    testing_case: Optional[str] = None  # Specific test case, or None for all
    num_episodes: int = 15
    max_steps: int = 8
    seed: int = 1234
    gui: bool = False
    visualize: bool = False
    save_results: bool = True
    results_dir: str = "benchmark_results"

    def __post_init__(self):
        """Auto-detect testing_case_dir if not provided."""
        if not self.testing_case_dir:
            from .a2 import get_testing_cases_path
            self.testing_case_dir = str(get_testing_cases_path(self.task, self.object_set))


class A2Benchmark:
    """Generalized benchmark evaluator for A2 environment.

    This evaluator is decoupled from specific policy implementations.
    Any policy implementing the GraspPlacePolicy protocol can be evaluated.
    """

    def __init__(
        self,
        policy: GraspPlacePolicy,
        config: BenchmarkConfig,
        action_generator: Optional[Callable] = None,
    ):
        """Initialize benchmark.

        Args:
            policy: Policy implementing GraspPlacePolicy protocol
            config: Benchmark configuration
            action_generator: Optional callable to generate candidate action poses.
                             If None, policy must generate poses from scratch.
                             Signature: (env, task) -> candidate_poses (N, 7)
        """
        self.policy = policy
        self.config = config
        self.action_generator = action_generator

        # Initialize environment
        from .a2_sim import Environment
        self.env = Environment(gui=config.gui, object_set=config.object_set)
        self.env.seed(config.seed)

        # Set random seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Results storage
        self.results: list[BenchmarkResult] = []

    def get_observations(self, cameras: list[str] = None) -> dict[str, Any]:
        """Get current observations from environment.

        Returns dict with:
            - color_images: Dict[camera_name, np.ndarray (H, W, 3)]
            - depth_images: Dict[camera_name, np.ndarray (H, W)]
            - point_cloud: np.ndarray (N, 6) with xyz + rgb
            - robot_state: Dict with ee_pos, ee_quat, joints, gripper
        """
        cameras = cameras or ["front", "left", "right"]

        # Get multi-view point cloud and images
        pcd, color_images, depth_images = self.env.get_multi_view_pointcloud(
            cameras=[0, 1, 2]  # front, left, right camera indices
        )

        # Convert Open3D point cloud to numpy with colors
        points = np.asarray(pcd.points)
        if len(pcd.colors) > 0:
            colors = np.asarray(pcd.colors) * 255.0  # Scale to 0-255
            point_cloud = np.concatenate([points, colors], axis=1).astype(np.float32)
        else:
            # No colors, pad with zeros
            point_cloud = np.concatenate([points, np.zeros_like(points)], axis=1).astype(np.float32)

        # Get robot state
        robot_state = self.env.get_robot_state()

        return {
            "color_images": color_images,
            "depth_images": depth_images,
            "point_cloud": point_cloud,
            "robot_state": robot_state,
        }

    def evaluate_grasp(self, test_case_path: str) -> BenchmarkResult:
        """Evaluate grasp task on a specific test case."""
        episode_results = []
        successes = []
        steps_list = []
        rewards_list = []

        for episode in range(self.config.num_episodes):
            # Reset and load test case
            self.env.reset(workspace="raw")
            success_load, lang_goal = self.env.add_object_push_from_file(test_case_path)

            if not success_load:
                print(f"Failed to load test case: {test_case_path}")
                continue

            print(f"  Episode {episode + 1}: {lang_goal}")

            episode_reward = 0.0
            episode_success = False
            steps = 0

            for step in range(self.config.max_steps):
                # Check if target objects still in workspace
                if not self._check_targets_in_workspace():
                    print("    Target objects out of workspace")
                    break

                # Get observations
                obs = self.get_observations()

                # Generate candidate poses if action generator provided
                candidate_poses = None
                if self.action_generator:
                    candidate_poses = self.action_generator(self.env, "grasp")

                # Get action from policy
                action, info = self.policy.select_grasp_action(
                    color_images=obs["color_images"],
                    depth_images=obs["depth_images"],
                    point_cloud=obs["point_cloud"],
                    lang_goal=lang_goal,
                    candidate_poses=candidate_poses,
                )

                # Execute grasp - pass 7D array directly
                grasp_success, grasped_id, _ = self.env.grasp(action[:7])

                reward = 1.0 if grasp_success and grasped_id in self.env.target_obj_ids else 0.0
                episode_reward += reward
                steps += 1

                print(f"    Step {step + 1}: grasp_success={grasp_success}, reward={reward}")

                if reward > 0:
                    episode_success = True
                    break

            successes.append(episode_success)
            steps_list.append(steps)
            rewards_list.append(episode_reward)
            episode_results.append({
                "episode": episode,
                "success": episode_success,
                "steps": steps,
                "reward": episode_reward,
            })

        # Compute metrics
        success_steps = [s for s, succ in zip(steps_list, successes) if succ]

        return BenchmarkResult(
            task="grasp",
            object_set=self.config.object_set,
            test_case=test_case_path,
            num_episodes=self.config.num_episodes,
            success_rate=sum(successes) / len(successes) if successes else 0.0,
            avg_steps=sum(steps_list) / len(steps_list) if steps_list else 0.0,
            avg_reward=sum(rewards_list) / len(rewards_list) if rewards_list else 0.0,
            success_steps=sum(success_steps) / len(success_steps) if success_steps else 0.0,
            episode_results=episode_results,
        )

    def evaluate_place(self, test_case_path: str) -> BenchmarkResult:
        """Evaluate place task on a specific test case."""
        episode_results = []
        successes = []
        steps_list = []
        rewards_list = []

        for episode in range(self.config.num_episodes):
            # Reset and load test case
            self.env.reset(workspace="raw")
            success_load, lang_goal = self.env.add_object_push_from_place_file(test_case_path)

            if not success_load:
                print(f"Failed to load test case: {test_case_path}")
                continue

            print(f"  Episode {episode + 1}: {lang_goal}")

            episode_reward = 0.0
            episode_success = False
            steps = 0

            for step in range(self.config.max_steps):
                # Check if reference objects still in workspace
                if not self._check_references_in_workspace():
                    print("    Reference objects out of workspace")
                    break

                # Get observations
                obs = self.get_observations()

                # Generate candidate poses if action generator provided
                candidate_poses = None
                if self.action_generator:
                    candidate_poses = self.action_generator(self.env, "place")

                # Get action from policy
                action, info = self.policy.select_place_action(
                    color_images=obs["color_images"],
                    depth_images=obs["depth_images"],
                    point_cloud=obs["point_cloud"],
                    lang_goal=lang_goal,
                    candidate_poses=candidate_poses,
                )

                # For place evaluation, we check if the pose is valid
                # (actual execution may not be needed for benchmark)
                pose = (tuple(action[:3]), tuple(action[3:7]))

                # Check if place is valid using environment's place validation
                # This is a simplified check - actual implementation depends on task
                place_success = self._validate_place(pose)

                reward = 1.0 if place_success else 0.0
                episode_reward += reward
                steps += 1

                print(f"    Step {step + 1}: place_valid={place_success}, reward={reward}")

                if reward > 0:
                    episode_success = True
                    break

            successes.append(episode_success)
            steps_list.append(steps)
            rewards_list.append(episode_reward)
            episode_results.append({
                "episode": episode,
                "success": episode_success,
                "steps": steps,
                "reward": episode_reward,
            })

        success_steps = [s for s, succ in zip(steps_list, successes) if succ]

        return BenchmarkResult(
            task="place",
            object_set=self.config.object_set,
            test_case=test_case_path,
            num_episodes=self.config.num_episodes,
            success_rate=sum(successes) / len(successes) if successes else 0.0,
            avg_steps=sum(steps_list) / len(steps_list) if steps_list else 0.0,
            avg_reward=sum(rewards_list) / len(rewards_list) if rewards_list else 0.0,
            success_steps=sum(success_steps) / len(success_steps) if success_steps else 0.0,
            episode_results=episode_results,
        )

    def evaluate_pickplace(self, test_case_path: str) -> BenchmarkResult:
        """Evaluate pick-and-place task on a specific test case."""
        episode_results = []
        successes = []
        grasp_successes = []
        place_successes = []
        steps_list = []
        rewards_list = []

        for episode in range(self.config.num_episodes):
            # Reset and load grasp scene
            self.env.reset(workspace="extend")
            success_load, grasp_lang, place_lang = self.env.add_object_push_from_pickplace_file(
                test_case_path, mode="grasp"
            )

            if not success_load:
                continue

            print(f"  Episode {episode + 1}: grasp={grasp_lang}")

            episode_reward = 0.0
            grasp_success = False
            place_success = False
            grasped_id = None
            steps = 0

            # Grasp phase
            for step in range(self.config.max_steps):
                if not self._check_targets_in_workspace():
                    break

                obs = self.get_observations()
                candidate_poses = None
                if self.action_generator:
                    candidate_poses = self.action_generator(self.env, "grasp")

                action, info = self.policy.select_grasp_action(
                    color_images=obs["color_images"],
                    depth_images=obs["depth_images"],
                    point_cloud=obs["point_cloud"],
                    lang_goal=grasp_lang,
                    candidate_poses=candidate_poses,
                )

                # Execute grasp - pass 7D array directly
                success, grasped_id, _ = self.env.grasp(action[:7], follow_place=True)

                steps += 1

                if success and grasped_id in self.env.target_obj_ids:
                    grasp_success = True
                    episode_reward += 2.0
                    break
                elif success:
                    # Grasped wrong object, place it out of workspace
                    self.env.place_out_of_workspace()
                    episode_reward -= 1.0

            # Place phase (only if grasp succeeded)
            if grasp_success:
                # Load place scene
                _, _, place_lang = self.env.add_object_push_from_pickplace_file(
                    test_case_path, mode="place"
                )
                print(f"    Place: {place_lang}")

                obs = self.get_observations()
                candidate_poses = None
                if self.action_generator:
                    candidate_poses = self.action_generator(self.env, "place")

                action, info = self.policy.select_place_action(
                    color_images=obs["color_images"],
                    depth_images=obs["depth_images"],
                    point_cloud=obs["point_cloud"],
                    lang_goal=place_lang,
                    candidate_poses=candidate_poses,
                    grasped_object_id=grasped_id,
                )

                pose = (tuple(action[:3]), tuple(action[3:7]))
                place_success = self._validate_place(pose)

                if place_success:
                    episode_reward += 2.0

            episode_success = grasp_success and place_success
            successes.append(episode_success)
            grasp_successes.append(grasp_success)
            place_successes.append(place_success)
            steps_list.append(steps)
            rewards_list.append(episode_reward)
            episode_results.append({
                "episode": episode,
                "success": episode_success,
                "grasp_success": grasp_success,
                "place_success": place_success,
                "steps": steps,
                "reward": episode_reward,
            })

            print(f"    Result: grasp={grasp_success}, place={place_success}")

        success_steps = [s for s, succ in zip(steps_list, successes) if succ]

        result = BenchmarkResult(
            task="pickplace",
            object_set=self.config.object_set,
            test_case=test_case_path,
            num_episodes=self.config.num_episodes,
            success_rate=sum(successes) / len(successes) if successes else 0.0,
            avg_steps=sum(steps_list) / len(steps_list) if steps_list else 0.0,
            avg_reward=sum(rewards_list) / len(rewards_list) if rewards_list else 0.0,
            success_steps=sum(success_steps) / len(success_steps) if success_steps else 0.0,
            episode_results=episode_results,
        )

        # Add extra metrics
        result.grasp_success_rate = sum(grasp_successes) / len(grasp_successes) if grasp_successes else 0.0
        result.place_success_rate = sum(place_successes) / len(place_successes) if place_successes else 0.0

        return result

    def run(self) -> list[BenchmarkResult]:
        """Run benchmark evaluation on all test cases."""
        # Get test case files
        if self.config.testing_case:
            test_cases = [os.path.join(self.config.testing_case_dir, self.config.testing_case)]
        else:
            test_cases = sorted([
                os.path.join(self.config.testing_case_dir, f)
                for f in os.listdir(self.config.testing_case_dir)
                if f.endswith(".txt")
            ])

        print(f"Running benchmark: task={self.config.task}, object_set={self.config.object_set}")
        print(f"Test cases: {len(test_cases)}")

        results = []
        for test_case in test_cases:
            print(f"\nTest case: {os.path.basename(test_case)}")

            if self.config.task == "grasp":
                result = self.evaluate_grasp(test_case)
            elif self.config.task == "place":
                result = self.evaluate_place(test_case)
            elif self.config.task == "pickplace":
                result = self.evaluate_pickplace(test_case)
            else:
                raise ValueError(f"Unknown task: {self.config.task}")

            results.append(result)
            print(f"  Success rate: {result.success_rate * 100:.1f}%")

        # Save results
        if self.config.save_results:
            self._save_results(results)

        self.results = results
        return results

    def _check_targets_in_workspace(self) -> bool:
        """Check if target objects are still in workspace."""
        from .a2_constants import WORKSPACE_LIMITS

        for obj_id in self.env.target_obj_ids:
            pos, _, _ = self.env.obj_info(obj_id)
            if (pos[0] < WORKSPACE_LIMITS[0, 0] or pos[0] > WORKSPACE_LIMITS[0, 1] or
                pos[1] < WORKSPACE_LIMITS[1, 0] or pos[1] > WORKSPACE_LIMITS[1, 1]):
                return False
        return True

    def _check_references_in_workspace(self) -> bool:
        """Check if reference objects are still in workspace."""
        from .a2_constants import WORKSPACE_LIMITS

        for obj_id in getattr(self.env, 'reference_obj_ids', []):
            pos, _, _ = self.env.obj_info(obj_id)
            if (pos[0] < WORKSPACE_LIMITS[0, 0] or pos[0] > WORKSPACE_LIMITS[0, 1] or
                pos[1] < WORKSPACE_LIMITS[1, 0] or pos[1] > WORKSPACE_LIMITS[1, 1]):
                return False
        return True

    def _validate_place(self, pose: tuple) -> bool:
        """Validate if a place pose is correct.

        This is a simplified validation. In practice, you'd check against
        the ground truth valid place regions.
        """
        # For now, just check if pose is in workspace
        from .a2_constants import WORKSPACE_LIMITS

        pos = pose[0]
        return (WORKSPACE_LIMITS[0, 0] <= pos[0] <= WORKSPACE_LIMITS[0, 1] and
                WORKSPACE_LIMITS[1, 0] <= pos[1] <= WORKSPACE_LIMITS[1, 1])

    def _save_results(self, results: list[BenchmarkResult]):
        """Save benchmark results to file."""
        os.makedirs(self.config.results_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.task}_{self.config.object_set}_{timestamp}.json"
        filepath = os.path.join(self.config.results_dir, filename)

        # Compute aggregate metrics
        aggregate = {
            "task": self.config.task,
            "object_set": self.config.object_set,
            "num_test_cases": len(results),
            "overall_success_rate": sum(r.success_rate for r in results) / len(results),
            "overall_avg_steps": sum(r.avg_steps for r in results) / len(results),
            "test_cases": [r.to_dict() for r in results],
        }

        with open(filepath, "w") as f:
            json.dump(aggregate, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        print(f"Overall success rate: {aggregate['overall_success_rate'] * 100:.1f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="A2 Benchmark Evaluation")

    parser.add_argument("--task", type=str, default="grasp",
                       choices=["grasp", "place", "pickplace"])
    parser.add_argument("--object_set", type=str, default="train",
                       choices=["train", "test"])
    parser.add_argument("--testing_case_dir", type=str,
                       default="testing_cases/grasp_testing_cases/seen")
    parser.add_argument("--testing_case", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--results_dir", type=str, default="benchmark_results")

    # Policy arguments
    parser.add_argument("--policy", type=str, default="random",
                       choices=["random", "a2", "custom"])
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--hf_repo", type=str, default="",
                       help="HuggingFace repo ID (e.g., dgrachev/a2_pretrained)")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create config - use auto-detection if testing_case_dir is default
    testing_case_dir = args.testing_case_dir
    if testing_case_dir.startswith("testing_cases/"):
        # Use auto-detection from assets
        testing_case_dir = ""

    config = BenchmarkConfig(
        task=args.task,
        object_set=args.object_set,
        testing_case_dir=testing_case_dir,
        testing_case=args.testing_case,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        gui=args.gui,
        results_dir=args.results_dir,
    )

    # Create policy
    if args.policy == "random":
        from .a2_collect import RandomPolicy
        policy = RandomPolicy(device=args.device)
    elif args.policy == "a2":
        from .a2_collect import create_a2_policy
        policy = create_a2_policy(
            model_path=args.model_path,
            hf_repo=args.hf_repo,
            device=args.device
        )
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    # Run benchmark
    benchmark = A2Benchmark(policy=policy, config=config)
    results = benchmark.run()


if __name__ == "__main__":
    main()
