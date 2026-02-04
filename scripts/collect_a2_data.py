#!/usr/bin/env python
"""A2 Data Collection Script.

This script collects manipulation data using the A2 environment with the same
episode logic as the benchmark. Supports grasp, place, and pickplace tasks.

Usage:
    # Collect 100 successful pickplace episodes with seen objects (random scenes)
    python scripts/collect_a2_data.py --task pickplace --episodes 100 --object_set seen

    # Collect using test cases for better success rate
    python scripts/collect_a2_data.py --task pickplace --episodes 100 --use_test_cases

    # Collect grasp data with both seen and unseen objects
    python scripts/collect_a2_data.py --task grasp --episodes 500 --object_set both

    # Collect to specific output directory
    python scripts/collect_a2_data.py --task pickplace --episodes 100 --output ./my_dataset
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pybullet as pb

# Import from a2_collect to reuse existing functions
from lerobot.envs.a2.a2_collect import (
    _get_observation_data,
    _predict_grasp_pose,
    _predict_place_pose,
    cleanup_gpu_memory,
    create_a2_policy,
)
from lerobot.envs.a2.a2_constants import GRASP_WORKSPACE_LIMITS, PLACE_WORKSPACE_LIMITS


def get_test_cases_dir(task: str, object_set: str) -> Path:
    """Get the directory containing test cases for a task and object set."""
    from lerobot.envs.a2.a2 import get_testing_cases_path
    return get_testing_cases_path(task, object_set)


def load_test_cases(task: str, object_set: str) -> list[Path]:
    """Load all test case files for a task and object set."""
    test_dir = get_test_cases_dir(task, object_set)
    if not test_dir.exists():
        print(f"Warning: Test cases directory not found: {test_dir}")
        return []

    # Get all .txt files
    cases = sorted(test_dir.glob("*.txt"))
    print(f"Found {len(cases)} test cases in {test_dir}")
    return cases


def setup_from_test_case(env, file_path: Path, task: str) -> dict:
    """Setup environment from a test case file.

    Returns:
        dict with lang_goal, grasp_lang_goal, place_lang_goal
    """
    result = {"lang_goal": "", "grasp_lang_goal": None, "place_lang_goal": None}

    # Clear existing objects
    for obj_id in list(env._env.obj_ids.get("rigid", [])):
        try:
            pb.removeBody(obj_id, physicsClientId=env._env._client_id)
        except Exception:
            pass
    env._env.obj_ids = {"fixed": env._env.obj_ids.get("fixed", []), "rigid": []}
    env._env.obj_labels = {}
    env._env.target_obj_ids = []
    env._env.reference_obj_ids = []

    try:
        if task == "grasp":
            success, lang_goal = env._env.add_object_push_from_file(str(file_path))
            if success:
                result["lang_goal"] = lang_goal
        elif task == "place":
            success, lang_goal = env._env.add_object_push_from_place_file(str(file_path))
            if success:
                result["lang_goal"] = lang_goal
                result["place_lang_goal"] = lang_goal
        elif task == "pickplace":
            # Load grasp scene
            ret = env._env.add_object_push_from_pickplace_file(str(file_path), mode="grasp")
            success, grasp_lang_goal, _ = ret
            if success:
                # Remove objects outside grasp workspace
                ref_ids = getattr(env._env, "reference_obj_ids", [])
                env._env.remove_out_of_workspace_objects(GRASP_WORKSPACE_LIMITS, exclude_ids=ref_ids)

                # Load place scene
                ret2 = env._env.add_object_push_from_pickplace_file(str(file_path), mode="place")
                success2, _, place_lang_goal = ret2
                if success2:
                    # Remove place objects outside workspace
                    target_ids = getattr(env._env, "target_obj_ids", [])
                    env._env.remove_out_of_workspace_objects(PLACE_WORKSPACE_LIMITS, exclude_ids=target_ids)

                    result["lang_goal"] = grasp_lang_goal
                    result["grasp_lang_goal"] = grasp_lang_goal
                    result["place_lang_goal"] = place_lang_goal
                    env._env.grasp_lang_goal = grasp_lang_goal
                    env._env.place_lang_goal = place_lang_goal
    except Exception as e:
        print(f"Error loading test case {file_path}: {e}")

    return result


def run_grasp_episode(
    env,
    policy,
    lang_goal: str,
    max_attempts: int,
    device: str,
    recorder=None,
) -> dict:
    """Run a single grasp episode (matching benchmark logic)."""
    result = {"success": False, "steps": 0, "reward": 0.0}

    for attempt in range(max_attempts):
        # Check if targets still in workspace
        if env._env.target_obj_ids:
            bounds = env._env.bounds
            in_workspace = [
                oid for oid in env._env.target_obj_ids
                if bounds[0][0] <= env._env.obj_info(oid)[0][0] <= bounds[0][1]
                and bounds[1][0] <= env._env.obj_info(oid)[0][1] <= bounds[1][1]
            ]
            if not in_workspace:
                print("  No targets in workspace")
                break

        color_images, depth_images, pcd = _get_observation_data(env)

        if hasattr(policy, "reset"):
            policy.reset()

        # Get target poses for GraspNet filtering
        object_poses = env._env.get_true_object_poses()
        target_poses = {oid: pose for oid, pose in object_poses.items()
                       if oid in (env._env.target_obj_ids or [])}

        grasp_pose, _ = _predict_grasp_pose(
            policy, color_images, depth_images, pcd, lang_goal, device,
            object_poses=target_poses if target_poses else None, env=env
        )

        if grasp_pose is None:
            print(f"  Attempt {attempt + 1}: No valid grasp")
            break

        print(f"  Attempt {attempt + 1}: ({grasp_pose[0]:.3f}, {grasp_pose[1]:.3f}, {grasp_pose[2]:.3f})")

        # Record frame if recorder provided
        if recorder:
            recorder._current_attempt = attempt
        env._env.set_current_action(grasp_pose)

        obs, reward, _, _, info = env.step(grasp_pose)
        result["steps"] += 1
        result["reward"] += reward

        if info.get("is_success", False):
            result["success"] = True
            print(f"    SUCCESS!")
            break
        else:
            print(f"    Failed, retrying...")

    return result


def run_place_episode(
    env,
    policy,
    lang_goal: str,
    max_attempts: int,
    device: str,
    recorder=None,
) -> dict:
    """Run a place episode with physical execution for data collection.

    Note: Original A2 benchmark doesn't physically execute place, but for data
    collection we need to execute the motion to record frames.
    """
    result = {"success": False, "steps": 0, "reward": 0.0}

    ref_ids = getattr(env._env, "reference_obj_ids", [])
    if not ref_ids:
        print("  No reference objects")
        return result

    color_images, depth_images, pcd = _get_observation_data(env)

    place_pose, info = _predict_place_pose(
        policy, color_images, depth_images, pcd, lang_goal, device, env=env
    )

    if place_pose is None:
        print("  No valid place pose")
        return result

    print(f"  Selected: ({place_pose[0]:.3f}, {place_pose[1]:.3f}, {place_pose[2]:.3f})")
    result["steps"] = 1

    # Execute place motion to record frames (for data collection)
    if recorder:
        recorder._current_attempt = 0
    env._env.set_current_action(place_pose)
    env._env.place(place_pose)  # This triggers frame recording
    env._env.go_home()

    if info.get("place_valid", False):
        print(f"    SUCCESS!")
        result["success"] = True
        result["reward"] = 2
    else:
        print(f"    FAIL (invalid placement)")

    return result


def run_pickplace_episode(
    env,
    policy,
    max_attempts: int,
    device: str,
    recorder=None,
) -> dict:
    """Run a pick-and-place episode (matching benchmark logic)."""
    GRASP_WORKSPACE_Y = (-0.336, 0.000)
    PLACE_WORKSPACE_Y = (0.000, 0.336)

    result = {"success": False, "steps": 0, "reward": 0.0}

    grasp_goal = getattr(env._env, "grasp_lang_goal", None) or getattr(env._env, "lang_goal", "grasp an object")
    place_goal = getattr(env._env, "place_lang_goal", None) or "place it somewhere"

    grasp_done = False

    # Phase 1: Grasp
    print(f"  [Grasp] Goal: {grasp_goal}")
    for attempt in range(max_attempts):
        # Check targets in grasp workspace
        if env._env.target_obj_ids:
            valid = [oid for oid in env._env.target_obj_ids
                    if GRASP_WORKSPACE_Y[0] <= env._env.obj_info(oid)[0][1] <= GRASP_WORKSPACE_Y[1]]
            if not valid:
                print("    No targets in grasp workspace")
                break

        color_images, depth_images, pcd = _get_observation_data(env)

        # Filter PCD to grasp workspace
        if pcd is not None and len(pcd) > 0:
            mask = (pcd[:, 1] >= GRASP_WORKSPACE_Y[0]) & (pcd[:, 1] <= GRASP_WORKSPACE_Y[1])
            pcd_grasp = pcd[mask] if mask.sum() >= 100 else pcd
        else:
            pcd_grasp = pcd

        # Target poses in grasp workspace
        object_poses = env._env.get_true_object_poses()
        target_poses = {}
        for oid, pose in object_poses.items():
            if oid in env._env.target_obj_ids:
                pos = pose[:3, 3] if isinstance(pose, np.ndarray) and pose.shape == (4, 4) else pose[:3]
                if GRASP_WORKSPACE_Y[0] <= pos[1] <= GRASP_WORKSPACE_Y[1]:
                    target_poses[oid] = pose

        grasp_pose, _ = _predict_grasp_pose(
            policy, color_images, depth_images, pcd_grasp, grasp_goal, device,
            object_poses=target_poses, env=env, is_pickplace=True
        )

        if grasp_pose is None:
            print(f"    Grasp {attempt + 1}: No pose")
            continue

        print(f"    Grasp {attempt + 1}: ({grasp_pose[0]:.3f}, {grasp_pose[1]:.3f}, {grasp_pose[2]:.3f})")

        # Set action for frame recording
        if recorder:
            recorder._current_attempt = attempt
        env._env.set_current_action(grasp_pose)
        success, grasped_id, _ = env._env.grasp(grasp_pose, follow_place=True)
        result["steps"] += 1

        if not success:
            print(f"      Failed")
            result["reward"] -= 1
            continue

        if grasped_id in env._env.target_obj_ids:
            print(f"      Grasped target!")
            result["reward"] += 2
            grasp_done = True
            break
        else:
            print(f"      Wrong object, dropping")
            env._env.place_out_of_workspace()

    if not grasp_done:
        print("  [Grasp] Failed")
        env._env.go_home()
        return result

    # Phase 2: Place
    print(f"  [Place] Goal: {place_goal}")
    remaining = max(1, max_attempts - result["steps"])

    for attempt in range(remaining):
        color_images, depth_images, pcd = _get_observation_data(env)

        # Filter PCD to place workspace
        if pcd is not None and len(pcd) > 0:
            mask = (pcd[:, 1] >= PLACE_WORKSPACE_Y[0]) & (pcd[:, 1] <= PLACE_WORKSPACE_Y[1])
            pcd_place = pcd[mask] if mask.sum() >= 100 else pcd
        else:
            pcd_place = pcd

        place_pose, info = _predict_place_pose(
            policy, color_images, depth_images, pcd_place, place_goal, device,
            env=env, is_pickplace=True
        )

        if place_pose is None:
            print(f"    Place {attempt + 1}: No pose")
            continue

        print(f"    Place {attempt + 1}: ({place_pose[0]:.3f}, {place_pose[1]:.3f}, {place_pose[2]:.3f})")

        # Set action for frame recording
        if recorder:
            recorder._current_attempt = result["steps"]
        env._env.set_current_action(place_pose)
        env._env.place(place_pose)
        result["steps"] += 1

        if info.get("place_valid", False):
            print(f"      SUCCESS!")
            result["success"] = True
            result["reward"] += 2
            break
        else:
            print(f"      FAIL (invalid)")
            result["reward"] -= 0.5

    env._env.go_home()
    return result


def collect_data(
    task: str,
    object_set: str,
    num_episodes: int,
    max_attempts: int | None,
    output_dir: str,
    device: str = "cuda",
    policy_path: str = "dgrachev/a2_pretrained",
    num_objects: int | None = None,
    env_reset_interval: int = 10,
    save_failed: bool = False,
    fps: int = 30,
    image_width: int = 640,
    image_height: int = 480,
    use_test_cases: bool = False,
):
    """Collect data for a task."""
    from lerobot.envs.a2.a2 import A2Env
    from lerobot.envs.a2.a2_recorder import VLARecorder

    # Set default num_objects based on task (matching original A2 paper)
    # Grasp: 15 objects, Place: 8 objects, Pickplace: handled by test cases or 15+8
    if num_objects is None:
        if task == "grasp":
            num_objects = 15  # Original A2: "15 objects randomly dropped"
        elif task == "place":
            num_objects = 8   # Original A2: "8 objects, centers ≥0.1m apart"
        else:  # pickplace
            num_objects = 15  # Grasp side objects (place side handled separately in test cases)

    # Set default max_attempts based on task (matching original A2 evaluator)
    if max_attempts is None:
        if task == "place":
            max_attempts = 1  # Place is single-shot pose selection
        else:
            max_attempts = 8  # Grasp/pickplace allow retries

    # Handle "both" object set by alternating
    if object_set == "both":
        object_sets = ["train", "test"]
    else:
        object_sets = [{"seen": "train", "unseen": "test"}.get(object_set, object_set)]

    print(f"\n{'=' * 60}")
    print(f"DATA COLLECTION: {task.upper()}")
    print(f"{'=' * 60}")
    print(f"  Object set: {object_set}")
    print(f"  Target episodes: {num_episodes}")
    print(f"  Max attempts: {max_attempts}")
    print(f"  Output: {output_dir}")
    print(f"  Save failed: {save_failed}")
    print(f"  Use test cases: {use_test_cases}")
    print(f"  Num objects: {num_objects}")
    print(f"  Max attempts: {max_attempts}")

    # Load test cases if requested
    test_cases = []
    if use_test_cases:
        for obj_set in object_sets:
            cases = load_test_cases(task, obj_set)
            test_cases.extend([(c, obj_set) for c in cases])
        if not test_cases:
            print("WARNING: No test cases found, falling back to random scenes")
            use_test_cases = False
        else:
            print(f"  Loaded {len(test_cases)} test cases total")
            random.shuffle(test_cases)

    # Load policy
    print("\nLoading policy...")
    policy = create_a2_policy(policy_path, device)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create recorder for successful episodes
    recorder = VLARecorder(
        output_dir=str(output_path),
        repo_id=f"local/{task}_{object_set}",
        fps=fps,
        image_size=(image_height, image_width),
        cameras=["front", "left", "right"],
    )

    # Stats
    stats = {
        "total_attempts": 0,
        "successes": 0,
        "failures": 0,
        "episodes_saved": 0,
    }

    # Progress file
    progress_file = output_path / "progress.json"

    current_set_idx = 0
    test_case_idx = 0
    episode = 0
    env = None
    cameras = ["front", "left", "right"]
    render_size = (image_width, image_height)

    def setup_frame_recorder(environment, rec):
        """Set up frame recording callback."""
        def frame_callback(images: dict, robot_state: dict):
            action = environment._env._current_action
            if action is None:
                action = np.zeros(7, dtype=np.float32)
            if rec.is_recording:
                rec.set_action(action, attempt_idx=getattr(rec, "_current_attempt", 0))
                rec.record_frame(images=images, robot_state=robot_state)

        environment._env.set_frame_recorder(frame_callback, fps=fps, cameras=cameras, render_size=render_size)

    try:
        while stats["episodes_saved"] < num_episodes:
            # Select object set
            if use_test_cases and test_cases:
                test_case_path, current_object_set = test_cases[test_case_idx % len(test_cases)]
                test_case_idx += 1
            else:
                current_object_set = object_sets[current_set_idx % len(object_sets)]
                current_set_idx += 1
                test_case_path = None

            # Create/recreate environment
            if env is None or (env_reset_interval > 0 and episode > 0 and episode % env_reset_interval == 0):
                if env is not None:
                    env.close()
                    cleanup_gpu_memory()
                    print("  [Memory cleanup]")

                env = A2Env(
                    task=task if task != "pickplace" else "pick_and_place",
                    num_objects=num_objects,
                    object_set=current_object_set,
                    gui=False,
                    observation_width=image_width,
                    observation_height=image_height,
                )
                setup_frame_recorder(env, recorder)

            episode += 1
            stats["total_attempts"] += 1

            print(f"\n--- Episode {episode} (saved: {stats['episodes_saved']}/{num_episodes}, set: {current_object_set}) ---")

            # Reset environment
            seed = int(time.time() * 1000) % (2**31)
            obs, info = env.reset(seed=seed)

            # Load test case if using test cases
            if use_test_cases and test_case_path:
                print(f"  Loading test case: {test_case_path.name}")
                case_info = setup_from_test_case(env, test_case_path, task)
                lang_goal = case_info.get("lang_goal") or "do the task"
            else:
                # Get language goal from random scene
                lang_goal = getattr(env._env, "lang_goal", None)
                grasp_goal = getattr(env._env, "grasp_lang_goal", None)
                place_goal = getattr(env._env, "place_lang_goal", None)
                if lang_goal is None:
                    lang_goal = grasp_goal or place_goal or "do the task"

            print(f"Goal: {lang_goal}")

            if hasattr(policy, "reset"):
                policy.reset()

            # Start recording
            recorder.start_episode(task=lang_goal)
            recorder._current_attempt = 0

            # Run episode
            if task == "grasp":
                result = run_grasp_episode(env, policy, lang_goal, max_attempts, device, recorder)
            elif task == "place":
                result = run_place_episode(env, policy, lang_goal, max_attempts, device, recorder)
            elif task == "pickplace":
                result = run_pickplace_episode(env, policy, max_attempts, device, recorder)
            else:
                raise ValueError(f"Unknown task: {task}")

            if result["success"]:
                stats["successes"] += 1
                recorder.end_episode(success=True, total_reward=result["reward"], num_attempts=result["steps"])
                stats["episodes_saved"] += 1
                print(f"  -> SUCCESS (saved: {stats['episodes_saved']}/{num_episodes})")
            else:
                stats["failures"] += 1
                if save_failed:
                    recorder.end_episode(success=False, total_reward=result["reward"], num_attempts=result["steps"])
                    stats["episodes_saved"] += 1
                else:
                    recorder.cancel_episode()
                print(f"  -> FAILED")

            # Save progress
            with open(progress_file, "w") as f:
                json.dump(stats, f, indent=2)

            # Print running stats
            success_rate = stats["successes"] / stats["total_attempts"] * 100
            print(f"  Running: {stats['successes']}/{stats['total_attempts']} ({success_rate:.1f}%)")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        if env is not None:
            env.close()
        recorder.finalize()

    # Final stats
    print(f"\n{'=' * 60}")
    print("COLLECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Episodes saved: {stats['episodes_saved']}")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Successes: {stats['successes']}")
    print(f"  Failures: {stats['failures']}")
    if stats["total_attempts"] > 0:
        print(f"  Success rate: {stats['successes']/stats['total_attempts']*100:.1f}%")
    print(f"  Output: {output_dir}")

    # Save final stats
    with open(output_path / "final_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="A2 Data Collection")
    parser.add_argument("--task", type=str, required=True, choices=["grasp", "place", "pickplace"],
                       help="Task to collect data for")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of successful episodes to collect")
    parser.add_argument("--object_set", type=str, default="seen", choices=["seen", "unseen", "both"],
                       help="Object set to use")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: datasets/a2_{task}_{object_set})")
    parser.add_argument("--max_attempts", type=int, default=None,
                       help="Max attempts per episode (default: 8 for grasp/pickplace, 1 for place)")
    parser.add_argument("--num_objects", type=int, default=None,
                       help="Number of objects in scene (default: 15 for grasp, 8 for place, 15+8 for pickplace)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--policy_path", type=str, default="dgrachev/a2_pretrained",
                       help="Policy checkpoint path")
    parser.add_argument("--env_reset_interval", type=int, default=10,
                       help="Episodes between environment recreation (for memory cleanup)")
    parser.add_argument("--save_failed", action="store_true",
                       help="Also save failed episodes")
    parser.add_argument("--fps", type=int, default=30,
                       help="Recording FPS")
    parser.add_argument("--image_width", type=int, default=640,
                       help="Image width")
    parser.add_argument("--image_height", type=int, default=480,
                       help="Image height")
    parser.add_argument("--use_test_cases", action="store_true",
                       help="Use predefined test cases instead of random scenes (higher success rate)")

    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        suffix = "_testcases" if args.use_test_cases else ""
        args.output = f"datasets/a2_{args.task}_{args.object_set}{suffix}"

    collect_data(
        task=args.task,
        object_set=args.object_set,
        num_episodes=args.episodes,
        max_attempts=args.max_attempts,
        output_dir=args.output,
        device=args.device,
        policy_path=args.policy_path,
        num_objects=args.num_objects,
        env_reset_interval=args.env_reset_interval,
        save_failed=args.save_failed,
        fps=args.fps,
        image_width=args.image_width,
        image_height=args.image_height,
        use_test_cases=args.use_test_cases,
    )


if __name__ == "__main__":
    main()
