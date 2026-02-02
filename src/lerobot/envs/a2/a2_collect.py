#!/usr/bin/env python
"""A2 Data Collection Script.

This script collects manipulation data using the A2 environment and saves it
in LeRobot dataset format. It supports any LeRobot registered policy.

Usage:
    # With A2 policy from HuggingFace
    python -m lerobot.envs.a2_collect --policy a2 --policy_path dgrachev/a2_pretrained

    # With ACT policy
    python -m lerobot.envs.a2_collect --policy act --policy_path lerobot/act_aloha

    # With random policy (for testing)
    python -m lerobot.envs.a2_collect --policy random --num_episodes 10
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.utils.import_utils import register_third_party_plugins

if TYPE_CHECKING:
    from .a2_recorder import VLARecorder


def cleanup_gpu_memory():
    """Force cleanup of GPU memory to prevent leaks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_progress_file(output_dir: str, repo_id: str) -> Path:
    """Get path to progress tracking file."""
    return Path(output_dir) / repo_id / "collection_progress.json"


def load_progress(output_dir: str, repo_id: str) -> dict:
    """Load collection progress from file."""
    progress_file = get_progress_file(output_dir, repo_id)
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"completed_episodes": 0, "success_count": 0, "total_attempts": 0}


def save_progress(output_dir: str, repo_id: str, completed: int, successes: int, attempts: int):
    """Save collection progress to file."""
    progress_file = get_progress_file(output_dir, repo_id)
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w") as f:
        json.dump(
            {
                "completed_episodes": completed,
                "success_count": successes,
                "total_attempts": attempts,
            },
            f,
        )


class RandomPolicy:
    """Random policy for testing data collection pipeline."""

    def __init__(self, device: str = "cpu"):
        self._device = torch.device(device)

    def reset(self):
        """Reset policy state between episodes."""
        pass

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select random action (delta_ee format).

        Returns:
            Action tensor (batch_size, 7): [dx, dy, dz, 0, 0, 0, gripper]
        """
        batch_size = 1
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].dim() >= 1:
                batch_size = batch[key].shape[0]
                break

        # Random delta position in [-1, 1] range
        action = torch.zeros(batch_size, 7, device=self._device)
        action[:, :3] = torch.rand(batch_size, 3, device=self._device) * 2 - 1
        action[:, 6] = (torch.rand(batch_size, device=self._device) > 0.5).float()

        return action


def create_a2_policy(
    model_path: str | None = None,
    hf_repo: str = "dgrachev/a2_pretrained",
    device: str = "cuda",
) -> Any:
    """Create an A2 policy with pretrained weights.

    Args:
        model_path: Local path to model checkpoint (takes priority over hf_repo)
        hf_repo: HuggingFace repo ID for pretrained weights
        device: Device to load policy on

    Returns:
        A2Policy instance
    """
    policy_path = model_path if model_path else hf_repo
    return load_policy(policy_type="a2", policy_path=policy_path, device=device)


def load_policy(
    policy_type: str,
    policy_path: str | None = None,
    device: str = "cuda",
) -> Any:
    """Load a LeRobot policy by type and path.

    Args:
        policy_type: Policy type name (e.g., "a2", "act", "diffusion", "random")
        policy_path: Pretrained path (HuggingFace repo ID or local path)
        device: Device to load policy on

    Returns:
        Policy instance with select_action() and reset() methods
    """
    if policy_type == "random":
        return RandomPolicy(device=device)

    # Register third-party plugins (like lerobot_policy_a2)
    register_third_party_plugins()

    # Import policy factory
    from lerobot.policies.factory import get_policy_class, make_policy_config

    # Get policy class
    policy_cls = get_policy_class(policy_type)

    # Create config
    config = make_policy_config(
        policy_type,
        device=device,
        pretrained_path=policy_path,
    )

    # Load policy
    if policy_path:
        print(f"Loading {policy_type} policy from: {policy_path}")
        policy = policy_cls.from_pretrained(
            pretrained_name_or_path=policy_path,
            config=config,
        )
    else:
        print(f"Creating {policy_type} policy from scratch (untrained)")
        policy = policy_cls(config)

    policy.to(device)
    policy.eval()

    return policy


def collect_data(
    policy,
    task: str = "grasp",
    object_set: str = "train",
    num_episodes: int = 100,
    max_attempts_per_episode: int = 8,
    num_objects: int = 8,
    output_dir: str = "data/a2_collected",
    repo_id: str = "local/a2_collected",
    cameras: list[str] | None = None,
    fps: int = 30,
    image_size: tuple[int, int] = (480, 640),
    seed: int = 42,
    gui: bool = False,
    device: str = "cuda",
    action_mode: str = "pose",
    start_episode: int = 0,
    resume: bool = False,
    oracle_grasp: bool = False,
    skip_failed_attempts: bool = False,
    distinguish_failed_attempts: bool = False,
    env_reset_interval: int = 10,
):
    """Collect manipulation data using the provided policy.

    Supports two action modes:
    - "pose": A2-style where policy outputs grasp/place poses executed as full trajectories
    - "delta_ee": VLA-style where policy outputs delta actions at each timestep

    Args:
        policy: Policy with predict_grasp/predict_place or select_action methods
        task: "grasp", "place", or "pick_and_place"
        object_set: "train" or "test"
        num_episodes: Number of episodes to collect
        max_attempts_per_episode: Maximum grasp/place attempts per episode (pose mode)
        num_objects: Number of objects in scene
        output_dir: Directory to save dataset
        repo_id: LeRobot dataset repository ID
        cameras: List of cameras to record
        fps: Recording FPS
        image_size: Image size (height, width)
        seed: Random seed
        gui: Show PyBullet GUI
        device: Device for policy inference
        action_mode: "pose" (A2-style) or "delta_ee" (VLA-style)
        start_episode: Episode number to start from (for resuming)
        resume: Auto-detect start_episode from progress file
        oracle_grasp: Use oracle grasp for place tasks
        skip_failed_attempts: Skip recording failed attempts (grasp/pick-and-place)
        distinguish_failed_attempts: Record full dataset plus success-only dataset
    """
    from .a2 import A2Env
    from .a2_recorder import VLARecorder

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Default cameras
    if cameras is None:
        cameras = ["front", "overview", "gripper"]

    # Use identity camera mapping so obs keys match recorder camera names
    camera_mapping = {cam: cam for cam in cameras}
    print(f"Creating A2 environment: task={task}, object_set={object_set}, action_mode={action_mode}")
    env = A2Env(
        task=task,
        object_set=object_set,
        num_objects=num_objects,
        camera_name=cameras,
        camera_name_mapping=camera_mapping,
        gui=gui,
        action_mode=action_mode,
        episode_length=500 if action_mode == "delta_ee" else max_attempts_per_episode,
    )

    # Setup VLA recorder
    print(f"Setting up VLA recorder: {output_dir}/{repo_id}")
    recorder = VLARecorder(
        output_dir=output_dir,
        repo_id=repo_id,
        fps=fps,
        image_size=image_size,
        cameras=cameras,
    )

    success_recorder = None
    if distinguish_failed_attempts:
        success_repo_id = f"{repo_id}_success"
        print(f"Setting up success-only recorder: {output_dir}/{success_repo_id}")
        success_recorder = VLARecorder(
            output_dir=output_dir,
            repo_id=success_repo_id,
            fps=fps,
            image_size=image_size,
            cameras=cameras,
        )

    # Handle resume logic
    if resume:
        progress = load_progress(output_dir, repo_id)
        start_episode = progress["completed_episodes"]
        print(f"Resuming from episode {start_episode} (prev: {progress['success_count']} successes)")

    if start_episode > 0:
        print(f"Starting from episode {start_episode}/{num_episodes}")

    if action_mode == "pose":
        _collect_pose_mode(
            env,
            policy,
            recorder,
            success_recorder,
            num_episodes,
            max_attempts_per_episode,
            seed,
            device,
            task,
            cameras,
            fps,
            image_size,
            output_dir,
            repo_id,
            start_episode,
            oracle_grasp,
            skip_failed_attempts,
            distinguish_failed_attempts,
            env_reset_interval,
        )
    else:
        _collect_delta_mode(
            env,
            policy,
            recorder,
            num_episodes,
            max_attempts_per_episode,
            seed,
            device,
            task,
            output_dir,
            repo_id,
            start_episode,
        )


def _run_pickplace_episode(
    env,
    policy,
    max_attempts,
    device,
    recorder=None,
    success_recorder=None,
):
    """Run a pick-and-place episode with separate grasp and place phases (A2_new style).

    Phase 1: Retry grasp until we successfully grasp a target object
    Phase 2: Retry place until we successfully place the object

    Returns:
        (success, total_reward, attempt_count)
    """
    grasp_lang_goal = getattr(env._env, "grasp_lang_goal", None) or getattr(env._env, "lang_goal", "grasp an object")
    place_lang_goal = getattr(env._env, "place_lang_goal", None) or "place it somewhere"

    grasp_done = False
    place_done = False
    total_reward = 0.0
    attempt_count = 0
    grasped_obj_id = None

    # Phase 1: Grasp attempts
    print(f"  [Grasp phase] Goal: {grasp_lang_goal}")
    for grasp_attempt in range(max_attempts):
        # Check if target objects still in workspace
        if env._env.target_obj_ids:
            valid_targets = []
            for obj_id in env._env.target_obj_ids:
                pos, _, _ = env._env.obj_info(obj_id)
                bounds = env._env.bounds
                if bounds[0][0] <= pos[0] <= bounds[0][1] and bounds[1][0] <= pos[1] <= bounds[1][1]:
                    valid_targets.append(obj_id)
            if not valid_targets:
                print("    Target objects not in workspace!")
                break

        # Get observation and predict grasp
        color_images, depth_images, pcd = _get_observation_data(env)

        object_poses = env._env.get_true_object_poses()
        reference_ids = getattr(env._env, "reference_obj_ids", None) or []
        graspable_ids = [oid for oid in env._env.obj_ids["rigid"] if oid not in reference_ids]
        target_object_poses = {oid: pose for oid, pose in object_poses.items() if oid in graspable_ids}

        grasp_action, _ = _predict_grasp_pose(
            policy, color_images, depth_images, pcd, grasp_lang_goal, device,
            object_poses=target_object_poses, env=env
        )

        if grasp_action is None:
            print(f"    Grasp attempt {grasp_attempt + 1}: No valid grasp")
            continue

        print(f"    Grasp attempt {grasp_attempt + 1}: pos=({grasp_action[0]:.3f}, {grasp_action[1]:.3f}, {grasp_action[2]:.3f})")

        # Execute grasp (keep holding for place)
        env._env.set_current_action(grasp_action)
        grasp_success, grasped_obj_id, _ = env._env.grasp(grasp_action, follow_place=True)
        attempt_count += 1

        if not grasp_success:
            print(f"      Grasp failed")
            total_reward -= 1
            continue

        # Check if we grasped a target object
        if grasped_obj_id in env._env.target_obj_ids:
            print(f"      Grasped target object!")
            total_reward += 2
            grasp_done = True
            break
        else:
            print(f"      Grasped wrong object, dropping...")
            total_reward += 0
            env._env.place_out_of_workspace()

    if not grasp_done:
        print("  [Grasp phase] Failed to grasp target")
        return False, total_reward, attempt_count

    # Phase 2: Place attempts
    print(f"  [Place phase] Goal: {place_lang_goal}")
    remaining_attempts = max_attempts - attempt_count
    for place_attempt in range(max(1, remaining_attempts)):
        # Get fresh observation for place
        color_images, depth_images, pcd = _get_observation_data(env)

        place_action, _ = _predict_place_pose(
            policy, color_images, depth_images, pcd, place_lang_goal, device
        )

        if place_action is None:
            print(f"    Place attempt {place_attempt + 1}: No valid place pose")
            continue

        print(f"    Place attempt {place_attempt + 1}: pos=({place_action[0]:.3f}, {place_action[1]:.3f}, {place_action[2]:.3f})")

        # Execute place
        env._env.set_current_action(place_action)
        place_success = env._env.place(place_action)
        attempt_count += 1

        if place_success:
            # Check if placement is correct (near reference object in correct direction)
            # For now, trust the environment's success signal
            print(f"      Place succeeded!")
            total_reward += 2
            place_done = True
            break
        else:
            print(f"      Place failed")
            total_reward -= 1

    env._env.go_home()

    success = grasp_done and place_done
    if success:
        print(f"  [SUCCESS] Pick-and-place completed in {attempt_count} attempts")
    else:
        print(f"  [FAILED] Pick-and-place failed after {attempt_count} attempts")

    return success, total_reward, attempt_count


def _collect_pose_mode(
    env,
    policy,
    recorder,
    success_recorder,
    num_episodes,
    max_attempts,
    seed,
    device,
    task,
    cameras,
    fps,
    render_size,
    output_dir="",
    repo_id="",
    start_episode=0,
    oracle_grasp=False,
    skip_failed_attempts=False,
    distinguish_failed_attempts=False,
    env_reset_interval=10,
):
    """A2-style collection: policy outputs poses, environment executes full trajectories."""
    # Load previous progress if resuming
    if start_episode > 0:
        progress = load_progress(output_dir, repo_id)
        success_count = progress.get("success_count", 0)
        total_attempts = progress.get("total_attempts", 0)
    else:
        success_count = 0
        total_attempts = 0

    recorders = [rec for rec in (recorder, success_recorder) if rec is not None]

    def setup_frame_recorder(environment):
        """Set up frame recording callback on the environment."""

        def frame_callback(images: dict, robot_state: dict):
            """Called by environment during trajectory execution."""
            action = environment._env._current_action
            if action is None:
                action = np.zeros(7, dtype=np.float32)
            for rec in recorders:
                if not rec.is_recording:
                    continue
                rec.set_action(
                    action, attempt_idx=rec._current_attempt if hasattr(rec, "_current_attempt") else 0
                )
                rec.record_frame(images=images, robot_state=robot_state)

        environment._env.set_frame_recorder(frame_callback, fps=fps, cameras=cameras, render_size=render_size)

    # Register the frame recorder
    setup_frame_recorder(env)

    for episode in range(start_episode, num_episodes):
        print(f"\n{'=' * 50}")
        print(f"Episode {episode + 1}/{num_episodes}")

        # Periodically recreate environment to prevent EGL memory leak
        if (
            env_reset_interval > 0
            and episode > start_episode
            and (episode - start_episode) % env_reset_interval == 0
        ):
            print("  [Memory cleanup] Recreating environment to release EGL memory...")
            # Save env config
            env_task = env.task
            env_num_objects = env.num_objects
            env_object_set = env.object_set
            env_obs_width = env.observation_width
            env_obs_height = env.observation_height
            # Close and recreate
            env.close()
            cleanup_gpu_memory()
            from lerobot.envs.a2.a2 import A2Env

            env = A2Env(
                task=env_task,
                num_objects=env_num_objects,
                object_set=env_object_set,
                gui=False,
                observation_width=env_obs_width,
                observation_height=env_obs_height,
            )
            setup_frame_recorder(env)
            print("  [Memory cleanup] Environment recreated")

        # Reset environment and policy
        obs, info = env.reset(seed=seed + episode)
        if hasattr(policy, "reset"):
            policy.reset()

        # Get language goal
        lang_goal = getattr(env._env, "lang_goal", None) or "place an object"
        print(f"Language goal: {lang_goal}")

        record_success_only_main = skip_failed_attempts and task in ("pick_and_place", "grasp")
        record_success_only_main = record_success_only_main and not distinguish_failed_attempts
        record_success_only_success = bool(success_recorder)

        # Start recording unless we are skipping failed attempts
        if recorder is not None and not record_success_only_main:
            recorder.start_episode(task=lang_goal)
        if recorder is not None:
            recorder._current_attempt = 0
        recorder_started = recorder.is_recording if recorder is not None else False

        success_recorder_started = False

        episode_reward = 0.0
        episode_success = False
        attempt_count = 0

        # Special handling for pick_and_place: two-phase execution (A2_new style)
        if task == "pick_and_place":
            episode_success, episode_reward, attempt_count = _run_pickplace_episode(
                env, policy, max_attempts, device, recorder, success_recorder
            )
            total_attempts += attempt_count
            if episode_success:
                success_count += 1

            # Handle recording
            if recorder is not None:
                if not record_success_only_main:
                    recorder.end_episode(success=episode_success, total_reward=episode_reward, num_attempts=attempt_count)
                elif episode_success:
                    recorder.start_episode(task=lang_goal)
                    recorder.end_episode(success=True, total_reward=episode_reward, num_attempts=attempt_count)

            # Save progress
            save_progress(output_dir, repo_id, episode + 1, success_count, total_attempts)

            # Summary
            print(f"Episode {episode + 1}: {'SUCCESS' if episode_success else 'FAIL'} | "
                  f"Running: {success_count}/{episode + 1 - start_episode} "
                  f"({100 * success_count / (episode + 1 - start_episode):.1f}%)")
            cleanup_gpu_memory()
            continue  # Skip the regular attempt loop

        for attempt in range(max_attempts):
            # Check if target objects still in workspace (grasp tasks only)
            if env._env.target_obj_ids:
                out_of_workspace = []
                for obj_id in env._env.target_obj_ids:
                    pos, _, _ = env._env.obj_info(obj_id)
                    bounds = env._env.bounds
                    if (
                        pos[0] < bounds[0][0]
                        or pos[0] > bounds[0][1]
                        or pos[1] < bounds[1][0]
                        or pos[1] > bounds[1][1]
                    ):
                        out_of_workspace.append(obj_id)

                if len(out_of_workspace) == len(env._env.target_obj_ids):
                    print("  Target objects are not in the scene!")
                    break

            # Get images and point cloud for policy
            color_images, depth_images, pcd = _get_observation_data(env)

            # Generate grasp/place pose using policy
            if task == "grasp":
                object_poses = env._env.get_true_object_poses()
                target_object_poses = {
                    obj_id: pose for obj_id, pose in object_poses.items() if obj_id in env._env.target_obj_ids
                }
                action, info_dict = _predict_grasp_pose(
                    policy,
                    color_images,
                    depth_images,
                    pcd,
                    lang_goal,
                    device,
                    object_poses=target_object_poses,
                    env=env,
                )
                place_action = None
                grasp_action = action
            elif task == "pick_and_place":
                object_poses = env._env.get_true_object_poses()
                reference_ids = getattr(env._env, "reference_obj_ids", None) or []
                graspable_ids = [oid for oid in env._env.obj_ids["rigid"] if oid not in reference_ids]
                target_object_poses = {
                    obj_id: pose for obj_id, pose in object_poses.items() if obj_id in graspable_ids
                }
                # Pass is_pickplace=True for workspace shift (original A2 uses --workspace_shift only for pickplace)
                grasp_action, _ = _predict_grasp_pose(
                    policy,
                    color_images,
                    depth_images,
                    pcd,
                    lang_goal,
                    device,
                    object_poses=target_object_poses,
                    env=env,
                    is_pickplace=True,
                )
                place_action, info_dict = _predict_place_pose(
                    policy, color_images, depth_images, pcd, lang_goal, device, env=env, is_pickplace=True
                )
                if grasp_action is None or place_action is None:
                    print("  Warning: No valid grasp/place action generated, skipping attempt")
                    break
                action = place_action
            else:
                # Standalone place - no workspace shift needed
                action, info_dict = _predict_place_pose(
                    policy, color_images, depth_images, pcd, lang_goal, device, env=env, is_pickplace=False
                )
                place_action = action
                grasp_action = None
                if oracle_grasp:
                    object_poses = env._env.get_true_object_poses()
                    reference_ids = getattr(env._env, "reference_obj_ids", None) or []
                    graspable_ids = [oid for oid in env._env.obj_ids["rigid"] if oid not in reference_ids]
                    target_object_poses = {
                        obj_id: pose for obj_id, pose in object_poses.items() if obj_id in graspable_ids
                    }
                    grasp_action, _ = _predict_grasp_pose(
                        policy,
                        color_images,
                        depth_images,
                        pcd,
                        lang_goal,
                        device,
                        object_poses=target_object_poses,
                        env=env,
                    )

            if action is None:
                print("  Warning: No valid action generated, skipping attempt")
                break

            print(f"  Attempt {attempt + 1}: action_pos={action[:3]}")

            if recorder is not None and record_success_only_main and not recorder_started:
                recorder.start_episode(task=lang_goal)
                recorder_started = True
            if success_recorder is not None and record_success_only_success and not success_recorder_started:
                success_recorder.start_episode(task=lang_goal)
                success_recorder_started = True

            # Set action for frame recording
            if recorder is not None:
                recorder._current_attempt = attempt
            if success_recorder is not None:
                success_recorder._current_attempt = attempt
            env._env.set_current_action(action)

            # Execute action (this triggers internal frame recording)
            if task == "pick_and_place":
                if oracle_grasp:
                    print("  Note: oracle_grasp ignored for pick_and_place")
                reward, success = env._env.step_place(
                    place_action,
                    grasp_pose=grasp_action,
                    oracle_grasp=False,
                    record_grasp=True,
                    keep_grasp_orientation=False,
                )
                if reward is None:
                    reward = 0.0
                info = {"is_success": bool(success)}
                terminated = bool(success)
                truncated = False
                obs = env._get_observation()
            elif task != "grasp" and oracle_grasp:
                record_grasp = task == "pick_and_place"
                keep_grasp_orientation = False
                reward, success = env._env.step_place(
                    action,
                    grasp_pose=grasp_action,
                    oracle_grasp=True,
                    record_grasp=record_grasp,
                    keep_grasp_orientation=keep_grasp_orientation,
                )
                if reward is None:
                    reward = 0.0
                info = {"is_success": bool(success)}
                terminated = bool(success)
                truncated = False
                obs = env._get_observation()
            else:
                obs, reward, terminated, truncated, info = env.step(action)

            # Update step result
            if recorder is not None and recorder_started:
                recorder.update_step_result(reward=reward, success=info.get("is_success", False))
            if success_recorder is not None and success_recorder_started:
                success_recorder.update_step_result(reward=reward, success=info.get("is_success", False))

            episode_reward += reward
            attempt_count += 1
            total_attempts += 1

            if info.get("is_success", False):
                episode_success = True
                print(f"  SUCCESS at attempt {attempt + 1}!")
                break

            if recorder is not None and record_success_only_main and recorder_started:
                recorder.cancel_episode()
                recorder_started = False
            if success_recorder is not None and record_success_only_success and success_recorder_started:
                success_recorder.cancel_episode()
                success_recorder_started = False

            if terminated or truncated:
                break

        # End episode
        if recorder is not None:
            if record_success_only_main and recorder_started and not episode_success:
                recorder.cancel_episode()
                recorder_started = False

            if not record_success_only_main or recorder_started:
                recorder.end_episode(
                    success=episode_success, total_reward=episode_reward, num_attempts=attempt_count
                )
            else:
                print("  [VLA] No successful attempt recorded for this episode")

        if success_recorder is not None:
            if success_recorder_started and not episode_success:
                success_recorder.cancel_episode()
                success_recorder_started = False
            if success_recorder_started:
                success_recorder.end_episode(
                    success=episode_success,
                    total_reward=episode_reward,
                    num_attempts=attempt_count,
                )

        if episode_success:
            success_count += 1

        # Save progress after each episode (for crash recovery)
        completed_episodes = episode + 1
        save_progress(output_dir, repo_id, completed_episodes, success_count, total_attempts)

        print(
            f"Episode result: success={episode_success}, reward={episode_reward:.2f}, "
            f"attempts={attempt_count}, progress={completed_episodes}/{num_episodes}"
        )

        # Cleanup GPU memory to prevent leaks
        cleanup_gpu_memory()

    # Finalize
    recorder.finalize()
    if success_recorder is not None:
        success_recorder.finalize()
    env.close()

    # Calculate actual episodes completed
    episodes_run = num_episodes - start_episode
    print(f"\n{'=' * 50}")
    print("Collection complete!")
    print(f"  Episodes this run: {episodes_run}")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Successful episodes: {success_count}")
    print(f"  Success rate: {success_count / num_episodes * 100:.1f}%")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Dataset saved to: {output_dir}/{repo_id}")


def _collect_delta_mode(
    env,
    policy,
    recorder,
    num_episodes,
    max_steps,
    seed,
    device,
    task,
    output_dir="",
    repo_id="",
    start_episode=0,
):
    """VLA-style collection: policy outputs delta actions at each timestep."""
    # Load previous progress if resuming
    if start_episode > 0:
        progress = load_progress(output_dir, repo_id)
        success_count = progress.get("success_count", 0)
        total_steps = progress.get("total_attempts", 0)  # reuse total_attempts for steps
    else:
        success_count = 0
        total_steps = 0

    for episode in range(start_episode, num_episodes):
        print(f"\n{'=' * 50}")
        print(f"Episode {episode + 1}/{num_episodes}")

        obs, info = env.reset(seed=seed + episode)
        if hasattr(policy, "reset"):
            policy.reset()

        lang_goal = getattr(env._env, "lang_goal", None) or "place an object"
        print(f"Language goal: {lang_goal}")

        recorder.start_episode(task=lang_goal)

        episode_reward = 0.0
        episode_success = False
        episode_steps = 0

        for step in range(max_steps):
            batch = _obs_to_batch(obs, lang_goal, device, env=env)

            with torch.no_grad():
                action = policy.select_action(batch)

            action_np = action.cpu().numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]

            if step % 50 == 0:
                ee_pos = obs.get("robot_state", {}).get("ee_pos", "N/A")
                print(f"    Step {step}: action={action_np[:3]}, ee_pos={ee_pos}")

            _record_frame(recorder, obs, action_np, attempt_idx=0)

            obs, reward, terminated, truncated, info = env.step(action_np)
            recorder.update_step_result(reward=reward, success=info.get("is_success", False))

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if info.get("is_success", False):
                episode_success = True

            if terminated or truncated:
                break

        recorder.end_episode(success=episode_success, total_reward=episode_reward, num_attempts=1)

        if episode_success:
            success_count += 1

        # Save progress after each episode
        completed_episodes = episode + 1
        save_progress(output_dir, repo_id, completed_episodes, success_count, total_steps)

        print(
            f"Episode result: success={episode_success}, reward={episode_reward:.2f}, "
            f"frames={episode_steps}, progress={completed_episodes}/{num_episodes}"
        )

        # Cleanup GPU memory to prevent leaks
        cleanup_gpu_memory()

    recorder.finalize()
    env.close()

    num_episodes - start_episode
    print(f"\n{'=' * 50}")
    print("Collection complete!")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Successful episodes: {success_count}")
    print(f"  Success rate: {success_count / num_episodes * 100:.1f}%")
    print(f"  Total steps: {total_steps}")


def _get_observation_data(env):
    """Get color images, depth images, and point cloud from environment.

    Returns:
        color_images: Dict mapping camera name to RGB image (H, W, 3)
        depth_images: Dict mapping camera name to depth image (H, W)
        point_cloud: Point cloud array (N, 6) with XYZ + RGB
    """
    # Get images from cameras
    color_images = {}
    depth_images = {}

    cam_names = ["front", "left", "right", "top", "side_left", "side_right", "overview"]
    for i, config in enumerate(env._env.agent_cams):
        cam_name = cam_names[i] if i < len(cam_names) else f"cam_{i}"
        # Only compute depth for cameras needed by GraspNet (front, left, right)
        if i < 3:
            color, depth, _ = env._env.render_camera(config)
            depth_images[cam_name] = depth
        else:
            # Fast RGB-only render for other cameras
            color = env._env.render_camera_fast(config, (480, 640))
        color_images[cam_name] = color

    # Get point cloud
    try:
        pcd = env._env.get_pointcloud_array(cameras=[0, 1, 2], max_points=20000)
    except Exception as e:
        print(f"Warning: Could not get point cloud: {e}")
        pcd = np.zeros((100, 6), dtype=np.float32)

    return color_images, depth_images, pcd


def _predict_grasp_pose(
    policy, color_images, depth_images, pcd, lang_goal, device, object_poses=None, env=None, is_pickplace=False
):
    """Use policy to predict a grasp pose.

    Args:
        is_pickplace: If True, apply workspace shift for pickplace task (original A2 uses
                     --workspace_shift only for pickplace, not standalone grasp)

    Returns:
        action: 7D grasp pose [x, y, z, qx, qy, qz, qw] or None if failed
        info: Dict with additional info
    """
    # Check if policy uses trained networks (direct_grounding=False)
    use_learned_networks = (
        hasattr(policy, "config")
        and hasattr(policy.config, "direct_grounding")
        and not policy.config.direct_grounding
    )

    # If using learned networks, prioritize _generate_grasp_target over oracle
    if use_learned_networks and hasattr(policy, "_generate_grasp_target"):
        batch = None
        try:
            # Build batch for internal method
            batch = {
                "task": lang_goal,
                "point_cloud": torch.from_numpy(pcd).float().unsqueeze(0).to(device),
                "is_pickplace": is_pickplace,  # For workspace shift (only for pickplace)
            }
            # Pass target object poses for post-grasp filtering
            if object_poses:
                batch["target_object_poses"] = object_poses

            # Add all camera images (for per-point CLIP feature extraction)
            if color_images:
                for cam_name, img in color_images.items():
                    if img is not None:
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0).to(device)

            # Add camera configs for point-to-pixel projection (required for per-point CLIP features)
            if env is not None and hasattr(env, "_env") and hasattr(env._env, "agent_cams"):
                camera_configs = []
                cam_names = ["front", "left", "right"]
                for i, cam_name in enumerate(cam_names):
                    if i < len(env._env.agent_cams):
                        camera_configs.append(env._env.agent_cams[i])
                batch["camera_configs"] = camera_configs
                batch["color_images"] = color_images  # Raw numpy images for feature field
                batch["depth_images"] = depth_images  # Depth images for depth-weighted fusion

            policy._generate_grasp_target(batch)
            if hasattr(policy, "_target_pose") and policy._target_pose is not None:
                grasp_pose = policy._target_pose.copy()
                # Ensure minimum grasp z height for reliable grasping
                if grasp_pose[2] < 0.025:
                    grasp_pose[2] = 0.025
                return grasp_pose, {"source": "learned_networks"}
        except Exception as e:
            print(f"Warning: learned networks grasp failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup batch tensors to prevent GPU memory leaks
            if batch is not None:
                del batch

    # Fallback to oracle object pose filtering when available (A2_new-style)
    if object_poses and hasattr(policy, "_get_grasp_generator"):
        try:
            grasp_generator = policy._get_grasp_generator()
            pts_pos = pcd[:, :3]
            pts_colors = None
            if pcd.shape[1] >= 6:
                pts_colors = pcd[:, 3:6] / 255.0

            grasp_poses, gg = grasp_generator.generate_grasps(pts_pos, pts_colors)
            if len(grasp_poses) > 0 and gg is not None:
                centers = np.array([pose[:3, 3] for pose in object_poses.values()])
                if centers.size > 0:
                    rs = gg.rotation_matrices
                    depths = gg.depths
                    ts = gg.translations + rs[:, :, 0] * np.vstack((depths, depths, depths)).T

                    # Find grasps near any target object (simpler, more robust)
                    all_dists = np.array([np.linalg.norm(ts - c, axis=1) for c in centers])
                    dists = np.min(all_dists, axis=0)  # Min distance to any target

                    angle_mask = rs[:, 2, 0] < -np.cos(np.deg2rad(15.0))
                    obj_mask = dists < 0.08
                    mask = obj_mask if np.sum(obj_mask & angle_mask) < 1 else obj_mask & angle_mask

                    if np.sum(mask) < 1:
                        mask = np.ones(len(ts), dtype=bool)

                    candidate_indices = np.where(mask)[0]
                    best_idx = candidate_indices[int(np.argmin(dists[candidate_indices]))]

                    grasp_pose = grasp_poses[best_idx].copy()
                    return grasp_pose, {"source": "oracle_object_filter"}
        except Exception as e:
            print(f"Warning: oracle grasp filtering failed: {e}")

    # Check if policy has predict_grasp_from_pointcloud method (A2Policy)
    if hasattr(policy, "predict_grasp_from_pointcloud"):
        try:
            action, info = policy.predict_grasp_from_pointcloud(
                point_cloud=pcd,
                lang_goal=lang_goal,
                color_images=color_images,
                depth_images=depth_images,
            )
            return action, info
        except Exception as e:
            print(f"Warning: predict_grasp_from_pointcloud failed: {e}")

    # Check if policy has _generate_grasp_target (internal A2Policy method)
    if hasattr(policy, "_generate_grasp_target"):
        batch = None
        try:
            # Build batch for internal method
            batch = {
                "task": lang_goal,
                "point_cloud": torch.from_numpy(pcd).float().unsqueeze(0).to(device),
            }

            # Add all camera images (for per-point CLIP feature extraction)
            if color_images:
                for cam_name, img in color_images.items():
                    if img is not None:
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0).to(device)

            # Add camera configs for point-to-pixel projection (required for per-point CLIP features)
            if env is not None and hasattr(env, "_env") and hasattr(env._env, "agent_cams"):
                camera_configs = []
                cam_names = ["front", "left", "right"]
                for i, cam_name in enumerate(cam_names):
                    if i < len(env._env.agent_cams):
                        camera_configs.append(env._env.agent_cams[i])
                batch["camera_configs"] = camera_configs
                batch["color_images"] = color_images  # Raw numpy images for feature field
                batch["depth_images"] = depth_images  # Depth images for depth-weighted fusion

            policy._generate_grasp_target(batch)
            if hasattr(policy, "_target_pose") and policy._target_pose is not None:
                grasp_pose = policy._target_pose.copy()
                # Ensure minimum grasp z height for reliable grasping
                if grasp_pose[2] < 0.025:
                    grasp_pose[2] = 0.025
                return grasp_pose, {"source": "internal"}
        except Exception as e:
            print(f"Warning: _generate_grasp_target failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Cleanup batch tensors to prevent GPU memory leaks
            if batch is not None:
                del batch

    # Fallback: random grasp in workspace
    print("Warning: Using random grasp fallback")
    x = np.random.uniform(0.3, 0.7)
    y = np.random.uniform(-0.2, 0.2)
    z = np.random.uniform(0.02, 0.10)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([x, y, z, *quat], dtype=np.float32), {"random": True}


def _topdown_place_quat() -> np.ndarray:
    """Quaternion (xyzw) for top-down EE orientation in this env."""
    # Derived from A2_new VLA place dataset: keeps EE x-axis down with correct yaw.
    return np.array([0.17277328, 0.98479404, 0.01092275, -0.01451854], dtype=np.float32)


def _predict_place_pose(policy, color_images, depth_images, pcd, lang_goal, device, env=None, is_pickplace=False):
    """Use policy to predict a place pose.

    Args:
        is_pickplace: If True, apply workspace shift for pickplace task (original A2 uses
                     --workspace_shift only for pickplace, not standalone place)

    Returns:
        action: 7D place pose [x, y, z, qx, qy, qz, qw] or None if failed
        info: Dict with additional info
    """
    # Check if policy has _generate_place_target method (internal A2Policy method)
    if hasattr(policy, "_generate_place_target"):
        batch = None
        try:
            # Build batch for internal method
            batch = {
                "task": lang_goal,
                "point_cloud": torch.from_numpy(pcd).float().unsqueeze(0).to(device),
                "is_pickplace": is_pickplace,  # For workspace shift (only for pickplace)
            }

            # Add all camera images
            if color_images:
                for cam_name, img in color_images.items():
                    if img is not None:
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0).to(device)

            # Add camera configs for point-to-pixel projection
            if env is not None and hasattr(env, "_env") and hasattr(env._env, "agent_cams"):
                camera_configs = []
                cam_names = ["front", "left", "right"]
                for i, cam_name in enumerate(cam_names):
                    if i < len(env._env.agent_cams):
                        camera_configs.append(env._env.agent_cams[i])
                batch["camera_configs"] = camera_configs
                batch["color_images"] = color_images
                batch["depth_images"] = depth_images

            # Add reference object info from environment
            if env is not None and hasattr(env, "_env"):
                # Get reference object information
                reference_obj_ids = getattr(env._env, "reference_obj_ids", [])
                reference_obj_dirs = getattr(env._env, "reference_obj_dirs", [])
                reference_direction = getattr(env._env, "reference_direction", None)
                reference_obj_name = getattr(env._env, "reference_obj_name", None)

                # Use first reference_obj_dir if reference_direction not set
                if not reference_direction and reference_obj_dirs:
                    reference_direction = reference_obj_dirs[0]

                if reference_obj_ids:
                    reference_positions = []
                    reference_sizes = []
                    for ref_id in reference_obj_ids:
                        pos, quat, size = env._env.obj_info(ref_id)
                        reference_positions.append(pos)
                        reference_sizes.append(size if size is not None else [0.05, 0.05, 0.05])

                    batch["reference_positions"] = reference_positions
                    batch["reference_sizes"] = reference_sizes

                if reference_direction:
                    batch["direction"] = reference_direction
                if reference_obj_name:
                    batch["reference_obj_name"] = reference_obj_name

                # Also get all object positions for place candidate generation
                if hasattr(env._env, "get_true_object_poses"):
                    object_poses = env._env.get_true_object_poses()
                    batch["object_poses"] = object_poses

            policy._generate_place_target(batch)
            if hasattr(policy, "_target_pose") and policy._target_pose is not None:
                action = policy._target_pose.copy()
                action[3:7] = _topdown_place_quat()
                # Check if selection is valid (matching original A2 evaluation)
                # Default to False if not set - conservative approach matching original A2
                is_valid = getattr(policy, "_place_selection_valid", False)
                return action, {"source": "internal", "place_valid": is_valid}
        except Exception as e:
            print(f"Warning: _generate_place_target failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if batch is not None:
                del batch

    # Check if policy has predict_place method (legacy interface)
    if hasattr(policy, "predict_place"):
        try:
            # Pass env for reference object info
            action, info = policy.predict_place(
                color_images, depth_images, pcd, lang_goal, env=env
            )
            if action is not None and len(action) >= 7:
                action = np.array(action, dtype=np.float32)
                action[3:7] = _topdown_place_quat()
            return action, info
        except Exception as e:
            print(f"Warning: predict_place failed: {e}")

    # Fallback: random place in workspace
    print("Warning: Using random place fallback")
    x = np.random.uniform(0.3, 0.7)
    y = np.random.uniform(-0.2, 0.2)
    z = 0.05
    action = np.array([x, y, z, *_topdown_place_quat()], dtype=np.float32)
    return action, {"random": True}


def _obs_to_batch(obs: dict, lang_goal: str, device: str, env=None) -> dict[str, torch.Tensor]:
    """Convert gymnasium observation to policy batch format.

    Args:
        obs: Observation from A2Env
        lang_goal: Language goal string
        device: Device for tensors
        env: Optional A2Env instance for point cloud access

    Returns:
        Batch dictionary with observation.state, observation.images.*, task
    """
    batch = {}

    # Convert robot state to observation.state
    # Format: [ee_pos(3), ee_quat(4), joints(6), gripper(1)] = 14D
    if "robot_state" in obs:
        robot_state = obs["robot_state"]
        state = np.concatenate(
            [
                robot_state["ee_pos"],
                robot_state["ee_quat"],
                robot_state["joints"],
                robot_state["gripper_angle"],
            ]
        )
        batch["observation.state"] = torch.from_numpy(state).float().unsqueeze(0).to(device)

    # Convert images to observation.images.*
    if "pixels" in obs:
        for key, img in obs["pixels"].items():
            # Convert HWC to CHW and normalize to [0, 1]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch[f"observation.images.{key}"] = img_tensor.unsqueeze(0).to(device)

    # Add depth if available
    if "depth" in obs:
        batch["observation.depth"] = torch.from_numpy(obs["depth"]).float().unsqueeze(0).to(device)

    # Add point cloud from environment if available
    if env is not None and hasattr(env, "_env"):
        try:
            pc = env._env.get_pointcloud_array(cameras=[0, 1, 2], max_points=20000)
            batch["point_cloud"] = torch.from_numpy(pc).float().unsqueeze(0).to(device)
        except Exception as e:
            print(f"Warning: Could not get point cloud: {e}")

    # Add task/language goal
    batch["task"] = lang_goal

    return batch


def _record_frame(recorder: VLARecorder, obs: dict, action: np.ndarray, attempt_idx: int = 0):
    """Record a single frame for the dataset.

    Args:
        recorder: VLARecorder instance
        obs: Current observation
        action: Action being executed
        attempt_idx: Current grasp/place attempt index (0 for first attempt)
    """
    # Set action for this frame - attempt_idx should be the grasp attempt number, not the step
    recorder.set_action(action, attempt_idx=attempt_idx)

    # Build images dict for recording
    images = {}
    if "pixels" in obs:
        images = obs["pixels"]

    # Build robot_state dict for recording
    robot_state = {}
    if "robot_state" in obs:
        rs = obs["robot_state"]
        robot_state = {
            "ee_pos": rs["ee_pos"],
            "ee_quat": rs["ee_quat"],
            "joints": rs["joints"],
            "gripper_angle": float(rs["gripper_angle"][0])
            if rs["gripper_angle"].ndim > 0
            else float(rs["gripper_angle"]),
        }
    else:
        # Default robot state if not available
        robot_state = {
            "ee_pos": np.zeros(3, dtype=np.float64),
            "ee_quat": np.array([0, 0, 0, 1], dtype=np.float64),
            "joints": np.zeros(6, dtype=np.float64),
            "gripper_angle": 0.0,
        }

    # Record the frame
    if images:
        recorder.record_frame(images=images, robot_state=robot_state)


def parse_args():
    parser = argparse.ArgumentParser(description="A2 Data Collection with LeRobot Policies")

    # Policy arguments
    parser.add_argument(
        "--policy", type=str, default="random", help="Policy type (e.g., 'a2', 'act', 'diffusion', 'random')"
    )
    parser.add_argument(
        "--policy_path", type=str, default="", help="Pretrained path (HuggingFace repo ID or local path)"
    )

    # Task arguments
    parser.add_argument(
        "--task", type=str, default="grasp", choices=["grasp", "place", "pick_and_place"], help="Task type"
    )
    parser.add_argument(
        "--object_set", type=str, default="train", choices=["train", "test"], help="Object set to use"
    )

    # Collection arguments
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument(
        "--max_attempts", type=int, default=8, help="Maximum grasp/place attempts per episode (pose mode)"
    )
    parser.add_argument("--num_objects", type=int, default=8, help="Number of objects in scene")
    parser.add_argument(
        "--action_mode",
        type=str,
        default="pose",
        choices=["pose", "delta_ee"],
        help="Action mode: 'pose' (A2-style full trajectory) or 'delta_ee' (VLA-style)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="data/a2_collected", help="Output directory for dataset"
    )
    parser.add_argument(
        "--repo_id", type=str, default="local/a2_collected", help="LeRobot dataset repository ID"
    )

    # Recording arguments
    parser.add_argument(
        "--cameras", type=str, default="front,overview,gripper", help="Comma-separated list of cameras"
    )
    parser.add_argument("--fps", type=int, default=30, help="Recording FPS")
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--image_width", type=int, default=640)

    parser.add_argument(
        "--skip_failed_attempts",
        action="store_true",
        help="Skip recording failed attempts (grasp/pick-and-place)",
    )
    parser.add_argument(
        "--distinguish_failed_attempts",
        action="store_true",
        help="Record full dataset plus success-only dataset",
    )

    # Oracle grasp (place/pick-and-place)
    parser.add_argument(
        "--oracle_grasp",
        action="store_true",
        help="Use oracle grasp during place/pick-and-place collection",
    )

    # Resume arguments
    parser.add_argument("--resume", action="store_true", help="Resume from last saved progress")
    parser.add_argument(
        "--start_episode", type=int, default=0, help="Episode number to start from (for manual resume)"
    )

    # Memory management
    parser.add_argument(
        "--env_reset_interval",
        type=int,
        default=10,
        help="Recreate environment every N episodes to prevent GPU memory leaks (0 to disable)",
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse cameras
    cameras = [c.strip() for c in args.cameras.split(",")]
    image_size = (args.image_height, args.image_width)

    # Determine device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load policy
    print(f"\nLoading policy: {args.policy}")
    policy = load_policy(
        policy_type=args.policy,
        policy_path=args.policy_path if args.policy_path else None,
        device=device,
    )
    print(f"Policy loaded: {type(policy).__name__}")

    # Run collection
    collect_data(
        policy=policy,
        task=args.task,
        object_set=args.object_set,
        num_episodes=args.num_episodes,
        max_attempts_per_episode=args.max_attempts,
        num_objects=args.num_objects,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        cameras=cameras,
        fps=args.fps,
        image_size=image_size,
        seed=args.seed,
        gui=args.gui,
        device=device,
        action_mode=args.action_mode,
        start_episode=args.start_episode,
        resume=args.resume,
        oracle_grasp=args.oracle_grasp,
        skip_failed_attempts=args.skip_failed_attempts,
        distinguish_failed_attempts=args.distinguish_failed_attempts,
        env_reset_interval=args.env_reset_interval,
    )


if __name__ == "__main__":
    main()
