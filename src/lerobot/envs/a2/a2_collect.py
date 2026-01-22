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

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.utils.import_utils import register_third_party_plugins


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
        json.dump({
            "completed_episodes": completed,
            "success_count": successes,
            "total_attempts": attempts,
        }, f)


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

    # Handle resume logic
    if resume:
        progress = load_progress(output_dir, repo_id)
        start_episode = progress["completed_episodes"]
        print(f"Resuming from episode {start_episode} (prev: {progress['success_count']} successes)")

    if start_episode > 0:
        print(f"Starting from episode {start_episode}/{num_episodes}")

    if action_mode == "pose":
        _collect_pose_mode(env, policy, recorder, num_episodes, max_attempts_per_episode,
                          seed, device, task, cameras, fps, output_dir, repo_id, start_episode)
    else:
        _collect_delta_mode(env, policy, recorder, num_episodes, max_attempts_per_episode,
                           seed, device, task, output_dir, repo_id, start_episode)


def _collect_pose_mode(env, policy, recorder, num_episodes, max_attempts, seed, device, task, cameras, fps,
                       output_dir="", repo_id="", start_episode=0):
    """A2-style collection: policy outputs poses, environment executes full trajectories."""
    # Load previous progress if resuming
    if start_episode > 0:
        progress = load_progress(output_dir, repo_id)
        success_count = progress.get("success_count", 0)
        total_attempts = progress.get("total_attempts", 0)
    else:
        success_count = 0
        total_attempts = 0

    # Set up frame recording callback on the underlying environment
    def frame_callback(images: dict, robot_state: dict):
        """Called by environment during trajectory execution."""
        # Get current action from environment
        action = env._env._current_action
        if action is None:
            action = np.zeros(7, dtype=np.float32)
        recorder.set_action(action, attempt_idx=recorder._current_attempt if hasattr(recorder, '_current_attempt') else 0)
        recorder.record_frame(images=images, robot_state=robot_state)

    # Register the frame recorder
    env._env.set_frame_recorder(frame_callback, fps=fps, cameras=cameras)

    for episode in range(start_episode, num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")

        # Reset environment and policy
        obs, info = env.reset(seed=seed + episode)
        if hasattr(policy, 'reset'):
            policy.reset()

        # Get language goal
        lang_goal = getattr(env._env, 'lang_goal', None) or f"{task} an object"
        print(f"Language goal: {lang_goal}")

        # Start recording
        recorder.start_episode(task=lang_goal)
        recorder._current_attempt = 0

        episode_reward = 0.0
        episode_success = False
        attempt_count = 0

        for attempt in range(max_attempts):
            # Check if target objects still in workspace
            out_of_workspace = []
            for obj_id in env._env.target_obj_ids:
                pos, _, _ = env._env.obj_info(obj_id)
                bounds = env._env.bounds
                if pos[0] < bounds[0][0] or pos[0] > bounds[0][1] \
                   or pos[1] < bounds[1][0] or pos[1] > bounds[1][1]:
                    out_of_workspace.append(obj_id)

            if len(out_of_workspace) == len(env._env.target_obj_ids):
                print("  Target objects are not in the scene!")
                break

            # Get images and point cloud for policy
            color_images, depth_images, pcd = _get_observation_data(env)

            # Generate grasp/place pose using policy
            if task == "grasp":
                action, info_dict = _predict_grasp_pose(policy, color_images, depth_images, pcd, lang_goal, device)
            else:
                action, info_dict = _predict_place_pose(policy, color_images, depth_images, pcd, lang_goal, device)

            if action is None:
                print("  Warning: No valid action generated, skipping attempt")
                break

            print(f"  Attempt {attempt + 1}: action_pos={action[:3]}")

            # Set action for frame recording
            recorder._current_attempt = attempt
            env._env.set_current_action(action)

            # Execute action (this triggers internal frame recording)
            obs, reward, terminated, truncated, info = env.step(action)

            # Update step result
            recorder.update_step_result(reward=reward, success=info.get("is_success", False))

            episode_reward += reward
            attempt_count += 1
            total_attempts += 1

            if info.get("is_success", False):
                episode_success = True
                print(f"  SUCCESS at attempt {attempt + 1}!")
                break

            if terminated or truncated:
                break

        # End episode
        recorder.end_episode(
            success=episode_success,
            total_reward=episode_reward,
            num_attempts=attempt_count
        )

        if episode_success:
            success_count += 1

        # Save progress after each episode (for crash recovery)
        completed_episodes = episode + 1
        save_progress(output_dir, repo_id, completed_episodes, success_count, total_attempts)

        print(f"Episode result: success={episode_success}, reward={episode_reward:.2f}, "
              f"attempts={attempt_count}, progress={completed_episodes}/{num_episodes}")

    # Finalize
    recorder.finalize()
    env.close()

    # Calculate actual episodes completed
    episodes_run = num_episodes - start_episode
    print(f"\n{'='*50}")
    print(f"Collection complete!")
    print(f"  Episodes this run: {episodes_run}")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Successful episodes: {success_count}")
    print(f"  Success rate: {success_count / num_episodes * 100:.1f}%")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Dataset saved to: {output_dir}/{repo_id}")


def _collect_delta_mode(env, policy, recorder, num_episodes, max_steps, seed, device, task,
                        output_dir="", repo_id="", start_episode=0):
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
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")

        obs, info = env.reset(seed=seed + episode)
        if hasattr(policy, 'reset'):
            policy.reset()

        lang_goal = getattr(env._env, 'lang_goal', None) or f"{task} an object"
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

        recorder.end_episode(
            success=episode_success,
            total_reward=episode_reward,
            num_attempts=1
        )

        if episode_success:
            success_count += 1

        # Save progress after each episode
        completed_episodes = episode + 1
        save_progress(output_dir, repo_id, completed_episodes, success_count, total_steps)

        print(f"Episode result: success={episode_success}, reward={episode_reward:.2f}, "
              f"frames={episode_steps}, progress={completed_episodes}/{num_episodes}")

    recorder.finalize()
    env.close()

    episodes_run = num_episodes - start_episode
    print(f"\n{'='*50}")
    print(f"Collection complete!")
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

    for i, config in enumerate(env._env.agent_cams):
        color, depth, _ = env._env.render_camera(config)
        cam_name = ["front", "overview", "gripper"][i] if i < 3 else f"cam_{i}"
        color_images[cam_name] = color
        depth_images[cam_name] = depth

    # Get point cloud
    try:
        pcd = env._env.get_pointcloud_array(cameras=[0, 1, 2], max_points=20000)
    except Exception as e:
        print(f"Warning: Could not get point cloud: {e}")
        pcd = np.zeros((100, 6), dtype=np.float32)

    return color_images, depth_images, pcd


def _predict_grasp_pose(policy, color_images, depth_images, pcd, lang_goal, device):
    """Use policy to predict a grasp pose.

    Returns:
        action: 7D grasp pose [x, y, z, qx, qy, qz, qw] or None if failed
        info: Dict with additional info
    """
    # Check if policy has predict_grasp_from_pointcloud method (A2Policy)
    if hasattr(policy, 'predict_grasp_from_pointcloud'):
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
    if hasattr(policy, '_generate_grasp_target'):
        try:
            # Build batch for internal method
            batch = {
                "task": lang_goal,
                "point_cloud": torch.from_numpy(pcd).float().unsqueeze(0).to(device),
            }
            policy._generate_grasp_target(batch)
            if hasattr(policy, '_target_pose') and policy._target_pose is not None:
                return policy._target_pose, {"source": "internal"}
        except Exception as e:
            print(f"Warning: _generate_grasp_target failed: {e}")
            import traceback
            traceback.print_exc()

    # Fallback: random grasp in workspace
    print("Warning: Using random grasp fallback")
    x = np.random.uniform(0.3, 0.7)
    y = np.random.uniform(-0.2, 0.2)
    z = np.random.uniform(0.02, 0.10)
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    return np.array([x, y, z, *quat], dtype=np.float32), {"random": True}


def _predict_place_pose(policy, color_images, depth_images, pcd, lang_goal, device):
    """Use policy to predict a place pose.

    Returns:
        action: 7D place pose [x, y, z, qx, qy, qz, qw] or None if failed
        info: Dict with additional info
    """
    # Check if policy has predict_place method
    if hasattr(policy, 'predict_place'):
        try:
            action, info = policy.predict_place(color_images, depth_images, pcd, lang_goal)
            return action, info
        except Exception as e:
            print(f"Warning: predict_place failed: {e}")

    # Fallback: random place in workspace
    print("Warning: Using random place fallback")
    x = np.random.uniform(0.3, 0.7)
    y = np.random.uniform(-0.2, 0.2)
    z = 0.05
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    return np.array([x, y, z, *quat], dtype=np.float32), {"random": True}


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
        state = np.concatenate([
            robot_state["ee_pos"],
            robot_state["ee_quat"],
            robot_state["joints"],
            robot_state["gripper_angle"],
        ])
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


def _record_frame(recorder: "VLARecorder", obs: dict, action: np.ndarray, attempt_idx: int = 0):
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
            "gripper_angle": float(rs["gripper_angle"][0]) if rs["gripper_angle"].ndim > 0 else float(rs["gripper_angle"]),
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
    parser.add_argument("--policy", type=str, default="random",
                        help="Policy type (e.g., 'a2', 'act', 'diffusion', 'random')")
    parser.add_argument("--policy_path", type=str, default="",
                        help="Pretrained path (HuggingFace repo ID or local path)")

    # Task arguments
    parser.add_argument("--task", type=str, default="grasp",
                        choices=["grasp", "place", "pick_and_place"],
                        help="Task type")
    parser.add_argument("--object_set", type=str, default="train",
                        choices=["train", "test"],
                        help="Object set to use")

    # Collection arguments
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--max_attempts", type=int, default=8,
                        help="Maximum grasp/place attempts per episode (pose mode)")
    parser.add_argument("--num_objects", type=int, default=8,
                        help="Number of objects in scene")
    parser.add_argument("--action_mode", type=str, default="pose",
                        choices=["pose", "delta_ee"],
                        help="Action mode: 'pose' (A2-style full trajectory) or 'delta_ee' (VLA-style)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="data/a2_collected",
                        help="Output directory for dataset")
    parser.add_argument("--repo_id", type=str, default="local/a2_collected",
                        help="LeRobot dataset repository ID")

    # Recording arguments
    parser.add_argument("--cameras", type=str, default="front,overview,gripper",
                        help="Comma-separated list of cameras")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording FPS")
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--image_width", type=int, default=640)

    # Resume arguments
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved progress")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="Episode number to start from (for manual resume)")

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
    )


if __name__ == "__main__":
    main()
