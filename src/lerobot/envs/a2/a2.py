#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A2 Environment for tabletop manipulation with UR5e robot.

This module provides a gymnasium-compatible environment for the A2 simulation,
supporting grasp and pick-and-place tasks with configurable object sets.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from huggingface_hub import snapshot_download


def get_a2_assets_path() -> Path:
    """Get path to A2 assets, downloading if necessary.

    Downloads from HuggingFace: dgrachev/a2_assets
    Contains: simplified_objects, unseen_objects, ur5e, workspace, testing_cases
    """
    cache_dir = Path.home() / ".cache" / "a2_assets"

    if not cache_dir.exists() or not (cache_dir / "simplified_objects").exists():
        print("Downloading A2 assets from HuggingFace...")
        snapshot_download(
            repo_id="dgrachev/a2_assets",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        print(f"A2 assets downloaded to {cache_dir}")

    return cache_dir


def get_testing_cases_path(task: str = "grasp", object_set: str = "train") -> Path:
    """Get path to testing cases for benchmark evaluation.

    Args:
        task: "grasp", "place", or "pickplace"
        object_set: "train" (seen) or "test" (unseen)

    Returns:
        Path to the testing cases directory
    """
    assets_path = get_a2_assets_path()

    task_dir_map = {
        "grasp": "grasp_testing_cases",
        "place": "place_testing_cases",
        "pickplace": "pp_testing_cases",
        "pick_and_place": "pp_testing_cases",
    }

    set_dir_map = {
        "train": "seen",
        "test": "unseen",
    }

    task_dir = task_dir_map.get(task, "grasp_testing_cases")
    set_dir = set_dir_map.get(object_set, "seen")

    return assets_path / "testing_cases" / task_dir / set_dir


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list, tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


# Camera name mapping for LeRobot convention
DEFAULT_CAMERA_MAPPING = {
    "front": "image",
    "overview": "image2",
    "gripper": "image3",
}


class A2Env(gym.Env):
    """A2 Environment for tabletop manipulation.

    Supports grasp and pick-and-place tasks with UR5e robot and Robotiq gripper.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "grasp",
        object_set: str = "train",
        num_objects: int = 8,
        camera_name: str | Sequence[str] = "front,overview,gripper",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 640,
        observation_height: int = 480,
        include_depth: bool = True,
        action_mode: str = "pose",
        camera_name_mapping: dict[str, str] | None = None,
        episode_length: int = 500,
        gui: bool = False,
    ):
        """Initialize A2 Environment.

        Args:
            task: Task type - "grasp" or "pick_and_place"
            object_set: Object set - "train" (simplified_objects) or "test" (unseen_objects)
            num_objects: Number of objects to spawn
            camera_name: Cameras to use (comma-separated or list)
            obs_type: Observation type - "pixels" or "pixels_agent_pos"
            render_mode: Render mode for gymnasium
            observation_width: Image width
            observation_height: Image height
            include_depth: Include depth observation from front camera
            action_mode: Action type - "pose", "delta_ee", or "joint"
            camera_name_mapping: Map camera names to observation keys
            episode_length: Maximum steps per episode
            gui: Show PyBullet GUI
        """
        super().__init__()

        self.task = task
        self.object_set = object_set
        self.num_objects = num_objects
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.include_depth = include_depth
        self.action_mode = action_mode
        self.gui = gui
        self._max_episode_steps = episode_length
        self._step_count = 0

        # Parse cameras
        self.camera_name = _parse_camera_names(camera_name)
        self.camera_name_mapping = camera_name_mapping or DEFAULT_CAMERA_MAPPING

        # Download assets if needed
        self.assets_path = get_a2_assets_path()

        # Initialize the underlying A2 environment
        self._env = None
        self._init_env()

        # Define observation space
        self._setup_observation_space()

        # Define action space
        self._setup_action_space()

    def _init_env(self):
        """Initialize the underlying PyBullet environment."""
        # Setup asset symlinks
        self._setup_asset_symlinks()

        # Import and create environment from local module
        from .a2_sim import Environment

        self._env = Environment(gui=self.gui, object_set=self.object_set)

        # Setup workspace based on task
        workspace = "extend" if self.task == "pick_and_place" else "raw"
        self._env.reset(workspace=workspace)

    def _setup_asset_symlinks(self):
        """Setup symlinks from cache to expected asset locations."""
        local_assets = Path("assets")
        if not local_assets.exists():
            local_assets.mkdir(parents=True, exist_ok=True)

        for asset_dir in ["simplified_objects", "unseen_objects", "ur5e", "workspace"]:
            src = self.assets_path / asset_dir
            dst = local_assets / asset_dir
            if src.exists() and not dst.exists():
                with contextlib.suppress(OSError):
                    dst.symlink_to(src)

    def _setup_observation_space(self):
        """Setup gymnasium observation space."""
        images = {}
        for cam in self.camera_name:
            key = self.camera_name_mapping.get(cam, cam)
            images[key] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        obs_dict = {"pixels": spaces.Dict(images)}

        if self.include_depth:
            obs_dict["depth"] = spaces.Box(
                low=0,
                high=10.0,
                shape=(self.observation_height, self.observation_width),
                dtype=np.float32,
            )

        if self.obs_type == "pixels_agent_pos":
            obs_dict["robot_state"] = spaces.Dict(
                {
                    "joints": spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float64),
                    "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                    "ee_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
                    "gripper_angle": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
                }
            )

        self.observation_space = spaces.Dict(obs_dict)

    def _setup_action_space(self):
        """Setup gymnasium action space."""
        if self.action_mode == "pose":
            # 6-DOF pose: xyz (3) + quaternion (4)
            self.action_space = spaces.Box(
                low=np.array([0.2, -0.5, 0.0, -1, -1, -1, -1], dtype=np.float32),
                high=np.array([0.8, 0.5, 0.5, 1, 1, 1, 1], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.action_mode == "delta_ee":
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(7,),
                dtype=np.float32,
            )
        elif self.action_mode == "joint":
            self.action_space = spaces.Box(
                low=-np.pi,
                high=np.pi,
                shape=(7,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

    def _get_observation(self) -> dict:
        """Get current observation from environment."""
        # Get camera images
        render_size = (self.observation_height, self.observation_width)
        camera_images = self._env.get_camera_images(
            cameras=self.camera_name,
            render_size=render_size,
        )

        # Map camera names to observation keys
        pixels = {}
        for cam, img in camera_images.items():
            key = self.camera_name_mapping.get(cam, cam)
            pixels[key] = img

        obs = {"pixels": pixels}

        # Get depth if requested
        if self.include_depth:
            front_config = self._env.agent_cams[0]
            _, depth, _ = self._env.render_camera(front_config)
            obs["depth"] = depth.astype(np.float32)

        # Get robot state if needed
        if self.obs_type == "pixels_agent_pos":
            robot_state = self._env.get_robot_state()
            obs["robot_state"] = {
                "joints": robot_state["joints"],
                "ee_pos": robot_state["ee_pos"],
                "ee_quat": robot_state["ee_quat"],
                "gripper_angle": np.array([robot_state["gripper_angle"]], dtype=np.float64),
            }

        return obs

    def reset(self, seed=None, **kwargs):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._step_count = 0

        if seed is not None:
            np.random.seed(seed)

        # Reset simulation
        workspace = "extend" if self.task == "pick_and_place" else "raw"
        self._env.reset(workspace=workspace)

        if self.task == "grasp":
            # Generate language goal first - this sets up target_obj_lst
            lang_goal = self._env.generate_lang_goal()

            # Add objects for grasp task (uses target_obj_lst)
            for _i in range(10):  # Retry up to 10 times
                self._env.add_objects(
                    num_obj=self.num_objects,
                    workspace_limits=self._env.bounds,
                )
                if len(self._env.target_obj_ids) > 0:
                    break
                # Reset and try again
                self._env.reset(workspace=workspace)
                lang_goal = self._env.generate_lang_goal()
        elif self.task == "place":
            # For place, add objects first then build place language goal
            self._env.add_objects_for_place(
                num_obj=self.num_objects,
                workspace_limits=self._env.bounds,
            )
            bbox_ids, bbox_centers, bbox_sizes = self._env.get_obj_bboxes_from_aabb()
            lang_goal, _, _, _, _ = self._env.generate_place_lang_goal(
                bbox_ids, bbox_centers, bbox_sizes, pp_file=False
            )
            if not lang_goal:
                lang_goal = "place an object"
        else:
            # For pick_and_place: generate BOTH grasp and place goals
            # First generate grasp goal (sets up target_obj_lst)
            grasp_lang_goal = self._env.generate_lang_goal()

            # Add objects using add_objects() to properly set target_obj_ids
            for _i in range(10):
                self._env.add_objects(
                    num_obj=self.num_objects,
                    workspace_limits=self._env.bounds,
                )
                if len(self._env.target_obj_ids) > 0:
                    break
                self._env.reset(workspace="extend")
                grasp_lang_goal = self._env.generate_lang_goal()

            # Generate place goal
            bbox_ids, bbox_centers, bbox_sizes = self._env.get_obj_bboxes_from_aabb()
            place_lang_goal, _, _, _, _ = self._env.generate_place_lang_goal(
                bbox_ids, bbox_centers, bbox_sizes, pp_file=True
            )

            # Combine goals: "grasp X and place it Y"
            if place_lang_goal:
                lang_goal = f"{grasp_lang_goal} and {place_lang_goal}"
            else:
                lang_goal = grasp_lang_goal

            # Store both for separate access
            self._env.grasp_lang_goal = grasp_lang_goal
            self._env.place_lang_goal = place_lang_goal

        observation = self._get_observation()
        info = {
            "is_success": False,
            "task": self.task,
            "object_set": self.object_set,
        }

        return observation, info

    def step(self, action: np.ndarray):
        """Execute action and return new observation."""
        self._step_count += 1

        if action.ndim != 1:
            raise ValueError(f"Expected 1-D action, got shape {action.shape}")

        success = False
        reward = 0.0

        if self.action_mode == "pose":
            # Pass action as 7D array (grasp/place expect this format)
            pose = action[:7]

            if self.task == "grasp":
                success, grasped_obj_id, _ = self._env.grasp(pose)
                reward = 1.0 if success and grasped_obj_id is not None else 0.0
            else:
                success = self._env.place(pose)
                reward = 1.0 if success else 0.0

        elif self.action_mode == "delta_ee":
            robot_state = self._env.get_robot_state()
            current_pos = robot_state["ee_pos"]
            current_quat = robot_state["ee_quat"]

            # Action is in [-1, 1] range, scale to meters (1.0 → 1cm movement)
            delta_pos = action[:3] * 0.01
            new_pos = current_pos + delta_pos

            pose = (tuple(new_pos), tuple(current_quat))
            # Use non-blocking VLA-style step (not blocking move_ee_pose)
            self._env.step_ee_pose(pose, num_sim_steps=8)

            if action[6] > 0.5:
                self._env.close_gripper(blocking=False)
            else:
                self._env.open_gripper(blocking=False)

        elif self.action_mode == "joint":
            joint_targets = action[:6]
            self._env.move_joints(joint_targets)

            if action[6] > 0.5:
                self._env.close_gripper()
            else:
                self._env.open_gripper()

        observation = self._get_observation()
        terminated = success
        truncated = self._step_count >= self._max_episode_steps

        info = {
            "is_success": success,
            "task": self.task,
            "step_count": self._step_count,
        }

        if terminated or truncated:
            info["final_info"] = {
                "is_success": success,
                "episode_length": self._step_count,
            }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render current frame."""
        images = self._env.get_camera_images(cameras=["front"])
        return images.get("front")

    def close(self):
        """Clean up environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    # ==================== Benchmark Methods ====================

    def load_test_case(self, file_path: str, test_type: str = "grasp"):
        """Load a benchmark test case from file.

        Args:
            file_path: Path to test case file
            test_type: Type of test - "grasp", "place", or "pickplace"

        Returns:
            For grasp: (success, lang_goal)
            For place: (success, lang_goal)
            For pickplace: (success, grasp_lang_goal, place_lang_goal)
        """
        workspace = "extend" if test_type == "pickplace" else "raw"
        self._env.reset(workspace=workspace)

        if test_type == "grasp":
            return self._env.add_object_push_from_file(file_path)
        elif test_type == "place":
            return self._env.add_object_push_from_place_file(file_path)
        elif test_type == "pickplace":
            return self._env.add_object_push_from_pickplace_file(file_path, mode="grasp")
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

    def get_lang_goal(self) -> str:
        """Get the current language goal."""
        return getattr(self._env, "lang_goal", None)

    def get_target_obj_ids(self) -> list:
        """Get list of target object IDs."""
        return getattr(self._env, "target_obj_ids", [])

    def get_reference_obj_ids(self) -> list:
        """Get list of reference object IDs (for place tasks)."""
        return getattr(self._env, "reference_obj_ids", [])

    # ==================== Data Collection Methods ====================

    def setup_vla_recorder(
        self,
        output_dir: str,
        repo_id: str = "local/a2_dataset",
        fps: int = 30,
        image_size: tuple = (480, 640),
        cameras: list = None,
    ):
        """Setup VLA recording for data collection.

        Args:
            output_dir: Directory to save the dataset
            repo_id: Repository ID for the LeRobot dataset
            fps: Recording frames per second
            image_size: Image size as (height, width)
            cameras: List of camera names to record

        Returns:
            VLARecorder instance
        """
        from .a2_recorder import VLARecorder

        cameras = cameras or list(self.camera_name)
        recorder = VLARecorder(
            output_dir=output_dir,
            repo_id=repo_id,
            fps=fps,
            image_size=image_size,
            cameras=cameras,
        )

        # Set up frame recording in the underlying environment
        render_size = (self.observation_height, self.observation_width)
        self._env.set_frame_recorder(
            recorder.record_frame,
            fps=fps,
            cameras=cameras,
            render_size=render_size,
        )

        return recorder

    def clear_vla_recorder(self):
        """Clear VLA recording setup."""
        self._env.clear_frame_recorder()

    # ==================== Low-level Access ====================

    @property
    def sim_env(self):
        """Access the underlying PyBullet simulation environment.

        This provides direct access to the Environment class for advanced
        operations like:
        - env.sim_env.grasp(pose)
        - env.sim_env.place(pose)
        - env.sim_env.get_true_object_poses()
        - env.sim_env.get_camera_images()
        """
        return self._env


def _make_env_fn(
    task: str,
    object_set: str,
    num_objects: int,
    camera_name: list[str],
    episode_length: int,
    gym_kwargs: dict[str, Any],
    episode_index: int = 0,
) -> Callable[[], A2Env]:
    """Create factory callable for A2Env."""

    def _make_env() -> A2Env:
        return A2Env(
            task=task,
            object_set=object_set,
            num_objects=num_objects,
            camera_name=camera_name,
            episode_length=episode_length,
            **gym_kwargs,
        )

    return _make_env


def create_a2_envs(
    task: str = "grasp",
    n_envs: int = 1,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "front,overview,gripper",
    object_set: str = "train",
    num_objects: int = 8,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int = 500,
) -> dict[str, dict[int, Any]]:
    """Create vectorized A2 environments.

    Returns:
        dict[task_name][task_id] -> vec_env
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    camera_names = _parse_camera_names(camera_name)

    print(f"Creating A2 envs | task={task} | n_envs={n_envs} | object_set={object_set}")

    fns = []
    for i in range(n_envs):
        fn = _make_env_fn(
            task=task,
            object_set=object_set,
            num_objects=num_objects,
            camera_name=camera_names,
            episode_length=episode_length,
            gym_kwargs=gym_kwargs,
            episode_index=i,
        )
        fns.append(fn)

    vec_env = env_cls(fns)
    return {task: {0: vec_env}}
