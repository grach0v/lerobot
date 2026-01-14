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

import os
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from huggingface_hub import snapshot_download


def get_a2_assets_path() -> Path:
    """Get path to A2 assets, downloading if necessary."""
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
        # Add A2 to path
        a2_path = Path(__file__).parent.parent.parent.parent.parent.parent / "A2_new"
        if not a2_path.exists():
            # Try current working directory
            a2_path = Path.cwd()
            if not (a2_path / "env" / "environment_sim.py").exists():
                a2_path = Path.cwd().parent

        if str(a2_path) not in sys.path:
            sys.path.insert(0, str(a2_path))

        # Setup asset symlinks
        self._setup_asset_symlinks()

        # Import and create environment
        from env.environment_sim import Environment
        self._env = Environment(gui=self.gui)

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
                try:
                    dst.symlink_to(src)
                except OSError:
                    pass

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
            obs_dict["robot_state"] = spaces.Dict({
                "joints": spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float64),
                "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "ee_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
                "gripper_angle": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
            })

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

        # Add objects based on object_set
        object_dir = "simplified_objects" if self.object_set == "train" else "unseen_objects"
        self._env.add_objects_for_place(
            num_obj=self.num_objects,
            workspace_limits=self._env.bounds,
        )

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
            # Convert action to pose tuple
            pos = tuple(action[:3])
            quat = tuple(action[3:7])
            pose = (pos, quat)

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

            delta_pos = action[:3] * 0.01
            new_pos = current_pos + delta_pos

            pose = (tuple(new_pos), tuple(current_quat))
            self._env.move_ee_pose(pose)

            if action[6] > 0.5:
                self._env.close_gripper()
            else:
                self._env.open_gripper()

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
            del self._env
            self._env = None


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
