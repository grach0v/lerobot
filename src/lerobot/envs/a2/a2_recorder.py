"""VLA (Vision-Language-Action) recorder for LeRobot dataset format.

This module provides the VLARecorder class for recording A2 environment
episodes to the LeRobot dataset format, supporting multi-view images,
robot state, and actions.
"""

import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


class VLARecorder:
    """Records episodes to LeRobot dataset format for VLA training.

    This class wraps LeRobot's dataset API to record robot manipulation
    episodes with multi-view images, robot state, and actions.

    One A2 episode = one LeRobot episode, with multiple attempts/steps recorded.
    """

    # Available camera names
    AVAILABLE_CAMERAS = ['front', 'left', 'right', 'top', 'side_left', 'side_right', 'overview', 'gripper']

    def __init__(
        self,
        output_dir: str,
        repo_id: str,
        fps: int = 30,
        image_size: tuple = (720, 1280),
        cameras: list = None,
        use_videos: bool = True,
    ):
        """Initialize VLA recorder.

        Args:
            output_dir: Directory to save the dataset.
            repo_id: Repository ID for the LeRobot dataset.
            fps: Recording frames per second.
            image_size: Image size as (height, width).
            cameras: List of camera names to record ['front', 'left', 'right'].
            use_videos: Whether to encode frames as videos (True) or images (False).
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import shutil

        self.fps = fps
        # Create unique dataset directory based on repo_id
        # LeRobotDataset.create expects this directory to not exist
        self.output_dir = Path(output_dir) / repo_id.replace("/", "_")
        if self.output_dir.exists():
            print(f"Warning: Removing existing dataset directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.image_size = image_size
        self.cameras = cameras if cameras else self.AVAILABLE_CAMERAS

        # Validate cameras
        for cam in self.cameras:
            if cam not in self.AVAILABLE_CAMERAS:
                raise ValueError(f"Unknown camera '{cam}'. Available: {self.AVAILABLE_CAMERAS}")

        self.current_action: Optional[np.ndarray] = None
        self.current_task: Optional[str] = None
        self.current_attempt: int = 0
        self.current_reward: float = 0.0
        self.current_success: bool = False
        self._episode_started = False
        self._frame_count = 0
        self._episode_count = 0

        # Define LeRobot features for our robot setup
        # Robot state: EE pose (7D) + joint positions (6D) + gripper (1D) = 14D
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": [
                    "ee_x", "ee_y", "ee_z",
                    "ee_qx", "ee_qy", "ee_qz", "ee_qw",
                    "joint_0", "joint_1", "joint_2",
                    "joint_3", "joint_4", "joint_5",
                    "gripper"
                ]
            },
            # Action: 7D grasp/place pose (position + quaternion)
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]
            },
            # Attempt index within episode (0-indexed)
            "attempt_idx": {
                "dtype": "int32",
                "shape": (1,),
                "names": ["attempt"]
            },
            # Step reward for current action
            "step_reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"]
            },
            # Whether this step was successful
            "step_success": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["success"]
            },
        }

        # Add only selected cameras
        for cam_name in self.cameras:
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video" if use_videos else "image",
                "shape": (3, image_size[0], image_size[1]),
                "names": ["channels", "height", "width"]
            }

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=str(self.output_dir),
            robot_type="ur5e_robotiq",
            use_videos=use_videos,
            image_writer_processes=0,  # Use threads only (more reliable than subprocesses)
            image_writer_threads=8,    # 8 threads for async image writing
        )

        print(f"VLARecorder initialized: {repo_id} at {output_dir}")
        print(f"  FPS: {fps}, Image size: {image_size[0]}x{image_size[1]}, Cameras: {self.cameras}")

    def start_episode(self, task: str):
        """Start recording a new episode.

        Args:
            task: Language description of the task (e.g., "grasp the red cube").
        """
        if self._episode_started:
            print("Warning: start_episode called while episode already in progress")
            return

        self.current_task = task
        self.current_action = np.zeros(7, dtype=np.float32)
        self.current_attempt = 0
        self.current_reward = 0.0
        self.current_success = False
        self._episode_started = True
        self._frame_count = 0
        self._episode_start_time = time.time()
        print(f"  [VLA] Recording episode {self._episode_count}: {task[:50]}...")

    def set_action(self, action: np.ndarray, attempt_idx: int):
        """Set the current action being executed.

        Args:
            action: The 7D action pose being executed.
            attempt_idx: The attempt index within this episode (0-indexed).
        """
        self.current_action = np.array(action, dtype=np.float32)
        self.current_attempt = attempt_idx

    def update_step_result(self, reward: float, success: bool):
        """Update the result of the current step/action.

        Args:
            reward: The reward received for this action.
            success: Whether this action was successful.
        """
        self.current_reward = reward
        self.current_success = success

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target size if needed."""
        h, w = img.shape[:2]
        target_h, target_w = self.image_size
        if h != target_h or w != target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return img

    def record_frame(self, images: dict[str, np.ndarray], robot_state: dict[str, Any]):
        """Record a single frame - called by environment at recording FPS.

        Args:
            images: Dict with camera images {'front': array, 'left': array, 'right': array}
                   Each array is HWC format (height, width, channels).
            robot_state: Dict with robot state:
                - 'ee_pos': np.ndarray of shape (3,) - end effector position
                - 'ee_quat': np.ndarray of shape (4,) - end effector quaternion (xyzw)
                - 'joints': np.ndarray of shape (6,) - joint positions
                - 'gripper_angle': float - gripper angle
        """
        if not self._episode_started:
            return

        # Build state vector: [ee_pos(3), ee_quat(4), joints(6), gripper(1)] = 14D
        state = np.concatenate([
            robot_state['ee_pos'],
            robot_state['ee_quat'],
            robot_state['joints'],
            np.array([robot_state['gripper_angle']])
        ]).astype(np.float32)

        # Build frame with state, action, and metadata
        frame = {
            "observation.state": state,
            "action": self.current_action,
            "attempt_idx": np.array([self.current_attempt], dtype=np.int32),
            "step_reward": np.array([self.current_reward], dtype=np.float32),
            "step_success": np.array([self.current_success], dtype=bool),
            "task": self.current_task,
        }

        # Add only selected cameras, with resizing if needed
        for cam_name in self.cameras:
            img = self._resize_image(images[cam_name])
            # Convert from HWC to CHW format for LeRobot
            frame[f"observation.images.{cam_name}"] = np.ascontiguousarray(img.transpose(2, 0, 1))

        self.dataset.add_frame(frame)
        self._frame_count += 1

        # Print progress every 10 frames
        if self._frame_count % 10 == 0:
            print(f"    [VLA] Recorded {self._frame_count} frames (attempt {self.current_attempt})...", end='\r')
            sys.stdout.flush()

    def end_episode(self, success: bool = True, total_reward: float = 0.0, num_attempts: int = 0):
        """End current episode and save.

        Args:
            success: Whether the episode was successful.
            total_reward: Total reward accumulated in this episode.
            num_attempts: Number of attempts/steps in this episode.
        """
        if not self._episode_started:
            return

        if self._frame_count > 0:
            elapsed = time.time() - self._episode_start_time
            self.dataset.save_episode()
            print(f"  [VLA] Saved episode {self._episode_count}: {self._frame_count} frames, "
                  f"{num_attempts} attempts, reward={total_reward:.2f}, success={success} ({elapsed:.1f}s)")
            self._episode_count += 1
        else:
            print(f"  [VLA] Discarded empty episode {self._episode_count}")

        self._episode_started = False
        self.current_task = None
        self.current_action = None
        self.current_attempt = 0
        self.current_reward = 0.0
        self.current_success = False
        self._frame_count = 0

    def cancel_episode(self):
        """Cancel current episode without saving."""
        if self._episode_started:
            print(f"Cancelled episode {self._episode_count}")
            # Clear the episode buffer without saving
            self.dataset.clear_episode_buffer()
            self._episode_started = False
            self.current_task = None
            self.current_action = None
            self._frame_count = 0

    def finalize(self):
        """Finalize dataset when done collecting."""
        if self._episode_started:
            self.end_episode(success=False)
        self.dataset.finalize()
        print(f"VLARecorder finalized: {self._episode_count} episodes saved")

    @property
    def episode_count(self) -> int:
        """Return number of saved episodes."""
        return self._episode_count

    @property
    def is_recording(self) -> bool:
        """Return whether an episode is currently being recorded."""
        return self._episode_started
