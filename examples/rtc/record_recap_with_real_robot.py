#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Record RTC rollouts on a real robot and save a RECAP-ready LeRobot dataset.

This is the data-collection entrypoint for RECAP. It runs a pretrained
action-chunking policy through the standard RTC inference loop, records every
frame (observation + action + RECAP columns) into a `LeRobotDataset`, and
writes per-episode success/failure labels as dataset columns. Nothing here
depends on the value network or policy fine-tuning.

Two rollout modes are supported:

- **Autonomous**: the policy drives the robot. Standard RTC flow.
- **Human intervention**: at any point during an episode the operator can
  press `i` to switch to teleop control (requires `--teleop.*` flags). While
  the human drives, the policy keeps inferring — its actions are ignored by
  the robot, but the corresponding frames are still recorded with
  `intervention=1.0`. Press `i` again to hand back to the policy.

After every episode the operator is prompted for `success / failure / redo /
quit`. On save, per-frame RECAP target columns (`episode_success`,
`episode_failure`, `recap_reward`, `recap_return`,
`recap_normalized_return`, `recap_value_target_bin`, `intervention`) are
written directly into the dataset so downstream training scripts can read
them with no extra build step.

Example:

    uv run python examples/rtc/record_recap_with_real_robot.py \\
        --policy.path=lerobot/pi05_base \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=so100_follower \\
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --task="Pick up the cube" \\
        --dataset.repo_id=my_user/my_rollouts \\
        --dataset.num_episodes=10 \\
        --dataset.episode_time_s=45

    # Intervention mode: add a teleop config (e.g. another arm) and press `i`
    # during an episode to take over.
    uv run python examples/rtc/record_recap_with_real_robot.py ... \\
        --teleop.type=so100_leader --teleop.port=/dev/ttyACM1
"""

import logging
import math
import select
import sys
import termios
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.algorithms.recap_targets import (
    EPISODE_FAILURE_FEATURE,
    EPISODE_SUCCESS_FEATURE,
    RECAP_INTERVENTION_FEATURE,
    RECAP_NORMALIZED_RETURN_FEATURE,
    RECAP_REWARD_FEATURE,
    RECAP_RETURN_FEATURE,
    RECAP_VALUE_TARGET_FEATURE,
    build_recap_targets_from_success,
    write_episode_labels_csv,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import RobotConfig  # noqa: F401
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.utils.hub import HubMixin
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class RecapDatasetConfig:
    repo_id: str = field(default="", metadata={"help": "LeRobot dataset repo id."})
    root: str | None = field(default=None, metadata={"help": "Optional local dataset root."})
    num_episodes: int = field(default=10)
    episode_time_s: float = field(default=60.0)
    reset_time_s: float = field(default=15.0)
    video: bool = field(default=True)
    push_to_hub: bool = field(default=False)
    private: bool = field(default=True)
    resume: bool = field(default=True)
    num_image_writer_processes: int = field(default=0)
    num_image_writer_threads_per_camera: int = field(default=4)
    video_encoding_batch_size: int = field(default=1)
    vcodec: str = field(default="libsvtav1")
    recap_num_bins: int = field(default=50)
    recap_max_episode_steps: int = field(
        default=0,
        metadata={"help": "Overrides episode_time_s * fps when computing return normalisation."},
    )
    recap_c_fail: float = field(
        default=0.0,
        metadata={"help": "Terminal failure penalty; 0 defaults to max_episode_steps."},
    )


@dataclass
class RecapRecordConfig(HubMixin):
    policy: PreTrainedConfig | None = None
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )
    dataset: RecapDatasetConfig = field(default_factory=RecapDatasetConfig)
    task: str = field(default="")
    fps: float = field(default=10.0)
    device: str | None = field(default=None)
    allow_early_success: bool = field(default=True)
    action_queue_size_to_get_new_actions: int = field(default=30)

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required (--policy.path=...).")
        if self.robot is None:
            raise ValueError("Robot configuration must be provided.")
        if not self.dataset.repo_id.strip():
            raise ValueError("dataset.repo_id must be provided.")
        if not self.task.strip():
            raise ValueError("task must be provided.")
        if self.fps <= 0:
            raise ValueError("fps must be > 0.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ── Terminal raw-mode helpers ────────────────────────────────────────────────


_original_terminal_settings = None


def _enable_char_mode() -> None:
    global _original_terminal_settings
    if sys.platform == "win32" or _original_terminal_settings is not None:
        return
    try:
        _original_terminal_settings = termios.tcgetattr(sys.stdin)
        new = list(_original_terminal_settings)
        new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new)
    except Exception:
        _original_terminal_settings = None


def _restore_terminal() -> None:
    global _original_terminal_settings
    if _original_terminal_settings is not None:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _original_terminal_settings)
        except Exception:
            pass
        _original_terminal_settings = None


def _read_char(timeout: float = 0.1) -> str | None:
    if sys.platform == "win32":
        import msvcrt

        return msvcrt.getwch() if msvcrt.kbhit() else None
    if select.select([sys.stdin], [], [], timeout)[0]:
        ch = sys.stdin.read(1)
        return ch if ch else None
    return None


# ── RobotWrapper + observation cache ─────────────────────────────────────────


class RobotWrapper:
    """Thread-safe wrapper around a `Robot` used by both actor and inference threads."""

    def __init__(self, robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self):
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self):
        with self.lock:
            return self.robot.observation_features

    def action_features(self):
        with self.lock:
            return self.robot.action_features


class ObservationCache:
    """Pass the latest observation frame between actor and inference threads without recapturing."""

    def __init__(self):
        self._lock = Lock()
        self._frame: dict[str, np.ndarray] | None = None

    def update(self, frame: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> dict[str, np.ndarray] | None:
        with self._lock:
            return None if self._frame is None else dict(self._frame)


# ── RECAP column helpers ─────────────────────────────────────────────────────


def _float_scalar(value: float) -> np.ndarray:
    return np.asarray([value], dtype=np.float32)


def _int_scalar(value: int) -> np.ndarray:
    return np.asarray([value], dtype=np.int64)


def _placeholder_recap_frame(intervention: float = 0.0) -> dict[str, np.ndarray]:
    return {
        EPISODE_SUCCESS_FEATURE: _float_scalar(0.0),
        EPISODE_FAILURE_FEATURE: _float_scalar(0.0),
        RECAP_INTERVENTION_FEATURE: _float_scalar(intervention),
        RECAP_REWARD_FEATURE: _float_scalar(0.0),
        RECAP_RETURN_FEATURE: _float_scalar(0.0),
        RECAP_NORMALIZED_RETURN_FEATURE: _float_scalar(0.0),
        RECAP_VALUE_TARGET_FEATURE: _int_scalar(0),
    }


def _recap_feature_spec() -> dict[str, dict[str, Any]]:
    return {
        EPISODE_SUCCESS_FEATURE: {"dtype": "float32", "shape": (1,), "names": None},
        EPISODE_FAILURE_FEATURE: {"dtype": "float32", "shape": (1,), "names": None},
        RECAP_INTERVENTION_FEATURE: {"dtype": "float32", "shape": (1,), "names": None},
        RECAP_REWARD_FEATURE: {"dtype": "float32", "shape": (1,), "names": None},
        RECAP_RETURN_FEATURE: {"dtype": "float32", "shape": (1,), "names": None},
        RECAP_NORMALIZED_RETURN_FEATURE: {"dtype": "float32", "shape": (1,), "names": None},
        RECAP_VALUE_TARGET_FEATURE: {"dtype": "int64", "shape": (1,), "names": None},
    }


def _finalize_episode_recap(
    dataset: LeRobotDataset,
    *,
    success: bool,
    intervention_flags: list[float],
    max_episode_steps: int,
    c_fail: float,
    num_bins: int,
) -> None:
    """Stamp per-frame RECAP targets onto the episode buffer just before save_episode."""
    buffer = dataset.episode_buffer
    if buffer is None:
        raise ValueError("No episode buffer to finalise.")
    length = int(buffer["size"])
    if length <= 0:
        raise ValueError("Cannot finalise an empty episode.")
    if len(intervention_flags) != length:
        raise ValueError(
            f"Intervention flags length {len(intervention_flags)} != episode length {length}"
        )

    episode_success = np.full(length, 1.0 if success else 0.0, dtype=np.float32)
    targets = build_recap_targets_from_success(
        episode_success=episode_success,
        episode_ranges=[(0, 0, length)],
        max_episode_steps=max_episode_steps,
        c_fail=c_fail,
        num_bins=num_bins,
    )

    buffer[EPISODE_SUCCESS_FEATURE] = [_float_scalar(1.0 if success else 0.0) for _ in range(length)]
    buffer[EPISODE_FAILURE_FEATURE] = [_float_scalar(0.0 if success else 1.0) for _ in range(length)]
    buffer[RECAP_INTERVENTION_FEATURE] = [_float_scalar(flag) for flag in intervention_flags]
    buffer[RECAP_REWARD_FEATURE] = [_float_scalar(float(v)) for v in targets[RECAP_REWARD_FEATURE]]
    buffer[RECAP_RETURN_FEATURE] = [_float_scalar(float(v)) for v in targets[RECAP_RETURN_FEATURE]]
    buffer[RECAP_NORMALIZED_RETURN_FEATURE] = [
        _float_scalar(float(v)) for v in targets[RECAP_NORMALIZED_RETURN_FEATURE]
    ]
    buffer[RECAP_VALUE_TARGET_FEATURE] = [
        _int_scalar(int(v)) for v in targets[RECAP_VALUE_TARGET_FEATURE]
    ]


# ── Inference thread ─────────────────────────────────────────────────────────


def _capture_observation_frame(
    *,
    robot: RobotWrapper,
    robot_observation_processor,
    observation_features: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    obs = robot.get_observation()
    obs_processed = robot_observation_processor(obs)
    frame = build_dataset_frame(observation_features, obs_processed, prefix=OBS_STR)
    return obs, frame


def _preprocess_inputs(frame, preprocessor, task, device) -> dict[str, Any]:
    batch = {k: v for k, v in frame.items()}
    batch["task"] = task
    batch = preprocessor(batch)
    return batch


def _get_actions_loop(
    *,
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    observation_features,
    observation_cache: ObservationCache,
    preprocessor,
    postprocessor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    stop_event: Event,
    thread_error: list[str],
    cfg: RecapRecordConfig,
    task: str,
) -> None:
    try:
        latency_tracker = LatencyTracker()
        time_per_chunk = 1.0 / cfg.fps
        policy_device = policy.config.device
        threshold = cfg.action_queue_size_to_get_new_actions
        if not cfg.rtc.enabled:
            threshold = 0
        elif threshold <= cfg.rtc.execution_horizon:
            threshold = cfg.rtc.execution_horizon + 1

        while not shutdown_event.is_set() and not stop_event.is_set():
            if action_queue.qsize() > threshold:
                time.sleep(0.005)
                continue

            start = time.perf_counter()
            action_index_before = action_queue.get_action_index()
            prev_actions = action_queue.get_left_over()
            inference_delay = math.ceil(latency_tracker.max() / time_per_chunk)

            frame = observation_cache.get()
            if frame is None:
                _, frame = _capture_observation_frame(
                    robot=robot,
                    robot_observation_processor=robot_observation_processor,
                    observation_features=observation_features,
                )
                observation_cache.update(frame)

            batch = _preprocess_inputs(frame, preprocessor, task, policy_device)
            with torch.no_grad():
                actions = policy.predict_action_chunk(
                    batch,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

            original_actions = actions.squeeze(0).clone()
            processed_actions = postprocessor(actions).squeeze(0)

            new_delay = math.ceil((time.perf_counter() - start) / time_per_chunk)
            latency_tracker.add(time.perf_counter() - start)
            action_queue.merge(
                original_actions, processed_actions, new_delay, action_index_before
            )
    except Exception as exc:
        message = f"[GET_ACTIONS] Fatal: {exc}\n{traceback.format_exc()}"
        logger.error(message)
        thread_error.append(message)
        stop_event.set()
        shutdown_event.set()


# ── Hotkey thread ────────────────────────────────────────────────────────────


def _control_listener(
    *,
    allow_early_success: bool,
    allow_intervention: bool,
    success_event: Event,
    intervention_event: Event,
    stop_event: Event,
    shutdown_event: Event,
) -> None:
    hotkeys: list[str] = []
    if allow_early_success:
        hotkeys.append("s=mark success and end episode")
    if allow_intervention:
        hotkeys.append("i=toggle human intervention")
    if not hotkeys:
        return

    _enable_char_mode()
    print(f"\n[RUNNING] Hotkeys: {' | '.join(hotkeys)}\n", flush=True)
    try:
        while not shutdown_event.is_set() and not stop_event.is_set():
            ch = _read_char(timeout=0.1)
            if ch is None:
                continue
            ch = ch.lower()
            if ch == "s" and allow_early_success and not success_event.is_set():
                success_event.set()
                logger.info("[CONTROL] Early success requested")
            elif ch == "i" and allow_intervention:
                if intervention_event.is_set():
                    intervention_event.clear()
                    logger.info("[CONTROL] Intervention OFF — policy is driving")
                else:
                    intervention_event.set()
                    logger.info("[CONTROL] Intervention ON — human is driving")
    finally:
        _restore_terminal()


# ── Actor loop ───────────────────────────────────────────────────────────────


@dataclass
class EpisodeResult:
    frames: int
    early_success: bool
    intervention_flags: list[float]


@safe_stop_image_writer
def _record_episode(
    *,
    robot: RobotWrapper,
    robot_observation_processor,
    observation_features,
    observation_cache: ObservationCache,
    robot_action_processor,
    dataset: LeRobotDataset,
    action_queue: ActionQueue,
    shutdown_event: Event,
    stop_event: Event,
    success_event: Event,
    intervention_event: Event,
    teleop,
    thread_error: list[str],
    cfg: RecapRecordConfig,
    task: str,
) -> EpisodeResult:
    try:
        action_names = list(dataset.features[ACTION]["names"])
        intervention_flags: list[float] = []
        episode_start: float | None = None
        warned_waiting = False

        while not shutdown_event.is_set() and not stop_event.is_set():
            if episode_start is not None and time.perf_counter() - episode_start >= cfg.dataset.episode_time_s:
                break

            loop_start = time.perf_counter()
            intervening = teleop is not None and intervention_event.is_set()

            if intervening:
                # Drop any queued policy actions so when the human hands back,
                # the next chunk is fresh.
                while action_queue.qsize() > 0:
                    action_queue.get()
                teleop_action = teleop.get_action()
                action_dict = {name: float(teleop_action[name]) for name in action_names}
            else:
                action = action_queue.get()
                if action is None:
                    if not warned_waiting:
                        logger.info("[ACTOR] Waiting for first RTC action chunk...")
                        warned_waiting = True
                    time.sleep(0.002)
                    continue
                warned_waiting = False
                action_cpu = action.detach().cpu()
                action_dict = {name: action_cpu[idx].item() for idx, name in enumerate(action_names)}

            if episode_start is None:
                episode_start = time.perf_counter()
                logger.info("[ACTOR] First action received, episode timer started")

            obs, observation_frame = _capture_observation_frame(
                robot=robot,
                robot_observation_processor=robot_observation_processor,
                observation_features=observation_features,
            )
            observation_cache.update(observation_frame)

            action_processed = robot_action_processor((action_dict, obs))
            robot.send_action(action_processed)

            action_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
            intervention_flag = 1.0 if intervening else 0.0
            frame = {
                **observation_frame,
                **action_frame,
                **_placeholder_recap_frame(intervention_flag),
                "task": task,
            }
            dataset.add_frame(frame)
            intervention_flags.append(intervention_flag)

            dt = time.perf_counter() - loop_start
            precise_sleep(max(1.0 / cfg.fps - dt, 0.0))

            if success_event.is_set() and intervention_flags:
                logger.info("[ACTOR] Early success hotkey — ending episode")
                break

        return EpisodeResult(
            frames=len(intervention_flags),
            early_success=success_event.is_set() and bool(intervention_flags),
            intervention_flags=intervention_flags,
        )
    except Exception as exc:
        message = f"[ACTOR] Fatal: {exc}\n{traceback.format_exc()}"
        logger.error(message)
        thread_error.append(message)
        stop_event.set()
        shutdown_event.set()
        return EpisodeResult(frames=0, early_success=False, intervention_flags=[])


def _run_single_episode(
    *,
    policy,
    robot_wrapper: RobotWrapper,
    teleop,
    robot_observation_processor,
    observation_features,
    robot_action_processor,
    preprocessor,
    postprocessor,
    dataset: LeRobotDataset,
    shutdown_event: Event,
    cfg: RecapRecordConfig,
    task: str,
) -> EpisodeResult:
    stop_event = Event()
    success_event = Event()
    intervention_event = Event()
    thread_error: list[str] = []
    action_queue = ActionQueue(cfg.rtc)
    observation_cache = ObservationCache()

    _, first_frame = _capture_observation_frame(
        robot=robot_wrapper,
        robot_observation_processor=robot_observation_processor,
        observation_features=observation_features,
    )
    observation_cache.update(first_frame)

    inference_thread = Thread(
        target=_get_actions_loop,
        kwargs=dict(
            policy=policy,
            robot=robot_wrapper,
            robot_observation_processor=robot_observation_processor,
            observation_features=observation_features,
            observation_cache=observation_cache,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            action_queue=action_queue,
            shutdown_event=shutdown_event,
            stop_event=stop_event,
            thread_error=thread_error,
            cfg=cfg,
            task=task,
        ),
        daemon=True,
        name="RTCGetActions",
    )
    inference_thread.start()

    control_thread = Thread(
        target=_control_listener,
        kwargs=dict(
            allow_early_success=cfg.allow_early_success,
            allow_intervention=teleop is not None,
            success_event=success_event,
            intervention_event=intervention_event,
            stop_event=stop_event,
            shutdown_event=shutdown_event,
        ),
        daemon=True,
        name="EpisodeControl",
    )
    control_thread.start()

    try:
        return _record_episode(
            robot=robot_wrapper,
            robot_observation_processor=robot_observation_processor,
            observation_features=observation_features,
            observation_cache=observation_cache,
            robot_action_processor=robot_action_processor,
            dataset=dataset,
            action_queue=action_queue,
            shutdown_event=shutdown_event,
            stop_event=stop_event,
            success_event=success_event,
            intervention_event=intervention_event,
            teleop=teleop,
            thread_error=thread_error,
            cfg=cfg,
            task=task,
        )
    finally:
        stop_event.set()
        inference_thread.join(timeout=5.0)
        control_thread.join(timeout=5.0)
        if thread_error:
            raise RuntimeError(thread_error[0])


# ── Dataset helpers ──────────────────────────────────────────────────────────


def _build_dataset_features(robot_wrapper: RobotWrapper, use_video: bool) -> dict[str, dict[str, Any]]:
    features = hw_to_dataset_features(robot_wrapper.observation_features(), OBS_STR, use_video=use_video)
    features.update(hw_to_dataset_features(robot_wrapper.action_features(), ACTION, use_video=use_video))
    features.update(_recap_feature_spec())
    return features


def _init_dataset(
    *,
    robot,
    robot_wrapper: RobotWrapper,
    control_fps: int,
    cfg: RecapRecordConfig,
) -> LeRobotDataset:
    features = _build_dataset_features(robot_wrapper, use_video=cfg.dataset.video)
    if cfg.dataset.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, control_fps, features)
        return dataset

    return LeRobotDataset.create(
        cfg.dataset.repo_id,
        control_fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        vcodec=cfg.dataset.vcodec,
    )


def _resolved_max_episode_steps(cfg: RecapRecordConfig) -> int:
    if cfg.dataset.recap_max_episode_steps > 0:
        return int(cfg.dataset.recap_max_episode_steps)
    return max(1, math.ceil(cfg.dataset.episode_time_s * round(cfg.fps)))


def _resolved_c_fail(cfg: RecapRecordConfig, max_episode_steps: int) -> float:
    return float(cfg.dataset.recap_c_fail) if cfg.dataset.recap_c_fail > 0 else float(max_episode_steps)


def _prompt_episode_outcome(episode_index: int) -> str:
    prompt = (
        f"Episode {episode_index} finished. Mark outcome: "
        "[s]uccess / [f]ailure / [r]edo / [q]uit: "
    )
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"s", "success"}:
            return "success"
        if answer in {"f", "failure"}:
            return "failure"
        if answer in {"r", "redo"}:
            return "redo"
        if answer in {"q", "quit"}:
            return "quit"
        print("Please enter one of: s, f, r, q.", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────


@parser.wrap()
def main(cfg: RecapRecordConfig) -> None:
    init_logging()
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    runtime_device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.policy.device = runtime_device
    control_fps = int(round(cfg.fps))

    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.eval()
    policy.to(runtime_device)

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_wrapper = RobotWrapper(robot)

    teleop = None
    if cfg.teleop is not None:
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()
        logger.info("Teleop connected — press 'i' during an episode to take over the robot.")

    dataset = _init_dataset(
        robot=robot,
        robot_wrapper=robot_wrapper,
        control_fps=control_fps,
        cfg=cfg,
    )
    logger.info("Dataset ready at %s", dataset.root)

    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()
    observation_features = hw_to_dataset_features(robot_wrapper.observation_features(), OBS_STR)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg.policy, dataset_stats=dataset.meta.stats
    )

    max_episode_steps = _resolved_max_episode_steps(cfg)
    c_fail = _resolved_c_fail(cfg, max_episode_steps)

    labels_path = Path(dataset.root) / "meta" / "episode_labels.csv"
    existing_labels: dict[int, bool] = {}
    if labels_path.exists():
        from lerobot.rl.algorithms.recap_targets import parse_episode_labels_csv

        try:
            existing_labels = parse_episode_labels_csv(labels_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read existing %s: %s", labels_path, exc)

    input("Press Enter to start recording RTC rollouts...")

    newly_recorded = 0
    try:
        with VideoEncodingManager(dataset):
            while newly_recorded < cfg.dataset.num_episodes and not shutdown_event.is_set():
                episode_index = int(dataset.num_episodes)
                logger.info("[MAIN] Starting episode %d (task=%s)", episode_index, cfg.task)

                result = _run_single_episode(
                    policy=policy,
                    robot_wrapper=robot_wrapper,
                    teleop=teleop,
                    robot_observation_processor=robot_observation_processor,
                    observation_features=observation_features,
                    robot_action_processor=robot_action_processor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    shutdown_event=shutdown_event,
                    cfg=cfg,
                    task=cfg.task,
                )
                if result.frames <= 0:
                    dataset.clear_episode_buffer()
                    if shutdown_event.is_set():
                        break
                    raise RuntimeError("Episode ended without recording any frames.")

                outcome = "success" if result.early_success else _prompt_episode_outcome(episode_index)
                if outcome == "redo":
                    dataset.clear_episode_buffer()
                    continue
                if outcome == "quit":
                    dataset.clear_episode_buffer()
                    break

                success = outcome == "success"
                _finalize_episode_recap(
                    dataset,
                    success=success,
                    intervention_flags=result.intervention_flags,
                    max_episode_steps=max_episode_steps,
                    c_fail=c_fail,
                    num_bins=cfg.dataset.recap_num_bins,
                )
                dataset.save_episode()
                existing_labels[episode_index] = success
                write_episode_labels_csv(labels_path, existing_labels)
                newly_recorded += 1

                n_intervention_frames = int(sum(result.intervention_flags))
                logger.info(
                    "[MAIN] Saved episode %d as %s (%d frames, %d intervention). Progress: %d/%d",
                    episode_index,
                    outcome,
                    result.frames,
                    n_intervention_frames,
                    newly_recorded,
                    cfg.dataset.num_episodes,
                )

                if newly_recorded < cfg.dataset.num_episodes and not shutdown_event.is_set():
                    logger.info("[MAIN] Resetting for %.1fs", cfg.dataset.reset_time_s)
                    time.sleep(cfg.dataset.reset_time_s)
    finally:
        if robot.is_connected:
            robot.disconnect()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if dataset is not None:
            dataset.finalize()
            if cfg.dataset.push_to_hub:
                dataset.push_to_hub(private=cfg.dataset.private)


if __name__ == "__main__":
    main()
