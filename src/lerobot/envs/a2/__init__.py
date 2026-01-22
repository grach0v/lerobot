"""A2 Environment package for LeRobot.

This package provides the A2 simulation environment for robotic manipulation
with UR5e + Robotiq gripper setup.

Components:
- A2Env: Gymnasium-compatible environment wrapper
- Environment: PyBullet simulation environment
- VLARecorder: Dataset recording for VLA training
- A2Benchmark: Benchmark evaluation framework
"""

from .a2 import A2Env, get_a2_assets_path
from .a2_sim import Environment
from .a2_recorder import VLARecorder
from .a2_benchmark import A2Benchmark, BenchmarkConfig
from .a2_constants import (
    LABEL,
    UNSEEN_LABEL,
    WORKSPACE_LIMITS,
    LANG_TEMPLATES,
    PLACE_LANG_TEMPLATES,
)
from .a2_collect import (
    RandomPolicy,
    load_policy,
    create_a2_policy,
    collect_data,
)

__all__ = [
    # Environment
    "A2Env",
    "get_a2_assets_path",
    "Environment",
    # Recording
    "VLARecorder",
    # Benchmark
    "A2Benchmark",
    "BenchmarkConfig",
    # Data Collection
    "RandomPolicy",
    "load_policy",
    "create_a2_policy",
    "collect_data",
    # Constants
    "LABEL",
    "UNSEEN_LABEL",
    "WORKSPACE_LIMITS",
    "LANG_TEMPLATES",
    "PLACE_LANG_TEMPLATES",
]
