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
from .a2_benchmark import A2Benchmark, BenchmarkConfig
from .a2_collect import (
    RandomPolicy,
    collect_data,
    create_a2_policy,
    load_policy,
)
from .a2_constants import (
    GRASP_WORKSPACE_LIMITS,
    IN_REGION,
    LABEL,
    LANG_TEMPLATES,
    OUT_REGION,
    PLACE_LANG_TEMPLATES,
    PLACE_WORKSPACE_LIMITS,
    PP_PIXEL_SIZE,
    PP_SHIFT_Y,
    PP_WORKSPACE_LIMITS,
    UNSEEN_LABEL,
    WORKSPACE_LIMITS,
)
from .a2_recorder import VLARecorder
from .a2_sim import Environment
from .a2_utils import (
    PlaceGenerator,
    generate_all_place_dist,
    generate_mask_map,
    generate_sector_mask,
    normalize_pos,
    preprocess_pp_unified,
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
    "GRASP_WORKSPACE_LIMITS",
    "PLACE_WORKSPACE_LIMITS",
    "PP_WORKSPACE_LIMITS",
    "PP_SHIFT_Y",
    "PP_PIXEL_SIZE",
    "IN_REGION",
    "OUT_REGION",
    # Utilities
    "PlaceGenerator",
    "generate_all_place_dist",
    "generate_mask_map",
    "generate_sector_mask",
    "preprocess_pp_unified",
    "normalize_pos",
]
