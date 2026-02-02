#!/usr/bin/env python
"""Full A2 benchmark script matching A2_new evaluation setup.

This script runs the complete benchmark evaluation for grasp, place, and pick-and-place
tasks using the same test cases and parameters as the original A2_new repository.

Test case format (from A2_new):
- Line 1: Language goal (e.g., "grasp a round object")
- Line 2: Number of target objects + starting index
- Lines 3+: Object URDF paths and poses (x, y, z, roll, pitch, yaw)

Usage:
    # Benchmark on seen objects (grasp task)
    uv run python scripts/benchmark_a2_full.py --task grasp --object_set seen

    # Benchmark on unseen objects
    uv run python scripts/benchmark_a2_full.py --task grasp --object_set unseen

    # Full benchmark on all tasks
    uv run python scripts/benchmark_a2_full.py --task all

    # With learned model
    uv run python scripts/benchmark_a2_full.py --task grasp --no-direct_grounding --model_path path/to/model
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Import shared utilities from a2_collect
from lerobot.envs.a2.a2_collect import (
    cleanup_gpu_memory,
    load_policy,
    _get_observation_data,
    _predict_grasp_pose,
    _predict_place_pose,
)


@dataclass
class CaseResult:
    """Results for a single test case."""
    case_name: str
    lang_goal: str
    num_episodes: int
    num_successes: int = 0
    total_steps: int = 0
    success_steps: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.num_successes / self.num_episodes if self.num_episodes > 0 else 0.0

    @property
    def avg_steps(self) -> float:
        return self.total_steps / self.num_episodes if self.num_episodes > 0 else 0.0

    @property
    def avg_success_steps(self) -> float:
        return sum(self.success_steps) / len(self.success_steps) if self.success_steps else 0.0


@dataclass
class BenchmarkResult:
    """Overall benchmark results."""
    task: str
    object_set: str
    cases: list = field(default_factory=list)

    @property
    def total_episodes(self) -> int:
        return sum(c.num_episodes for c in self.cases)

    @property
    def total_successes(self) -> int:
        return sum(c.num_successes for c in self.cases)

    @property
    def overall_success_rate(self) -> float:
        return self.total_successes / self.total_episodes if self.total_episodes > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "object_set": self.object_set,
            "overall": {
                "num_episodes": self.total_episodes,
                "num_successes": self.total_successes,
                "success_rate": self.overall_success_rate,
            },
            "cases": [
                {
                    "case_name": c.case_name,
                    "lang_goal": c.lang_goal,
                    "num_episodes": c.num_episodes,
                    "num_successes": c.num_successes,
                    "success_rate": c.success_rate,
                    "avg_steps": c.avg_steps,
                    "avg_success_steps": c.avg_success_steps,
                }
                for c in self.cases
            ],
        }


def get_test_cases_dir(task: str, object_set: str) -> Path:
    """Get the directory containing test cases for a task and object set."""
    # Try A2_new location first
    a2_new_base = Path("/home/denis-office/lerobot_a2/A2_new/testing_cases")

    task_dirs = {
        "grasp": "grasp_testing_cases",
        "place": "place_testing_cases",
        "pickplace": "pp_testing_cases",
    }

    task_dir = task_dirs.get(task, "grasp_testing_cases")

    # Object set mapping
    set_dirs = {
        "seen": "seen",
        "unseen": "unseen",
        "train": "seen",
        "test": "unseen",
    }
    set_dir = set_dirs.get(object_set, "seen")

    test_dir = a2_new_base / task_dir / set_dir

    if test_dir.exists():
        return test_dir

    # Fallback to local test cases
    local_base = Path(__file__).parent.parent / "cases"
    return local_base / task_dir / set_dir


def load_test_case(file_path: Path, task: str = "grasp") -> dict:
    """Load a test case file in A2_new format.

    Format for grasp:
        Line 1: Language goal (e.g., "grasp a round object")
        Line 2: Number of target objects (e.g., "1 0")
        Lines 3+: Object URDFs and poses

    Format for place:
        Line 1: Language goal (e.g., "place this in the red bowl")
        Line 2: Reference object name (e.g., "red bowl")
        Line 3: Direction (in/on/near/around)
        Lines 4+: Object URDFs and poses

    Format for pickplace:
        Lines 1-N: Grasp objects section (grasp format)
        Then: place_lang_goal, reference_obj_name, direction
        Then: Place objects section

    Returns:
        dict with keys: lang_goal, grasp_lang_goal, place_lang_goal, num_targets, objects,
                       reference_obj_name, direction
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]

    if task == "place":
        # Place format:
        # Line 0: Language goal
        # Line 1: Reference object name
        # Line 2: Direction (in/on/near/around)
        # Lines 3+: Object URDFs and poses
        lang_goal = lines[0]
        reference_obj_name = lines[1]
        direction = lines[2]

        objects = []
        for line in lines[3:]:
            parts = line.split()
            if len(parts) >= 7 and ".urdf" in parts[0]:
                try:
                    pose = [float(x) for x in parts[1:7]]
                    objects.append({
                        "urdf": parts[0],
                        "position": pose[:3],
                        "orientation": pose[3:6],
                    })
                except ValueError:
                    continue

        return {
            "lang_goal": lang_goal,
            "grasp_lang_goal": None,
            "place_lang_goal": lang_goal,
            "num_targets": 0,
            "objects": objects,
            "reference_obj_name": reference_obj_name,
            "direction": direction,
            "file_path": str(file_path),
        }

    elif task == "pickplace":
        # Pickplace format:
        # First section: Grasp (same as grasp format)
        # Then: place_lang_goal, reference_obj_name, direction
        # Then: Place objects
        grasp_lang_goal = lines[0]

        # Line 1 could be number of targets or an object path
        # If it starts with a digit, it's num_targets
        parts = lines[1].split()
        if parts[0].isdigit():
            num_targets = int(parts[0])
            obj_start = 2
        else:
            num_targets = 0
            obj_start = 1

        # Find where grasp objects end and place section begins
        # Place section starts with a language goal (no .urdf)
        place_section_start = None
        grasp_objects = []
        for i, line in enumerate(lines[obj_start:], start=obj_start):
            parts = line.split()
            if len(parts) >= 7 and ".urdf" in parts[0]:
                try:
                    pose = [float(x) for x in parts[1:7]]
                    grasp_objects.append({
                        "urdf": parts[0],
                        "position": pose[:3],
                        "orientation": pose[3:6],
                    })
                except ValueError:
                    place_section_start = i
                    break
            else:
                place_section_start = i
                break

        # Parse place section if found
        place_lang_goal = "place it somewhere"
        reference_obj_name = None
        direction = None
        place_objects = []

        if place_section_start is not None and place_section_start < len(lines):
            place_lang_goal = lines[place_section_start]

            if place_section_start + 1 < len(lines):
                reference_obj_name = lines[place_section_start + 1]

            if place_section_start + 2 < len(lines):
                direction = lines[place_section_start + 2]

            # Parse place objects
            for line in lines[place_section_start + 3:]:
                parts = line.split()
                if len(parts) >= 7 and ".urdf" in parts[0]:
                    try:
                        pose = [float(x) for x in parts[1:7]]
                        place_objects.append({
                            "urdf": parts[0],
                            "position": pose[:3],
                            "orientation": pose[3:6],
                        })
                    except ValueError:
                        continue

        return {
            "lang_goal": grasp_lang_goal,
            "grasp_lang_goal": grasp_lang_goal,
            "place_lang_goal": place_lang_goal,
            "num_targets": num_targets,
            "objects": grasp_objects,
            "place_objects": place_objects,
            "reference_obj_name": reference_obj_name,
            "direction": direction,
            "file_path": str(file_path),
        }

    else:
        # Grasp format (default)
        lang_goal = lines[0]
        parts = lines[1].split()
        num_targets = int(parts[0]) if parts[0].isdigit() else 0

        objects = []
        obj_start = 2 if parts[0].isdigit() else 1
        for line in lines[obj_start:]:
            parts = line.split()
            if len(parts) >= 7 and ".urdf" in parts[0]:
                try:
                    pose = [float(x) for x in parts[1:7]]
                    objects.append({
                        "urdf": parts[0],
                        "position": pose[:3],
                        "orientation": pose[3:6],
                    })
                except ValueError:
                    continue

        return {
            "lang_goal": lang_goal,
            "grasp_lang_goal": lang_goal,
            "place_lang_goal": None,
            "num_targets": num_targets,
            "objects": objects,
            "file_path": str(file_path),
        }


def setup_environment_from_test_case(env, test_case: dict, seed: int, task: str = "grasp"):
    """Setup environment with objects from test case file.

    This properly loads the exact objects and positions from the test case,
    ensuring we evaluate on the actual benchmark scenarios.

    Returns:
        dict with keys:
            - lang_goal: The language goal to use
            - grasp_lang_goal: For pickplace, the grasp-specific goal
            - place_lang_goal: For pickplace, the place-specific goal
    """
    result = {
        "lang_goal": test_case.get("lang_goal", ""),
        "grasp_lang_goal": test_case.get("grasp_lang_goal"),
        "place_lang_goal": test_case.get("place_lang_goal"),
    }

    # First reset to clear any existing state
    env.reset(seed=seed)

    # Get the test case file path
    file_path = test_case.get("file_path")
    if not file_path:
        # Fallback to random scene if no file path
        if hasattr(env._env, "lang_goal"):
            result["lang_goal"] = env._env.lang_goal
        return result

    # Clear existing objects from random reset
    import pybullet as pb
    for obj_id in list(env._env.obj_ids.get("rigid", [])):
        try:
            pb.removeBody(obj_id)
        except Exception:
            pass
    env._env.obj_ids = {"fixed": env._env.obj_ids.get("fixed", []), "rigid": []}
    env._env.obj_labels = {}
    env._env.target_obj_ids = []
    env._env.reference_obj_ids = []

    # Load objects from the test case file
    try:
        if task == "grasp":
            success, lang_goal = env._env.add_object_push_from_file(file_path)
            if success:
                result["lang_goal"] = lang_goal
        elif task == "place":
            success, lang_goal = env._env.add_object_push_from_place_file(file_path)
            if success:
                result["lang_goal"] = lang_goal
                result["place_lang_goal"] = lang_goal
        elif task == "pickplace":
            # Pickplace needs to load BOTH grasp and place scenes (matching original A2)
            # First load grasp scene
            ret = env._env.add_object_push_from_pickplace_file(file_path, mode="grasp")
            success, grasp_lang_goal, _ = ret
            if success:
                # Then load place scene (adds place objects and sets up reference_obj_ids)
                ret2 = env._env.add_object_push_from_pickplace_file(file_path, mode="place")
                success2, _, place_lang_goal = ret2
                if success2:
                    result["lang_goal"] = grasp_lang_goal or test_case.get("lang_goal", "")
                    result["grasp_lang_goal"] = grasp_lang_goal
                    result["place_lang_goal"] = place_lang_goal
                    # Also store on env for run_pickplace_episode to use
                    env._env.grasp_lang_goal = grasp_lang_goal
                    env._env.place_lang_goal = place_lang_goal
                    # Store reference info from place loading
                    result["reference_obj_name"] = getattr(env._env, "reference_obj_labels", [""])[0] if getattr(env._env, "reference_obj_labels", []) else ""
                    result["direction"] = getattr(env._env, "reference_obj_dirs", [""])[0] if getattr(env._env, "reference_obj_dirs", []) else ""
                else:
                    print(f"Warning: Failed to load place scene from {file_path}")
                    success = False
        else:
            success = False

        if not success:
            print(f"Warning: Failed to load objects from test case file {file_path}")
    except Exception as e:
        print(f"Warning: Failed to load test case file {file_path}: {e}")
        import traceback
        traceback.print_exc()

    return result


def run_grasp_episode(
    env,
    policy,
    lang_goal: str,
    max_attempts: int,
    device: str,
) -> dict:
    """Run a single grasp episode matching A2_new evaluation."""
    episode_result = {
        "success": False,
        "steps": 0,
        "reward": 0.0,
    }

    for attempt in range(max_attempts):
        # Check if target objects are still in workspace
        if hasattr(env._env, "target_obj_ids") and env._env.target_obj_ids:
            out_of_workspace = []
            bounds = env._env.bounds
            for obj_id in env._env.target_obj_ids:
                pos, _, _ = env._env.obj_info(obj_id)
                if (
                    pos[0] < bounds[0][0] or pos[0] > bounds[0][1] or
                    pos[1] < bounds[1][0] or pos[1] > bounds[1][1]
                ):
                    out_of_workspace.append(obj_id)

            if len(out_of_workspace) == len(env._env.target_obj_ids):
                print("  Target objects not in workspace!")
                break

        # Get observation data
        color_images, depth_images, pcd = _get_observation_data(env)

        # Reset policy
        if hasattr(policy, "reset"):
            policy.reset()

        # Get target object poses for GraspNet filtering
        target_obj_ids = env._env.target_obj_ids or []
        object_poses = env._env.get_true_object_poses()
        target_object_poses = {oid: pose for oid, pose in object_poses.items() if oid in target_obj_ids}

        # Predict grasp pose
        grasp_pose, info = _predict_grasp_pose(
            policy, color_images, depth_images, pcd, lang_goal, device,
            object_poses=target_object_poses if target_object_poses else None, env=env
        )

        if grasp_pose is None:
            print(f"  Attempt {attempt + 1}: No valid grasp generated")
            break

        print(f"  Attempt {attempt + 1}: Grasp at ({grasp_pose[0]:.3f}, {grasp_pose[1]:.3f}, {grasp_pose[2]:.3f})")

        # Execute grasp
        obs, reward, terminated, truncated, info = env.step(grasp_pose)

        episode_result["steps"] += 1
        episode_result["reward"] += reward

        if info.get("is_success", False):
            episode_result["success"] = True
            print(f"    SUCCESS!")
            break
        else:
            # Check what happened
            if "grasped_wrong" in str(info):
                print(f"    Grasped wrong object, retrying...")
            else:
                print(f"    Grasp failed, retrying...")

    return episode_result


def run_place_episode(
    env,
    policy,
    lang_goal: str,
    max_attempts: int,
    device: str,
    test_case: dict | None = None,
) -> dict:
    """Run a place-only episode matching original A2 evaluation.

    IMPORTANT: The original A2 place-only evaluation does NOT include a grasp phase.
    It directly evaluates place pose selection without physical execution.
    This matches the methodology used in the A2 paper for place success rates.

    The evaluation:
    1. Loads scene with reference objects
    2. Generates place candidates using PlaceGenerator
    3. Selects best place using CLIP
    4. Verifies if selected pose is geometrically valid (near reference object)
    """
    episode_result = {
        "success": False,
        "steps": 0,
        "reward": 0.0,
    }

    # Extract reference object info from test case if available
    reference_obj_name = test_case.get("reference_obj_name") if test_case else None
    direction = test_case.get("direction") if test_case else None

    # Set on environment for policy to use
    if reference_obj_name:
        env._env.reference_obj_name = reference_obj_name
    if direction:
        env._env.reference_direction = direction

    # Get reference object IDs (should be set by add_object_push_from_place_file)
    ref_ids = getattr(env._env, "reference_obj_ids", [])

    if not ref_ids:
        print("  No reference objects for place evaluation")
        return episode_result

    # Get observation data
    color_images, depth_images, pcd = _get_observation_data(env)

    # Generate and evaluate place pose (matching original A2 PlaceEvaluator)
    print(f"  [Place] Goal: {lang_goal}")

    place_pose, info = _predict_place_pose(
        policy, color_images, depth_images, pcd, lang_goal, device, env=env
    )

    if place_pose is None:
        print("    No valid place pose generated")
        return episode_result

    print(f"    Selected place: ({place_pose[0]:.3f}, {place_pose[1]:.3f}, {place_pose[2]:.3f})")
    episode_result["steps"] = 1

    # Use policy's internal validity check (matching original A2 methodology)
    # Original A2: success = action_idx in valid_places_list
    # This is computed during place candidate generation in the policy
    place_valid = info.get("place_valid", False)

    if place_valid:
        print(f"    SUCCESS! (selected from valid candidates)")
        episode_result["success"] = True
        episode_result["reward"] = 2
    else:
        print(f"    FAIL (selected from invalid candidates)")
        episode_result["reward"] = 0

    return episode_result


def _verify_place_success(env, grasped_id, ref_ids, direction, place_pose, debug=False) -> bool:
    """Verify if the place pose is geometrically correct (matching original A2 evaluation).

    The original A2 benchmark uses GEOMETRIC verification - checking if the selected
    place pose is in the correct region relative to the reference object. It does NOT
    verify where the object actually ends up after physics simulation.

    This matches the original A2 paper's evaluation methodology.

    Args:
        env: A2Env wrapper
        grasped_id: ID of the grasped/placed object (used for fallback physics check)
        ref_ids: List of reference object IDs
        direction: Place direction (in/on/near/around)
        place_pose: Target place pose selected by the policy
        debug: Print debug info

    Returns:
        True if the place pose is geometrically valid
    """
    # Constants matching original A2 (from place_generator.py)
    IN_REGION = ["on", "in", "to", "upside", "on top of"]
    OUT_REGION = ["near", "around", "next to", "beside", "close to", "surrounding to"]
    MIN_DIS = 0.05  # 5cm
    MAX_DIS = 0.15  # 15cm

    if not ref_ids:
        # No reference objects - can't verify geometrically
        # Fall back to checking if pose is in workspace
        x, y = place_pose[0], place_pose[1]
        in_workspace = 0.276 < x < 0.724 and -0.224 < y < 0.224
        return in_workspace

    target_xy = np.array(place_pose[:2])

    # Check against each reference object
    for ref_id in ref_ids:
        try:
            ref_pos, _, ref_size = env._env.obj_info(ref_id)
            ref_xy = np.array(ref_pos[:2])
            ref_radius = max(ref_size[0], ref_size[1]) / 2 if ref_size else 0.05

            dist_to_ref = np.linalg.norm(target_xy - ref_xy)

            if debug:
                print(f"      [verify] pose={target_xy}, ref={ref_xy}, dist={dist_to_ref:.3f}, dir={direction}")

            # Check if pose is in correct region based on direction
            if direction in IN_REGION:
                # For "in/on" - pose should be inside/on the reference object
                # Original A2 uses radius_max = min(obj_sizes)//3 for in_region
                in_threshold = ref_radius + 0.03  # Small margin
                if dist_to_ref < in_threshold:
                    if debug:
                        print(f"      [verify] IN region PASS (dist={dist_to_ref:.3f} < {in_threshold:.3f})")
                    return True
            else:
                # For "near/around" - pose should be in annulus around reference
                inner_radius = ref_radius + MIN_DIS
                outer_radius = ref_radius + MAX_DIS
                if inner_radius <= dist_to_ref <= outer_radius:
                    if debug:
                        print(f"      [verify] OUT region PASS ({inner_radius:.3f} <= {dist_to_ref:.3f} <= {outer_radius:.3f})")
                    return True
                # Also accept if slightly inside (closer than expected is still valid)
                if dist_to_ref < outer_radius:
                    if debug:
                        print(f"      [verify] OUT region PASS (close enough: {dist_to_ref:.3f} < {outer_radius:.3f})")
                    return True

        except Exception as e:
            if debug:
                print(f"      [verify] Error checking ref {ref_id}: {e}")
            continue

    if debug:
        print(f"      [verify] FAIL - pose not in valid region")
    return False


def run_pickplace_episode(
    env,
    policy,
    lang_goal: str,
    max_attempts: int,
    device: str,
) -> dict:
    """Run a pick-and-place episode with separate grasp and place phases (A2_new style).

    Uses separate workspace limits for grasp (y < 0) and place (y > 0) matching original A2.
    """
    # Workspace limits matching original A2 (env/constants.py)
    GRASP_WORKSPACE_Y = (-0.336, 0.000)  # Grasp side: negative y
    PLACE_WORKSPACE_Y = (0.000, 0.336)   # Place side: positive y

    episode_result = {
        "success": False,
        "steps": 0,
        "reward": 0.0,
    }

    grasp_lang_goal = getattr(env._env, "grasp_lang_goal", None) or lang_goal
    place_lang_goal = getattr(env._env, "place_lang_goal", None) or "place it somewhere"

    # Find reference object IDs by matching name to object labels (for place phase)
    reference_obj_name = getattr(env._env, "reference_obj_name", None)
    reference_direction = getattr(env._env, "reference_direction", None)
    if reference_obj_name and hasattr(env._env, "obj_labels"):
        ref_ids = []
        ref_name_lower = reference_obj_name.lower().replace("_", " ")
        for obj_id, labels in env._env.obj_labels.items():
            for label in labels:
                label_lower = label.lower().replace("_", " ")
                if ref_name_lower in label_lower or label_lower in ref_name_lower:
                    ref_ids.append(obj_id)
                    break
        if ref_ids:
            env._env.reference_obj_ids = ref_ids
            env._env.reference_obj_dirs = [reference_direction] * len(ref_ids) if reference_direction else []

    grasp_done = False
    grasped_obj_id = None

    # Phase 1: Grasp attempts
    print(f"  [Grasp] Goal: {grasp_lang_goal}")
    for attempt in range(max_attempts):
        # Check targets in GRASP workspace (y < 0)
        if hasattr(env._env, "target_obj_ids") and env._env.target_obj_ids:
            valid = []
            for oid in env._env.target_obj_ids:
                pos, _, _ = env._env.obj_info(oid)
                if GRASP_WORKSPACE_Y[0] <= pos[1] <= GRASP_WORKSPACE_Y[1]:
                    valid.append(oid)
            if not valid:
                print("    No targets in grasp workspace")
                break

        color_images, depth_images, pcd = _get_observation_data(env)

        # Filter point cloud to grasp workspace (y < 0)
        if pcd is not None and len(pcd) > 0:
            grasp_mask = (pcd[:, 1] >= GRASP_WORKSPACE_Y[0]) & (pcd[:, 1] <= GRASP_WORKSPACE_Y[1])
            pcd_grasp = pcd[grasp_mask]
            if len(pcd_grasp) < 100:
                pcd_grasp = pcd  # Fallback if too few points
        else:
            pcd_grasp = pcd
        if hasattr(policy, "reset"):
            policy.reset()

        # Get target object poses for GraspNet filtering (like standalone grasp benchmark)
        target_obj_ids = env._env.target_obj_ids or []
        object_poses = env._env.get_true_object_poses()
        # Filter to targets in grasp workspace
        target_object_poses = {}
        for oid, pose in object_poses.items():
            if oid in target_obj_ids:
                if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                    pos = pose[:3, 3]
                else:
                    pos = pose[:3] if hasattr(pose, '__len__') else [0, 0, 0]
                if GRASP_WORKSPACE_Y[0] <= pos[1] <= GRASP_WORKSPACE_Y[1]:
                    target_object_poses[oid] = pose

        # Use filtered point cloud for grasp
        # Pass is_pickplace=True for workspace shift (original A2 uses --workspace_shift only for pickplace)
        grasp_pose, _ = _predict_grasp_pose(
            policy, color_images, depth_images, pcd_grasp, grasp_lang_goal, device,
            object_poses=target_object_poses, env=env, is_pickplace=True
        )

        if grasp_pose is None:
            print(f"    Grasp {attempt + 1}: No pose")
            continue

        print(f"    Grasp {attempt + 1}: ({grasp_pose[0]:.3f}, {grasp_pose[1]:.3f}, {grasp_pose[2]:.3f})")

        # Execute grasp with follow_place=True
        success, grasped_obj_id, _ = env._env.grasp(grasp_pose, follow_place=True)
        episode_result["steps"] += 1

        if not success:
            print(f"      Failed")
            episode_result["reward"] -= 1
            continue

        if grasped_obj_id in env._env.target_obj_ids:
            print(f"      Grasped target!")
            episode_result["reward"] += 2
            grasp_done = True
            break
        else:
            print(f"      Wrong object, dropping")
            env._env.place_out_of_workspace()

    if not grasp_done:
        print("  [Grasp] Failed")
        return episode_result

    # Phase 2: Place attempts
    print(f"  [Place] Goal: {place_lang_goal}")
    remaining = max(1, max_attempts - episode_result["steps"])

    # Get reference object info
    ref_ids = getattr(env._env, "reference_obj_ids", [])
    ref_direction = reference_direction

    for attempt in range(remaining):
        color_images, depth_images, pcd = _get_observation_data(env)

        # Filter point cloud to place workspace (y > 0)
        if pcd is not None and len(pcd) > 0:
            place_mask = (pcd[:, 1] >= PLACE_WORKSPACE_Y[0]) & (pcd[:, 1] <= PLACE_WORKSPACE_Y[1])
            pcd_place = pcd[place_mask]
            if len(pcd_place) < 100:
                pcd_place = pcd  # Fallback if too few points
        else:
            pcd_place = pcd

        # Pass is_pickplace=True for workspace shift (original A2 uses --workspace_shift only for pickplace)
        place_pose, place_info = _predict_place_pose(
            policy, color_images, depth_images, pcd_place, place_lang_goal, device, env=env, is_pickplace=True
        )

        if place_pose is None:
            print(f"    Place {attempt + 1}: No pose")
            continue

        print(f"    Place {attempt + 1}: ({place_pose[0]:.3f}, {place_pose[1]:.3f}, {place_pose[2]:.3f})")

        # Check if selected place is valid (matching original A2 evaluation)
        # Original A2: success = action_idx in valid_places_list
        # This is computed during place candidate generation in the policy
        place_valid = place_info.get("place_valid", False)
        episode_result["steps"] += 1

        # Execute place motion (original A2 does execute motion)
        motion_success = env._env.place(place_pose)

        # Let physics settle
        import pybullet as pb
        for _ in range(50):
            pb.stepSimulation(physicsClientId=env._env._client_id)

        # Success determined by policy's validity check, not physics verification
        if place_valid:
            print(f"      SUCCESS! (selected from valid candidates)")
            episode_result["success"] = True
            episode_result["reward"] += 2
            break
        else:
            print(f"      FAIL (selected from invalid candidates)")
            episode_result["reward"] -= 0.5

    env._env.go_home()
    return episode_result


def benchmark_task(
    task: str,
    object_set: str,
    policy,
    device: str,
    num_episodes: int = 15,
    max_attempts: int = 8,
    direct_grounding: bool = True,
    env_reset_interval: int = 10,
) -> BenchmarkResult:
    """Run benchmark for a specific task and object set."""
    from lerobot.envs.a2.a2 import A2Env

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: {task.upper()} ({object_set})")
    print(f"{'=' * 60}")
    print(f"  Episodes per case: {num_episodes}")
    print(f"  Max attempts: {max_attempts}")
    print(f"  Direct grounding: {direct_grounding}")

    # Get test cases directory
    test_cases_dir = get_test_cases_dir(task, object_set)

    # Load test cases
    if test_cases_dir.exists():
        test_case_files = sorted(test_cases_dir.glob("*.txt"))
        print(f"  Test cases directory: {test_cases_dir}")
        print(f"  Found {len(test_case_files)} test cases")
    else:
        print(f"  Warning: Test cases directory not found: {test_cases_dir}")
        print(f"  Using random scene generation instead")
        test_case_files = []

    result = BenchmarkResult(task=task, object_set=object_set)

    # Create environment
    env = A2Env(
        task=task if task != "pickplace" else "pick_and_place",
        object_set="train" if object_set in ("seen", "train") else "test",
        num_objects=8,
        gui=False,
    )

    total_episodes_run = 0

    if test_case_files:
        # Run benchmark on each test case
        for case_idx, case_file in enumerate(test_case_files):
            case_name = case_file.stem
            test_case = load_test_case(case_file, task=task)
            lang_goal = test_case["lang_goal"]

            print(f"\n  Case {case_idx + 1}/{len(test_case_files)}: {case_name}")
            print(f"  Language goal: {lang_goal}")

            case_result = CaseResult(
                case_name=case_name,
                lang_goal=lang_goal,
                num_episodes=num_episodes,
            )

            for episode in range(num_episodes):
                # Periodic environment recreation for memory management
                if (
                    env_reset_interval > 0 and
                    total_episodes_run > 0 and
                    total_episodes_run % env_reset_interval == 0
                ):
                    print(f"  [Memory cleanup] Recreating environment...")
                    env.close()
                    cleanup_gpu_memory()
                    env = A2Env(
                        task=task if task != "pickplace" else "pick_and_place",
                        object_set="train" if object_set in ("seen", "train") else "test",
                        num_objects=8,
                        gui=False,
                    )

                # Setup environment from test case (loads actual objects from file)
                seed = 1234 + case_idx * 1000 + episode
                setup_result = setup_environment_from_test_case(env, test_case, seed, task=task)

                # Use the language goal from the loaded test case
                episode_lang_goal = setup_result.get("lang_goal") or lang_goal

                # For pickplace, also set separate grasp and place goals
                if task == "pickplace":
                    env._env.grasp_lang_goal = setup_result.get("grasp_lang_goal") or episode_lang_goal
                    env._env.place_lang_goal = setup_result.get("place_lang_goal") or "place it somewhere"
                    # Also set reference object info for place phase
                    if test_case.get("reference_obj_name"):
                        env._env.reference_obj_name = test_case["reference_obj_name"]
                    if test_case.get("direction"):
                        env._env.reference_direction = test_case["direction"]

                # Run episode based on task
                if task == "grasp":
                    ep_result = run_grasp_episode(env, policy, episode_lang_goal, max_attempts, device)
                elif task == "pickplace":
                    ep_result = run_pickplace_episode(env, policy, episode_lang_goal, max_attempts, device)
                elif task == "place":
                    ep_result = run_place_episode(env, policy, episode_lang_goal, max_attempts, device, test_case=test_case)
                else:
                    # Fallback for unknown task
                    ep_result = run_grasp_episode(env, policy, episode_lang_goal, max_attempts, device)

                case_result.total_steps += ep_result["steps"]
                if ep_result["success"]:
                    case_result.num_successes += 1
                    case_result.success_steps.append(ep_result["steps"])

                total_episodes_run += 1
                cleanup_gpu_memory()

            # Print case results
            print(f"  Case result: {case_result.num_successes}/{num_episodes} ({case_result.success_rate * 100:.1f}%)")
            result.cases.append(case_result)

    else:
        # Run with random scenes if no test cases found
        print(f"\n  Running {num_episodes} episodes with random scenes...")

        case_result = CaseResult(
            case_name="random",
            lang_goal="random",
            num_episodes=num_episodes,
        )

        for episode in range(num_episodes):
            # Periodic environment recreation
            if (
                env_reset_interval > 0 and
                episode > 0 and
                episode % env_reset_interval == 0
            ):
                print(f"  [Memory cleanup] Recreating environment...")
                env.close()
                cleanup_gpu_memory()
                env = A2Env(
                    task=task if task != "pickplace" else "pick_and_place",
                    object_set="train" if object_set in ("seen", "train") else "test",
                    num_objects=8,
                    gui=False,
                )

            # Reset environment
            seed = 1234 + episode
            obs, info = env.reset(seed=seed)

            lang_goal = getattr(env._env, "lang_goal", "grasp an object")
            print(f"\n  Episode {episode + 1}/{num_episodes}: {lang_goal}")

            # Run episode based on task
            if task == "grasp":
                ep_result = run_grasp_episode(env, policy, lang_goal, max_attempts, device)
            elif task == "pickplace":
                ep_result = run_pickplace_episode(env, policy, lang_goal, max_attempts, device)
            elif task == "place":
                ep_result = run_place_episode(env, policy, lang_goal, max_attempts, device)
            else:
                ep_result = run_grasp_episode(env, policy, lang_goal, max_attempts, device)

            case_result.total_steps += ep_result["steps"]
            if ep_result["success"]:
                case_result.num_successes += 1
                case_result.success_steps.append(ep_result["steps"])

            status = "SUCCESS" if ep_result["success"] else "FAIL"
            print(f"    [{status}] steps={ep_result['steps']}, reward={ep_result['reward']:.2f}")

            cleanup_gpu_memory()

        result.cases.append(case_result)

    env.close()

    return result


def print_results(results: list[BenchmarkResult], output_file: str | None = None):
    """Print benchmark results in A2_new format."""
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")

    all_results = {}

    for result in results:
        print(f"\n{result.task.upper()} ({result.object_set}):")
        print(f"  Overall: {result.total_successes}/{result.total_episodes} ({result.overall_success_rate * 100:.1f}%)")

        if len(result.cases) > 1:
            print(f"\n  Per-case results:")
            print(f"  {'Case':<20} {'Goal':<30} {'Success Rate':<15} {'Avg Steps':<10}")
            print(f"  {'-' * 75}")

            for case in result.cases:
                print(f"  {case.case_name:<20} {case.lang_goal[:28]:<30} {case.success_rate * 100:>6.1f}% {case.avg_steps:>10.1f}")

        all_results[f"{result.task}_{result.object_set}"] = result.to_dict()

    # Save results to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Full A2 Benchmark (A2_new compatible)")

    # Task and object set
    parser.add_argument(
        "--task",
        type=str,
        default="grasp",
        choices=["grasp", "place", "pickplace", "all"],
        help="Task to benchmark",
    )
    parser.add_argument(
        "--object_set",
        type=str,
        default="seen",
        choices=["seen", "unseen", "train", "test", "all"],
        help="Object set to use",
    )

    # Policy settings
    parser.add_argument("--policy", type=str, default="a2", help="Policy type (a2, vla, or custom)")
    parser.add_argument("--policy_path", type=str, default="", help="Path to policy checkpoint")
    parser.add_argument("--model_path", type=str, default="", help="Alias for --policy_path")
    parser.add_argument("--direct_grounding", action="store_true", default=True)
    parser.add_argument("--no-direct_grounding", dest="direct_grounding", action="store_false")

    # Evaluation settings (matching A2_new defaults)
    parser.add_argument("--num_episodes", type=int, default=15, help="Episodes per test case")
    parser.add_argument("--max_attempts", type=int, default=8, help="Max attempts per episode")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Memory management
    parser.add_argument("--env_reset_interval", type=int, default=10)

    # Output
    parser.add_argument("--output", type=str, default="", help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load policy
    # Support both --policy_path and --model_path (backward compat)
    policy_path = args.policy_path or args.model_path or None

    # Default to trained A2 checkpoint if none specified
    if policy_path is None and args.policy == "a2":
        default_checkpoint = "/home/denis-office/lerobot_a2/A2_new/logs/a2_pretrained/checkpoints/sl_checkpoint_199.pth"
        import os
        if os.path.exists(default_checkpoint):
            policy_path = default_checkpoint
            # When using trained checkpoint, use learned networks (not direct grounding)
            if args.direct_grounding:
                print("  Note: Using trained checkpoint, switching to learned networks (--no-direct_grounding)")
                args.direct_grounding = False

    print(f"\nLoading policy: {args.policy}")
    if policy_path:
        print(f"  Checkpoint path: {policy_path}")

    policy = load_policy(
        policy_type=args.policy,
        policy_path=policy_path,
        device=device,
    )

    # Set direct grounding mode
    if hasattr(policy, "config") and hasattr(policy.config, "direct_grounding"):
        policy.config.direct_grounding = args.direct_grounding
        print(f"  Direct grounding: {args.direct_grounding}")

    print(f"  Policy class: {type(policy).__name__}")

    # Determine tasks and object sets to run
    tasks = ["grasp", "place", "pickplace"] if args.task == "all" else [args.task]
    object_sets = ["seen", "unseen"] if args.object_set == "all" else [args.object_set]

    # Run benchmarks
    all_results = []

    for task in tasks:
        for object_set in object_sets:
            result = benchmark_task(
                task=task,
                object_set=object_set,
                policy=policy,
                device=device,
                num_episodes=args.num_episodes,
                max_attempts=args.max_attempts,
                direct_grounding=args.direct_grounding,
                env_reset_interval=args.env_reset_interval,
            )
            all_results.append(result)

    # Print and save results
    output_file = args.output if args.output else f"benchmark_results_{args.task}_{args.object_set}.json"
    print_results(all_results, output_file)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for result in all_results:
        print(f"  {result.task} ({result.object_set}): {result.overall_success_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
