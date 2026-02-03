#!/usr/bin/env python
"""Debug script for pickplace grasp execution."""

import sys

sys.path.insert(0, "src")
from lerobot.envs.a2.a2_sim import Environment
import numpy as np

env = Environment(gui=False, object_set="train")
env.reset(workspace="extend")
success, grasp_lang, place_lang = env.add_object_push_from_pickplace_file(
    "/home/denis-office/.cache/a2_assets/testing_cases/pp_testing_cases/seen/case00.txt", mode="grasp"
)
print(f"Success: {success}")
print(f"Grasp lang: {grasp_lang}")
print(f"Target obj ids: {env.target_obj_ids}")

# Get target object info
target_id = env.target_obj_ids[0]
pos, quat, size = env.obj_info(target_id)
print(f"Target object: id={target_id}, pos={pos}, size={size}")

# Try a grasp at target position
grasp_pose = np.array([pos[0], pos[1], pos[2] + 0.02, 1.0, 0.0, 0.0, 0.0])
print(f"Attempting grasp at: {grasp_pose[:3]}")

grasp_success, grasped_id, min_dist = env.grasp(grasp_pose, follow_place=True)
print(f"Grasp result: success={grasp_success}, grasped_id={grasped_id}, min_dist={min_dist}")

if grasped_id in env.target_obj_ids:
    print("SUCCESS: Grasped target object!")
else:
    print(f"FAIL: Grasped wrong object (expected one of {env.target_obj_ids}, got {grasped_id})")

env.close()
