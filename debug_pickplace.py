#!/usr/bin/env python
"""Debug script for pickplace target objects."""

import sys

sys.path.insert(0, "src")
from lerobot.envs.a2.a2_sim import Environment

env = Environment(gui=False, object_set="train")
env.reset(workspace="extend")
success, grasp_lang, place_lang = env.add_object_push_from_pickplace_file(
    "/home/denis-office/.cache/a2_assets/testing_cases/pp_testing_cases/seen/case00.txt", mode="grasp"
)
print(f"Success: {success}")
print(f"Grasp lang: {grasp_lang}")
print(f"Target obj ids: {env.target_obj_ids}")
print(f"Number of objects: {len(env.obj_ids['rigid'])}")
for obj_id in env.target_obj_ids[:3]:
    pos, quat, size = env.obj_info(obj_id)
    print(f"  Target {obj_id}: pos={pos}, size={size}")
env.close()
