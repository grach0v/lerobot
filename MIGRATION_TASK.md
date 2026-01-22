# A2 Environment Migration Task

## Goal
Move ALL A2 environment code from A2_new into this lerobot fork so that:
- `pip install lerobot[a2]` gives you everything needed to run A2 simulation
- NO dependency on A2_new repository
- Benchmark and data collection scripts work standalone
- Can optionally use lerobot_policy_a2 for running trained policies

---

## Current State

### Already exists but BROKEN:
- `src/lerobot/envs/a2.py` - A2Env wrapper, but imports from A2_new
- `src/lerobot/envs/configs.py` - A2EnvConfig
- `src/lerobot/scripts/eval_a2.py` - Basic evaluation script (placeholder)

### Problem:
```python
# In a2.py - THIS MUST BE REMOVED:
a2_path = Path(__file__).parent.parent.parent.parent.parent.parent / "A2_new"
from env.environment_sim import Environment  # Imports from A2_new!
```

---

## Files to move FROM A2_new:

### 1. Environment core (CRITICAL):
```
A2_new/env/environment_sim.py      → src/lerobot/envs/a2/environment.py
A2_new/env/unseen_environment_sim.py → merge into environment.py
A2_new/env/constants.py            → src/lerobot/envs/a2/constants.py
A2_new/env/robot_sim.py            → src/lerobot/envs/a2/robot.py
A2_new/env/camera_sim.py           → src/lerobot/envs/a2/camera.py
```

### 2. Benchmark/Evaluation:
```
A2_new/a2/evaluate/evaluator.py    → src/lerobot/scripts/eval_a2.py (rewrite)
A2_new/a2/evaluate/test_pick.py    → integrate into eval_a2.py
A2_new/a2/evaluate/test_pickplace.py → integrate into eval_a2.py
```

### 3. Data Collection:
```
A2_new/data_collection/collect_data_grasp.py → src/lerobot/scripts/collect_a2.py
A2_new/data_collection/collect_data_place.py → integrate into collect_a2.py
```

### 4. Feature extraction (if needed by env):
```
A2_new/feature_extractor/feature_field_builder.py → src/lerobot/envs/a2/feature_field.py
```

### 5. Utils (cherry-pick what's needed):
```
A2_new/utils/utils.py → src/lerobot/envs/a2/utils.py (only env-related functions)
```

---

## After migration, directory structure:
```
src/lerobot/envs/
├── a2/
│   ├── __init__.py
│   ├── environment.py      # Main PyBullet simulation
│   ├── robot.py            # UR5e + Robotiq gripper control
│   ├── camera.py           # Camera rendering
│   ├── constants.py        # Workspace limits, etc.
│   ├── feature_field.py    # Optional: feature extraction
│   └── utils.py            # Environment utilities
├── a2.py                   # Gymnasium wrapper (update imports)
├── configs.py              # A2EnvConfig
└── factory.py              # Environment factory

src/lerobot/scripts/
├── eval_a2.py              # Benchmark evaluation
└── collect_a2.py           # Data collection for VLA training
```

---

## Assets
- Downloaded from HuggingFace: `dgrachev/a2_assets`
- Function `get_a2_assets_path()` in a2.py handles this
- Contains: simplified_objects, unseen_objects, ur5e, workspace

---

## Dependencies to add to pyproject.toml [a2] extra:
- pybullet
- open3d
- scipy
- (graspnetAPI only if feature_field needs it)

---

## Testing after migration:
```bash
# Install
cd lerobot_grach0v
uv pip install -e ".[a2]"

# Test environment
python -c "from lerobot.envs.a2 import A2Env; env = A2Env(); env.reset()"

# Run benchmark (without policy - random actions)
python -m lerobot.scripts.eval_a2 --task grasp --num_episodes 5

# Run benchmark WITH policy
pip install lerobot_policy_a2
python -m lerobot.scripts.eval_a2 --task grasp --policy lerobot_policy_a2
```

---

## README update needed:
Remove any references to cloning A2 or A2_bench repositories.
