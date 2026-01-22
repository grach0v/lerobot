# A2 Environment for LeRobot

A PyBullet-based tabletop manipulation environment for 6-DOF grasp and place tasks with UR5e robot.

## Overview

The A2 environment provides:
- **Robot**: UR5e with Robotiq 2F-85 gripper
- **Tasks**: `grasp`, `place`, `pick_and_place`
- **Object Sets**: `train` (66 simplified objects), `test` (17 unseen objects)
- **Cameras**: `front`, `overview`, `gripper`
- **Assets**: Automatically downloaded from [dgrachev/a2_assets](https://huggingface.co/datasets/dgrachev/a2_assets)

## Installation

```bash
# Clone LeRobot fork with A2 support
git clone https://github.com/grach0v/lerobot.git lerobot_grach0v
cd lerobot_grach0v

# Install with uv (recommended)
uv sync --extra a2

# Or with pip
pip install -e ".[a2]"

# Install A2 policy package (optional, for A2 policy)
uv pip install -e /path/to/lerobot_policy_a2
```

## Quick Start

### Basic Environment Usage

```python
import os
os.environ['A2_DISABLE_EGL'] = 'true'  # For headless rendering

from lerobot.envs.a2 import A2Env

# Create environment
env = A2Env(
    task="grasp",           # grasp, place, or pick_and_place
    object_set="train",     # train or test
    num_objects=8,          # Number of objects in scene
    camera_name="front",    # front, overview, gripper, or comma-separated
    gui=False,              # Show PyBullet GUI
)

# Reset and get observation
obs, info = env.reset()

# Sample random action (7D pose: xyz + quaternion)
action = env.action_space.sample()

# Step environment
obs, reward, terminated, truncated, info = env.step(action)

# Close environment
env.close()
```

### Data Collection

```bash
cd lerobot_grach0v

# Collect data with random policy (for testing)
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_collect \
    --policy random \
    --task grasp \
    --num_episodes 10 \
    --max_steps 8 \
    --num_objects 8 \
    --output_dir /tmp/a2_test \
    --repo_id local/a2_test \
    --fps 30

# Collect data with pretrained A2 policy from HuggingFace
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_collect \
    --policy a2 \
    --hf_repo dgrachev/a2_pretrained \
    --task grasp \
    --num_episodes 100 \
    --max_steps 8 \
    --output_dir data/a2_collected \
    --repo_id local/a2_grasp \
    --device cuda

# Collect pick-and-place data
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_collect \
    --policy a2 \
    --hf_repo dgrachev/a2_pretrained \
    --task pick_and_place \
    --num_episodes 100 \
    --output_dir data/a2_pickplace
```

### Benchmark Evaluation

```bash
cd lerobot_grach0v

# Evaluate on grasp task (seen objects)
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_benchmark \
    --task grasp \
    --object_set train \
    --policy a2 \
    --hf_repo dgrachev/a2_pretrained \
    --num_episodes 15 \
    --max_steps 8 \
    --results_dir benchmark_results \
    --device cuda

# Evaluate on unseen objects
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_benchmark \
    --task grasp \
    --object_set test \
    --policy a2 \
    --hf_repo dgrachev/a2_pretrained

# Evaluate specific test case
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_benchmark \
    --task grasp \
    --testing_case case00-round.txt \
    --policy a2 \
    --hf_repo dgrachev/a2_pretrained
```

## Command Line Options

### a2_collect.py

| Option | Default | Description |
|--------|---------|-------------|
| `--policy` | `random` | Policy type: `random`, `a2`, or `custom` |
| `--model_path` | `""` | Local path to model checkpoint |
| `--hf_repo` | `""` | HuggingFace repo ID (e.g., `dgrachev/a2_pretrained`) |
| `--task` | `grasp` | Task: `grasp`, `place`, `pick_and_place` |
| `--object_set` | `train` | Object set: `train` or `test` |
| `--num_episodes` | `100` | Number of episodes to collect |
| `--max_steps` | `8` | Maximum steps per episode |
| `--num_objects` | `8` | Number of objects in scene |
| `--output_dir` | `data/vla_collected` | Output directory |
| `--repo_id` | `local/a2_collected` | LeRobot dataset repository ID |
| `--cameras` | `front,overview,gripper` | Cameras to record |
| `--fps` | `30` | Recording FPS |
| `--device` | `cuda` | Device for policy inference |
| `--gui` | `False` | Show PyBullet GUI |

### a2_benchmark.py

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | `grasp` | Task: `grasp`, `place`, `pickplace` |
| `--object_set` | `train` | Object set: `train` or `test` |
| `--testing_case` | `None` | Specific test case file (e.g., `case00-round.txt`) |
| `--num_episodes` | `15` | Episodes per test case |
| `--max_steps` | `8` | Maximum steps per episode |
| `--policy` | `random` | Policy type: `random`, `a2`, `custom` |
| `--model_path` | `""` | Local path to model checkpoint |
| `--hf_repo` | `""` | HuggingFace repo ID |
| `--results_dir` | `benchmark_results` | Results output directory |
| `--device` | `cuda` | Device for policy inference |
| `--gui` | `False` | Show PyBullet GUI |

## Testing Cases

Testing cases are included in the assets and define specific object configurations:

```
testing_cases/
├── grasp_testing_cases/
│   ├── seen/          # 10 test cases with train objects
│   │   ├── case00-round.txt
│   │   ├── case01-eat.txt
│   │   └── ...
│   └── unseen/        # 5 test cases with test objects
├── place_testing_cases/
│   ├── seen/          # 20 test cases
│   └── unseen/        # 10 test cases
└── pp_testing_cases/  # Pick-and-place
    ├── seen/          # 8 test cases
    └── unseen/        # 4 test cases
```

Each test case file contains:
- Line 1: Language instruction (e.g., "grasp a round object")
- Line 2: Target object indices
- Lines 3+: Object URDFs with poses

## Environment Details

### Observation Space

| Key | Shape | Description |
|-----|-------|-------------|
| `observation.images.front` | (480, 640, 3) | Front camera RGB |
| `observation.images.overview` | (480, 640, 3) | Overview camera RGB |
| `observation.images.gripper` | (480, 640, 3) | Gripper camera RGB |

### Action Space

| Shape | Description |
|-------|-------------|
| (7,) | 6-DOF pose: xyz position (3) + quaternion orientation (4) |

### Reward

- `1.0` for successful grasp/place of target object
- `0.0` otherwise

## HuggingFace Resources

| Resource | URL | Description |
|----------|-----|-------------|
| **Environment Assets** | [dgrachev/a2_assets](https://huggingface.co/datasets/dgrachev/a2_assets) | 3D objects, URDFs, testing cases |
| **Pretrained A2 Policy** | [dgrachev/a2_pretrained](https://huggingface.co/dgrachev/a2_pretrained) | Trained policy weights |

## Troubleshooting

### EGL Rendering Issues

If you encounter EGL/OpenGL errors in headless environments:

```bash
# Disable EGL rendering
export A2_DISABLE_EGL=true

# Or set in Python before importing
import os
os.environ['A2_DISABLE_EGL'] = 'true'
```

### PyBullet Segmentation Faults

If PyBullet crashes with segmentation faults:

1. Ensure proper disconnection:
```python
env.close()  # Always close environment when done
```

2. Run with EGL disabled:
```bash
A2_DISABLE_EGL=true python your_script.py
```

### Missing Assets

Assets are automatically downloaded from HuggingFace on first use. To force re-download:

```python
from lerobot.envs.a2 import get_a2_assets_path
import shutil

# Remove cached assets
cache_dir = get_a2_assets_path()
shutil.rmtree(cache_dir, ignore_errors=True)

# Assets will be re-downloaded on next environment creation
```

## File Structure

```
lerobot/envs/
├── a2.py                 # Gymnasium-compatible A2Env wrapper
├── a2_sim.py             # Core PyBullet simulation
├── a2_constants.py       # Constants (workspace limits, etc.)
├── a2_cameras.py         # Camera configurations
├── a2_utils.py           # Utility functions
├── a2_collect.py         # Data collection script
├── a2_benchmark.py       # Benchmark evaluation script
├── a2_recorder.py        # VLA dataset recorder
└── A2_README.md          # This file
```

## License

Apache-2.0
