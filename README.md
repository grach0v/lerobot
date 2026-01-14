<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry so that everyone can contribute to and benefit from shared datasets and pretrained models.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 State-of-the-art policies that have been shown to transfer to the real-world ready for training and deployment.

🤗 Comprehensive support for the open-source ecosystem to democratize physical AI.

## Quick Start

LeRobot can be installed directly from PyPI.

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> For detailed installation guide, please see the [Installation Documentation](https://huggingface.co/docs/lerobot/installation).

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices.

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**Supported Hardware:** SO100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones, OpenARM, Unitree G1.

While these devices are natively integrated into the LeRobot codebase, the library is designed to be extensible. You can easily implement the Robot interface to utilize LeRobot's data collection, training, and visualization tools for your own custom robot.

For detailed hardware setup guides, see the [Hardware Documentation](https://huggingface.co/docs/lerobot/integrate_hardware).

## LeRobot Dataset

To solve the data fragmentation problem in robotics, we utilize the **LeRobotDataset** format.

- **Structure:** Synchronized MP4 videos (or images) for vision and Parquet files for state/action data.
- **HF Hub Integration:** Explore thousands of robotics datasets on the [Hugging Face Hub](https://huggingface.co/lerobot).
- **Tools:** Seamlessly delete episodes, split by indices/fractions, add/remove features, and merge multiple datasets.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

Learn more about it in the [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## SoTA Models

LeRobot implements state-of-the-art policies in pure PyTorch, covering Imitation Learning, Reinforcement Learning, and Vision-Language-Action (VLA) models, with more coming soon. It also provides you with the tools to instrument and inspect your training process.

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

Training a policy is as simple as running a script configuration:

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| Category                   | Models                                                                                                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Imitation Learning**     | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md)                                                             |
| **Reinforcement Learning** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL (coming soon)                                                                                                  |
| **VLAs Models**            | [Pi0Fast](./docs/source/pi0fast.mdx), [Pi0.5](./docs/source/pi05.mdx), [GR00T N1.5](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx) |

Similarly to the hardware, you can easily implement your own policy & leverage LeRobot's data collection, training, and visualization tools, and share your model to the HF Hub

For detailed policy setup guides, see the [Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies).

## Inference & Evaluation

Evaluate your policies in simulation or on real hardware using the unified evaluation script. LeRobot supports standard benchmarks like **LIBERO**, **MetaWorld** and more to come.

```bash
# Evaluate a policy on the LIBERO benchmark
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

Learn how to implement your own simulation environment or benchmark and distribute it from the HF Hub by following the [EnvHub Documentation](https://huggingface.co/docs/lerobot/envhub)

---

## A2 Pick-and-Place Benchmark

This fork includes the **A2 Pick-and-Place Benchmark** - a language-conditioned manipulation benchmark using PyBullet simulation with UR5e robot and Robotiq gripper.

### Quick Setup from Scratch

```bash
# 1. Clone this repository
git clone https://github.com/grach0v/lerobot.git
cd lerobot
git checkout a2_pp_bench

# 2. Create uv environment and install dependencies
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[a2]"

# 3. Clone and install A2 policy package
cd ..
git clone https://github.com/grach0v/lerobot_policy_a2.git
cd lerobot_policy_a2
uv pip install -e .
cd ../lerobot

# 4. Clone A2 simulation (for benchmark scripts and assets)
cd ..
git clone https://github.com/grach0v/A2.git A2_bench
cd A2_bench
uv pip install -e .

# 5. Download simulation assets from HuggingFace
python scripts/setup/download_resources.py
```

### Running the Benchmark

The A2 benchmark evaluates language-conditioned grasping and pick-and-place policies.

#### Grasp (Pick) Evaluation

```bash
cd A2_bench

# Run grasp evaluation with trained model
python -m a2.evaluate.test_pick \
    --load_model \
    --model_path path/to/checkpoint.pth \
    --num_episode 50 \
    --use_rope

# Run with CLIP baseline (no trained model)
python -m a2.evaluate.test_pick \
    --direct_grounding \
    --num_episode 50

# Run on unseen objects
python -m a2.evaluate.test_pick \
    --unseen \
    --load_model \
    --model_path path/to/checkpoint.pth
```

#### Pick-and-Place Evaluation

```bash
# Run full pick-and-place evaluation
python -m a2.evaluate.test_pickplace \
    --load_model \
    --model_path path/to/checkpoint.pth \
    --num_episode 50 \
    --use_rope

# With GUI visualization
python -m a2.evaluate.test_pickplace \
    --gui \
    --load_model \
    --model_path path/to/checkpoint.pth
```

### Using A2 Environment with LeRobot

```python
from lerobot.envs.configs import A2EnvConfig
from lerobot.envs.factory import make_env

# Create A2 environment
config = A2EnvConfig(
    task="pick_and_place",  # or "grasp"
    object_set="train",     # or "test" for unseen objects
    num_objects=8,
    camera_name="front,overview,gripper",
)

env = make_env(config)
obs, info = env.reset()

# Run episode
done = False
while not done:
    action = policy.select_action(obs)
    obs, reward, done, truncated, info = env.step(action)
```

### Data Collection for VLA Training

Record demonstrations in LeRobot dataset format:

```bash
# Collect grasp demonstrations
python data_collection/collect_data_grasp.py \
    --num_episode 1000 \
    --num_obj 8 \
    --record_vla \
    --vla_output data/a2_grasp_vla

# Collect pick-and-place demonstrations
python data_collection/collect_data_pp.py \
    --num_episode 1000 \
    --record_vla \
    --vla_output data/a2_pp_vla
```

### Model Checkpoints

Pre-trained A2 policy checkpoints are available on HuggingFace:

| Model | Task | Success Rate | Link |
|-------|------|--------------|------|
| A2-ViLGP3D-Grasp | Grasping | ~85% | `dgrachev/a2_grasp` |
| A2-ViLGP3D-PP | Pick-and-Place | ~70% | `dgrachev/a2_pickplace` |

### Benchmark Tasks

The A2 benchmark includes:

1. **Language-Conditioned Grasping**: "Give me the banana", "Grasp a red object", etc.
2. **Spatial Place**: "Put it next to the cup", "Place this near the bowl"
3. **Pick-and-Place**: Combined grasp + place with language instructions

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_episode` | Number of evaluation episodes | 15 |
| `--num_obj` | Objects per scene | 8-15 |
| `--use_rope` | Use RoPE position encoding | False |
| `--unseen` | Evaluate on unseen objects | False |
| `--gui` | Show PyBullet GUI | False |
| `--direct_grounding` | Use CLIP baseline | False |

---

## Resources

- **[Documentation](https://huggingface.co/docs/lerobot/index):** The complete guide to tutorials & API.
- **[Discord](https://discord.gg/q8Dzzpym3f):** Join the `LeRobot` server to discuss with the community.
- **[X](https://x.com/LeRobotHF):** Follow us on X to stay up-to-date with the latest developments.
- **[Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial):** A free, hands-on course to learn robot learning using LeRobot.

## Citation

If you use LeRobot in your research, please cite:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

## Contribute

We welcome contributions from everyone in the community! To get started, please read our [CONTRIBUTING.md](./CONTRIBUTING.md) guide. Whether you're adding a new feature, improving documentation, or fixing a bug, your help and feedback are invaluable. We're incredibly excited about the future of open-source robotics and can't wait to work with you on what's next—thank you for your support!

<p align="center">
  <img alt="SO101 Video" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>Built by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>
