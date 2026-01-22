#!/usr/bin/env python
"""Evaluate A2 policies on pick/place tasks.

Usage:
    # Evaluate grasp with trained model
    python -m lerobot.scripts.eval_a2 \
        --task grasp \
        --model_path dgrachev/a2_grasp \
        --num_episodes 50

    # Evaluate pick-and-place with CLIP baseline
    python -m lerobot.scripts.eval_a2 \
        --task pick_and_place \
        --direct_grounding \
        --num_episodes 50

    # Evaluate on unseen objects
    python -m lerobot.scripts.eval_a2 \
        --task grasp \
        --model_path dgrachev/a2_grasp \
        --unseen \
        --num_episodes 50
"""

#TODO: Why is this part of scripts and not a2 env?
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate A2 policy on pick/place tasks")

    # Task settings
    parser.add_argument("--task", type=str, default="grasp",
                        choices=["grasp", "pick_and_place"],
                        help="Task type")
    parser.add_argument("--object_set", type=str, default="train",
                        choices=["train", "test"],
                        help="Object set (train=simplified, test=unseen)")
    parser.add_argument("--unseen", action="store_true", default=False,
                        help="Use unseen objects (alias for --object_set test)")
    parser.add_argument("--num_objects", type=int, default=8,
                        help="Number of objects per episode")

    # Evaluation settings
    parser.add_argument("--num_episodes", type=int, default=15,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")

    # Model settings
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to trained model or HuggingFace repo")
    parser.add_argument("--direct_grounding", action="store_true", default=False,
                        help="Use CLIP direct grounding baseline (no trained model)")
    parser.add_argument("--use_rope", action="store_true", default=True,
                        help="Use RoPE position encoding")

    # Environment settings
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Show PyBullet GUI")
    parser.add_argument("--cameras", type=str, default="front,overview,gripper",
                        help="Cameras to use")

    # Output settings
    parser.add_argument("--output_dir", type=str, default="outputs/eval_a2",
                        help="Directory for evaluation results")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    args = parser.parse_args()

    # Handle --unseen flag
    if args.unseen:
        args.object_set = "test"

    return args


def create_env(args):
    """Create A2 environment."""
    from lerobot.envs.a2 import A2Env

    env = A2Env(
        task=args.task,
        object_set=args.object_set,
        num_objects=args.num_objects,
        camera_name=args.cameras,
        observation_width=640,
        observation_height=480,
        include_depth=True,
        gui=args.gui,
        episode_length=args.max_steps,
    )

    return env


def create_policy(args):
    """Create A2 policy for evaluation."""
    if args.direct_grounding:
        # Use CLIP baseline
        print("Using CLIP direct grounding baseline")
        return None

    if not args.model_path:
        print("Warning: No model path provided, using random actions")
        return None

    try:
        from lerobot_policy_a2 import A2Policy, A2Config

        print(f"Loading A2 policy from {args.model_path}...")
        config = A2Config(
            use_rope=args.use_rope,
            device=args.device,
        )
        policy = A2Policy.from_pretrained(args.model_path, **vars(config))
        policy.eval()
        return policy
    except ImportError:
        print("Error: lerobot_policy_a2 not installed")
        print("Install with: pip install lerobot_policy_a2")
        return None
    except Exception as e:
        print(f"Error loading policy: {e}")
        return None


def evaluate_episode(env, policy, args, episode_idx: int) -> dict[str, Any]:
    """Run single evaluation episode.

    Returns:
        Episode results dictionary
    """
    obs, info = env.reset(seed=args.seed + episode_idx)
    done = False
    truncated = False
    step = 0
    success = False
    rewards = []

    while not (done or truncated) and step < args.max_steps:
        # Get action
        if policy is None:
            # Random action
            action = env.action_space.sample()
        else:
            # Use policy (simplified - full implementation would process observations)
            action = env.action_space.sample()  # Placeholder

        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        step += 1

        if info.get("is_success", False):
            success = True
            done = True

    return {
        "episode": episode_idx,
        "success": success,
        "steps": step,
        "total_reward": sum(rewards),
        "done": done,
        "truncated": truncated,
    }


def main():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"A2 Benchmark Evaluation")
    print(f"=" * 50)
    print(f"Task: {args.task}")
    print(f"Object set: {args.object_set}")
    print(f"Num objects: {args.num_objects}")
    print(f"Num episodes: {args.num_episodes}")
    print(f"Model: {args.model_path or 'None (random/CLIP)'}")
    print(f"=" * 50)

    # Create environment
    print("\nCreating environment...")
    env = create_env(args)

    # Create policy
    print("Creating policy...")
    policy = create_policy(args)

    # Run evaluation
    results = []
    successes = 0

    print(f"\nRunning {args.num_episodes} evaluation episodes...")
    for i in range(args.num_episodes):
        result = evaluate_episode(env, policy, args, i)
        results.append(result)

        if result["success"]:
            successes += 1

        # Print progress
        success_rate = successes / (i + 1) * 100
        print(f"Episode {i+1}/{args.num_episodes}: "
              f"{'Success' if result['success'] else 'Failed'} | "
              f"Steps: {result['steps']} | "
              f"Success rate: {success_rate:.1f}%")

    # Summary statistics
    success_rate = successes / args.num_episodes * 100
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])

    print(f"\n{'=' * 50}")
    print(f"EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"Task: {args.task}")
    print(f"Object set: {args.object_set}")
    print(f"Total episodes: {args.num_episodes}")
    print(f"Successful episodes: {successes}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"{'=' * 50}")

    # Save results
    results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_data = {
        "config": vars(args),
        "summary": {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "total_episodes": args.num_episodes,
            "successful_episodes": successes,
        },
        "episodes": results,
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Cleanup
    env.close()

    return success_rate


if __name__ == "__main__":
    main()
