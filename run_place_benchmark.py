#!/usr/bin/env python
"""Run full place benchmark to measure actual success rate."""

import sys

sys.path.insert(0, "src")
from lerobot.envs.a2.a2_benchmark import A2Benchmark, BenchmarkConfig
from lerobot.envs.a2.a2_collect import create_a2_policy
import glob

# Create policy
policy = create_a2_policy(hf_repo="dgrachev/a2_pretrained", device="cuda")

# Create config
config = BenchmarkConfig(
    task="place",
    object_set="train",
    num_episodes=3,  # 3 episodes per test case
    max_steps=1,
    testing_case_dir="/home/denis-office/.cache/a2_assets/testing_cases/place_testing_cases/seen",
)

# Get test cases
test_cases = sorted(glob.glob(config.testing_case_dir + "/case*.txt"))[:20]  # First 20 cases
print(f"Running benchmark on {len(test_cases)} test cases, {config.num_episodes} episodes each")
print(f"Total evaluations: {len(test_cases) * config.num_episodes}")
print()

# Run benchmark
benchmark = A2Benchmark(policy=policy, config=config)

all_results = []
for test_case in test_cases:
    case_name = test_case.split("/")[-1]
    result = benchmark.evaluate_place(test_case)
    all_results.append(result)
    print(f"{case_name}: {result.success_rate * 100:.1f}% ({result.num_episodes} episodes)")

# Calculate overall success rate
total_episodes = sum(r.num_episodes for r in all_results)
total_successes = sum(r.success_rate * r.num_episodes for r in all_results)
overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0

print()
print(f"Overall success rate: {overall_success_rate * 100:.1f}%")
print(f"Total episodes: {int(total_episodes)}")
