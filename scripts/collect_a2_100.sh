#!/usr/bin/env bash
# Collect 100 episodes of each A2 task
#
# Usage:
#   ./scripts/collect_a2_100.sh                # Use seen objects (default)
#   ./scripts/collect_a2_100.sh --unseen       # Use unseen objects
#   ./scripts/collect_a2_100.sh --object_set test  # Same as --unseen

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
OUT_DIR="${ROOT_DIR}/dataset_a2_100"
OBJECT_SET="train"  # "train" = seen, "test" = unseen

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --object_set|--object-set)
            OBJECT_SET="$2"
            shift 2
            ;;
        --seen)
            OBJECT_SET="train"
            shift
            ;;
        --unseen)
            OBJECT_SET="test"
            shift
            ;;
        --output)
            OUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --object_set SET   Object set: 'train' (seen) or 'test' (unseen)"
            echo "  --seen             Use seen objects (default)"
            echo "  --unseen           Use unseen objects"
            echo "  --output DIR       Output directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

echo "========================================"
echo "A2 Data Collection (100 episodes)"
echo "========================================"
echo "Object set: $OBJECT_SET ($([ "$OBJECT_SET" = "train" ] && echo 'seen' || echo 'unseen') objects)"
echo "Output: ${OUT_DIR}"
echo "========================================"

uv run python -m lerobot.envs.a2.a2_collect \
  --policy a2 \
  --policy_path dgrachev/a2_pretrained \
  --task grasp \
  --object_set "$OBJECT_SET" \
  --num_episodes 100 \
  --max_attempts 8 \
  --num_objects 15 \
  --action_mode pose \
  --output_dir "${OUT_DIR}" \
  --repo_id "local/a2_grasp_100" \
  --device cuda \
  --distinguish_failed_attempts \
  2>&1 | tee "${LOG_DIR}/a2_collect_grasp_100.log"

uv run python -m lerobot.envs.a2.a2_collect \
  --policy a2 \
  --policy_path dgrachev/a2_pretrained \
  --task place \
  --object_set "$OBJECT_SET" \
  --num_episodes 100 \
  --max_attempts 8 \
  --num_objects 15 \
  --action_mode pose \
  --output_dir "${OUT_DIR}" \
  --repo_id "local/a2_place_100" \
  --device cuda \
  --oracle_grasp \
  --distinguish_failed_attempts \
  2>&1 | tee "${LOG_DIR}/a2_collect_place_100.log"

uv run python -m lerobot.envs.a2.a2_collect \
  --policy a2 \
  --policy_path dgrachev/a2_pretrained \
  --task pick_and_place \
  --object_set "$OBJECT_SET" \
  --num_episodes 100 \
  --max_attempts 8 \
  --num_objects 15 \
  --action_mode pose \
  --output_dir "${OUT_DIR}" \
  --repo_id "local/a2_pick_and_place_100" \
  --device cuda \
  --distinguish_failed_attempts \
  2>&1 | tee "${LOG_DIR}/a2_collect_pick_and_place_100.log"

echo "========================================"
echo "Collection complete!"
echo "========================================"
