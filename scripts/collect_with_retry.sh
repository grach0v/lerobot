#!/bin/bash
# A2 Data Collection with Auto-Retry
# This script runs data collection and automatically restarts on segfaults (EGL crashes)

set -e

# Default values
POLICY="${POLICY:-a2}"
POLICY_PATH="${POLICY_PATH:-dgrachev/a2_pretrained}"
TASK="${TASK:-grasp}"
NUM_EPISODES="${NUM_EPISODES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-data/a2_collected}"
REPO_ID="${REPO_ID:-local/a2_grasp}"
MAX_RETRIES="${MAX_RETRIES:-50}"

echo "=========================================="
echo "A2 Data Collection with Auto-Retry"
echo "=========================================="
echo "Policy: $POLICY"
echo "Policy Path: $POLICY_PATH"
echo "Task: $TASK"
echo "Total Episodes: $NUM_EPISODES"
echo "Output: $OUTPUT_DIR/$REPO_ID"
echo "Max Retries: $MAX_RETRIES"
echo "=========================================="

retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    echo ""
    echo "[$(date '+%H:%M:%S')] Starting collection attempt $((retry_count + 1))..."

    # Run collection with --resume flag
    set +e
    uv run python -m lerobot.envs.a2.a2_collect \
        --policy "$POLICY" \
        --policy_path "$POLICY_PATH" \
        --task "$TASK" \
        --num_episodes "$NUM_EPISODES" \
        --output_dir "$OUTPUT_DIR" \
        --repo_id "$REPO_ID" \
        --resume \
        "$@"

    exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "[$(date '+%H:%M:%S')] Collection completed successfully!"
        exit 0
    elif [ $exit_code -eq 139 ]; then
        # Segfault - likely EGL crash
        retry_count=$((retry_count + 1))
        echo ""
        echo "[$(date '+%H:%M:%S')] Segfault detected (exit code 139). Restarting..."
        echo "Retry $retry_count of $MAX_RETRIES"
        sleep 2  # Brief pause before restart
    else
        echo ""
        echo "[$(date '+%H:%M:%S')] Collection failed with exit code $exit_code"
        exit $exit_code
    fi
done

echo ""
echo "[$(date '+%H:%M:%S')] Max retries ($MAX_RETRIES) exceeded. Exiting."
exit 1
