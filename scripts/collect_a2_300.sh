#!/bin/bash
# Collect 1000 episodes of each A2 task (grasp, place, pick_and_place)
#
# Usage:
#   ./scripts/collect_a2_300.sh                    # Run all tasks sequentially
#   ./scripts/collect_a2_300.sh --parallel         # Run all tasks in parallel
#   ./scripts/collect_a2_300.sh --task grasp       # Run single task
#
# Output: dataset_a2_1000/{grasp,place,pick_and_place}_1000/

set -e

# Configuration
NUM_EPISODES=100
MAX_ATTEMPTS=4
NUM_OBJECTS=8
FPS=30
IMAGE_HEIGHT=480
IMAGE_WIDTH=640
DEVICE="cuda"
OUTPUT_BASE="dataset_a2_100"
OBJECT_SET="train"  # "train" = seen objects, "test" = unseen objects
# Environment reset interval to prevent GPU memory leak (PyBullet EGL leak)
# Memory grows ~300-500MB per episode per process
# Sequential: 10 episodes (~5GB peak before reset)
# Parallel (3 procs): 5 episodes (~2.5GB each = ~7.5GB total peak)
ENV_RESET_INTERVAL=10

# Parse arguments
PARALLEL=false
SINGLE_TASK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --task)
            SINGLE_TASK="$2"
            shift 2
            ;;
        --episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
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
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel              Run tasks in parallel"
            echo "  --task TASK             Single task: grasp, place, or pick_and_place"
    echo "  --episodes N            Number of episodes per task (default: 1000)"
    echo "  --output DIR            Output directory (default: dataset_a2_1000)"
            echo "  --object_set SET        Object set: 'train' (seen) or 'test' (unseen)"
            echo "  --seen                  Use seen objects (alias for --object_set train)"
            echo "  --unseen                Use unseen objects (alias for --object_set test)"
            echo "  -h, --help              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--parallel] [--task grasp|place|pick_and_place] [--episodes N] [--output DIR] [--object_set train|test]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to run collection for a task
collect_task() {
    local task=$1
    local extra_args=$2
    local output_dir="${OUTPUT_BASE}/${task}_${NUM_EPISODES}"
    local repo_id="local/${task}_${NUM_EPISODES}"
    local log_file="${OUTPUT_BASE}/${task}.log"

    echo "[$task] Starting collection of $NUM_EPISODES episodes..."
    echo "[$task] Output: $output_dir"
    echo "[$task] Log: $log_file"

    uv run python -m lerobot.envs.a2.a2_collect \
        --policy a2 \
        --policy_path dgrachev/a2_pretrained \
        --task "$task" \
        --object_set "$OBJECT_SET" \
        --num_episodes "$NUM_EPISODES" \
        --max_attempts "$MAX_ATTEMPTS" \
        --num_objects "$NUM_OBJECTS" \
        --fps "$FPS" \
        --image_height "$IMAGE_HEIGHT" \
        --image_width "$IMAGE_WIDTH" \
        --device "$DEVICE" \
        --output_dir "$output_dir" \
        --repo_id "$repo_id" \
        --env_reset_interval "$ENV_RESET_INTERVAL" \
        --distinguish_failed_attempts \
        $extra_args \
        2>&1 | tee "$log_file"

    echo "[$task] Completed!"
}

# Task-specific arguments
GRASP_ARGS=""
PLACE_ARGS=""
PP_ARGS=""

# Determine which tasks to run
if [[ -n "$SINGLE_TASK" ]]; then
    TASKS=("$SINGLE_TASK")
else
    TASKS=("grasp" "place" "pick_and_place")
fi

echo "========================================"
echo "A2 Data Collection"
echo "========================================"
echo "Episodes per task: $NUM_EPISODES"
echo "Tasks: ${TASKS[*]}"
echo "Object set: $OBJECT_SET ($([ "$OBJECT_SET" = "train" ] && echo 'seen' || echo 'unseen') objects)"
echo "Parallel: $PARALLEL"
echo "Output: $OUTPUT_BASE"
echo "========================================"

if $PARALLEL; then
    echo "Running tasks in parallel..."
    # Use more aggressive reset interval for parallel (3 processes sharing GPU)
    ENV_RESET_INTERVAL=5
    echo "Using env_reset_interval=$ENV_RESET_INTERVAL for parallel mode"
    echo ""

    # Check number of GPUs
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo "Detected $NUM_GPUS GPU(s)"

    # Start each task in background
    PIDS=()
    GPU_IDX=0

    for task in "${TASKS[@]}"; do
        case $task in
            grasp)
                extra_args="$GRASP_ARGS"
                ;;
            place)
                extra_args="$PLACE_ARGS"
                ;;
            pick_and_place)
                extra_args="$PP_ARGS"
                ;;
            *)
                echo "Unknown task: $task"
                continue
                ;;
        esac

        # Run in background with different CUDA device if multiple GPUs available
        (
            if [[ $NUM_GPUS -gt 1 ]]; then
                export CUDA_VISIBLE_DEVICES=$GPU_IDX
                echo "[$task] Using GPU $GPU_IDX"
            fi
            collect_task "$task" "$extra_args"
        ) &
        PIDS+=($!)
        echo "Started $task (PID: ${PIDS[-1]})"

        # Rotate GPU assignment
        GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))

        # Small delay to avoid race conditions
        sleep 5
    done

    echo ""
    echo "Waiting for all tasks to complete..."
    echo "PIDs: ${PIDS[*]}"

    # Wait for all background processes
    FAILED=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || ((FAILED++))
    done

    if [[ $FAILED -gt 0 ]]; then
        echo "WARNING: $FAILED task(s) failed!"
        exit 1
    fi
else
    echo "Running tasks sequentially..."

    for task in "${TASKS[@]}"; do
        case $task in
            grasp)
                extra_args="$GRASP_ARGS"
                ;;
            place)
                extra_args="$PLACE_ARGS"
                ;;
            pick_and_place)
                extra_args="$PP_ARGS"
                ;;
            *)
                echo "Unknown task: $task"
                continue
                ;;
        esac

        collect_task "$task" "$extra_args"
        echo ""
    done
fi

echo "========================================"
echo "Collection complete!"
echo "========================================"
echo "Output directory: $OUTPUT_BASE"
ls -la "$OUTPUT_BASE"
