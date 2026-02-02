#!/bin/bash
# A2 Pick-and-Place Benchmark Suite
# Runs all benchmarks matching A2_new evaluation setup
#
# Usage:
#   ./scripts/run_a2_benchmarks.sh                    # Run with A2 policy (default)
#   ./scripts/run_a2_benchmarks.sh --policy vla       # Run with VLA policy
#   ./scripts/run_a2_benchmarks.sh --policy /path/to/checkpoint  # Custom checkpoint
#   ./scripts/run_a2_benchmarks.sh --quick            # Quick test (2 episodes per case)

set -e

# Default settings
POLICY="a2"
NUM_EPISODES=15
MAX_ATTEMPTS=8
DEVICE="cuda"
OUTPUT_DIR="benchmark_results"
TASKS="grasp place pickplace"
OBJECT_SETS="seen unseen"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --policy)
            POLICY="$2"
            shift 2
            ;;
        --episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --max-attempts)
            MAX_ATTEMPTS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --object-sets)
            OBJECT_SETS="$2"
            shift 2
            ;;
        --quick)
            NUM_EPISODES=2
            shift
            ;;
        --grasp-only)
            TASKS="grasp"
            shift
            ;;
        --seen-only)
            OBJECT_SETS="seen"
            shift
            ;;
        --help|-h)
            echo "A2 Pick-and-Place Benchmark Suite"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --policy POLICY       Policy to use: 'a2', 'vla', or path to checkpoint"
            echo "                        (default: a2)"
            echo "  --episodes N          Number of episodes per test case (default: 15)"
            echo "  --max-attempts N      Max grasp attempts per episode (default: 8)"
            echo "  --device DEVICE       Device to use: 'cuda' or 'cpu' (default: cuda)"
            echo "  --output-dir DIR      Directory for results (default: benchmark_results)"
            echo "  --tasks 'TASKS'       Tasks to run: 'grasp place pickplace' (default: all)"
            echo "  --object-sets 'SETS'  Object sets: 'seen unseen' (default: both)"
            echo "  --quick               Quick test mode (2 episodes per case)"
            echo "  --grasp-only          Only run grasp task"
            echo "  --seen-only           Only run seen object set"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Full benchmark with A2 policy"
            echo "  $0 --policy vla                       # Full benchmark with VLA policy"
            echo "  $0 --quick                            # Quick test (2 episodes)"
            echo "  $0 --policy /path/to/checkpoint       # Custom checkpoint"
            echo "  $0 --grasp-only --seen-only --quick   # Quick grasp test on seen objects"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
POLICY_NAME=$(basename "$POLICY")
RESULTS_DIR="${OUTPUT_DIR}/${POLICY_NAME}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "A2 PICK-AND-PLACE BENCHMARK SUITE"
echo "============================================================"
echo "Policy: $POLICY"
echo "Episodes per case: $NUM_EPISODES"
echo "Max attempts: $MAX_ATTEMPTS"
echo "Device: $DEVICE"
echo "Tasks: $TASKS"
echo "Object sets: $OBJECT_SETS"
echo "Results directory: $RESULTS_DIR"
echo "============================================================"
echo ""

# Build common arguments
COMMON_ARGS="--num_episodes $NUM_EPISODES --max_attempts $MAX_ATTEMPTS --device $DEVICE"

# Add policy argument based on type
if [[ "$POLICY" == "a2" ]]; then
    # Default A2 policy (direct grounding with GraspNet + CLIP)
    COMMON_ARGS="$COMMON_ARGS"
elif [[ "$POLICY" == "vla" ]]; then
    # VLA policy - requires pretrained checkpoint
    # TODO: Update path when VLA checkpoint is available
    echo "Note: VLA policy requires a pretrained checkpoint."
    echo "Please specify the checkpoint path with --policy /path/to/vla/checkpoint"
    COMMON_ARGS="$COMMON_ARGS --policy_path lerobot/vla"
elif [[ -d "$POLICY" ]] || [[ -f "$POLICY" ]]; then
    # Custom checkpoint path
    COMMON_ARGS="$COMMON_ARGS --policy_path $POLICY"
else
    echo "Warning: Policy '$POLICY' not recognized. Using as checkpoint path."
    COMMON_ARGS="$COMMON_ARGS --policy_path $POLICY"
fi

# Function to run a single benchmark
run_benchmark() {
    local task=$1
    local object_set=$2
    local log_file="${RESULTS_DIR}/${task}_${object_set}.log"
    local json_file="benchmark_results_${task}_${object_set}.json"

    echo ""
    echo "------------------------------------------------------------"
    echo "Running: $task ($object_set)"
    echo "Started: $(date)"
    echo "Log: $log_file"
    echo "------------------------------------------------------------"

    # Run benchmark and capture output
    if uv run python scripts/benchmark_a2_full.py \
        --task "$task" \
        --object_set "$object_set" \
        $COMMON_ARGS \
        2>&1 | tee "$log_file"; then
        echo "[PASS] $task ($object_set)"
    else
        echo "[FAIL] $task ($object_set)"
    fi

    # Move JSON results if generated
    if [[ -f "$json_file" ]]; then
        mv "$json_file" "$RESULTS_DIR/"
    fi

    echo "Completed: $(date)"
}

# Track start time
START_TIME=$(date +%s)

echo ""
echo "Starting benchmarks..."

# Run benchmarks for each task and object set combination
for task in $TASKS; do
    for object_set in $OBJECT_SETS; do
        run_benchmark "$task" "$object_set"
    done
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "BENCHMARK SUITE COMPLETE"
echo "============================================================"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Results saved to: $RESULTS_DIR"
echo ""

# Generate summary
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
{
    echo "A2 Benchmark Summary"
    echo "==================="
    echo "Policy: $POLICY"
    echo "Episodes per case: $NUM_EPISODES"
    echo "Max attempts: $MAX_ATTEMPTS"
    echo "Date: $(date)"
    echo ""
    echo "Results:"
    echo "--------"
} > "$SUMMARY_FILE"

# Extract results from JSON files
for json_file in "$RESULTS_DIR"/*.json; do
    if [[ -f "$json_file" ]]; then
        # Extract overall success rate using python
        uv run python -c "
import json
import sys
try:
    with open('$json_file') as f:
        data = json.load(f)
        task = data.get('task', 'unknown')
        obj_set = data.get('object_set', 'unknown')
        overall = data.get('overall', {})
        success = overall.get('num_successes', 0)
        total = overall.get('num_episodes', 0)
        rate = (success / total * 100) if total > 0 else 0
        print(f'  {task:12} ({obj_set:6}): {success:3}/{total:3} ({rate:5.1f}%)')
except Exception as e:
    print(f'  Error parsing $(basename "$json_file"): {e}', file=sys.stderr)
" >> "$SUMMARY_FILE" 2>/dev/null || echo "  (could not parse $(basename "$json_file"))" >> "$SUMMARY_FILE"
    fi
done

{
    echo ""
    echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
} >> "$SUMMARY_FILE"

echo ""
cat "$SUMMARY_FILE"
echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo "Full logs in: $RESULTS_DIR/"
