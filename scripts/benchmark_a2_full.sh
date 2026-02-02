#!/bin/bash
# Full A2 Policy Benchmark Suite
#
# Runs grasp, place, and pick-and-place benchmarks with the A2 policy
# on both seen (train) and unseen (test) object sets.
#
# Usage:
#   ./scripts/benchmark_a2_full.sh                    # Full benchmark (15 episodes)
#   ./scripts/benchmark_a2_full.sh --quick            # Quick test (3 episodes)
#   ./scripts/benchmark_a2_full.sh --episodes 10      # Custom episode count
#   ./scripts/benchmark_a2_full.sh --train-only       # Only train objects
#   ./scripts/benchmark_a2_full.sh --policy-path dgrachev/a2_pretrained  # Use pretrained

set -e

# Default configuration
NUM_EPISODES=15
DEVICE="cuda"
POLICY_PATH=""
OUTPUT_DIR="benchmark_results"
TRAIN_ONLY=false
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            NUM_EPISODES=3
            shift
            ;;
        --episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --policy-path)
            POLICY_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --help)
            echo "A2 Policy Full Benchmark Suite"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick              Quick test mode (3 episodes per task)"
            echo "  --episodes N         Number of episodes per task (default: 15)"
            echo "  --device DEVICE      Device to use: cuda or cpu (default: cuda)"
            echo "  --policy-path PATH   Path to pretrained model (HuggingFace repo or local)"
            echo "  --output-dir DIR     Output directory for results (default: benchmark_results)"
            echo "  --train-only         Only benchmark on train (seen) objects"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                      # Full benchmark"
            echo "  $0 --quick                              # Quick 3-episode test"
            echo "  $0 --policy-path dgrachev/a2_pretrained # Use pretrained weights"
            echo "  $0 --episodes 10 --train-only           # 10 episodes, train only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build policy path argument
POLICY_ARG=""
if [ -n "$POLICY_PATH" ]; then
    POLICY_ARG="--policy_path $POLICY_PATH"
fi

# Print configuration
echo "========================================"
echo "A2 POLICY FULL BENCHMARK"
echo "========================================"
echo "Date: $(date)"
echo "Episodes per task: $NUM_EPISODES"
echo "Device: $DEVICE"
echo "Policy path: ${POLICY_PATH:-'(untrained)'}"
echo "Output directory: $OUTPUT_DIR"
echo "Train only: $TRAIN_ONLY"
echo "========================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/benchmark_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

# Function to run benchmark and log results
run_benchmark() {
    local task=$1
    local object_set=$2
    local description="$3"

    echo ""
    echo "----------------------------------------"
    echo "Running: $description"
    echo "  Task: $task, Objects: $object_set, Episodes: $NUM_EPISODES"
    echo "----------------------------------------"

    local start_time=$(date +%s)

    if uv run python scripts/benchmark_a2.py \
        --policy a2 \
        $POLICY_ARG \
        --task "$task" \
        --object_set "$object_set" \
        --num_episodes "$NUM_EPISODES" \
        --device "$DEVICE" \
        --output_dir "$OUTPUT_DIR" \
        --save 2>&1 | tee -a "$LOG_FILE"; then
        local status="PASS"
    else
        local status="FAIL"
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    echo "[$status] $description (${duration}s)"
    echo ""

    # Return status
    [ "$status" = "PASS" ]
}

# Track results
declare -A RESULTS
TOTAL_START=$(date +%s)

# ============================================
# PART 1: Train (Seen) Objects
# ============================================
echo ""
echo "========================================"
echo "PART 1: TRAIN (SEEN) OBJECTS"
echo "========================================"

run_benchmark "grasp" "train" "Grasp - Train Objects"
RESULTS["grasp_train"]=$?

run_benchmark "place" "train" "Place - Train Objects"
RESULTS["place_train"]=$?

run_benchmark "pickplace" "train" "Pick-and-Place - Train Objects"
RESULTS["pickplace_train"]=$?

# ============================================
# PART 2: Test (Unseen) Objects
# ============================================
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "========================================"
    echo "PART 2: TEST (UNSEEN) OBJECTS"
    echo "========================================"

    run_benchmark "grasp" "test" "Grasp - Unseen Objects"
    RESULTS["grasp_test"]=$?

    run_benchmark "place" "test" "Place - Unseen Objects"
    RESULTS["place_test"]=$?

    run_benchmark "pickplace" "test" "Pick-and-Place - Unseen Objects"
    RESULTS["pickplace_test"]=$?
fi

# ============================================
# Summary
# ============================================
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "========================================"
echo "BENCHMARK SUMMARY"
echo "========================================"
echo ""

# Count results
PASSED=0
FAILED=0

echo "Train (Seen) Objects:"
for task in grasp place pickplace; do
    key="${task}_train"
    if [ "${RESULTS[$key]}" = "0" ]; then
        echo "  [PASS] $task"
        ((PASSED++))
    else
        echo "  [FAIL] $task"
        ((FAILED++))
    fi
done

if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "Test (Unseen) Objects:"
    for task in grasp place pickplace; do
        key="${task}_test"
        if [ "${RESULTS[$key]}" = "0" ]; then
            echo "  [PASS] $task"
            ((PASSED++))
        else
            echo "  [FAIL] $task"
            ((FAILED++))
        fi
    done
fi

echo ""
echo "----------------------------------------"
echo "Total: $PASSED passed, $FAILED failed"
echo "Total time: ${TOTAL_DURATION}s ($(( TOTAL_DURATION / 60 ))m $(( TOTAL_DURATION % 60 ))s)"
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "========================================"

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Some benchmarks failed!"
    exit 1
else
    echo ""
    echo "All benchmarks completed successfully!"
    exit 0
fi
