#!/bin/bash
# Quick start script for Graph Heuristic Search with rotation optimization
# Updated for CoDesign-InHand directory structure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODESIGN_ENV_FILE="${CODESIGN_ENV_FILE:-$REPO_ROOT/.codesign_env.sh}"

# Configuration - Updated paths for your structure
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="ghs_rotation_optimization/run_${RUN_ID}"
ITERATIONS=40
CANDIDATES=50
EVAL_EPISODES=1
EVAL_ENVS=2048
PLACEMENT_MODE="both"
EVAL_GROUPS="all"   # "all" or comma-separated subset, e.g. "sym5" or "sym3,sym4,sym5"
SHUFFLE_GROUPS=1    # 1: reshuffle selected group order each cycle (multi-group only), 0: fixed order
BATCH_SIZE=32
EPSILON=0.4
DEBUG=0

# Paths for your actual file locations
BUILD_SCRIPT="$REPO_ROOT/Generation/converters/batch_build_rand_hands2.sh"
ISAAC_LAB_DIR="$REPO_ROOT/IsaacLab"
PLAY_SCRIPT="$ISAAC_LAB_DIR/scripts/rl_games/play.py"
CODESIGN_SOURCE_DIR="$ISAAC_LAB_DIR/source/codesign"
DEFAULT_VENV_PY="$HOME/.venvs/codesign/bin/python"
LEGACY_VENV_PY="$HOME/.venvs/catchingbot/bin/python"
DEFAULT_CONDA_PY="/opt/conda/envs/catchingbot/bin/python3"

cd "$SCRIPT_DIR"

if [[ -f "$CODESIGN_ENV_FILE" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$CODESIGN_ENV_FILE"
    set -u
fi

python_supports_runner() {
    local py="$1"
    "$py" - <<'PY' >/dev/null 2>&1
import importlib.util
required = ("torch", "numpy", "scipy", "trimesh", "coacd")
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
PY
}

resolve_runner_python() {
    local -a candidates=()
    local candidate

    [[ -n "${CODESIGN_RUNNER_PYTHON:-}" ]] && candidates+=("$CODESIGN_RUNNER_PYTHON")
    [[ -n "${CODESIGN_PYTHON:-}" ]] && candidates+=("$CODESIGN_PYTHON")
    [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]] && candidates+=("${VIRTUAL_ENV}/bin/python")
    [[ -n "${CODESIGN_VENV_DIR:-}" && -x "${CODESIGN_VENV_DIR}/bin/python" ]] && candidates+=("${CODESIGN_VENV_DIR}/bin/python")
    [[ -x "$DEFAULT_VENV_PY" ]] && candidates+=("$DEFAULT_VENV_PY")
    [[ -x "$LEGACY_VENV_PY" ]] && candidates+=("$LEGACY_VENV_PY")
    [[ -x "$DEFAULT_CONDA_PY" ]] && candidates+=("$DEFAULT_CONDA_PY")
    [[ -n "${CODESIGN_ISAAC_PYTHON:-}" ]] && candidates+=("$CODESIGN_ISAAC_PYTHON")
    [[ -x "$ISAAC_LAB_DIR/_isaac_sim/python.sh" ]] && candidates+=("$ISAAC_LAB_DIR/_isaac_sim/python.sh")
    [[ -x /workspace/isaaclab/_isaac_sim/python.sh ]] && candidates+=("/workspace/isaaclab/_isaac_sim/python.sh")
    command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")
    command -v python >/dev/null 2>&1 && candidates+=("$(command -v python)")

    declare -A seen=()
    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        [[ -n "${seen[$candidate]:-}" ]] && continue
        seen["$candidate"]=1
        if python_supports_runner "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

echo "=========================================="
echo "Graph Heuristic Search - Rotation Focus"
echo "=========================================="
echo "Running from: $(pwd)"
echo "Build script: $BUILD_SCRIPT"
echo "Output: $OUTPUT_DIR"
echo "Iterations: $ITERATIONS"
echo "Candidates per iteration: $CANDIDATES"
echo "Eval groups: $EVAL_GROUPS"
if [[ -f "$CODESIGN_ENV_FILE" ]]; then
    echo "Env file: $CODESIGN_ENV_FILE"
fi
echo "Expected runtime: ~45 minutes"
echo ""

# Verify directory structure
if [ ! -d "$ISAAC_LAB_DIR" ]; then
    echo "Error: IsaacLab directory not found at $ISAAC_LAB_DIR"
    echo "Current directory structure:"
    ls -la "$REPO_ROOT"
    exit 1
fi

if [ ! -d "$REPO_ROOT/Generation" ]; then
    echo "Error: Generation directory not found at $REPO_ROOT/Generation"
    echo "Current directory structure:"
    ls -la "$REPO_ROOT"
    exit 1
fi

# Verify build script exists
if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "Error: Build script not found at $BUILD_SCRIPT"
    echo "Available files in Generation/converters:"
    ls -la "$REPO_ROOT/Generation/converters/" || echo "Directory not accessible"
    exit 1
fi

# Verify play script exists
if [ ! -f "$PLAY_SCRIPT" ]; then
    echo "Error: Play script not found at $PLAY_SCRIPT"
    echo "Available files in IsaacLab/scripts:"
    ls -la "$ISAAC_LAB_DIR/scripts/" || echo "Directory not accessible"
    exit 1
fi

echo "✓ Build script: $BUILD_SCRIPT"
echo "✓ Play script: $PLAY_SCRIPT"
echo ""

RUNNER_PYTHON="$(resolve_runner_python || true)"
if [ -z "$RUNNER_PYTHON" ]; then
    echo "Error: Could not find a Python interpreter with required runner/build dependencies"
    echo "Expected modules: torch, numpy, scipy, trimesh, coacd"
    echo "Set CODESIGN_RUNNER_PYTHON explicitly after running setup.sh if needed."
    exit 1
fi

export CODESIGN_PYTHON="$RUNNER_PYTHON"
export CODESIGN_RUNNER_PYTHON="$RUNNER_PYTHON"
export PYTHONPATH="$CODESIGN_SOURCE_DIR:$ISAAC_LAB_DIR${PYTHONPATH:+:$PYTHONPATH}"

RUNNER_BIN="$(dirname "$(readlink -f "$RUNNER_PYTHON")")"
if [ -d "$RUNNER_BIN" ]; then
    export PATH="$RUNNER_BIN:$PATH"
fi

if [ -f "$RUNNER_BIN/activate" ]; then
    export VIRTUAL_ENV="$(cd "$RUNNER_BIN/.." && pwd)"
fi

if ! command -v xacrodoc >/dev/null 2>&1 && [ -x "/opt/conda/envs/catchingbot/bin/xacrodoc" ]; then
    export PATH="/opt/conda/envs/catchingbot/bin:$PATH"
fi

if command -v xacrodoc >/dev/null 2>&1; then
    export CODESIGN_XACRODOC="$(command -v xacrodoc)"
fi

# Test build script
echo "Testing build script..."
if ! bash "$BUILD_SCRIPT" --help &>/dev/null; then
    echo "Warning: Build script help not available, but proceeding..."
    chmod +x "$BUILD_SCRIPT"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting optimization..."

# source /opt/conda/etc/profile.d/conda.sh
# conda activate catchingbot

PYTHON_PATH="$RUNNER_PYTHON"

echo "Using Python: $PYTHON_PATH"
echo "Python version: $($PYTHON_PATH --version)"

# Test if all runner/build dependencies are available with this python
if ! "$PYTHON_PATH" - <<'PY' 2>/dev/null
import importlib
import sys

required = ("torch", "numpy", "scipy", "trimesh", "coacd")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing modules:", ", ".join(missing))
    raise SystemExit(1)

torch = importlib.import_module("torch")
numpy = importlib.import_module("numpy")
print(f"Torch available: {torch.__version__}")
print(f"NumPy available: {numpy.__version__}")
PY
then
    echo "Error: Runner Python is missing required build dependencies"
    echo "Expected modules: torch, numpy, scipy, trimesh, coacd"
    echo "Interpreter: $PYTHON_PATH"
    echo "Trying to diagnose the issue..."
    
    echo "Python executable exists: $(test -f "$PYTHON_PATH" && echo 'Yes' || echo 'No')"
    echo "Python executable permissions: $(ls -la "$PYTHON_PATH")"
    echo "Python import paths:"
    "$PYTHON_PATH" -c "import sys; print('\\n'.join(sys.path))" || echo "Failed to run python"
    
    exit 1
else
    echo "✓ Runner build dependencies are available"
fi

# Keep IsaacLab repo root and codesign source visible to subprocesses.
export PYTHONPATH="$CODESIGN_SOURCE_DIR:$ISAAC_LAB_DIR${PYTHONPATH:+:$PYTHONPATH}"

DEBUG_FLAG=""
if [ "${DEBUG:-0}" -ne 0 ]; then
    DEBUG_FLAG="--debug"
fi

GROUP_SHUFFLE_FLAG="--no-shuffle-groups"
if [ "${SHUFFLE_GROUPS:-1}" -ne 0 ]; then
    GROUP_SHUFFLE_FLAG="--shuffle-groups"
fi

export CODESIGN_FIXED_DR_LEVEL=5
export CODESIGN_DISABLE_ADR_ADAPTATION=1
export CODESIGN_GHS_EVALUATION=1  # If you also want fixed episode lengths

# Use the specific python path that works
$PYTHON_PATH ghs_runner.py \
    --iterations $ITERATIONS \
    --candidates $CANDIDATES \
    --video-every-n-cycles 1 \
    --video-length 800 \
    --wandb \
    --wandb-project codesign-ghs \
    --epsilon $EPSILON \
    --eval-episodes $EVAL_EPISODES \
    --eval-envs $EVAL_ENVS \
    --task "Codesign-Reorientation-Direct-v0" \
    --output-dir "$OUTPUT_DIR" \
    --play-script "$PLAY_SCRIPT" \
    --build-script "$BUILD_SCRIPT" \
    --focus-metric "rotation" \
    --device "auto" \
    --placement-mode "$PLACEMENT_MODE" \
    --eval-groups "$EVAL_GROUPS" \
    $GROUP_SHUFFLE_FLAG \
    --batch-size $BATCH_SIZE \
    $DEBUG_FLAG

echo ""
echo "=========================================="
echo "OPTIMIZATION COMPLETE!"
echo "=========================================="

# Check if results exist and display summary
if [ -f "$OUTPUT_DIR/final_results/best_design.json" ]; then
    echo "Best design details:"
    $PYTHON_PATH -c "
import json
import sys
sys.path.append('$ISAAC_LAB_DIR')
try:
    with open('$OUTPUT_DIR/final_results/best_design.json', 'r') as f:
        data = json.load(f)
    print(f'Design: {data[\"design_string\"]}')
    print(f'Rotation score: {data[\"reward\"]:.4f}')
    print(f'Grammar codes: {data[\"grammar_codes\"]}')
except Exception as e:
    print(f'Error reading results: {e}')
"
    echo ""
    echo "Files created:"
    echo "- Results: $OUTPUT_DIR/final_results/"
    if [ -f "$OUTPUT_DIR/final_results/best_design_hand.urdf" ]; then
        echo "- Best URDF: $OUTPUT_DIR/final_results/best_design_hand.urdf"
    fi
    if [ -f "$OUTPUT_DIR/final_results/evaluate_best_design.sh" ]; then
        echo "- Evaluation script: $OUTPUT_DIR/final_results/evaluate_best_design.sh"
    fi
    echo ""
    echo "To test the best design:"
    echo "cd $OUTPUT_DIR/final_results && ./evaluate_best_design.sh"
else
    echo "No results found. Check the log above for errors."
fi
