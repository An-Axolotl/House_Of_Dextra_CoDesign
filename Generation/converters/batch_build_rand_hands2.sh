#!/usr/bin/env bash
# batch_build_rand_hands2.sh
# Build many MuJoCo hand variants with randomized grammar stacks and servo counts.
#
# Modes:
#   1) uniform-all   : all five fingers share SAME 3-digit code (g1 g2 s); build all combos in ranges
#   2) thumb-random  : fingers 2..5 share the SAME code; thumb gets any code (all combos)
#   3) all-random    : each finger gets an independent random code (optionally uniqueness per-hand)
#   4) manual        : provide exact 5 codes with -C (repeat -C to add more hands)
#
# Output:
#   - Final URDFs in:  -o <dir>
#   - Intermediates in: <out>/_build/<hand_name>/...
#
# Requirements:
#   - CODESIGN_PYTHON or python3
#   - CODESIGN_XACRODOC or xacrodoc on PATH
#
# Assumes alongside this script:
#   converters/generate_rand_gram_joint_finger.py
#   converters/mjcf_to_xacro_gram_joints.py
#
# For runnable examples and preset generation commands, see:
#   Generation/converters/README.md

set -euo pipefail

# ---------- paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_FINGER_PY="${SCRIPT_DIR}/generate_rand_gram_joint_finger.py"
CONV_PY="${SCRIPT_DIR}/mjcf_to_xacro_gram_joints.py"
GEN_PALM_PY="${SCRIPT_DIR}/generate_palm_mesh.py"
WRITE_METADATA_PY="${SCRIPT_DIR}/write_hand_metadata.py"
WRITE_HAND_XML_PY="${SCRIPT_DIR}/write_complete_hand_xml.py"
WRITE_HAND_XACRO_PY="${SCRIPT_DIR}/write_complete_hand_xacro.py"
PYTHON_BIN="${CODESIGN_PYTHON:-python3}"
XACRODOC_BIN="${CODESIGN_XACRODOC:-xacrodoc}"

# ---------- defaults ----------
HAND_XML=""
OUT_DIR=""
MODE=""                          # uniform-all | thumb-random | all-random | manual
NUM=0                            # how many hands to generate in all-random
MING=1                           # grammar min (inclusive)
MAXG=10                           # grammar max (inclusive)
MINS=1                           # servos min (inclusive)  {1,2,3}
MAXS=3                           # servos max (inclusive)
DX="-0.00009"; DY="0.00254"; DZ="-0.0"  # grammar Δg
CX="0.0015"; CY="-0.03147"; CZ="0.00103"   # compensation when a servo is removed
THUMB_INDEX=1                    # which finger is "thumb" (1..5) for thumb-random
UNIQUE_FINGERS=0                 # enforce all five finger codes differ (all-random only)
SEED=""                          # optional seed for bash RNG

# generator/converter fine-tuning
GHOST_MODE="tiny"                # generator: tiny | zero
GHOST_EPS_GEN="1e-8"             # generator: epsilon for [0,eps]
GHOST_EPS_CONV="1e-6"            # converter: threshold to detect ghost
FLATTEN_STACK=0                  # converter: --flatten-stack-offsets
KEEP_G1_GHOST=0                  # converter: --keep-grammar1-ghost (rarely needed)

MANUAL_CODES=()                  # for --mode manual; each -C is "abc,def,ghi,jkl,mno"

# Palm parameters
PALM_FINGERS=5
PALM_RADIUS="0.07"
PALM_THICKNESS="0.0381"
PALM_SEED_MODE="random"          # "random" | "fixed" | "incremental"
PALM_BASE_SEED=42
# Add finger count range parameters
MIN_PALM_FINGERS=3
MAX_PALM_FINGERS=5
PLACEMENT_MODE="symmetric"     # asymmetric | symmetric | anthro-top-heavy
MIN_ANGLE_DEG="12.0"
SYMM_START_DEG="0.0"
SYMM_JITTER_DEG="0.0"
THUMB_BOTTOM_DEG_LO="210.0"
THUMB_BOTTOM_DEG_HI="330.0"
TOP_BAND_DEG_LO="300.0"
TOP_BAND_DEG_HI="60.0"
Y_MIRRORED="0"
# Discrete-slot controls (palm)
USE_DISCRETE_SLOTS=1     # 1=on, 0=off
SLOT_COUNT=36
MIN_SEP_SLOTS=4
JITTER_DEG=0.0           # keep 0.0 for reproducible meshes; add at PPO runtime if desired
MOUNT_SLOTS=""           # CSV of slot indices, e.g. "0,6,12,18,24"
# Anthro controls (fixed top + fixed/random thumb)
ANTHRO_TOP_FIXED_SLOTS=""   # CSV for non-thumb fingers in anthro mode, len = fingers-1
THUMB_FIXED_SLOT=""         # single integer slot for thumb
THUMB_FIXED_SERVOS=""            # empty => no override; else 1|2|3

# Add these new parameters after the existing palm parameters
SAVE_XML=0                           # whether to save intermediate XML files
XML_DIR=""                           # directory for XML files (if empty, use OUT_DIR/xml)

# Fingertip type parameters
FINGERTIP_TYPE="standard"            # standard | wedged | rounded
FINGERTIP_F1="standard"              # fingertip type for finger 1 (thumb)
FINGERTIP_F2="standard"              # fingertip type for finger 2
FINGERTIP_F3="standard"              # fingertip type for finger 3
FINGERTIP_F4="standard"              # fingertip type for finger 4
FINGERTIP_F5="standard"              # fingertip type for finger 5
FINGERTIP_INDIVIDUAL=0               # whether individual fingertip types are set
FINGERTIP_RANDOM=0                   # whether to randomize fingertips per hand
FINGERTIP_TYPES=("standard" "wedged" "rounded" "thinner") # available fingertip types

# ---------- helpers ----------
print_help() {
  cat <<EOF
Usage:
  $(basename "$0") -o <out_dir> --mode <uniform-all|thumb-random|all-random|manual> [options]

Palm options:
  --palm-fingers N        Number of fingers on palm (default: 5)
  --min-palm-fingers N    Minimum fingers per palm for random generation (default: 3)
  --max-palm-fingers N    Maximum fingers per palm for random generation (default: 5)
  --palm-radius R         Palm radius in meters (default: 0.12)
  --palm-thickness T      Palm thickness in meters (default: 0.0254)
  --palm-seed-mode MODE   Palm seed strategy: random|fixed|incremental (default: random)
  --palm-base-seed S      Base seed for palm generation (default: 42)
  --anthro-top-fixed-slots CSV  Fixed slots for non-thumb fingers in anthro mode (len=fingers-1)
  --thumb-fixed-slot N          Fix the thumb to slot N (anthro mode)


XML generation options:
  --save-xml              Save intermediate MJCF XML files
  --xml-dir DIR           Directory for XML files (default: <out_dir>/xml)

Fingertip type options:
  --fingertip-type TYPE      Set all fingertips to same type: standard|wedged|rounded|thinner (default: standard)
  --fingertip-random         Randomize fingertip types for each finger on each hand
  --fingertip-f1 TYPE        Set finger 1 (thumb) fingertip type: standard|wedged|rounded|thinner
  --fingertip-f2 TYPE        Set finger 2 fingertip type: standard|wedged|rounded|thinner
  --fingertip-f3 TYPE        Set finger 3 fingertip type: standard|wedged|rounded|thinner
  --fingertip-f4 TYPE        Set finger 4 fingertip type: standard|wedged|rounded|thinner
  --fingertip-f5 TYPE        Set finger 5 fingertip type: standard|wedged|rounded|thinner

Common options:
  --delta DX DY DZ            Grammar step Δg (default: -8e-5 3.82e-3 -3e-5)
  --comp  CX CY CZ            Compensation vector when upstream servo removed (default: 7.1e-4 -3.274e-2 1.41e-3)
  --min M                     Minimum grammar per site (default: 1)
  --max M                     Maximum grammar per site (default: 10)
  --min-servos K              Minimum present servos per finger in {1,2,3} (default: 1)
  --max-servos K              Maximum present servos per finger in {1,2,3} (default: 3)
  --seed S                    Seed bash RNG
  --thumb-index K             (thumb-random) which finger is the thumb (1..N), default 1
  --unique-fingers            (all-random) enforce all finger codes differ within a hand

Generator ghost options:
  --ghost-mode {tiny|zero}    tiny => ranges [0,eps] (default), zero => [0,0]
  --ghost-eps-gen E           eps for generator tiny mode (default: 1e-8)

Converter options:
  --ghost-eps-conv E          eps to detect ghost joints (default: 1e-6)
  --flatten-stack-offsets     zero out fixed offsets from stacking
  --keep-grammar1-ghost       insert a ghost slot link between base_lever and middle

Modes:
  --mode uniform-all
      Build ALL combinations over g1,g2 in [--min..--max] AND servos in [--min-servos..--max-servos],
      applying the SAME code to all fingers.

  --mode thumb-random
      For each base code (g1,g2,s), build all hands where the thumb (finger N) gets ANY code.

  --mode all-random -n N
      Sample N hands; each finger gets an independent random code (optionally enforce uniqueness).

  --mode manual
      -C "abc,def,ghi,jkl,mno"  (repeat -C)
      Each token is a 3-digit code: g1 g2 s with g1,g2 in [--min..--max], s in [--min-servos..--max-servos].

Examples:
  $(basename "$0") -o sets/uniform --mode uniform-all
  $(basename "$0") -o sets/thumb_rand --mode thumb-random --seed 42
  $(basename "$0") -o sets/all_rand --mode all-random -n 200
  $(basename "$0") -o sets/manual --mode manual -C 221,332,141,231,112
  $(basename "$0") -o sets/wedged --mode all-random -n 50 --fingertip-type wedged
  $(basename "$0") -o sets/mixed --mode all-random -n 50 --fingertip-f1 rounded --fingertip-f2 wedged
EOF
}

die() { echo "Error: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

rand_int() { # rand_int MIN MAX  -> inclusive
  local lo="$1" hi="$2"
  echo $(( lo + (RANDOM % (hi - lo + 1)) ))
}

rand_code() { # returns 3-digit "g1g2s" where g1,g2 are 0-9 (representing 1-10 stacks)
  local a b s
  a="$(rand_int $((MING-1)) $((MAXG-1)))"  # Convert 1-10 range to 0-9
  b="$(rand_int $((MING-1)) $((MAXG-1)))"  # Convert 1-10 range to 0-9  
  s="$(rand_int "${MINS}" "${MAXS}")"      # Servo count stays 1-3
  printf "%d%d%d" "$a" "$b" "$s"
}

# unique set of 5 codes
rand_codes_unique5() {
  local -a pool=()
  local -A used=()
  local c
  while ((${#pool[@]} < 5)); do
    c="$(rand_code)"
    [[ -n "${used[$c]:-}" ]] && continue
    used["$c"]=1
    pool+=("$c")
  done
  echo "${pool[*]}"
}

# Validate code string "abc,def,ghi,jkl,mno" based on actual finger structure rules
validate_five_codes() {
  local s="$1"
  
  # Split into array, allowing empty entries
  IFS=',' read -r C1 C2 C3 C4 C5 <<< "$s"
  
  # Process each code (including empty ones)
  for i in 1 2 3 4 5; do
    local c
    case $i in
      1) c="$C1" ;;
      2) c="$C2" ;;
      3) c="$C3" ;;
      4) c="$C4" ;;
      5) c="$C5" ;;
    esac
    
    # Check if it's a ghost finger (empty string)
    if [[ -z "$c" ]]; then
      continue  # Valid ghost finger - skip validation
    fi
    
    # Check if it's a ghost finger (empty string)
    if [[ -z "$c" ]]; then
      continue  # Valid ghost finger - skip validation
    fi

    # Determine servo count from number of digits
    local num_digits=${#c}
    case "$num_digits" in
      1)
        # 1 servo: only end grammar (always 0)
        [[ "$c" == "0" ]] || return 1
        ;;
      2) 
        # 2 servos: middle grammar + end grammar (always 0)
        [[ "$c" =~ ^[0-9]0$ ]] || return 1
        local g2_encoded="${c:0:1}"
        local g2=$((g2_encoded + 1))
        (( g2 >= MING && g2 <= MAXG )) || return 1
        ;;
      3)
        # 3 servos: base grammar + middle grammar + end grammar (always 0)  
        [[ "$c" =~ ^[0-9][0-9]0$ ]] || return 1
        local g1_encoded="${c:0:1}" g2_encoded="${c:1:1}"
        local g1=$((g1_encoded + 1)) g2=$((g2_encoded + 1))
        (( g1 >= MING && g1 <= MAXG )) || return 1
        (( g2 >= MING && g2 <= MAXG )) || return 1
        ;;
      *)
        return 1  # Invalid digit count
        ;;
    esac
  done
  
  # Return the parsed codes (preserving empty entries)
  echo "$C1,$C2,$C3,$C4,$C5"
}

# Map (g1,g2,s) -> filename token reflecting active servos; end grammar is always 0
# s=3: base+middle+end -> g1 g2 0
# s=2: middle+end      ->     g2 0
# s=1: end only        ->           0
fname_token_from_g1g2s() {
  local g1_encoded="$1" g2_encoded="$2" s="$3"
  # For the filename, show only the grammar stacks that will be present:
  # s=3: base(g1) + middle(g2) + end(0) = g1g20 (3 digits)
  # s=2: middle(g2) + end(0) = g20 (2 digits)
  # s=1: end(0) = 0 (1 digit)
  case "$s" in
    3) printf "%s%s0" "$g1_encoded" "$g2_encoded" ;;
    2) printf "%s0" "$g2_encoded" ;;
    1) printf "0" ;;
    *) printf "0" ;;  # fallback for invalid servo counts
  esac
}

trim() { awk '{$1=$1;print}'; }

# ---------- arg parse ----------
[[ $# -eq 0 ]] && { print_help; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -H) HAND_XML="$2"; shift 2 ;;
    -o) OUT_DIR="$2"; shift 2 ;;
    -n) NUM="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --palm-fingers) PALM_FINGERS="$2"; shift 2 ;;
    --min-palm-fingers) MIN_PALM_FINGERS="$2"; shift 2 ;;
    --max-palm-fingers) MAX_PALM_FINGERS="$2"; shift 2 ;;
    --palm-radius) PALM_RADIUS="$2"; shift 2 ;;
    --palm-thickness) PALM_THICKNESS="$2"; shift 2 ;;
    --palm-seed-mode) PALM_SEED_MODE="$2"; shift 2 ;;
    --palm-base-seed) PALM_BASE_SEED="$2"; shift 2 ;;
    --min) MING="$2"; shift 2 ;;
    --max) MAXG="$2"; shift 2 ;;
    --min-servos) MINS="$2"; shift 2 ;;
    --max-servos) MAXS="$2"; shift 2 ;;
    --delta) DX="$2"; DY="$3"; DZ="$4"; shift 4 ;;
    --comp)  CX="$2"; CY="$3"; CZ="$4"; shift 4 ;;
    --seed) SEED="$2"; shift 2 ;;
    --thumb-index) THUMB_INDEX="$2"; shift 2 ;;
    --unique-fingers) UNIQUE_FINGERS=1; shift ;;
    --ghost-mode) GHOST_MODE="$2"; shift 2 ;;
    --ghost-eps-gen) GHOST_EPS_GEN="$2"; shift 2 ;;
    --ghost-eps-conv) GHOST_EPS_CONV="$2"; shift 2 ;;
    --flatten-stack-offsets) FLATTEN_STACK=1; shift ;;
    --keep-grammar1-ghost)   KEEP_G1_GHOST=1; shift ;;
    -C) MANUAL_CODES+=("$2"); shift 2 ;;
    --save-xml) SAVE_XML=1; shift ;;
    --xml-dir) XML_DIR="$2"; shift 2 ;;
    --fingertip-type) FINGERTIP_TYPE="$2"; shift 2 ;;
    --fingertip-random) FINGERTIP_RANDOM=1; shift ;;
    --fingertip-f1) FINGERTIP_F1="$2"; FINGERTIP_INDIVIDUAL=1; shift 2 ;;
    --fingertip-f2) FINGERTIP_F2="$2"; FINGERTIP_INDIVIDUAL=1; shift 2 ;;
    --fingertip-f3) FINGERTIP_F3="$2"; FINGERTIP_INDIVIDUAL=1; shift 2 ;;
    --fingertip-f4) FINGERTIP_F4="$2"; FINGERTIP_INDIVIDUAL=1; shift 2 ;;
    --fingertip-f5) FINGERTIP_F5="$2"; FINGERTIP_INDIVIDUAL=1; shift 2 ;;
    --placement-mode) PLACEMENT_MODE="$2"; shift 2 ;;
    --min-angle-deg) MIN_ANGLE_DEG="$2"; shift 2 ;;
    --symmetric-start-deg) SYMM_START_DEG="$2"; shift 2 ;;
    --symmetric-jitter-deg) SYMM_JITTER_DEG="$2"; shift 2 ;;
    --thumb-bottom-deg) THUMB_BOTTOM_DEG_LO="$2"; THUMB_BOTTOM_DEG_HI="$3"; shift 3 ;;
    --top-band-deg) TOP_BAND_DEG_LO="$2"; TOP_BAND_DEG_HI="$3"; shift 3 ;;
    --y-mirrored) Y_MIRRORED=1; shift ;;
    --slot-count) SLOT_COUNT="$2"; shift 2 ;;
    --min-sep-slots) MIN_SEP_SLOTS="$2"; shift 2 ;;
    --no-discrete-slots) USE_DISCRETE_SLOTS=0; shift ;;
    --mount-slots) MOUNT_SLOTS="$2"; shift 2 ;;
    --jitter-deg) JITTER_DEG="$2"; shift 2 ;;
    --anthro-top-fixed-slots) ANTHRO_TOP_FIXED_SLOTS="$2"; shift 2 ;;
    --thumb-fixed-slot) THUMB_FIXED_SLOT="$2"; shift 2 ;;
    --thumb-fixed-servos) THUMB_FIXED_SERVOS="$2"; shift 2 ;;



    -h|--help) print_help; exit 0 ;;
    *) die "Unknown arg: $1 (see --help)";;
  esac
done

# Validation
[[ -z "$OUT_DIR" || -z "$MODE" ]] && { print_help; exit 1; }
# Remove hand.xml requirement since we're generating palms directly
# [[ -f "$HAND_XML" ]] || die "hand.xml not found: $HAND_XML"
[[ -f "$GEN_FINGER_PY" ]] || die "Finger generator not found: $GEN_FINGER_PY"
[[ -f "$CONV_PY" ]]  || die "Converter not found: $CONV_PY"
[[ -f "$GEN_PALM_PY" ]] || die "Palm generator not found: $GEN_PALM_PY"
[[ -f "$WRITE_METADATA_PY" ]] || die "Metadata writer not found: $WRITE_METADATA_PY"
[[ -f "$WRITE_HAND_XML_PY" ]] || die "Hand XML writer not found: $WRITE_HAND_XML_PY"
[[ -f "$WRITE_HAND_XACRO_PY" ]] || die "Hand Xacro writer not found: $WRITE_HAND_XACRO_PY"
have "$PYTHON_BIN" || die "Python interpreter not found: $PYTHON_BIN"
have "$XACRODOC_BIN" || die "xacrodoc not found: $XACRODOC_BIN"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
required = ("numpy", "scipy", "trimesh", "coacd")
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
PY
then
  die "Python interpreter missing required generation modules: $PYTHON_BIN (need numpy, scipy, trimesh, coacd)"
fi

(( MING <= MAXG )) || die "--min must be <= --max"
(( MINS <= MAXS )) || die "--min-servos must be <= --max-servos"
(( THUMB_INDEX >= 1 && THUMB_INDEX <= 5 )) || die "--thumb-index must be 1..5"
[[ "$GHOST_MODE" == "tiny" || "$GHOST_MODE" == "zero" ]] || die "--ghost-mode must be tiny|zero"
(( MIN_PALM_FINGERS >= 1 && MIN_PALM_FINGERS <= MAX_PALM_FINGERS )) || die "--min-palm-fingers must be >= 1 and <= --max-palm-fingers"
(( MAX_PALM_FINGERS <= 10 )) || die "--max-palm-fingers must be <= 10"

if [[ -n "$THUMB_FIXED_SERVOS" ]]; then
  (( THUMB_FIXED_SERVOS >= 1 && THUMB_FIXED_SERVOS <= 3 )) || die "--thumb-fixed-servos must be 1, 2, or 3"
fi

# Validate fingertip types
validate_fingertip_type() {
  local type="$1"
  case "$type" in
    standard|wedged|rounded|thinner) return 0 ;;
    *) return 1 ;;
  esac
}

validate_fingertip_type "$FINGERTIP_TYPE" || die "--fingertip-type must be standard|wedged|rounded|thinner"
validate_fingertip_type "$FINGERTIP_F1" || die "--fingertip-f1 must be standard|wedged|rounded|thinner"
validate_fingertip_type "$FINGERTIP_F2" || die "--fingertip-f2 must be standard|wedged|rounded|thinner"
validate_fingertip_type "$FINGERTIP_F3" || die "--fingertip-f3 must be standard|wedged|rounded|thinner"
validate_fingertip_type "$FINGERTIP_F4" || die "--fingertip-f4 must be standard|wedged|rounded|thinner"
validate_fingertip_type "$FINGERTIP_F5" || die "--fingertip-f5 must be standard|wedged|rounded|thinner"

# Convert filename token back to (g1,g2,servos) for finger generation
decode_finger_token() {
  local token="$1"
  
  if [[ -z "$token" ]]; then
    # Ghost finger
    echo "0 0 0"  # Will be handled as ghost
    return
  fi

  local num_digits=${#token}
  case "$num_digits" in
    1)
      # 1 servo: only end grammar
      echo "0 0 1"  # g1=0 (unused), g2=0 (unused), servos=1
      ;;
    2)
      # 2 servos: middle + end  
      local g2_encoded="${token:0:1}"
      echo "0 $g2_encoded 2"  # g1=0 (unused), g2=encoded, servos=2
      ;;
    3)
      # 3 servos: base + middle + end
      local g1_encoded="${token:0:1}" g2_encoded="${token:1:1}"
      echo "$g1_encoded $g2_encoded 3"
      ;;
    *)
      echo "0 0 0"  # Invalid - treat as ghost
      ;;
  esac
}

# Add helper function for random fingertip selection
rand_fingertip_type() {
  local types=("standard" "wedged")
  local idx=$(rand_int 0 1)
  echo "${types[idx]}"
}

# set_s_digit CODE NEW_S  -> returns CODE with last digit replaced by NEW_S
set_s_digit() {
  local code="$1" new_s="$2"
  printf "%s%d" "${code:0:2}" "$new_s"
}


# Function to get fingertip type for a specific finger
get_fingertip_type() {
  local finger_num="$1"
  
  # Priority: 1) Random (if enabled), 2) Individual settings, 3) Global setting
  if [[ $FINGERTIP_RANDOM -eq 1 ]]; then
    rand_fingertip_type
  elif [[ $FINGERTIP_INDIVIDUAL -eq 1 ]]; then
    case "$finger_num" in
      1) echo "$FINGERTIP_F1" ;;
      2) echo "$FINGERTIP_F2" ;;
      3) echo "$FINGERTIP_F3" ;;
      4) echo "$FINGERTIP_F4" ;;
      5) echo "$FINGERTIP_F5" ;;
      *) echo "$FINGERTIP_TYPE" ;;
    esac
  else
    echo "$FINGERTIP_TYPE"
  fi
}

mkdir -p "$OUT_DIR"
BUILD_DIR="${OUT_DIR}/_build"
METADATA_DIR="${OUT_DIR}/metadata"
mkdir -p "$BUILD_DIR"
mkdir -p "$METADATA_DIR"

# Set up XML directory if saving XML files
if [[ $SAVE_XML -eq 1 ]]; then
  [[ -z "$XML_DIR" ]] && XML_DIR="${OUT_DIR}/xml"
  mkdir -p "$XML_DIR"
  echo "XML files will be saved to: $XML_DIR"
fi

# seed bash RNG if requested
if [[ -n "$SEED" ]]; then
  # shellcheck disable=SC2034
  RANDOM=$(( SEED & 0x7fff ))
fi

# ---------- palm mesh generator ----------
# Input: hand name (for seed), output palm OBJ file
get_palm_seed() {
  local hand_name="$1"
  case "$PALM_SEED_MODE" in
    "random") echo $((RANDOM + $(date +%s))) ;;
    "fixed") echo "$PALM_BASE_SEED" ;;
    "incremental") 
      local hash=$(echo "$hand_name" | cksum | cut -d' ' -f1)
      echo $((PALM_BASE_SEED + hash % 10000)) ;;
    *) die "Invalid palm seed mode: $PALM_SEED_MODE" ;;
  esac
}

get_palm_finger_count() {
  local hand_name="$1"
  # Always randomize finger count within the specified range for variable generation
  echo $(rand_int "${MIN_PALM_FINGERS}" "${MAX_PALM_FINGERS}")
}

# ---------- finger mesh generator ----------
# Input: hand name (for seed), output finger XML file, g1, g2, servos

# ---------- metadata generator ----------
# Generate JSON metadata for a hand configuration
generate_hand_metadata() {
  local hand_name="$1"
  local actual_finger_count="$2"
  local -a final_codes=("${@:3:5}")  # codes array: arguments 3-7
  local -a fingertip_types=("${@:8}")  # fingertip types array: arguments 8-12
  
  local metadata_file="${METADATA_DIR}/${hand_name}.meta.json"
  local frames_json="${BUILD_DIR}/${hand_name}/${hand_name}_palm_frames.json"
  
  # Determine palm joint limits based on finger count
  local palm_limit_lo palm_limit_hi
  case "$actual_finger_count" in
    3) palm_limit_lo="-1.047197"; palm_limit_hi="1.047197" ;;  # -60 to 60 degrees
    4) palm_limit_lo="-0.872664"; palm_limit_hi="0.872664" ;;  # -50 to 50 degrees
    5) palm_limit_lo="-0.785398"; palm_limit_hi="0.785398" ;;  # -45 to 45 degrees
    *) palm_limit_lo="-0.785398"; palm_limit_hi="0.785398" ;;  # default to 5-finger case
  esac
  
  "${PYTHON_BIN}" "${WRITE_METADATA_PY}" \
    --metadata-file "$metadata_file" \
    --hand-name "$hand_name" \
    --actual-finger-count "$actual_finger_count" \
    --palm-limit-lo "$palm_limit_lo" \
    --palm-limit-hi "$palm_limit_hi" \
    --frames-json "$frames_json" \
    --codes "${final_codes[@]}" \
    --fingertip-types "${fingertip_types[@]}"
}

# ---------- per-hand builder ----------
# Input: five 3-digit codes like "221 223 332 141 311"
build_one_hand() {
  local -a codes=("$@")
  
  # Always expect exactly 5 codes for consistent structure
  (( ${#codes[@]} == 5 )) || die "Expected exactly 5 codes, got ${#codes[@]}"
  
  # For manual mode, determine finger count from codes (count non-"000" codes)
    # For other modes, use random generation
    local actual_finger_count
    if [[ "$MODE" == "manual" ]]; then
    # Count non-ghost fingers from the codes
    actual_finger_count=0
    for code in "${codes[@]}"; do
        if [[ "$code" != "000" ]]; then
            actual_finger_count=$((actual_finger_count + 1))
        fi
    done
    # Ensure we have at least 1 finger
    ((actual_finger_count >= 1)) || actual_finger_count=1
    else
    actual_finger_count="$(get_palm_finger_count "dummy")"
    fi

  if (( USE_DISCRETE_SLOTS )); then
    if (( actual_finger_count * MIN_SEP_SLOTS > SLOT_COUNT )); then
      die "Infeasible slots: fingers=${actual_finger_count}, min_sep=${MIN_SEP_SLOTS}, slots=${SLOT_COUNT}"
    fi
  fi
  
  # Generate fingertip types for this hand (BEFORE processing fingers)
  local -a hand_fingertip_types=()
  for i in $(seq 1 5); do
    hand_fingertip_types+=("$(get_fingertip_type "$i")")
  done
  
  # Prepare codes: active fingers get real codes, missing fingers get ghost code "000"
  local -a final_codes=()
    if [[ "$MODE" == "manual" ]]; then
    # Manual mode: use codes exactly as specified
    final_codes=("${codes[@]}")
    else
    # Other modes: apply finger count limit (active fingers get real codes, missing fingers get ghost code "000")
    for ((i=0; i<5; i++)); do
        if ((i < actual_finger_count)); then
        final_codes+=("${codes[i]}")
        else
        final_codes+=("000")  # Ghost finger code
        fi
    done
    fi

  if [[ "$MODE" == "manual" ]]; then
    # compute actual_finger_count first
    for i in $(seq 1 $actual_finger_count); do
      [[ "${final_codes[$((i-1))]}" != "000" ]] || die "Manual mode requires first $actual_finger_count fingers be non-ghost."
    done
    for i in $(seq $((actual_finger_count+1)) 5); do
      [[ "${final_codes[$((i-1))]}" == "000" ]] || die "Manual mode forbids holes; ghosts must be trailing."
    done
  fi
  
  # Generate display tokens for ALL 5 finger positions
  local -a tokens=()
  for i in 1 2 3 4 5; do
    local code="${final_codes[$((i-1))]}"
    if [[ "$code" == "000" ]]; then
      tokens+=("")  # Empty token for ghost fingers
    else
      local g1_encoded="${code:0:1}" g2_encoded="${code:1:1}" s="${code:2:1}"
      # Pass encoded values (0-9) directly to filename function
      tokens+=("$(fname_token_from_g1g2s "$g1_encoded" "$g2_encoded" "$s")")
    fi
  done
  
  # Build hand name showing all 5 finger positions
  local hand_name="hand"
  for i in 1 2 3 4 5; do
    local token="${tokens[$((i-1))]}"
    if [[ -n "$token" ]]; then
      hand_name="${hand_name}_f${i}_${token}"
    else
      hand_name="${hand_name}_f${i}"  # Just f1, f2, etc. for ghost fingers
    fi
  done
  
  local vroot="${BUILD_DIR}/${hand_name}"
  local vurdf="${OUT_DIR}/${hand_name}.urdf"
  
  # Create single meshes directory for all .obj files
  local meshes_dir="${OUT_DIR}/meshes"
  mkdir -p "${meshes_dir}"

  # Copy shared robot mesh assets used by generated URDFs.
  # Prefer repo-relative paths, keep pod path as fallback.
  local -a robot_meshes_src_candidates=(
    "${SCRIPT_DIR}/../meshes/lego_hand/robot_meshes"
    "/workspace/CoDesign-InHand/Generation/meshes/lego_hand/robot_meshes"
  )
  local robot_meshes_src=""
  for candidate in "${robot_meshes_src_candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      robot_meshes_src="$candidate"
      break
    fi
  done
  local robot_meshes_dest="${OUT_DIR}/robot_meshes"
  if [[ ! -d "$robot_meshes_dest" ]]; then
    [[ -n "$robot_meshes_src" ]] || die "Could not locate robot_meshes source. Tried: ${robot_meshes_src_candidates[*]}"
    echo "Copying robot_meshes to output directory..."
    cp -r "$robot_meshes_src" "$robot_meshes_dest"
  fi

    # Also copy to XML directory if saving XML files.
    # Reuse already-copied output meshes when available.
    if [[ $SAVE_XML -eq 1 ]]; then
    local xml_robot_meshes_dest="${XML_DIR}/robot_meshes"
    if [[ ! -d "$xml_robot_meshes_dest" ]]; then
        local xml_robot_meshes_src="$robot_meshes_dest"
        [[ -d "$xml_robot_meshes_src" ]] || xml_robot_meshes_src="$robot_meshes_src"
        [[ -d "$xml_robot_meshes_src" ]] || die "Could not locate robot_meshes for XML directory copy."
        echo "Copying robot_meshes to XML directory..."
        cp -r "$xml_robot_meshes_src" "$xml_robot_meshes_dest"
    fi
    fi
  
  echo "=== Building ${hand_name} (${actual_finger_count} active fingers)"
  mkdir -p "${vroot}"
  
  # 1) Generate palm mesh with ONLY the actual number of attachment points
  local palm_seed
  palm_seed="$(get_palm_seed "$hand_name")"
  local palm_obj="${meshes_dir}/${hand_name}_palm.obj"
  echo "Generating palm (seed=$palm_seed, fingers=$actual_finger_count, radius=$PALM_RADIUS)"
  
  palm_args=(
    --out "${palm_obj}"
    --fingers "$actual_finger_count"
    --palm-radius "$PALM_RADIUS"
    --thickness "$PALM_THICKNESS"
    --seed "$palm_seed"
    --placement-mode "$PLACEMENT_MODE"
    --min-angle-deg "$MIN_ANGLE_DEG"
    --symmetric-start-deg "$SYMM_START_DEG"
    --symmetric-jitter-deg "$SYMM_JITTER_DEG"
    --thumb-bottom-deg "$THUMB_BOTTOM_DEG_LO" "$THUMB_BOTTOM_DEG_HI"
    --top-band-deg "$TOP_BAND_DEG_LO" "$TOP_BAND_DEG_HI"
  )
  (( Y_MIRRORED )) && palm_args+=( --y-mirrored )
  [[ -n "$ANTHRO_TOP_FIXED_SLOTS" ]] && palm_args+=( --anthro-top-fixed-slots "$ANTHRO_TOP_FIXED_SLOTS" )
  [[ -n "$THUMB_FIXED_SLOT" ]]       && palm_args+=( --thumb-fixed-slot "$THUMB_FIXED_SLOT" )

  if [[ "$PLACEMENT_MODE" != "anthro-top-heavy" ]]; then
    [[ -n "$ANTHRO_TOP_FIXED_SLOTS" ]] && echo "Note: --anthro-top-fixed-slots is ignored unless anthro-top-heavy."
    [[ -n "$THUMB_FIXED_SLOT" ]] && echo "Note: --thumb-fixed-slot is ignored unless anthro-top-heavy."
  fi

  if (( USE_DISCRETE_SLOTS )); then
    palm_args+=( --slot-count "$SLOT_COUNT" --min-sep-slots "$MIN_SEP_SLOTS" --jitter-deg "$JITTER_DEG" )
    [[ -n "$MOUNT_SLOTS" ]] && palm_args+=( --mount-slots "$MOUNT_SLOTS" )
  else
    palm_args+=( --no-discrete-slots )
  fi

  if ! "${PYTHON_BIN}" "${GEN_PALM_PY}" "${palm_args[@]}"; then
    echo "Error: Failed to generate palm for ${hand_name}" >&2
    return 1
  fi
  
  # Fix the expected file paths
  local palm_base="${palm_obj%%.obj}"
  [[ -f "${palm_base}_frames.json" ]] || die "Palm frame data not generated: ${palm_base}_frames.json"
  [[ -f "${palm_base}.urdf.xacro" ]] || die "Palm URDF/Xacro not generated: ${palm_base}.urdf.xacro"
  
  # Copy frames file to build directory for metadata generation
  cp "${palm_base}_frames.json" "${vroot}/${hand_name}_palm_frames.json"
  
  # 2) Generate finger MJCF files - for ALL 5 fingers (including ghosts)
  for i in $(seq 1 5); do
    local code="${final_codes[$((i-1))]}"
    local fdir="${vroot}/f${i}"
    mkdir -p "${fdir}"
    
    # Get fingertip type for this finger from the pre-generated array
    local fingertip_type="${hand_fingertip_types[$((i-1))]}"
    
    # Check if this should be a ghost finger (either explicitly "000" or beyond actual finger count)
    if [[ "$code" == "000" ]] || ((i > actual_finger_count)); then
      echo "Generating ghost finger ${i}"
      # Generate ghost finger
      if ! "${PYTHON_BIN}" "${GEN_FINGER_PY}" \
        --out "${fdir}/random_finger.xml" \
        --seed "$SEED" \
        --ghost-finger \
        --ghost-mode "$GHOST_MODE" \
        --ghost-eps "$GHOST_EPS_GEN" \
        --tip "$fingertip_type"; then
        echo "Error: Failed to generate ghost finger ${i} for ${hand_name}" >&2
        return 1
      fi
    else
      # Parse the 3-digit code and convert 0-9 back to 1-10 for grammar counts
      local g1_encoded="${code:0:1}" g2_encoded="${code:1:1}" s="${code:2:1}"
      local g1=$((g1_encoded + 1))  # Convert 0-9 back to 1-10
      local g2=$((g2_encoded + 1))  # Convert 0-9 back to 1-10
      
      # Validate servo count - if it's 0, make it a ghost finger instead
      if ((s == 0)); then
        echo "Converting finger ${i} with 0 servos to ghost finger"
        if ! "${PYTHON_BIN}" "${GEN_FINGER_PY}" \
          --out "${fdir}/random_finger.xml" \
          --ghost-finger \
          --ghost-mode "$GHOST_MODE" \
          --ghost-eps "$GHOST_EPS_GEN" \
          --tip "$fingertip_type"; then
          echo "Error: Failed to generate ghost finger ${i} for ${hand_name}" >&2
          return 1
        fi
      else
        echo "Generating finger ${i}: g1=$g1, g2=$g2, servos=$s, fingertip=$fingertip_type"
        if ! "${PYTHON_BIN}" "${GEN_FINGER_PY}" \
          --out "${fdir}/random_finger.xml" \
          --g1 "$g1" --g2 "$g2" --servos "$s" \
          --delta "$DX" "$DY" "$DZ" \
          --comp "$CX" "$CY" "$CZ" \
          --ghost-mode "$GHOST_MODE" \
          --ghost-eps "$GHOST_EPS_GEN" \
          --tip "$fingertip_type"; then
          echo "Error: Failed to generate finger ${i} for ${hand_name}" >&2
          return 1
        fi
      fi
    fi
    
    # Copy XML file if requested
    if [[ $SAVE_XML -eq 1 ]]; then
      cp "${fdir}/random_finger.xml" "${XML_DIR}/${hand_name}_f${i}.xml"
    fi
  done

  # 2.5) Generate complete hand XML file with palm and attachment frames
  if [[ $SAVE_XML -eq 1 ]]; then
    local hand_xml="${XML_DIR}/${hand_name}_complete.xml"
    echo "Generating complete hand XML: ${hand_xml}"
    
    if ! "${PYTHON_BIN}" "${WRITE_HAND_XML_PY}" \
      --frames-json "${palm_base}_frames.json" \
      --out "$hand_xml" \
      --hand-name "$hand_name" \
      --actual-finger-count "$actual_finger_count"
    then
      echo "Error: Failed to generate complete hand XML for ${hand_name}" >&2
      return 1
    fi
  fi

  # 3) Convert finger MJCF to xacro - for ALL 5 fingers
  for i in $(seq 1 5); do
    local code="${final_codes[$((i-1))]}"
    local token="${tokens[$((i-1))]}"
    local fdir="${vroot}/f${i}"
    local xacro="${fdir}/random_finger.urdf.xacro"
    
    local conv_flags=()
    [[ $FLATTEN_STACK -eq 1 ]] && conv_flags+=("--flatten-stack-offsets")
    [[ $KEEP_G1_GHOST -eq 1 ]]  && conv_flags+=("--keep-grammar1-ghost")
    conv_flags+=("--ghost-range-eps" "${GHOST_EPS_CONV}")
    
    if ! "${PYTHON_BIN}" "${CONV_PY}" "${fdir}/random_finger.xml" -o "${xacro}" "${conv_flags[@]}"; then
      echo "Error: Failed to convert finger ${i} MJCF to xacro for ${hand_name}" >&2
      return 1
    fi
    
    # Create appropriate macro name based on whether it's a ghost finger
    if [[ -n "$token" ]]; then
      local newmacro="random_finger_f${i}_${token}"
    else
      local newmacro="random_finger_f${i}"  # No suffix for ghost fingers
    fi
    sed -i.bak -E "s/(<xacro:macro name=\")random_finger(\")/\1${newmacro}\2/" "${xacro}"
    rm -f "${xacro}.bak"
  done

  # 4) Build complete hand xacro from palm + fingers + frame data
  local hand_xacro="${vroot}/${hand_name}.urdf.xacro"
  if ! "${PYTHON_BIN}" "${WRITE_HAND_XACRO_PY}" \
    --palm-xacro "${palm_base}.urdf.xacro" \
    --frames-json "${palm_base}_frames.json" \
    --out "$hand_xacro" \
    --vroot "$vroot" \
    --hand-name "$hand_name" \
    --actual-finger-count "$actual_finger_count"
  then
    echo "Error: Failed to create complete hand xacro for ${hand_name}" >&2
    return 1
  fi
  
  # 5) Convert xacro to URDF
  if ! (cd "${vroot}" && "${XACRODOC_BIN}" "${hand_name}.urdf.xacro") > "${vurdf}"; then
    echo "Error: xacrodoc failed for ${hand_name}" >&2
    return 1
  fi

  # xacrodoc resolves mesh paths to absolute file:// URIs rooted under the per-hand
  # build directory. Isaac's URDF importer expects these assets to stay relative to
  # the final URDF location, where we already keep shared meshes/ and robot_meshes/.
  sed -E -i \
    -e 's#filename="file://[^"]*/meshes/#filename="meshes/#g' \
    -e 's#filename="file://[^"]*/robot_meshes/#filename="robot_meshes/#g' \
    "${vurdf}"
  
  # 6) Generate metadata JSON
  local -a fingertip_types=()
  for i in $(seq 1 5); do
    fingertip_types+=("$(get_fingertip_type "$i")")
  done
  generate_hand_metadata "$hand_name" "$actual_finger_count" "${final_codes[@]}" "${hand_fingertip_types[@]}"

  echo "→ ${vurdf}"
  echo "→ Palm mesh: ${palm_obj}"
  echo "→ Active fingers: ${actual_finger_count}/5"
  echo "→ Metadata: ${METADATA_DIR}/${hand_name}.meta.json"
}

# ---------- drive per mode ----------
case "$MODE" in
  uniform-all)
    echo "[Mode] uniform-all: all fingers use same code, finger count varies ${MIN_PALM_FINGERS}-${MAX_PALM_FINGERS}"
    for a in $(seq "$MING" "$MAXG"); do
      for b in $(seq "$MING" "$MAXG"); do
        for s in $(seq "$MINS" "$MAXS"); do
          code="${a}${b}${s}"
          # Always pass 5 codes
          build_one_hand "$code" "$code" "$code" "$code" "$code"
        done
      done
    done
    ;;
    
  thumb-random)
    echo "[Mode] thumb-random: base=fingers except f${THUMB_INDEX}; thumb can be any code"
    count=0
    for a in $(seq "$MING" "$MAXG"); do
      for b in $(seq "$MING" "$MAXG"); do
        for s in $(seq "$MINS" "$MAXS"); do
          base_code="${a}${b}${s}"
          for ta in $(seq "$MING" "$MAXG"); do
            for tb in $(seq "$MING" "$MAXG"); do
              for ts in $(seq "$MINS" "$MAXS"); do
                thumb_code="${ta}${tb}${ts}"
                c1="$base_code"; c2="$base_code"; c3="$base_code"; c4="$base_code"; c5="$base_code"
                eval "c${THUMB_INDEX}='${thumb_code}'"
                build_one_hand "$c1" "$c2" "$c3" "$c4" "$c5"
                count=$((count + 1))
              done
            done
          done
        done
      done
    done
    echo "Generated $count hand combinations"
    ;;
    
  all-random)
    (( NUM > 0 )) || die "-n is required for all-random"
    echo "[Mode] all-random: N=${NUM}, grammars=[${MING}..${MAXG}], servos=[${MINS}..${MAXS}], unique_fingers=${UNIQUE_FINGERS}"

    generated_file=$(mktemp)
    trap "rm -f '$generated_file'" EXIT
    generated_count=0
    attempts=0
    max_attempts=$((NUM * 20))

    while (( generated_count < NUM && attempts < max_attempts )); do
      if (( UNIQUE_FINGERS )); then
        read -r c1 c2 c3 c4 c5 <<<"$(rand_codes_unique5)"
      else
        c1="$(rand_code)"; c2="$(rand_code)"; c3="$(rand_code)"; c4="$(rand_code)"; c5="$(rand_code)"
      fi

      # If requested, force thumb servos
      if [[ -n "$THUMB_FIXED_SERVOS" ]]; then
        case "$THUMB_INDEX" in
          1) c1="$(set_s_digit "$c1" "$THUMB_FIXED_SERVOS")" ;;
          2) c2="$(set_s_digit "$c2" "$THUMB_FIXED_SERVOS")" ;;
          3) c3="$(set_s_digit "$c3" "$THUMB_FIXED_SERVOS")" ;;
          4) c4="$(set_s_digit "$c4" "$THUMB_FIXED_SERVOS")" ;;
          5) c5="$(set_s_digit "$c5" "$THUMB_FIXED_SERVOS")" ;;
        esac
      fi


      hand_id="${c1},${c2},${c3},${c4},${c5}"
      if grep -Fxq "$hand_id" "$generated_file" 2>/dev/null; then
        attempts=$((attempts + 1)); continue
      fi
      echo "$hand_id" >> "$generated_file"
      build_one_hand "$c1" "$c2" "$c3" "$c4" "$c5"
      generated_count=$((generated_count + 1))
      attempts=$((attempts + 1))
      if (( generated_count % 50 == 0 )); then
        echo "Generated $generated_count/$NUM hands (attempts: $attempts)"
      fi
    done

    if (( generated_count < NUM )); then
      echo "Warning: Only generated $generated_count unique hands (requested $NUM)"
    else
      echo "Successfully generated $generated_count hands"
    fi
    ;;
    
  manual)
  ((${#MANUAL_CODES[@]} > 0 )) || die "Provide at least one -C 'abc,def,ghi,jkl,mno'"
  manual_count=${#MANUAL_CODES[@]}
  echo "[Mode] manual: $manual_count hands"
  
  for s in "${MANUAL_CODES[@]}"; do
    ok="$(validate_five_codes "$s")" || die "Bad -C '$s' (format: 1-digit=1servo, 2-digit=2servos, 3-digit=3servos, empty=ghost)"
    IFS=',' read -r c1 c2 c3 c4 c5 <<< "$ok"
    
    # Convert tokens to codes for build_one_hand (which expects 3-digit codes)
    codes=()
    for token in "$c1" "$c2" "$c3" "$c4" "$c5"; do
      if [[ -z "$token" ]]; then
        codes+=("000")  # Ghost finger
      else
        read -r g1_enc g2_enc servos <<< "$(decode_finger_token "$token")"
        codes+=("${g1_enc}${g2_enc}${servos}")
      fi
    done
    
    build_one_hand "${codes[@]}"
  done
  ;;
  *)
    die "--mode must be one of: uniform-all | thumb-random | all-random | manual"
    ;;
esac

echo "All done. URDFs are in: ${OUT_DIR}"
echo "Metadata files are in: ${METADATA_DIR}"
if [[ $SAVE_XML -eq 1 ]]; then
  echo "XML files are in: ${XML_DIR}"
fi
echo "Palm parameters: fingers=${MIN_PALM_FINGERS}-${MAX_PALM_FINGERS}, radius=$PALM_RADIUS, thickness=$PALM_THICKNESS"
echo "Finger parameters: grammar=[$MING..$MAXG], servos=[$MINS..$MAXS]"
if [[ $FINGERTIP_RANDOM -eq 1 ]]; then
  echo "Fingertip types: randomized per hand"
elif [[ $FINGERTIP_INDIVIDUAL -eq 1 ]]; then
  echo "Fingertip types: f1=$FINGERTIP_F1, f2=$FINGERTIP_F2, f3=$FINGERTIP_F3, f4=$FINGERTIP_F4, f5=$FINGERTIP_F5"
else
  echo "Fingertip type: $FINGERTIP_TYPE (all fingers)"
fi
