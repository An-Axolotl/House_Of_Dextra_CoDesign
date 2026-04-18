import os, re
from typing import Dict, Tuple, List, Optional

def _digits_to_links(digits: str, L_MAX: int, G_MAX: int):
    # map rightmost -> END, then MIDDLE, then BASE (right-align)
    raw = [int(c) for c in digits if c.isdigit()]
    # Map 0->1, 1->2, 2->3, ..., 9->10 (shift by +1 for grammar stacks)
    raw = [x + 1 for x in raw]
    raw = [min(max(x, 1), G_MAX) for x in raw]  # clamp to [1, G_MAX]

    # right-align to L_MAX (pad on the LEFT with zeros for missing segments)
    if len(raw) < L_MAX:
        raw = [0]*(L_MAX - len(raw)) + raw
    else:
        raw = raw[-L_MAX:]

    # indices: 0=base, 1=middle, 2=end
    scales = raw

    # Servo presence based on number of digits (not values)
    num_digits = len([c for c in digits if c.isdigit()])
    
    # Determine which servos are present based on digit count
    servo_present = [False, False, False]  # [base, middle, end]
    
    if num_digits == 3:
        servo_present = [True, True, True]   # all servos present
    elif num_digits == 2:
        servo_present = [False, True, True]  # base omitted
    elif num_digits == 1:
        servo_present = [False, False, True] # base and middle omitted
    # if 0 digits, all remain False
    
    # Count present servos
    n_links = sum(servo_present)
    
    return n_links, scales, servo_present

# Update parse_morph_from_filename to handle the new return value
def parse_morph_from_filename(basename: str, F_MAX: int, L_MAX: int, G_MAX: int) -> Dict:
    out = {"F_MAX": F_MAX, "L_MAX": L_MAX, "G_SCALE_MAX": G_MAX}
    for f in range(1, F_MAX + 1):
        out[f"F{f}_ACTIVE"] = 0
        out[f"F{f}_NUM_LINKS"] = 0
        for l in range(1, L_MAX + 1):
            out[f"F{f}_LINK{l}_SCALE"] = 0
            out[f"F{f}_LINK{l}_HAS_SERVO"] = 0

    # Updated regex to handle optional digits (empty string allowed)
    for m in re.finditer(r"f([1-5])_([0-9]*)", basename.lower()):
        f_idx = int(m.group(1))
        digits = m.group(2)
        n_links, scales, servo_present = _digits_to_links(digits, L_MAX, G_MAX)

        out[f"F{f_idx}_ACTIVE"] = 1 if n_links > 0 else 0
        out[f"F{f_idx}_NUM_LINKS"] = n_links
        for l in range(1, L_MAX + 1):
            out[f"F{f_idx}_LINK{l}_SCALE"] = scales[l - 1]
            out[f"F{f_idx}_LINK{l}_HAS_SERVO"] = 1 if servo_present[l - 1] else 0
    return out


# If you want to go straight to the morphology vector:
def morph_vec_from_filename(
    basename: str,
    F_MAX: int,
    L_MAX: int,
    G_MAX: int
):
    cfg = parse_morph_from_filename(basename, F_MAX=F_MAX, L_MAX=L_MAX, G_MAX=G_MAX)
    return build_morph_vector(cfg, scale_mode="frac0")  # 0.0 absent; 1..G -> 1/G..1.0

def build_morph_vector(cfg, *, scale_mode="frac0"):
    """
    scale_mode='frac0' => raw 0 -> 0.0 (absent), raw in [1..G] -> raw/G in (0,1].
    """
    F_MAX  = int(cfg["F_MAX"])
    L_MAX  = int(cfg["L_MAX"])
    G_MAX  = int(cfg["G_SCALE_MAX"])

    vec = []

    # presence (1 if any distal link kept)
    presence = []
    per_finger_masks = []
    per_finger_scales = []

    for f in range(1, F_MAX + 1):
        scales = [int(cfg.get(f"F{f}_LINK{l}_SCALE", 0)) for l in range(1, L_MAX + 1)]  # [base, middle, end]
        mask = [1 if s > 0 else 0 for s in scales]
        per_finger_masks.append(mask)
        per_finger_scales.append(scales)
        presence.append(1 if any(mask) else 0)

    vec.extend(presence)

    # then for each finger: link_mask[L_MAX], scale_norm[L_MAX]
    for f in range(F_MAX):
        mask  = per_finger_masks[f]
        scales= per_finger_scales[f]

        # mask
        vec.extend(mask)

        # normalized scales
        for s in scales:
            vec.append(0.0 if s == 0 else (s / float(G_MAX)))

    return vec

def build_hand_morph_obs(cfg: Dict) -> List[float]:
    F_MAX  = int(cfg["F_MAX"]); L_MAX = int(cfg["L_MAX"]); G_MAX = int(cfg["G_SCALE_MAX"])
    vec: List[float] = []

    # [which fingers are active] (any servo present per finger)
    active = [float(cfg.get(f"F{f}_ACTIVE", 0)) for f in range(1, F_MAX + 1)]
    vec.extend(active)

    # Per-finger: [servo_present base..end] + [grammar_norm base..end]
    for f in range(1, F_MAX + 1):
        present = [float(cfg.get(f"F{f}_LINK{l}_HAS_SERVO", 0)) for l in range(1, L_MAX + 1)]
        vec.extend(present)
        scales  = [int(cfg.get(f"F{f}_LINK{l}_SCALE", 0)) for l in range(1, L_MAX + 1)]
        # Normalize: 0→0.0 (no servo), 1-10→0.1-1.0
        vec.extend([0.0 if s == 0 else (s / float(G_MAX)) for s in scales])
    return vec