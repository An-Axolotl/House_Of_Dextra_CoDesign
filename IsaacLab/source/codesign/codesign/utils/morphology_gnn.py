# gnn_morphology.py
from __future__ import annotations
import json, math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

# ---------- utils ----------
def _safe_unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    n = torch.linalg.norm(v)
    return (v / n) if n > 1e-8 else torch.zeros_like(v)

def yaw_sin_cos_from_quat_wxyz(q):  # q: [w,x,y,z]
    # Project to palm plane (assume palm Z is "up"); we'll use standard yaw from quaternion
    w, x, y, z = q
    # Yaw from wxyz (intrinsic z rotation)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    s = 2.0 * (w*z + x*y)
    c = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(s, c)
    return math.sin(yaw), math.cos(yaw)

def norm_xy(pos_xy, denom=0.05):  # denom ~ palm radius (m) → tune to your scale
    x, y = pos_xy
    return float(x/denom), float(y/denom)

ROLE_TO_ONEHOT = {"palm":[1,0,0,0], "base":[0,1,0,0], "middle":[0,0,1,0], "end":[0,0,0,1]}

# Add fingertip type mapping
FINGERTIP_TO_ONEHOT = {
    "standard": [1,0,0],
    "wedged": [0,1,0], 
    "rounded": [0,0,1]
}

# Add this (replace norm_limits with center/span)
ROLE_MAX_SPAN = {"palm": 1.0472, "base": 1.5708, "middle": 1.5708, "end": 1.5708}  # ~±60°, 0..90°, etc.

def limits_center_span(lo, hi, role):
    lo = float(lo); hi = float(hi)
    span0 = max(ROLE_MAX_SPAN.get(role, 1.0), 1e-6)
    center = 0.5 * (lo + hi) / span0       # ~[-1, 1]
    span   = 0.5 * (hi - lo) / span0       # ~[0, 1]
    return center, span


# ---------- JSON → per-joint descriptors ----------
def build_descriptors_from_meta(
    meta: Dict,
    F_MAX: int = 5,
    roles: Tuple[str,...] = ("palm","base","middle","end"),
    G_MAX: int = 10,
    use_finger_id_onehot: bool = False,
    palm_radius_m: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows: List[List[float]] = []
    mask: List[float] = []

    fingers = meta.get("fingers", [])
    for f_idx in range(1, F_MAX+1):
        f_info = next((f for f in fingers if int(f.get("finger_idx", -1)) == f_idx), None)

        # If finger block missing → all four roles absent:
        if f_info is None:
            for role in roles:
                role_oh = ROLE_TO_ONEHOT[role]
                fingertip_oh = [0.0, 0.0, 0.0]  # absent fingertip type
                feats = [0.0] + role_oh + fingertip_oh + [0.0,0.0, 0.0,0.0,  # base (x,y), yaw(sin,cos)
                                           0.0,               # grammar_norm
                                           0.0,0.0,0.0,       # axis xyz
                                           0.0,0.0,           # limits: center, span
                                           0.0]               # torque_scale (kept for future-proofing)
                if use_finger_id_onehot:
                    feats += [0.0]*5
                rows.append(feats); mask.append(0.0)
            continue

        present_finger = bool(f_info.get("present", False))
        base_pose = f_info.get("base_pose", {"pos_palm_m":[0,0,0], "quat_wxyz":[1,0,0,0]})
        px, py, _ = base_pose.get("pos_palm_m", [0.0,0.0,0.0])
        sx, sy = norm_xy((px, py), denom=palm_radius_m)
        ys, yc = yaw_sin_cos_from_quat_wxyz(base_pose.get("quat_wxyz", [1,0,0,0]))
        
        # Extract fingertip type
        fingertip_type = f_info.get("fingertip_type", "standard")
        fingertip_oh = FINGERTIP_TO_ONEHOT.get(fingertip_type, [1,0,0])  # default to standard
        if not present_finger:
            fingertip_oh = [0.0, 0.0, 0.0]  # zero out if finger not present
        
        id_oh = [0.0]*5
        if use_finger_id_onehot and present_finger:
            id_oh[f_idx-1] = 1.0

        j_list = f_info.get("joints", [])
        def find_joint(role_name):
            return next((j for j in j_list if j.get("role")==role_name), None)

        for role in roles:
            j = find_joint(role) or {}
            j_present = 1.0 if (present_finger and bool(j.get("present", False))) else 0.0

            # grammar: force palm to 0
            if role == "palm":
                grammar = 0.0
            else:
                grammar = float(j.get("grammar", 0.0))
            grammar_norm = 0.0 if grammar <= 0 else min(1.0, grammar / float(G_MAX))

            # axis: support both keys; default zeros
            axis = j.get("axis_xyz", j.get("axis_palm_xyz", [0.0,0.0,0.0]))
            axis = _safe_unit(axis).tolist()

            # limits → center/span
            lo = j.get("limit_lo", 0.0)
            hi = j.get("limit_hi", 0.0)
            ctr, spn = limits_center_span(lo, hi, role)

            torque = float(j.get("torque_scale", 0.0))  # ok if missing

            role_oh = ROLE_TO_ONEHOT[role]
            feats = [j_present] + role_oh + fingertip_oh + [sx, sy, ys, yc, grammar_norm] + axis + [ctr, spn, torque]
            if use_finger_id_onehot:
                feats += id_oh
            rows.append(feats)
            mask.append(j_present)

    J_desc = torch.tensor(rows, dtype=torch.float32).view(20, -1)  # [20, Fdesc]
    J_mask = torch.tensor(mask, dtype=torch.float32)               # [20]
    return J_desc, J_mask


def load_meta_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)
    
    
def build_hand_attn_bias(F_MAX=5, chain_bonus=0.5, same_finger_bonus=0.2, cross_finger_penalty=1.0):
    """
    Returns a (S,S) float mask added to attention logits before softmax.
    Positive entries increase attention; negative decrease. S = 4*F_MAX.
    Token layout: [palms(0..F-1), bases(F..2F-1), middles(2F..3F-1), ends(3F..4F-1)].
    """
    S = 4 * F_MAX
    bias = torch.zeros(S, S, dtype=torch.float32)

    def finger_of(idx):
        # idx in [0..S-1]; role block size is F_MAX
        return idx % F_MAX

    # base bonuses within the same finger
    for i in range(S):
        for j in range(S):
            if i == j:
                continue
            if finger_of(i) == finger_of(j):
                bias[i, j] += same_finger_bonus
            else:
                bias[i, j] -= cross_finger_penalty

    # extra chain bonus for direct neighbors along the kinematic chain per finger
    for f in range(F_MAX):
        palm   = 0*F_MAX + f
        base   = 1*F_MAX + f
        middle = 2*F_MAX + f
        end    = 3*F_MAX + f
        chain_pairs = [(palm, base), (base, middle), (middle, end)]
        for a, b in chain_pairs:
            bias[a, b] += chain_bonus
            bias[b, a] += chain_bonus

    return bias

class MorphEncoder(nn.Module):
    def __init__(self, feat_dim: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_joint: int = 32, d_global: int = 64,
                 # new knobs:
                 chain_bonus: float = 0.5, same_finger_bonus: float = 0.2, cross_finger_penalty: float = 1.0,
                 F_MAX: int = 5):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=4*d_model,
                                       batch_first=True, activation="gelu")
            for _ in range(n_layers)
        ])
        self.to_joint  = nn.Linear(d_model, d_joint)
        self.to_global = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_global)
        )
        # precompute graph-aware attention bias (SxS, S=20 when F_MAX=5)
        bias = build_hand_attn_bias(F_MAX=F_MAX,
                                    chain_bonus=chain_bonus,
                                    same_finger_bonus=same_finger_bonus,
                                    cross_finger_penalty=cross_finger_penalty)
        # register so it moves with .to(device)
        self.register_buffer("attn_bias", bias)              # optional (float, not used by the layer)
        attn_block = (bias < 0)                              # True => block attention
        attn_block.fill_diagonal_(False)                     # allow self-attention
        self.register_buffer("attn_block", attn_block)       # bool [S,S]

    def forward(self, J_desc: torch.Tensor, J_mask: torch.Tensor):
        """
        J_desc: [B, 20, Fdesc]
        J_mask: [B, 20]  (1=present, 0=pad)
        """
        x = self.inp(J_desc)
        pad = (J_mask == 0)  # bool [B,S]  True => pad

        for layer in self.layers:
            x = layer(x, src_mask=self.attn_block, src_key_padding_mask=pad)

        Z_joint = self.to_joint(x)               # [B,20,d_joint]
        w = J_mask.unsqueeze(-1).float()
        denom = w.sum(dim=1).clamp_min(1.0)
        pooled = (x * w).sum(dim=1) / denom
        Z_global = self.to_global(pooled)
        return Z_joint, Z_global



class FrozenMorphEncoder(nn.Module):
    """
    Same backbone, but exposes a no-grad forward that returns only a global embedding.
    Use this for the precompute-and-append path.
    """
    def __init__(self, feat_dim: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, out_dim: int = 64):
        super().__init__()
        self.backbone = MorphEncoder(feat_dim, d_model, n_heads, n_layers,
                                     d_joint=32, d_global=out_dim)

    @torch.no_grad()
    def forward(self, J_desc: torch.Tensor, J_mask: torch.Tensor) -> torch.Tensor:
        # Accept [N,20,F] or [20,F]
        if J_desc.dim() == 2:
            J_desc = J_desc.unsqueeze(0)
            J_mask = J_mask.unsqueeze(0)
        _, Zg = self.backbone(J_desc, J_mask)  # [N, out_dim]
        return Zg
        return Zg