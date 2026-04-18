#!/usr/bin/env python3
"""
Integration script to run Graph Heuristic Search with your existing PPO pipeline.
Uses your pre-trained checkpoints and play.py script for evaluation.
"""

import os
import sys
import argparse
import subprocess
import shutil
import random
import time
from pathlib import Path
import json
import torch
from typing import Dict, List, Tuple, Optional
import re
import csv
import wandb
import hashlib

# Import the core GHS algorithm
from graph_heuristic_search import GraphHeuristicSearch, DesignGraph, HandDesignGenerator, FINGERTIP_CHOICES
from graph_heuristic_search import canonicalize_fingers


from hand_groups import HandGroup
from utils.isaac_python import resolve_isaac_python
from utils.eval_logging import EvaluatorLoggingMixin
from utils.clean import run_managed_subprocess


DEFAULT_GROUP_CYCLE = [
    HandGroup.SYM3, HandGroup.SYM4, HandGroup.SYM5,
    HandGroup.ANTH21, HandGroup.ANTH27, HandGroup.ANTH33,
]


def parse_eval_groups(raw: str) -> List[HandGroup]:
    """
    Parse --eval-groups into an ordered list of HandGroup values.
    Accepts:
      - "all" (default): all 6 groups
      - comma-separated subset: e.g. "sym5" or "sym3,sym4,sym5"
    """
    if raw is None:
        return DEFAULT_GROUP_CYCLE.copy()

    token = raw.strip().lower()
    if token in ("all", "*"):
        return DEFAULT_GROUP_CYCLE.copy()

    by_name = {g.value: g for g in HandGroup}
    names = [part.strip().lower() for part in token.split(",") if part.strip()]
    if not names:
        raise ValueError("Empty --eval-groups. Use 'all' or comma-separated group names.")

    invalid = sorted({name for name in names if name not in by_name})
    if invalid:
        valid = ", ".join(g.value for g in DEFAULT_GROUP_CYCLE)
        raise ValueError(f"Invalid group(s): {', '.join(invalid)}. Valid values: {valid}, or 'all'.")

    # Deduplicate while preserving user order.
    out: List[HandGroup] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(by_name[name])

    return out


def _canonicalize_with_effective_metadata(design: DesignGraph) -> DesignGraph:
    """Canonicalize a design while preserving effective/evaluated annotations."""
    original = design
    canonical = canonicalize_fingers(design)
    for attr in ("_effective_key", "_effective_graph", "_effective_group", "evaluated_group"):
        if hasattr(original, attr):
            setattr(canonical, attr, getattr(original, attr))
    return canonical

class CodesignPPOEvaluator(EvaluatorLoggingMixin):
    """
    PPO evaluator that uses your pre-trained checkpoint and play.py script.
    Focuses on rotation performance metrics.
    """
    
    def __init__(self, 
                 build_script: str = "../Generation/converters/batch_build_rand_hands2.sh",
                 play_script: str = "../IsaacLab/scripts/rl_games/play.py",
                 output_dir: str = "ghs_outputs",
                 isaac_python: Optional[str] = None,
                 eval_episodes: int = 3,
                 eval_envs: int = 128,
                 eval_timeout: int = 30,
                 env=None,
                 placement_mode: str = "symmetric",
                 debug: bool = False,
                 wandb_run: Optional[object] = None,
                 video_every_n_cycles: Optional[int] = None,
                 video_length: int = 800,
                 seed: Optional[int] = None,
                 eval_groups: Optional[List[HandGroup]] = None,
                 shuffle_groups: bool = True):
        
        self.debug = debug
        self.wandb_run = wandb_run
        self.video_every_n_cycles = video_every_n_cycles
        self.video_length = video_length            
        self.build_script = build_script
        self.play_script = play_script
        self.output_dir = output_dir
        self.eval_episodes = eval_episodes
        self.eval_envs = eval_envs
        self.eval_timeout = eval_timeout
        self.seed = seed
        self.env = os.environ.copy() if env is None else env
        self.placement_mode = placement_mode # "symmetric" | "anthro-top-heavy" | "both"
        self.thumb_slot_choices = [21, 27, 33]
        self.last_mode_by_design: Dict[str, str] = {}
        self.last_codes_by_design: Dict[str, str] = {}
        self.iteration_stats = []  # list of dicts we also flush to disk
        self.stats_path_csv = os.path.join(self.output_dir, "final_results", "iteration_group_stats.csv")
        self.stats_path_jsonl = os.path.join(self.output_dir, "final_results", "iteration_group_stats.jsonl")
        
        # Cycle-level tracking
        self.cycle_stats = []  # list of dicts for cycle-level aggregates
        self.cycle_stats_path_csv = os.path.join(self.output_dir, "final_results", "cycle_group_stats.csv")
        self.cycle_stats_path_jsonl = os.path.join(self.output_dir, "final_results", "cycle_group_stats.jsonl")
        self._cycle_count = 0  # Track cycle number independently for wandb
        
        # Track per-group stats within each cycle
        self.current_cycle_group_data: Dict[str, Dict[str, List[float]]] = {}  # {group_name: {"scores": [...], "rewards": [...]}}
        
        os.makedirs(os.path.dirname(self.stats_path_csv), exist_ok=True)
        
        # running bests across all iterations
        self.global_best_score: Optional[float] = None
        self.global_best_reward: Optional[float] = None
        
        # running worst across all iterations
        self.global_worst_score: Optional[float] = None
        self.global_worst_reward: Optional[float] = None
        
        self._iter_idx = 0
        self._group_cycle = (eval_groups or DEFAULT_GROUP_CYCLE).copy()
        if not self._group_cycle:
            raise ValueError("eval_groups must include at least one group.")
        self._num_groups = len(self._group_cycle)
        self._shuffle_groups = bool(shuffle_groups and self._num_groups > 1)
        if self._shuffle_groups:
            random.shuffle(self._group_cycle)
        self._reshuffle_every = self._num_groups

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Verify required files exist
        if not os.path.exists(build_script):
            raise FileNotFoundError(f"Build script not found: {build_script}")
        if not os.path.exists(play_script):
            raise FileNotFoundError(f"Play script not found: {play_script}")

        self.isaac_python = isaac_python or resolve_isaac_python(
            play_script=self.play_script,
            env=self.env,
        )
    
    @staticmethod
    def _safe_mean(vals):
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))]
        return sum(vals) / len(vals) if vals else None
    
    @staticmethod
    def _safe_max(vals):
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))]
        return max(vals) if vals else None
    
    @staticmethod
    def _stable_id(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    
    @staticmethod
    def _safe_min(vals):
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))]
        return min(vals) if vals else None
        
    def _current_group(self) -> HandGroup:
        # reshuffle cycle every N iterations to keep exploration fresh
        if self._shuffle_groups and self._iter_idx > 0 and (self._iter_idx % self._reshuffle_every == 0):
            random.shuffle(self._group_cycle)
        return self._group_cycle[self._iter_idx % self._num_groups]
    
    def finalize_incomplete_cycle(self):
        """Log any remaining cycle data that wasn't logged due to incomplete cycle"""
        if self.current_cycle_group_data:
            current_cycle = self._iter_idx // self._num_groups
            print(f"\n⚠️  Incomplete cycle {current_cycle} detected - logging partial results")
            self._log_cycle_stats(current_cycle)

    def _checkpoint_for_group(self, g: HandGroup) -> str:
        if g == HandGroup.SYM3:
            return "../IsaacLab/logs/rl_games/cd_hand_direct/group4_4096envs/nn/cd_hand_direct.pth"
        if g == HandGroup.SYM4:
            return "../IsaacLab/logs/rl_games/cd_hand_direct/group2_4096envs_11-13obj/nn/cd_hand_direct.pth"
        if g == HandGroup.SYM5:
            return "../IsaacLab/logs/rl_games/cd_hand_direct/group3_4096envs/nn/cd_hand_direct.pth"
        if g == HandGroup.ANTH21:
            return "../IsaacLab/logs/rl_games/cd_hand_direct/group9_2048envs/nn/cd_hand_direct.pth"
        if g == HandGroup.ANTH27:
            return "../IsaacLab/logs/rl_games/cd_hand_direct/group8_2048envs/nn/cd_hand_direct.pth"
        if g == HandGroup.ANTH33:
            return "../IsaacLab/logs/rl_games/cd_hand_direct/group7_2048envs/nn/cd_hand_direct.pth"
        return self.checkpoint_path  # fallback
    
    def codes_to_designgraph(
        self,
        finger_codes: list[str],
        finger_tips: list[str],
        mode: str,
        thumb_slot: Optional[int]
    ) -> DesignGraph:
        # Build a canonical 6-node graph: palm + 5 fingers
        nodes = [{'finger_id': 0, 'is_base': True, 'is_terminal': True}]
        for i, code in enumerate(finger_codes, start=1):
            s = 0 if code == "" else (2 if len(code)==2 else 3)
            g1 = int(code[0]) + 1 if s==3 else 0
            g2 = int(code[-2]) + 1 if s>=2 else 0
            nodes.append({
                'finger_id': i,
                'grammar_1_count': g1,
                'grammar_2_count': g2,
                'servo_count': s,
                'fingertip_type': finger_tips[i-1] if i-1 < len(finger_tips) else 'standard',
                'is_terminal': (s==0),
                'is_base': False,
            })
        edges = [(0,i) for i in range(1,6)]
        terminals = {0} | {i for i in range(1,6) if nodes[i]['is_terminal']}
        non_terminals = {i for i in range(1,6) if i not in terminals}

        thumb_str = f"_{thumb_slot}" if thumb_slot is not None else ""
        # 🔑 Include fingertip type with each code to avoid key collisions
        code_tip_pairs = ",".join(f"{c}:{t}" for c, t in zip(finger_codes, finger_tips))
        design_string = f"eff_{mode}{thumb_str}_" + code_tip_pairs

        return DesignGraph(nodes, edges, terminals, non_terminals, design_string)



    def effective_codes_for(self, design: DesignGraph, group: HandGroup) -> tuple[list[str], str, Optional[int], list[str]]:
        """
        Single source of truth for converting a design to effective codes for a given group.
        Returns (codes, mode, thumb_slot, tips)
        """
        finger_codes_raw = self.design_to_grammar_codes(design)
        finger_tips = self.design_to_fingertips(design)

        finger_codes, selected_mode, fixed_thumb_slot = self._force_group_constraints(finger_codes_raw, group)
        return finger_codes, selected_mode, fixed_thumb_slot, finger_tips
    
    def _force_group_constraints(
        self, finger_codes: List[str], g: HandGroup
    ) -> Tuple[List[str], str, Optional[int]]:
        """
        Returns (adjusted_codes, placement_mode, thumb_slot)
        - For symmetric groups: force exactly 3/4/5 active fingers.
        - For anthro groups: force anthro-top-heavy + fixed thumb slot (21/27/33).
        """
        codes = finger_codes[:] + [""] * (5 - len(finger_codes))
        def active_count(cc): return sum(1 for c in cc if c)

        if g in (HandGroup.SYM3, HandGroup.SYM4, HandGroup.SYM5):
            target = {HandGroup.SYM3: 3, HandGroup.SYM4: 4, HandGroup.SYM5: 5}[g]
            # Make exactly `target` non-empty codes (leftmost kept first)
            act_idx = [i for i, c in enumerate(codes) if c]
            ghost_idx = [i for i, c in enumerate(codes) if not c]
            if len(act_idx) > target:
                for i in act_idx[target:]:
                    codes[i] = ""  # ghost it
            elif len(act_idx) < target:
                # activate ghosts using a safe default (2-servo): "20"
                need = target - len(act_idx)
                for i in ghost_idx[:need]:
                    codes[i] = "20"
            # clip length and return symmetric
            return codes[:5], "symmetric", None

        # Anthro groups: exactly 5 active, last has 3 servos; fixed thumb slot
        thumb_map = {
            HandGroup.ANTH21: 21, HandGroup.ANTH27: 27, HandGroup.ANTH33: 33
        }
        ts = thumb_map[g]
        
        # Ensure 5 active fingers first
        for i in range(5):
            if not codes[i]:
                # print(f"  Activating ghost finger {i+1} with 2 servos.")
                codes[i] = "20"   # activate with safe 2-servo default
            if len(codes[i]) > 3:
                codes[i] = codes[i][:3]
        
        three_servo_idx = None
        for i in range(5):
            if len(codes[i]) == 3:
                three_servo_idx = i
                break

        # If we found a 3-servo finger not already at position 0, swap it
        if three_servo_idx is not None and three_servo_idx != 0:
            print(f"DEBUG: Swapping finger {three_servo_idx+1} (3-servo) to position 1 for anthro-top-heavy.")
            codes[0], codes[three_servo_idx] = codes[three_servo_idx], codes[0]

        # Now ensure position 0 has 3 servos
        if len(codes[0]) == 2:
            print("DEBUG: Upgrading 2-servo to 3-servo on finger 1 for anthro-top-heavy.")
            codes[0] = "3" + codes[0]
        elif len(codes[0]) < 2:
            print("DEBUG: Forcing 3-servo on finger 1 for anthro-top-heavy.")
            codes[0] = "330"
        elif len(codes[0]) > 3:
            print("DEBUG: Truncating finger 1 code to 3-servo for anthro-top-heavy.")
            codes[0] = codes[0][:3]
            
        return codes[:5], "anthro-top-heavy", ts

    
    def design_to_grammar_codes(self, design: DesignGraph, placement_mode: Optional[str] = None) -> List[str]:
        canonical_design = canonicalize_fingers(design)
        finger_codes = []
        for i, node in enumerate(canonical_design.nodes):
            if i == 0:
                continue

            g1 = min(max(node.get('grammar_1_count', 3), 1), 10) - 1  # 0..9
            g2 = min(max(node.get('grammar_2_count', 3), 1), 10) - 1  # 0..9
            s  = node.get('servo_count', 3)

            # Hard guard: disallow 1-servo by coercing to 2
            if s == 1:
                print("  WARN: servo_count==1 encountered; coercing to 2.")
                s = 2

            if s <= 0:
                code = ""                # ghost
            elif s == 2:
                code = f"{g2}0"          # middle + end
            else:  # s >= 3
                code = f"{g1}{g2}0"      # base + middle + end

            finger_codes.append(code)

        while len(finger_codes) < 5:
            finger_codes.append("")

        # if self.debug:
        #     print("  DEBUG (post-canonicalization):")
        #     for idx, code in enumerate(finger_codes, 1):
        #         print(f"    Finger {idx}: code='{code}', servos={len(code) if code else 0}")
            
        seen_empty = False
        for idx, c in enumerate(finger_codes, 1):
            if c == "": seen_empty = True
            elif seen_empty:
                raise RuntimeError("Non-ghost after a ghost: graph not canonicalized.")

        mode = (placement_mode or self.placement_mode)
        if mode == "anthro-top-heavy":
            # Enforce exactly 5 active fingers and 3-servos on the FIRST finger (thumb)
            # Force first code (thumb) to have 3 servos
            first = finger_codes[0]
            if len(first) < 3:
                finger_codes[0] = "330"  # default 3-servo code
            elif len(first) == 2:
                # upgrade 2-servo to 3-servo using a default base joint
                finger_codes[0] = f"3{first}"
            elif len(first) > 3:
                finger_codes[0] = first[:3]
            
            # Make remaining 4 non-empty with at least 2 servos
            for j in range(1, 5):
                cj = finger_codes[j]
                if not cj or len(cj) < 2:
                    finger_codes[j] = "20"  # default 2-servo code
                elif len(cj) > 3:
                    finger_codes[j] = cj[:3]
        elif mode == "symmetric":       
            pass

        return finger_codes[:5]
    
    def design_to_fingertips(self, design: DesignGraph, placement_mode: Optional[str] = None) -> List[str]:
        """
        Return a 5-length list of fingertip types (strings) in canonical finger order.
        Ghost fingers still get a value (default 'standard'); the builder may ignore if finger is ghost.
        """
        canonical_design = canonicalize_fingers(design)
        tips: List[str] = []
        for i, node in enumerate(canonical_design.nodes):
            if i == 0:
                continue
            t = str(node.get('fingertip_type', 'standard'))
            if t not in FINGERTIP_CHOICES:
                t = 'standard'
            tips.append(t)
        while len(tips) < 5:
            tips.append('standard')
        return tips[:5]
    
    def _choose_mode_for_design(self, finger_codes: List[str]) -> str:
        if self.placement_mode == "both":
            return "anthro-top-heavy" if random.random() < 0.5 else "symmetric"
        return self.placement_mode

    def _build_mode_flags(self, mode: str, active_count: int, thumb_slot: Optional[int] = None) -> List[str]:
        palm_radius = "0.06" if (mode == "symmetric" and active_count == 3) else "0.07"
        seed = self.seed if self.seed is not None else 10
        seed_flag = ["--seed", str(seed)]
        if mode == "symmetric":
            return [
                "--placement-mode", "symmetric",
                "--palm-radius", palm_radius,
                "--symmetric-start-deg", "0",
                "--symmetric-jitter-deg", "0",
                "--slot-count", "36",
                "--min-sep-slots", "4",
                "--save-xml",
                *seed_flag,
            ]
        elif mode == "anthro-top-heavy":
            ts = thumb_slot if thumb_slot is not None else random.choice(self.thumb_slot_choices)
            return [
                "--placement-mode", "anthro-top-heavy",
                "--palm-radius", "0.07",
                "--slot-count", "36",
                "--min-sep-slots", "4",
                "--thumb-bottom-deg", "180", "360",
                "--top-band-deg", "0", "180",
                "--anthro-top-fixed-slots", "15,11,7,3",
                "--thumb-fixed-slot", str(ts),
                "--thumb-fixed-servos", "3",
                "--save-xml",
                *seed_flag,
            ]
        return [*seed_flag]

    def generate_hand_urdf(self, design: DesignGraph, placement_mode: Optional[str] = None, 
                          group: Optional[HandGroup] = None) -> Optional[str]:
        """Generate URDF for the design using existing build pipeline with enhanced error logging.
        
        Args:
            design: The design graph to build
            placement_mode: Optional override for placement mode
            group: Optional HandGroup to use for constraint enforcement (preferred)
        """
        # **FIX A + C: Use effective_codes_for if we know the group**
        finger_codes, selected_mode, fixed_thumb_slot, finger_tips = self.effective_codes_for(design, group)
        
        # 2) Build effective DesignGraph + key
        eff_dg = self.codes_to_designgraph(finger_codes, finger_tips, selected_mode, fixed_thumb_slot)
        eff_key = eff_dg.design_string
        codes_str = ",".join(finger_codes)

        # **FIX D: Record under BOTH effective key AND raw key**
        self.last_mode_by_design[eff_key] = selected_mode
        self.last_codes_by_design[eff_key] = codes_str
        
        raw_key = getattr(design, "design_string", eff_key)
        self.last_mode_by_design.setdefault(raw_key, selected_mode)
        self.last_codes_by_design.setdefault(raw_key, codes_str)

        # 4) Attach for later reads
        setattr(design, "_effective_key", eff_key)
        setattr(design, "_effective_graph", eff_dg)

        # 5) Paths + logs (hash by eff_key so artifacts are 1:1 with effective design)
        design_hash = self._stable_id(eff_key)
        design_dir = os.path.join(self.output_dir, f"design_{design_hash}")
        os.makedirs(design_dir, exist_ok=True)
        error_log_dir = os.path.join(self.output_dir, "error_logs")
        os.makedirs(error_log_dir, exist_ok=True)

        # fingertip flags
        tips_flags = []
        for j, tip in enumerate(finger_tips, start=1):
            tips_flags += [f"--fingertip-f{j}", tip]

        print(f"  🔧 Build details:")
        print(f"    Effective key: {eff_key}")
        print(f"    Codes: {codes_str}")
        print(f"    Mode: {selected_mode}")
        print(f"    Output dir: {design_dir}")

        try:
            cmd = [
                "bash", self.build_script, "-o", design_dir, "--mode", "manual", "-C", codes_str,
            ]
            active_count = sum(1 for c in finger_codes if c)
            # **Use fixed_thumb_slot if available (from group constraints)**
            if fixed_thumb_slot is not None:
                mode_flags = self._build_mode_flags(selected_mode, active_count, fixed_thumb_slot)
            else:
                mode_flags = self._build_mode_flags(selected_mode, active_count)
            cmd += mode_flags
            cmd += tips_flags
            print(f"    Command: {' '.join(cmd)}")

            build_start = time.time()
            result = run_managed_subprocess(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=self.env,
                label=f"build {design_hash}",
            )
            build_time = time.time() - build_start
            print(f"    Build completed in {build_time:.1f}s")
            print(f"    Return code: {result.returncode}")

            if result.returncode != 0:
                print(f"    ❌ Build FAILED")
                error_file_base = os.path.join(error_log_dir, f"build_error_{design_hash}")
                with open(f"{error_file_base}_stdout.log", 'w') as f:
                    f.write(f"Raw key: {raw_key}\nEff key: {eff_key}\nCodes: {codes_str}\n")
                    f.write(f"Command: {' '.join(cmd)}\nReturn code: {result.returncode}\n")
                    f.write(f"Build time: {build_time:.1f}s\n{'='*80}\nSTDOUT:\n{result.stdout}")
                with open(f"{error_file_base}_stderr.log", 'w') as f:
                    f.write(f"Raw key: {raw_key}\nEff key: {eff_key}\nCodes: {codes_str}\n")
                    f.write(f"Command: {' '.join(cmd)}\nReturn code: {result.returncode}\n")
                    f.write(f"Build time: {build_time:.1f}s\n{'='*80}\nSTDERR:\n{result.stderr}")
                print(f"    STDOUT preview: {result.stdout[:200]}...")
                print(f"    STDERR preview: {result.stderr[:200]}...")
                print(f"    📄 Logs: {error_file_base}_stdout.log / _stderr.log")
                return None

            urdf_files = list(Path(design_dir).glob("*.urdf"))
            other_files = list(Path(design_dir).glob("*"))
            print(f"    Files generated: {len(other_files)} total")
            print(f"    URDF files: {len(urdf_files)}")

            if not urdf_files:
                print(f"    ❌ No URDF files found!")
                success_log = os.path.join(error_log_dir, f"success_no_urdf_{design_hash}.log")
                with open(success_log, 'w') as f:
                    f.write(f"Eff key: {eff_key}\nCodes: {codes_str}\n")
                    f.write(f"Command: {' '.join(cmd)}\nReturn code: {result.returncode} (SUCCESS)\n")
                    f.write(f"Build time: {build_time:.1f}s\nFiles generated: {len(other_files)}\n")
                    f.write(f"Directory contents: {[p.name for p in other_files]}\n{'='*80}\n")
                    f.write("STDOUT:\n" + result.stdout + "\n" + '='*80 + "\nSTDERR:\n" + result.stderr)
                print(f"    📄 Success log (no URDF): {success_log}")
                return None

            urdf_path = str(urdf_files[0])
            urdf_size = os.path.getsize(urdf_path)
            print(f"    ✓ URDF found: {urdf_path} ({urdf_size} bytes)")
            return urdf_path

        except subprocess.TimeoutExpired:
            print(f"    ⏰ Build TIMED OUT after 300s")
            timeout_log = os.path.join(error_log_dir, f"timeout_{design_hash}.log")
            with open(timeout_log, 'w') as f:
                f.write(f"Eff key: {eff_key}\nCodes: {codes_str}\n")
                f.write(f"Command: {' '.join(cmd)}\nBuild TIMED OUT after 300s\n")
            print(f"    📄 Timeout log: {timeout_log}")
            return None
        except Exception as e:
            print(f"    💥 Build ERROR: {e}")
            exception_log = os.path.join(error_log_dir, f"exception_{design_hash}.log")
            with open(exception_log, 'w') as f:
                f.write(f"Eff key: {eff_key}\nCodes: {codes_str}\n")
                f.write(f"Command: {' '.join(cmd)}\nException: {e}\nType: {type(e).__name__}\n")
            print(f"    📄 Exception log: {exception_log}")
            return None
        
    def evaluate_designs_parallel(self, designs: List[DesignGraph]) -> List[float]:
        """Evaluate multiple designs in parallel using one simulation"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {len(designs)} DESIGNS IN PARALLEL")
        print(f"{'='*60}")
        
        # Rotate groups every iteration
        group = self._current_group()
        
        for d in designs:
            d.evaluated_group = group.value
        
        print(f"\n🔀 Iteration {self._iter_idx+1} using group: {group.value}")
        
        # Calculate which cycle we're in
        current_cycle = self._iter_idx // self._num_groups
        position_in_cycle = self._iter_idx % self._num_groups
        is_last_iteration_of_cycle = (position_in_cycle == self._num_groups - 1)
        
        # Determine if this cycle should record videos
        should_record_video = False
        if self.video_every_n_cycles is not None and current_cycle > 0:
            if current_cycle % self.video_every_n_cycles == 0:
                should_record_video = True
                print(
                    f"[INFO] Recording video: cycle {current_cycle}, "
                    f"iteration {position_in_cycle+1}/{self._num_groups} of cycle ({group.value})"
                )

        # Create iteration-specific batch folder
        iteration_id = f"iter_{self._iter_idx+1:04d}_{group.value}"
        shared_dir = os.path.join(self.output_dir, iteration_id, "batch_candidates")
        os.makedirs(shared_dir, exist_ok=True)

        # Video setup
        video_dir = None
        video_name = None
        if should_record_video:
            video_dir = os.path.join(
                self.output_dir, 
                "eval_videos", 
                f"cycle_{current_cycle:03d}",
                group.value
            )
            os.makedirs(video_dir, exist_ok=True)
            video_name = f"iter_{self._iter_idx+1:04d}_{group.value}"
            print(f"[INFO] Recording video for cycle {current_cycle}, group {group.value}: {video_name}.mp4")

        print(f"🔧 Using shared directory: {shared_dir}")
        
        # Step 1: Generate all URDF files
        urdf_paths = []
        design_hashes = []
        build_commands = []
        codes_by_idx = {}
        mode_by_idx = {}
        
        print(f"\n🔧 Generating URDFs for {len(designs)} designs...")
        for i, design in enumerate(designs):
            if self.debug:
                print(f"  Processing design {i+1}/{len(designs)}: {design.design_string}")
            
            finger_codes, selected_mode, fixed_thumb_slot, finger_tips = self.effective_codes_for(design, group)

            eff_dg = self.codes_to_designgraph(finger_codes, finger_tips, selected_mode, fixed_thumb_slot)
            eff_key = eff_dg.design_string
            design._effective_graph = eff_dg
            design._effective_group = group.value
            design._effective_key = eff_key

            tips_flags = []
            for j, tip in enumerate(finger_tips, start=1):
                tips_flags += [f"--fingertip-f{j}", tip]

            codes_str = ",".join(finger_codes)
            codes_by_idx[i] = codes_str
            mode_by_idx[i] = selected_mode
            
            self.last_mode_by_design[eff_key] = selected_mode
            self.last_codes_by_design[eff_key] = codes_str
            
            raw_key = design.design_string
            self.last_mode_by_design.setdefault(raw_key, selected_mode)
            self.last_codes_by_design.setdefault(raw_key, codes_str)
            
            design_hash = self._stable_id(eff_key)
            design_hashes.append(design_hash)

            if self.debug:
                print(f"    Group={group.value} | mode={selected_mode} | codes={codes_str}")

            design_prefix = f"design_{i:03d}_{design_hash}"
            cmd = [
                "bash", self.build_script,
                "-o", shared_dir,
                "--mode", "manual",
                "-C", codes_str,
            ]
            active_count = sum(1 for c in finger_codes if c)
            cmd += self._build_mode_flags(selected_mode, active_count, fixed_thumb_slot)
            cmd += tips_flags
            build_commands.append((cmd, design_prefix, i, codes_str))

            if self.debug:
                print(f"    Will build as: {design_prefix}")
        
        # Step 2: Execute all build commands and track successes
        print(f"\n🔨 Building {len(build_commands)} URDFs...")
        successful_builds = []  # List of (design_idx, urdf_path, design_prefix)
        design_to_actual_usd = {}  # Maps design_idx to actual USD basename
        failed_design_indices = set()  # Use set to avoid duplicates

        for cmd, design_prefix, design_idx, codes_str in build_commands:
            urdf_files_before = set(Path(shared_dir).glob("*.urdf"))
            
            try:
                if self.debug:
                    print(f"  Building design {design_idx+1}: {design_prefix}")
                    print(f"    Grammar codes: {codes_str}")

                result = run_managed_subprocess(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=self.env,
                    label=f"batch build {design_prefix}",
                )
                
                if result.returncode == 0:
                    urdf_files_after = set(Path(shared_dir).glob("*.urdf"))
                    new_urdf_files = urdf_files_after - urdf_files_before
                    
                    if new_urdf_files:
                        new_urdf = list(new_urdf_files)[0]
                        urdf_path = str(new_urdf)
                        actual_usd_basename = os.path.splitext(os.path.basename(urdf_path))[0]
                        
                        successful_builds.append((design_idx, urdf_path, design_prefix))
                        design_to_actual_usd[design_idx] = actual_usd_basename
                        
                        if self.debug:
                            print(f"    ✓ URDF: {os.path.basename(urdf_path)}")
                            print(f"    → Will become USD: {actual_usd_basename}.usd")
                    else:
                        print(f"    ❌ No new URDF found for {design_prefix}")
                        print(f"    Directory contents: {[f.name for f in Path(shared_dir).iterdir()]}")
                        
                        # Print detailed design info for failed builds:
                        design = designs[design_idx]
                        finger_codes, selected_mode, fixed_thumb_slot, finger_tips = self.effective_codes_for(design, group)
                        self.print_design_details(design, finger_codes, finger_tips, selected_mode, fixed_thumb_slot)
                        
                        failed_design_indices.add(design_idx)
                else:
                    print(f"    ❌ Build failed for {design_prefix}: {result.returncode}")
                    if result.stderr:
                        print(f"    Error output: {result.stderr[:200]}")
                    failed_design_indices.add(design_idx)
                        
            except subprocess.TimeoutExpired:
                print(f"    ⏱️  Build timeout for {design_prefix}")
                failed_design_indices.add(design_idx)
            except Exception as e:
                print(f"    💥 Build error for {design_prefix}: {e}")
                failed_design_indices.add(design_idx)

        num_successful = len(successful_builds)
        num_failed = len(failed_design_indices)

        print(f"✓ Successfully built {num_successful}/{len(designs)} URDFs")
        if num_failed > 0:
            print(f"⚠️  {num_failed} designs failed to build: {sorted(failed_design_indices)}")

        # **FAILSAFE 1: Check if we have enough successful builds to continue**
        if num_successful == 0:
            print("❌ No designs built successfully - aborting iteration")
            self._iter_idx += 1
            return [-1000.0] * len(designs)

        # Continue with only successful builds
        print(f"📋 Proceeding with {num_successful} successfully built designs")
        
        if self.debug:
            print(f"🗺️  Design index to actual USD mapping:")
            for design_idx, actual_usd_basename in design_to_actual_usd.items():
                print(f"    Design {design_idx} → {actual_usd_basename}.usd")
        
        # Step 3: Single batch USD conversion
        print(f"\n🔄 Converting all URDFs to USD in batch...")
        
        try:
            python_exe = self.isaac_python
            convert_stdout = ""
            convert_stderr = ""
            convert_env = self.env.copy()
            # The converter is stable with Isaac Sim's default loader behavior.
            # Keeping the CUDA preload here can trigger native allocator crashes.
            convert_env.pop("LD_PRELOAD", None)
            
            cmd = [
                python_exe,
                "source/codesign/codesign/utils/convert_urdf_usd.py",
                "--in-dir", os.path.abspath(shared_dir),
                "--out-dir", os.path.abspath(shared_dir),
                "--merge-joints", 
                "--headless"
            ]
            
            print(f"    Command: {' '.join(cmd)}")
            
            convert_start = time.time()
            result = run_managed_subprocess(
                cmd,
                capture_output=True,
                text=True,
                timeout=360,
                cwd="../IsaacLab",
                env=convert_env,
                label=f"batch USD conversion {iteration_id}",
            )
            convert_time = time.time() - convert_start
            convert_stdout = result.stdout or ""
            convert_stderr = result.stderr or ""
            
            print(f"    Batch conversion completed in {convert_time:.1f}s")
            print(f"    Return code: {result.returncode}")

            converter_failed = result.returncode != 0
            if "Traceback (most recent call last):" in convert_stdout or "Traceback (most recent call last):" in convert_stderr:
                converter_failed = True

            if converter_failed:
                print(f"    ❌ Batch conversion FAILED")
                if convert_stdout.strip():
                    print("    Converter stdout:")
                    for line in convert_stdout.strip().splitlines()[-40:]:
                        print(f"      {line}")
                if convert_stderr.strip():
                    print("    Converter stderr:")
                    for line in convert_stderr.strip().splitlines()[-40:]:
                        print(f"      {line}")
                self._iter_idx += 1
                return [-1000.0] * len(designs)
            
        except Exception as e:
            print(f"💥 Batch conversion ERROR: {e}")
            self._iter_idx += 1
            return [-1000.0] * len(designs)
        
        # Step 4: Collect USD files - ONLY for successful builds
        print(f"\n📋 Collecting USD files for {num_successful} successful builds...")
        usd_paths = [None] * len(designs)  # Full list including failures

        all_usd_files = list(Path(shared_dir).glob("*.usd"))
        if not all_usd_files:
            all_usd_files = list(Path(shared_dir).glob("**/*.usd"))

        print(f"Found {len(all_usd_files)} USD files total")

        # Map USD files ONLY to successful builds
        matched_count = 0
        for design_idx, urdf_path, design_prefix in successful_builds:
            urdf_basename = os.path.splitext(os.path.basename(urdf_path))[0]
            matched_usd = None
            
            for usd_file in all_usd_files:
                usd_basename = os.path.splitext(usd_file.name)[0]
                if usd_basename == urdf_basename:
                    matched_usd = str(usd_file)
                    break

            if matched_usd:
                usd_paths[design_idx] = matched_usd
                design_to_actual_usd[design_idx] = os.path.splitext(os.path.basename(matched_usd))[0]
                matched_count += 1
                if self.debug:
                    print(f"  Design {design_idx+1}: {os.path.basename(matched_usd)} (exact match)")
            else:
                print(f"  Design {design_idx+1}: No USD match for {urdf_basename}")
                failed_design_indices.add(design_idx)  # Add to failed set

        print(f"✓ Matched {matched_count} USD files to successful builds")
        print(f"✗ Total failed designs: {len(failed_design_indices)} - {sorted(failed_design_indices)}")

        valid_usd_count = sum(1 for p in usd_paths if p is not None)
        if valid_usd_count == 0:
            print("❌ No valid USD files found after conversion!")
            if convert_stdout.strip():
                print("    Converter stdout:")
                for line in convert_stdout.strip().splitlines()[-40:]:
                    print(f"      {line}")
            if convert_stderr.strip():
                print("    Converter stderr:")
                for line in convert_stderr.strip().splitlines()[-40:]:
                    print(f"      {line}")
            self._iter_idx += 1
            return [-1000.0] * len(designs)

        print(f"✓ {valid_usd_count} valid USD files ready for evaluation")

        # Step 5: Set environment variables ONLY for valid paths
        for i in range(100):
            env_var = f'CODESIGN_HAND_USD_PATH_{i}'
            if env_var in os.environ:
                del os.environ[env_var]

        # Map only successful designs to sequential environment indices
        design_idx_to_env_idx = {}
        env_idx = 0
        for design_idx, usd_path in enumerate(usd_paths):
            if usd_path is not None:
                absolute_usd_path = os.path.abspath(usd_path)
                os.environ[f'CODESIGN_HAND_USD_PATH_{env_idx}'] = absolute_usd_path
                design_idx_to_env_idx[design_idx] = env_idx
                env_idx += 1

        # Step 6: Run parallel evaluation
        print(f"\n🎮 Running parallel evaluation with {self.eval_envs} environments...")

        try:
            python_exe = self.isaac_python
            checkpoint_for_group = self._checkpoint_for_group(group)
            
            metrics_file = os.path.join(shared_dir, "ghs_metrics.jsonl")
            try:
                if os.path.exists(metrics_file):
                    os.remove(metrics_file)
            except Exception as _e:
                print(f"[warn] could not clear old metrics file: {metrics_file}: {_e}")

            
            cmd = [
                python_exe, self.play_script,
                f"--task=Codesign-Reorientation-Direct-v0",
                f"--num_envs={self.eval_envs}",
                f"--checkpoint={checkpoint_for_group}",
                f"--run_name=ghs_parallel_eval_{iteration_id}",
                "--headless",
                f"--max_episodes={self.eval_episodes}",
                f"--timeout={self.eval_timeout}",
                "--use_unexported"
            ]
            
            # if should_record_video and video_dir:
            #     cmd.extend([
            #         "--video",
            #         "--video_length", str(self.video_length),
            #         "--video_folder", video_dir,
            #         "--video_name", video_name,
            #     ])
            
            print(f"    Command: {' '.join(cmd)}")
            
            cmd.extend(["--metrics_out", metrics_file])
            child_env = self.env.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["GHS_METRICS_OUT"] = metrics_file  # backup path if args parsing differs
            child_env["KIT_DISABLE_LOG_COLORS"] = "1"
            child_env["OMNI_LOG_DISABLE_CONSOLE_COLORS"] = "1"

            # Clear old paths
            for k in list(child_env.keys()):
                if k.startswith('CODESIGN_HAND_USD_PATH'):
                    del child_env[k]
            
            # Set new paths using the mapping we created
            for design_idx, env_idx in design_idx_to_env_idx.items():
                usd_path = usd_paths[design_idx]
                child_env[f'CODESIGN_HAND_USD_PATH_{env_idx}'] = os.path.abspath(usd_path)
                    
            child_env["CODESIGN_HAND_GROUP"] = group.value
            child_env["CODESIGN_JOINT_PRESET"] = "anthro" if group in (
                HandGroup.ANTH21, HandGroup.ANTH27, HandGroup.ANTH33
            ) else "symmetric"
            child_env["CODESIGN_GHS_EVALUATION"] = "1"
            
            result = run_managed_subprocess(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
                cwd=os.getcwd(),
                env=child_env,
                label=f"parallel evaluation {iteration_id}",
            )

            if result.returncode == 0:
                print("✓ Parallel evaluation completed successfully")
                
                combined = result.stdout or ""
                score = None
                if os.path.exists(metrics_file) and os.path.getsize(metrics_file) > 0:
                    with open(metrics_file, "r", encoding="utf-8", errors="replace") as f:
                        file_text = f.read()
                    score = self._parse_rotation_performance_with_debugging(file_text, shared_dir, 0)

                if score is None:  # or use `if score is None` if you change parser to return None on “not found”
                    score = self._parse_rotation_performance_with_debugging(combined, shared_dir, 0)
                
                hand_metrics = {}
                if hasattr(self, 'last_hand_design_metrics') and 0 in self.last_hand_design_metrics:
                    hand_metrics = self.last_hand_design_metrics[0]
                
                if hand_metrics:
                    print(f"✓ Successfully parsed metrics for {len(hand_metrics)} hand designs")
                    self.print_hand_design_performance_summary()
                    
                    # Initialize scores for ALL designs (including failures)
                    scores = [-1000.0] * len(designs)  # Default failure score
                    research_data = [{}] * len(designs)  # Default empty data
                    
                    # Map hand metrics to scores ONLY for successful designs
                    for design_idx in range(len(designs)):
                        if design_idx in failed_design_indices:
                            print(f"  Design {design_idx+1}: FAILED BUILD - using sentinel score -1000.0")
                            continue
                            
                        if design_idx not in design_to_actual_usd:
                            print(f"  Design {design_idx+1}: NO USD MAPPING - using sentinel score -1000.0")
                            continue
                        
                        actual_usd_basename = design_to_actual_usd[design_idx]
                        design_score = 0.0
                        raw_metrics = {}
                        
                        # Find matching metrics
                        matched = False
                        for design_id, metrics in hand_metrics.items():
                            if design_id == actual_usd_basename:
                                matched = True
                                nr = metrics.get("norm_reward", None)
                                if nr is not None:
                                    design_score = float(nr)
                                else:
                                    # fallback: derive from average_reward if present
                                    avg_r = metrics.get("average_reward", None)
                                    if avg_r is not None:
                                        cap = float(metrics.get("reward_cap", 5.0))  # must match env cfg default
                                        design_score = max(0.0, min(float(avg_r) / cap, 1.0 - 1e-6))
                                    else:
                                        # final fallbacks (omega/rps) so old runs still work
                                        omega_score = metrics.get("omega_score", None)
                                        if omega_score is not None:
                                            design_score = float(omega_score)
                                        else:
                                            rps_from_omega = metrics.get("rps_from_omega", None)
                                            if rps_from_omega is not None:
                                                rps_cap = 3.0
                                                design_score = max(0.0, min(float(rps_from_omega) / rps_cap, 1.0 - 1e-6))
                                            else:
                                                rps = metrics.get("turns_per_second", None)
                                                if rps is None:
                                                    tpm = metrics.get("turns_per_minute", None)
                                                    rps = (tpm / 60.0) if isinstance(tpm, (int, float)) else None
                                                if rps is None:
                                                    sr = float(metrics.get("success_rate", 0.0))
                                                    if sr > 1.5:
                                                        sr /= 100.0
                                                    rps = max(0.0, min(sr, 1.0))
                                                design_score = float(rps)
                                
                                raw_metrics = {
                                    "design_id": design_id,
                                    "norm_reward": metrics.get("norm_reward", 0.0),
                                    "reward_cap": metrics.get("reward_cap", 0.0),
                                    "omega_score": metrics.get("omega_score", 0.0),
                                    "raw_omega_z_rad_per_sec": metrics.get("raw_omega_z_rad_per_sec", 0.0),
                                    "raw_omega_z_deg_per_sec": metrics.get("raw_omega_z_deg_per_sec", 0.0),
                                    "raw_omega_z_rpm": metrics.get("raw_omega_z_rpm", 0.0),
                                    "rps_from_omega": metrics.get("rps_from_omega", 0.0),
                                    "turns_per_second": metrics.get("turns_per_second", 0.0),
                                    "step_fraction_evaluated": metrics.get("step_fraction_evaluated", 0.0),
                                    "contact_steps": metrics.get("contact_steps", 0),
                                    "evaluation_time_s": metrics.get("evaluation_time_s", 0.0),
                                    "average_reward": (lambda v: max(0.0, float(v)) if v is not None else 0.0)(
                                        metrics.get("average_reward", 0.0)
                                    ),
                                    "total_episodes": metrics.get("total_episodes", 0),
                                }
                                
                                print(f"  Design {design_idx+1} → {actual_usd_basename}: score={design_score:.3f}")
                                break
                        
                        if not matched:
                            print(f"  Design {design_idx+1} → {actual_usd_basename}: no metrics found, using default score 0.0")
                        
                        scores[design_idx] = design_score
                        research_data[design_idx] = raw_metrics
                    
                    # Build URDF dict
                    urdf_by_idx = {idx: urdf for (idx, urdf, _) in successful_builds}
                    
                    iter_no = self._iter_idx
                    valid_scores = [s for s in scores if isinstance(s, (int, float)) and s > -999.0]
                    avg_score = self._safe_mean(valid_scores)

                    rewards_this_iter = []
                    for di, usd_base in design_to_actual_usd.items():
                        m = hand_metrics.get(usd_base)
                        if m is not None and "average_reward" in m:
                            try:
                                val = float(m["average_reward"])
                                # Clamp negative rewards to 0 for logging/aggregation
                                rewards_this_iter.append(max(0.0, val))
                            except Exception:
                                pass
                    avg_reward = self._safe_mean(rewards_this_iter)
                    
                    best_score_iter = self._safe_max(valid_scores)
                    best_reward_iter = self._safe_max(rewards_this_iter)

                    valid_scores_for_worst = [s for s in valid_scores if s > 0.001]
                    worst_score_iter = self._safe_min(valid_scores_for_worst) if valid_scores_for_worst else None

                    valid_rewards_for_worst = [r for r in rewards_this_iter if r > 0.001]
                    worst_reward_iter = self._safe_min(valid_rewards_for_worst) if valid_rewards_for_worst else None

                    if best_score_iter is not None:
                        self.global_best_score = best_score_iter if self.global_best_score is None else max(self.global_best_score, best_score_iter)
                    if best_reward_iter is not None:
                        self.global_best_reward = best_reward_iter if self.global_best_reward is None else max(self.global_best_reward, best_reward_iter)
                    
                    if worst_score_iter is not None:
                        self.global_worst_score = worst_score_iter if self.global_worst_score is None else min(self.global_worst_score, worst_score_iter)
                    if worst_reward_iter is not None:
                        self.global_worst_reward = worst_reward_iter if self.global_worst_reward is None else min(self.global_worst_reward, worst_reward_iter)

                    print(f"[iter {iter_no:03d} | {group.value}] best_score={best_score_iter if best_score_iter is not None else 'n/a'} "
                        f"(running={self.global_best_score if self.global_best_score is not None else 'n/a'}), "
                        f"avg_score={avg_score if avg_score is not None else 'n/a'}, "
                        f"worst_score={worst_score_iter if worst_score_iter is not None else 'n/a'} "
                        f"(running={self.global_worst_score if self.global_worst_score is not None else 'n/a'}), "
                        f"best_reward={best_reward_iter if best_reward_iter is not None else 'n/a'} "
                        f"(running={self.global_best_reward if self.global_best_reward is not None else 'n/a'}), "
                        f"avg_reward={avg_reward if avg_reward is not None else 'n/a'}")

                    # Track per-group cycle stats
                    group_name = group.value
                    if group_name not in self.current_cycle_group_data:
                        self.current_cycle_group_data[group_name] = {"scores": [], "rewards": []}
                    
                    for score in valid_scores:
                        self.current_cycle_group_data[group_name]["scores"].append(score)
                    for reward in rewards_this_iter:
                        self.current_cycle_group_data[group_name]["rewards"].append(reward)

                    if self.debug:
                        print(f"  [cycle-tracking] Added {len(valid_scores)} scores to {group_name} (total: {len(self.current_cycle_group_data[group_name]['scores'])})")

                    # Logging
                    self._wandb_log_iteration_payload(
                        iter_no=iter_no,
                        group=group,
                        scores=valid_scores if avg_score is not None else scores,
                        rewards=rewards_this_iter,
                        codes_by_idx=codes_by_idx,
                        mode_by_idx=mode_by_idx,
                        usd_paths=usd_paths,
                        research_data=research_data
                    )
                    
                    # if should_record_video and video_dir and self.wandb_run and wandb:
                    #     self._upload_videos_to_wandb(video_dir, iteration_id, scores, codes_by_idx, group)

                    self._log_iteration_stats(
                        iteration_idx=iter_no,
                        iteration_id=iteration_id,
                        group=group,
                        avg_score=avg_score,
                        avg_reward=avg_reward,
                        n_candidates=len(designs),
                        best_score_of_iter=best_score_iter,
                        best_reward_of_iter=best_reward_iter,
                        best_score_so_far=self.global_best_score,
                        best_reward_so_far=self.global_best_reward,
                        worst_score_of_iter=worst_score_iter,
                        worst_reward_of_iter=worst_reward_iter,
                        worst_score_so_far=self.global_worst_score,
                        worst_reward_so_far=self.global_worst_reward,
                    )
                    
                    if is_last_iteration_of_cycle:
                        self._log_cycle_stats(current_cycle)

                    self._write_manifests(
                        iteration_id=iteration_id,
                        designs=designs,
                        codes_by_idx=codes_by_idx,
                        mode_by_idx=mode_by_idx,
                        scores=scores,
                        research_data=research_data,
                        urdf_by_idx=urdf_by_idx,
                        usd_paths=usd_paths,
                        design_to_actual_usd=design_to_actual_usd,
                        design_to_env_var=design_idx_to_env_idx,
                        shared_dir=shared_dir
                    )
                    self._iter_idx += 1
                            
                    return scores

                # Keep the search alive even if evaluation exited cleanly but failed to
                # emit any metrics due to asset/runtime issues.
                self._log_iteration_stats(
                    iteration_idx=self._iter_idx,
                    iteration_id=iteration_id,
                    group=group,
                    avg_score=None,
                    avg_reward=None,
                    n_candidates=len(designs),
                    best_score_of_iter=None,
                    best_reward_of_iter=None,
                    best_score_so_far=self.global_best_score,
                    best_reward_so_far=self.global_best_reward,
                    worst_score_of_iter=None,
                    worst_reward_of_iter=None,
                    worst_score_so_far=self.global_worst_score,
                    worst_reward_so_far=self.global_worst_reward,
                )

                parse_log_file = os.path.join(shared_dir, "episode_0_parse_log.txt")
                print("⚠️ Parallel evaluation finished without emitting per-hand metrics; using sentinel scores -1000.0")
                if os.path.exists(parse_log_file):
                    print(f"   Parser details: {parse_log_file}")
                self._iter_idx += 1
                return [-1000.0] * len(designs)
            else:
                # Persist a failure iteration entry so plots keep x-axis contiguous
                self._log_iteration_stats(
                    iteration_idx=self._iter_idx,
                    iteration_id=iteration_id,
                    group=group,
                    avg_score=None,
                    avg_reward=None,
                    n_candidates=len(designs),
                    best_score_of_iter=None,
                    best_reward_of_iter=None,
                    best_score_so_far=self.global_best_score,
                    best_reward_so_far=self.global_best_reward,
                    worst_score_of_iter=None,
                    worst_reward_of_iter=None,
                    worst_score_so_far=self.global_worst_score,
                    worst_reward_so_far=self.global_worst_reward,
                )

                print(f"❌ Parallel evaluation FAILED (return code {result.returncode})")
                # Show tails so the logs are useful but not huge
                print(f"STDOUT (tail): {result.stdout[-1000:]}")
                print(f"STDERR (tail): {result.stderr[-2000:]}")
                # Return sentinel scores so the search can proceed
                self._iter_idx += 1
                return [-1000.0] * len(designs)
                
        except Exception as e:
            # Persist a failure iteration entry so plots keep x-axis contiguous
            self._log_iteration_stats(
                iteration_idx=self._iter_idx,
                iteration_id=iteration_id,
                group=group,
                avg_score=None,
                avg_reward=None,
                n_candidates=len(designs),
                best_score_of_iter=None,
                best_reward_of_iter=None,
                best_score_so_far=self.global_best_score,
                best_reward_so_far=self.global_best_reward,
                worst_score_of_iter=None,
                worst_reward_of_iter=None,
                worst_score_so_far=self.global_worst_score,
                worst_reward_so_far=self.global_worst_reward,
            )


            print(f"💥 Parallel evaluation ERROR: {e}")
            self._iter_idx += 1
            return [-1000.0] * len(designs)
        
    def _copy_assets_for_design(self, candidate_roots, dest_root, design_id):
        """
        Copy meshes/ and robot_meshes/ for a design into:
        <dest_root>/<design_id>/{meshes,robot_meshes}
        candidate_roots: list of dirs to search (first hit wins for each subfolder).
        """
        import shutil, os
        from pathlib import Path

        copied_any = False
        for sub in ("meshes", "robot_meshes"):
            # find the first candidate that has this subfolder
            src = next((os.path.join(root, sub) for root in candidate_roots
                        if os.path.isdir(os.path.join(root, sub))), None)
            if src:
                dst = os.path.join(dest_root, design_id, sub)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"✓ Copied {sub} for {design_id} -> {dst}")
                copied_any = True
            else:
                print(f"⚠️  {sub} not found for {design_id}; checked: {candidate_roots}")
        return copied_any
    
    def _write_manifests(self, iteration_id, designs, codes_by_idx, mode_by_idx, scores, research_data, urdf_by_idx, usd_paths, design_to_actual_usd, design_to_env_var, shared_dir):
        # Write batch manifest
        batch_csv = os.path.join(shared_dir, "manifest.csv")
        with open(batch_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration_id","design_idx","raw_design_string","effective_design_key","effective_group", "design_string","codes", "placement_mode", "training_score",
                "raw_omega_z_rad_per_sec","raw_omega_z_deg_per_sec","raw_omega_z_rpm","rps_from_omega","turns_per_second",
                "step_fraction_evaluated","average_reward","total_episodes",
                "urdf_path","usd_path","usd_basename","env_var_index",
            ])
            for di in range(len(designs)):
                d = designs[di]
                raw = research_data[di] if di < len(research_data) else {}
                raw_key = d.design_string
                eff_key = getattr(d, "_effective_key", raw_key)
                eff_group = getattr(d, "_effective_group", "")
                writer.writerow([
                    iteration_id,
                    di,
                    raw_key,
                    eff_key,
                    eff_group,
                    d.design_string,
                    codes_by_idx.get(di, ""), mode_by_idx.get(di, ""),
                    scores[di] if di < len(scores) else "",
                    raw.get("raw_omega_z_rad_per_sec", ""),
                    raw.get("raw_omega_z_deg_per_sec", ""), 
                    raw.get("raw_omega_z_rpm", ""),
                    raw.get("rps_from_omega", ""),
                    raw.get("turns_per_second", ""),
                    raw.get("step_fraction_evaluated", ""),
                    raw.get("average_reward", ""),
                    raw.get("total_episodes", ""),
                    urdf_by_idx.get(di, ""),
                    usd_paths[di] or "",
                    design_to_actual_usd.get(di, ""),
                    design_to_env_var.get(di, ""),
                ])

        # Append to global manifest
        global_csv = os.path.join(self.output_dir, "final_results", "design_asset_map.csv")
        os.makedirs(os.path.dirname(global_csv), exist_ok=True)
        first = not os.path.exists(global_csv)
        with open(global_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if first:
                writer.writerow([
                    "iteration_id","design_idx","design_string","codes", "placement_mode","training_score",
                    "raw_omega_z_rad_per_sec","raw_omega_z_deg_per_sec","raw_omega_z_rpm","rps_from_omega","turns_per_second",
                    "step_fraction_evaluated","average_reward","total_episodes",
                    "urdf_path","usd_path","usd_basename","env_var_index"
                ])
            for di in range(len(designs)):
                d = designs[di]
                raw = research_data[di] if di < len(research_data) else {}
                writer.writerow([
                    iteration_id,
                    di,
                    d.design_string,
                    codes_by_idx.get(di, ""),
                    mode_by_idx.get(di, ""),
                    scores[di] if di < len(scores) else "",
                    raw.get("raw_omega_z_rad_per_sec", ""),
                    raw.get("raw_omega_z_deg_per_sec", ""),
                    raw.get("raw_omega_z_rpm", ""),
                    raw.get("rps_from_omega", ""),
                    raw.get("turns_per_second", ""),
                    raw.get("step_fraction_evaluated", ""),
                    raw.get("average_reward", ""),
                    raw.get("total_episodes", ""),
                    urdf_by_idx.get(di, ""),
                    usd_paths[di] or "",
                    design_to_actual_usd.get(di, ""),
                    design_to_env_var.get(di, ""),
                ])
    
    def _parse_rotation_performance_with_debugging(self, stdout: str, debug_dir: str, episode: int, stderr: str = "") -> float:
        """Parse rotation performance with enhanced debugging and file output"""
        prefix_final = "GHS_METRICS_JSON_FINAL "
        prefix_last = "GHS_METRICS_JSON "
        prefix_hand_metrics = "GHS_HAND_METRICS_JSON "  # New: for hand-specific metrics

        final_line, last_line, hand_metrics_line = None, None, None
        all_json_lines = []
        
        lines = stdout.splitlines()
        print(f"    [parser] Scanning {len(lines)} lines for metrics...", flush=True)
        
        # Save detailed parsing info to file
        parse_log_file = os.path.join(debug_dir, f"episode_{episode}_parse_log.txt")
        with open(parse_log_file, 'w') as f:
            f.write(f"Parser log for episode {episode}\n")
            f.write(f"Total lines: {len(lines)}\n")
            f.write("=" * 50 + "\n\n")
            
            for line_num, line in enumerate(lines):
                if prefix_final in line:
                    i = line.find(prefix_final)
                    final_line = line[i + len(prefix_final):].strip()
                    all_json_lines.append(f"FINAL: {final_line[:100]}...")
                    f.write(f"Found FINAL line at {line_num}: {final_line[:100]}...\n")
                    print(f"    [parser] Found FINAL line at {line_num}: {final_line[:100]}...", flush=True)
                elif prefix_hand_metrics in line:  # New: parse hand metrics
                    i = line.find(prefix_hand_metrics)
                    hand_metrics_line = line[i + len(prefix_hand_metrics):].strip()
                    all_json_lines.append(f"HAND_METRICS: {hand_metrics_line[:100]}...")
                    f.write(f"Found HAND_METRICS line at {line_num}: {hand_metrics_line[:100]}...\n")
                    print(f"    [parser] Found HAND_METRICS line at {line_num}: {hand_metrics_line[:100]}...", flush=True)
                elif prefix_last in line:
                    i = line.find(prefix_last)
                    last_line = line[i + len(prefix_last):].strip()
                    all_json_lines.append(f"REGULAR: {last_line[:100]}...")
                    f.write(f"Found REGULAR line at {line_num}: {last_line[:100]}...\n")
                    print(f"    [parser] Found REGULAR line at {line_num}: {last_line[:100]}...", flush=True)
                
                # Look for debug lines that might give us clues
                if "DEBUG:" in line:
                    f.write(f"DEBUG line {line_num}: {line}\n")
                elif "Episodes completed:" in line:
                    f.write(f"Episode completion line {line_num}: {line}\n")
                elif "Exception" in line or "Error" in line:
                    f.write(f"Error line {line_num}: {line}\n")
            
            f.write(f"\nTotal JSON lines found: {len(all_json_lines)}\n")
            for i, json_line in enumerate(all_json_lines):
                f.write(f"JSON {i+1}: {json_line}\n")

        # Parse hand metrics if available
        hand_design_metrics = {}
        if hand_metrics_line:
            try:
                hand_metrics_cleaned = re.sub(r'\bNaN\b', 'null', hand_metrics_line)
                hand_metrics_cleaned = re.sub(r'\b(?:Infinity|-?Inf)\b', 'null', hand_metrics_cleaned)
                hand_metrics_payload = json.loads(hand_metrics_cleaned)
                hand_design_metrics = hand_metrics_payload.get("hand_design_metrics", {})
                print(f"    [parser] Successfully parsed hand metrics for {len(hand_design_metrics)} designs", flush=True)
                
                # Clamp negative average rewards to 0 so downstream logging/aggregation is non-negative
                for d_id, m in hand_design_metrics.items():
                    if "average_reward" in m:
                        try:
                            m["average_reward"] = max(0.0, float(m.get("average_reward", 0.0)))
                        except Exception:
                            m["average_reward"] = 0.0
                
                # Log hand metrics to file for debugging
                with open(parse_log_file, 'a') as f:
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("HAND DESIGN METRICS:\n")
                    for design_id, metrics in hand_design_metrics.items():
                        f.write(f"Design '{design_id}':\n")
                        f.write(f"  Average Reward: {metrics.get('average_reward', 'N/A')}\n")
                        f.write(f"  Success Rate: {metrics.get('success_rate', 'N/A')}\n")
                        f.write(f"  Episodes: {metrics.get('total_episodes', 'N/A')}\n")
                        f.write(f"  Environments: {len(metrics.get('env_ids', []))}\n")
                        
            except Exception as e:
                print(f"    [parser] Failed to parse hand metrics: {e}", flush=True)
                with open(parse_log_file, 'a') as f:
                    f.write(f"\nFailed to parse hand metrics: {e}\n")
                    f.write(f"Raw hand metrics line: {hand_metrics_line[:500]}...\n")

        # Parse main metrics (existing logic)
        js = final_line or last_line
        if not js:
            print("    [parser] No GHS_METRICS_JSON* line found; checking for fallback patterns...", flush=True)
            
            # Look for any JSON-like lines as fallback
            for line in lines:
                if '"episode_count"' in line and '"score"' in line:
                    print(f"    [parser] Found potential JSON fallback: {line[:100]}...", flush=True)
                    try:
                        # Try to extract just the JSON part
                        start_idx = line.find('{')
                        if start_idx >= 0:
                            js = line[start_idx:].strip()
                            print(f"    [parser] Extracted JSON: {js[:100]}...", flush=True)
                            break
                    except Exception as e:
                        print(f"    [parser] Fallback extraction failed: {e}", flush=True)
            
            if not js:
                print("    [parser] No metrics found anywhere. Checking for clues...", flush=True)
                
                # Save analysis of what we found
                with open(parse_log_file, 'a') as f:
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("NO JSON FOUND - Analysis:\n")
                    f.write(f"Lines containing 'Episodes completed': {sum(1 for line in lines if 'Episodes completed' in line)}\n")
                    f.write(f"Lines containing 'DEBUG': {sum(1 for line in lines if 'DEBUG' in line)}\n")
                    f.write(f"Lines containing 'Exception': {sum(1 for line in lines if 'Exception' in line)}\n")
                    f.write(f"Lines containing 'Error': {sum(1 for line in lines if 'Error' in line)}\n")
                    f.write(f"Lines containing 'timeout': {sum(1 for line in lines if 'timeout' in line.lower())}\n")
                    
                    # Show last 10 lines for clues
                    f.write("\nLast 10 lines of output:\n")
                    for i, line in enumerate(lines[-10:], start=len(lines)-10):
                        f.write(f"{i}: {line}\n")
                
                print(f"    [parser] Analysis saved to: {parse_log_file}")
                return 0.0

        # Clean up the JSON string
        js = re.sub(r'\bNaN\b', 'null', js)
        js = re.sub(r'\b(?:Infinity|-?Inf)\b', 'null', js)
        
        try:
            payload = json.loads(js)
            print(f"    [parser] Successfully parsed JSON payload", flush=True)
        except Exception as e:
            print(f"    [parser] JSON load failed. Raw js excerpt:\n    {js[:400]}...\n    Error: {e}", flush=True)
            return 0.0

        # Extract metrics from payload - handle both old and new formats
        metrics = payload.get("avg_metrics") or payload.get("metrics") or {}
        score = payload.get("score")
        
        print(f"    [parser] Payload keys: {list(payload.keys())}", flush=True)
        print(f"    [parser] Metrics keys: {list(metrics.keys()) if metrics else 'None'}", flush=True)
        print(f"    [parser] Direct score field: {score}", flush=True)

        # Store hand design metrics for later use (you can add this to the class)
        if hand_design_metrics:
            # You can store this in the evaluator instance for later processing
            if not hasattr(self, 'last_hand_design_metrics'):
                self.last_hand_design_metrics = {}
            self.last_hand_design_metrics[episode] = hand_design_metrics
            
            # Print summary of hand metrics
            print(f"    [parser] Hand design performance summary:", flush=True)
            for design_id, metrics in hand_design_metrics.items():
                avg_reward = metrics.get('average_reward', 0.0)
                success_rate = metrics.get('success_rate', 0.0)
                episodes = metrics.get('total_episodes', 0)
                print(f"      {design_id}: reward={avg_reward:.3f}, success={success_rate:.1%}, eps={episodes}", flush=True)
        
        # Use normalized reward
        for container in (payload, metrics):
            if isinstance(container, dict):
                if "norm_reward" in container:
                    v = float(container["norm_reward"])
                    print(f"    [parser] Using norm_reward: {v}", flush=True)
                    return v

        print("    [parser] No suitable score found, returning 0.0", flush=True)
        return 0.0

class CodesignGraphHeuristicSearch(GraphHeuristicSearch):
    """
    Graph Heuristic Search specialized for the codesign hand optimization.
    """
    
    def __init__(self, **kwargs):
        # Override default generator
        super().__init__(**kwargs)
        self.generator = HandDesignGenerator()
        # Evaluator will be set externally
        self.worst_design = None
        self.worst_reward = None
    
    def save_results(self, save_dir: str = "ghs_results"):
        """Save comprehensive results including design details and performance"""
        os.makedirs(save_dir, exist_ok=True)

        def _resolve_mode_and_codes(design: DesignGraph) -> Tuple[Optional[str], List[str]]:
            if not self.evaluator:
                return None, []

            key = getattr(design, "_effective_key", design.design_string)
            mode = self.evaluator.last_mode_by_design.get(key, self.evaluator.placement_mode)
            codes_csv = self.evaluator.last_codes_by_design.get(
                key,
                ",".join(self.evaluator.design_to_grammar_codes(design, placement_mode=mode)),
            )
            return mode, codes_csv.split(",")

        def _write_design_json(path: str, design: Optional[DesignGraph], reward: Optional[float]) -> None:
            if design is None:
                return

            mode, codes = _resolve_mode_and_codes(design)
            design_data = {
                "design_string": design.design_string,
                "reward": reward,
                "nodes": design.nodes,
                "edges": design.edges,
                "placement_mode": mode,
                "grammar_codes": codes,
                "total_iterations": len(self.seen_designs),
            }
            with open(path, "w") as f:
                json.dump(design_data, f, indent=2)

        _write_design_json(os.path.join(save_dir, "best_design.json"), self.best_design, self.best_reward)
        _write_design_json(os.path.join(save_dir, "worst_design.json"), self.worst_design, self.worst_reward)

        # Save lookup table for analysis (JSON-safe: expand tuple keys)
        lt_export = [
            {"design_string": k[0], "group": k[1], "value": float(v)}
            for k, v in self.lookup_table.items()
        ]
        with open(os.path.join(save_dir, "lookup_table.json"), 'w') as f:
            json.dump(lt_export, f, indent=2)

        # Save CSV of evaluated designs and rewards
        csv_path = os.path.join(save_dir, "rewards.csv")
        rows = {}
        for d in self.seen_designs:
            dkey = getattr(d, "_effective_key", d.design_string)
            if dkey in rows:
                continue
            group = getattr(d, "_effective_group", "default")
            reward = float(self.lookup_table.get((dkey, group), float("nan")))
            codes_csv = self.evaluator.last_codes_by_design.get(
                dkey,
                ",".join(self.evaluator.design_to_grammar_codes(
                    d,
                    placement_mode=self.evaluator.last_mode_by_design.get(dkey, self.evaluator.placement_mode)
                ))
            )
            rows[dkey] = (dkey, reward, codes_csv)

            
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["design_string", "reward", "codes"])
            writer.writerows(rows.values())

        # Save model checkpoint
        torch.save({
            'value_network_state': self.value_network.state_dict(),
            'best_reward': self.best_reward,
            'num_designs_evaluated': len(self.seen_designs)
        }, os.path.join(save_dir, "final_model.pt"))

        print(f"Results saved to {save_dir}")

def main():
    
    cmd_env = os.environ.copy()

    def prepend_env_path(key: str, *entries: str) -> None:
        existing = cmd_env.get(key, "")
        parts = []
        seen = set()

        for entry in [*entries, *existing.split(os.pathsep)]:
            if not entry:
                continue
            if entry in seen:
                continue
            seen.add(entry)
            parts.append(entry)

        if parts:
            cmd_env[key] = os.pathsep.join(parts)

    # remove conda + cuda-toolkit stubs from the search path
    for k in ("CONDA_PREFIX","CONDA_DEFAULT_ENV","_CE_CONDA","_CE_M",
            "MAMBA_DEFAULT_ENV","MAMBA_ROOT_PREFIX","CUDA_HOME","CUDA_PATH"):
        cmd_env.pop(k, None)

    # put driver lib dirs first; keep existing LD_LIBRARY_PATH after
    driver_libs = ["/usr/lib/x86_64-linux-gnu", "/usr/local/nvidia/lib64"]
    existing = cmd_env.get("LD_LIBRARY_PATH", "")
    cmd_env["LD_LIBRARY_PATH"] = ":".join([*driver_libs, existing]) if existing else ":".join(driver_libs)

    # torchvision warning is harmless, but you can silence the extension load attempt
    cmd_env["TORCHVISION_DISABLE_IMAGE"] = "1"
    
    # strongly pin the real driver to avoid stubs being picked first
    cmd_env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    
    parser = argparse.ArgumentParser(description="Run Graph Heuristic Search for hand design optimization using pre-trained PPO checkpoint")
    parser.add_argument("--iterations", type=int, default=25, help="Number of search iterations")
    parser.add_argument("--candidates", type=int, default=6, help="Number of candidate designs per iteration")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Epsilon for epsilon-greedy selection")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Number of evaluation episodes per design")
    parser.add_argument("--eval-envs", type=int, default=16, help="Number of environments for evaluation")
    parser.add_argument("--task", type=str, default="Codesign-Reorientation-Direct-v0", help="Base task name")
    parser.add_argument("--output-dir", type=str, default="ghs_outputs", help="Output directory")
    parser.add_argument("--play-script", type=str, default="../IsaacLab/scripts/rl_games/play.py", help="Path to play script")
    parser.add_argument("--build-script", type=str, default="../Generation/converters/batch_build_rand_hands2.sh", help="Path to hand build script")
    parser.add_argument(
        "--isaac-python",
        type=str,
        default=None,
        help=(
            "Python executable for Isaac Lab subprocesses (play.py, URDF->USD conversion). "
            "If omitted, auto-detected from env vars/common IsaacLab paths."
        ),
    )
    parser.add_argument("--resume-checkpoint", type=str, help="Resume GHS from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--focus-metric", type=str, default="rotation", choices=["rotation", "reward", "success"], 
                       help="Primary metric to optimize for")
    parser.add_argument("--placement-mode", type=str, default="symmetric",
                       choices=["symmetric", "anthro-top-heavy", "both"],
                       help="Finger placement mode for builder. 'both' randomly picks per design.")
    parser.add_argument(
        "--eval-groups",
        type=str,
        default="all",
        help=(
            "Evaluation groups: 'all' or comma-separated subset of "
            "{sym3,sym4,sym5,anth21,anth27,anth33}. "
            "Examples: 'sym5' or 'sym3,sym4,sym5'."
        ),
    )
    parser.add_argument(
        "--shuffle-groups",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to reshuffle group order at cycle boundaries in multi-group mode.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="codesign-ghs", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/org (optional)")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--tags", type=str, nargs="*", default=None, help="Optional W&B tags")

    parser.add_argument(
        "--video-every-n-cycles",
        type=int,
        default=None,
        help="Save videos every N cycles (1 cycle = len(eval_groups) iterations). None = no videos",
    )
    parser.add_argument("--video-length", type=int, default=800, 
                    help="Length of recorded videos in steps")
    
    parser.add_argument("--eval-timeout", type=int, default=40, help="Evaluation timeout in seconds")

    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    try:
        eval_groups = parse_eval_groups(args.eval_groups)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"No seed provided; using randomly chosen seed: {args.seed}")
    
    # Set all random seeds if provided
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
        # Make PyTorch deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for subprocess seeds (Isaac Lab, etc.)
        cmd_env["PYTHONHASHSEED"] = str(args.seed)
    
    # Validate required files exist
    if not os.path.exists(args.play_script):
        print(f"Error: Play script not found: {args.play_script}")
        return 1
    
    if not os.path.exists(args.build_script):
        print(f"Error: Build script not found: {args.build_script}")
        return 1

    try:
        play_script_path = Path(args.play_script).resolve()
        if len(play_script_path.parents) >= 3:
            isaaclab_root = play_script_path.parents[2]
            prepend_env_path(
                "PYTHONPATH",
                str(isaaclab_root),
                str(isaaclab_root / "source" / "codesign"),
            )
    except Exception as exc:
        print(f"[WARN] Could not extend PYTHONPATH from play script: {exc}")

    try:
        isaac_python = resolve_isaac_python(
            play_script=args.play_script,
            env=cmd_env,
            preferred=args.isaac_python,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    cmd_env["CODESIGN_ISAAC_PYTHON"] = isaac_python
    # Preserve the venv launcher path instead of resolving the symlink back to
    # the system interpreter. The generation/build scripts need the runner
    # environment's installed modules, not /usr/bin/python3.x.
    cmd_env["CODESIGN_PYTHON"] = sys.executable
    cmd_env["CODESIGN_RUNNER_PYTHON"] = sys.executable
    xacrodoc_exe = shutil.which("xacrodoc")
    if xacrodoc_exe:
        cmd_env["CODESIGN_XACRODOC"] = xacrodoc_exe
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Isaac Python: {isaac_python}")
    print(f"Optimizing for: {args.focus_metric}")
    print(f"Evaluation groups: {', '.join(g.value for g in eval_groups)}")
    print(f"Group reshuffling: {'enabled' if args.shuffle_groups and len(eval_groups) > 1 else 'disabled'}")
    
    # --- W&B INIT ---
    wandb_run = None
    if args.wandb and wandb is not None:
        wandb_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name or f"ghs_rotation_{int(time.time())}",
            config=vars(args),
            reinit=True,
        )
        # Respect WANDB_MODE=offline if user sets it
        wandb_run = wandb.init(**wandb_kwargs)
        if args.tags:
            wandb_run.tags = args.tags
    elif args.wandb and wandb is None:
        print("[WARN] --wandb was set but 'wandb' is not installed. `pip install wandb` to enable logging.")

    if wandb_run:
        # iteration metrics share iter/index
        wandb.define_metric("iter/index")
        wandb.define_metric("iter/*", step_metric="iter/index")

        # cycle metrics share cycle/index
        wandb.define_metric("cycle/index")
        for group in eval_groups:
            wandb.define_metric(f"cycle/{group.value}/*", step_metric="cycle/index")
    
    # Initialize search
    ghs = CodesignGraphHeuristicSearch(
        num_iterations=args.iterations,
        num_candidates=args.candidates,
        epsilon=args.epsilon,
        device=device,
        learning_rate=3e-4,
        opt_iterations=200,
        batch_size=args.batch_size,
        debug=args.debug,
    )
    
    # Configure evaluator
    ghs.evaluator = CodesignPPOEvaluator(
        build_script=args.build_script,
        play_script=args.play_script,
        output_dir=args.output_dir,
        isaac_python=isaac_python,
        eval_episodes=args.eval_episodes,
        eval_envs=args.eval_envs,
        eval_timeout=args.eval_timeout,
        env=cmd_env,
        placement_mode=args.placement_mode,
        debug=args.debug,
        wandb_run=wandb_run,
        video_every_n_cycles=args.video_every_n_cycles,
        video_length=args.video_length, 
        seed=args.seed,
        eval_groups=eval_groups,
        shuffle_groups=args.shuffle_groups,
    )
    
    # Load checkpoint if provided
    if args.resume_checkpoint:
        print(f"Resuming from GHS checkpoint: {args.resume_checkpoint}")
        ghs.load_checkpoint(args.resume_checkpoint)
    
    # Run the search
    print("\nStarting Graph Heuristic Search with pre-trained PPO evaluation...")
    print(f"Will evaluate {args.iterations * args.candidates} total designs")
    print(f"Estimated time: {args.iterations * args.candidates * args.eval_episodes * 5 / 60:.1f} minutes")
    
    best_design, best_score = ghs.run()
    
    if hasattr(ghs.evaluator, 'finalize_incomplete_cycle'):
        ghs.evaluator.finalize_incomplete_cycle()
    
    # Track worst design from lookup table
    if ghs.lookup_table:
        # Filter out invalid scores (<=0.001) before finding worst
        valid_lookup_items = [(k, v) for k, v in ghs.lookup_table.items() if v > 0.001]
        
        if valid_lookup_items:
            (worst_key, worst_score) = min(valid_lookup_items, key=lambda x: x[1])
            worst_design_string, worst_group = worst_key
            for design in ghs.seen_designs:
                if getattr(design, "_effective_key", design.design_string) == worst_design_string \
                and getattr(design, "_effective_group", "default") == worst_group:
                    ghs.worst_design = design
                    ghs.worst_reward = worst_score
                    break
        else:
            print("⚠️  No valid worst design found (all scores <= 0.001)")

    # Save results
    results_dir = os.path.join(args.output_dir, "final_results")
    ghs.save_results(results_dir)

    def _resolve_build_context(
        design: DesignGraph, *, label: str
    ) -> Tuple[DesignGraph, Optional[HandGroup], str]:
        """Resolve canonical design + group/mode context for final URDF generation."""
        prepared = _canonicalize_with_effective_metadata(design)
        group_value = getattr(prepared, "_effective_group", getattr(prepared, "evaluated_group", None))

        if group_value:
            group = HandGroup(group_value)
            if label == "best":
                print(f"Using evaluated group: {group.value}")
            _, mode, _, _ = ghs.evaluator.effective_codes_for(prepared, group)
            return prepared, group, mode

        key = getattr(prepared, "_effective_key", prepared.design_string)
        raw_key = prepared.design_string
        mode = (
            ghs.evaluator.last_mode_by_design.get(key)
            or ghs.evaluator.last_mode_by_design.get(raw_key)
            or ghs.evaluator.placement_mode
        )
        if label == "best":
            print(f"⚠️  No evaluated group found, using fallback mode: {mode}")
        return prepared, None, mode
    
    # Generate final URDF for best design
    if best_design is not None:
        print("\nGenerating final URDF for best design...")
        best_design, best_group, best_mode = _resolve_build_context(best_design, label="best")
        if best_group is not None:
            final_urdf = ghs.evaluator.generate_hand_urdf(best_design, group=best_group)
        else:
            final_urdf = ghs.evaluator.generate_hand_urdf(best_design, placement_mode=best_mode)
        
        
        if final_urdf:
            # Copy to results directory
            final_urdf_dest = os.path.join(results_dir, "best_design_hand.urdf")
            shutil.copy2(final_urdf, final_urdf_dest)

            builder_dir = os.path.dirname(final_urdf)
            urdf_stem = Path(final_urdf).stem

            print(f"[best-eval] URDF is: {final_urdf}")
            print(f"[best-eval] Looking for USD under: {builder_dir}")

            def _find_best_usd(builder_dir: str, urdf_stem: str) -> str | None:
                root = Path(builder_dir)
                # 1) local *.usd
                candidates = list(root.glob("*.usd"))
                # 2) any nested *.usd (handles per-model subfolders)
                if not candidates:
                    candidates = list(root.glob("**/*.usd"))

                if not candidates:
                    return None

                # Prefer one whose stem matches the URDF stem
                for p in candidates:
                    if p.stem == urdf_stem:
                        return str(p)

                # Otherwise just take the first
                return str(candidates[0])

            # First try to find an existing USD (maybe builder already produced one)
            best_usd_path = _find_best_usd(builder_dir, urdf_stem)

            if not best_usd_path or not os.path.exists(best_usd_path):
                print("\nConverting best-design URDF to USD...")
                python_exe = ghs.evaluator.isaac_python
                convert_cmd = [
                    python_exe,
                    "source/codesign/codesign/utils/convert_urdf_usd.py",
                    "--in-dir", os.path.abspath(builder_dir),
                    "--out-dir", os.path.abspath(builder_dir),
                    "--merge-joints",
                    "--headless",
                ]
                print("  " + " ".join(convert_cmd))
                try:
                    convert_env = cmd_env.copy()
                    convert_env.pop("LD_PRELOAD", None)
                    run_managed_subprocess(
                        convert_cmd,
                        check=True,
                        cwd="../IsaacLab",
                        env=convert_env,
                        label="best-design USD conversion",
                    )
                except subprocess.CalledProcessError as e:
                    print(f"[WARN] USD conversion for best design failed: {e}")
                    best_usd_path = None
                else:
                    # Re-scan after conversion
                    best_usd_path = _find_best_usd(builder_dir, urdf_stem)

            if best_usd_path and os.path.exists(best_usd_path):
                print(f"Best design USD: {best_usd_path}")
            else:
                print("[WARN] No USD file found for best design; final video will use default hand")
                best_usd_path = None


            
            # Save assets for the best design (single-design build)
            best_design_id = Path(final_urdf).stem
            best_candidates = [os.path.dirname(final_urdf)]

            ghs.evaluator._copy_assets_for_design(
                best_candidates, results_dir, best_design_id
            )
            print(f"Best design URDF saved to: {final_urdf_dest}")
            
            preset_for_best = "anthro" if best_mode == "anthro-top-heavy" else "symmetric"
            
            # Determine the correct checkpoint based on the best design's configuration
            best_codes = ghs.evaluator.design_to_grammar_codes(best_design, placement_mode=best_mode)
            active_count = sum(1 for c in best_codes if c)

            if best_group is None:
                if best_mode == "symmetric":
                    best_group = {
                        3: HandGroup.SYM3,
                        4: HandGroup.SYM4,
                        5: HandGroup.SYM5,
                    }.get(active_count, HandGroup.SYM5)
                else:
                    # Fallback when anthro group metadata is unavailable.
                    best_group = HandGroup.ANTH21

            best_checkpoint = ghs.evaluator._checkpoint_for_group(best_group)
            
            # Make a dedicated folder for final best-design videos
            video_dir = os.path.join(results_dir, "best_eval_videos")
            os.makedirs(video_dir, exist_ok=True)
            video_name = f"best_{best_group.value}_{int(time.time())}"

            # --- 1) Run final best-design eval NOW (16 envs + video) ---
            python_exe = ghs.evaluator.isaac_python
            final_cmd = [
                python_exe,
                args.play_script,
                f"--task={args.task}",
                "--num_envs=16",
                "--headless",
                f"--checkpoint={best_checkpoint}",
                "--run_name=ghs_best_design_eval",
                f"--max_episodes={args.eval_episodes}",
                f"--timeout={args.eval_timeout}",
                "--video",
                "--video_length", str(args.video_length),
                "--video_folder", video_dir,
                "--video_name", video_name,
                "--use_unexported",
            ]

            print("Running final best-design evaluation with video:")
            print("  " + " ".join(final_cmd))

            # Use a child env that also tells the task which hand USD to load
            final_env = cmd_env.copy()
            if best_usd_path and os.path.exists(best_usd_path):
                final_env["CODESIGN_HAND_USD_PATH_0"] = os.path.abspath(best_usd_path)
                final_env["CODESIGN_HAND_GROUP"] = best_group.value
                final_env["CODESIGN_JOINT_PRESET"] = preset_for_best
                final_env["CODESIGN_GHS_EVALUATION"] = "1"
            else:
                print("[WARN] No best_usd_path set; using default task hand asset")

            try:
                run_managed_subprocess(
                    final_cmd,
                    check=True,
                    env=final_env,
                    label="final best-design evaluation",
                )
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Final best-design eval failed: {e}")


            print(f"Final best-design video should be at: {video_dir}/{video_name}.mp4")

            # --- 2) Upload the video to W&B (now that it exists) ---
            if wandb_run and wandb:
                best_codes_csv = ",".join(
                    ghs.evaluator.design_to_grammar_codes(best_design, placement_mode=best_mode)
                )
                final_scores = [best_score]
                final_codes_by_idx = {0: best_codes_csv}

                ghs.evaluator._upload_videos_to_wandb(
                    video_dir=video_dir,
                    iteration_id="final_best_eval",
                    scores=final_scores,
                    codes_by_idx=final_codes_by_idx,
                    group=best_group,
                )

            # --- 3) Also create a reusable shell script for later reruns ---
            best_usd_for_script = (
                os.path.abspath(best_usd_path)
                if best_usd_path
                else "<REPLACE_WITH_ABSOLUTE_USD_PATH>"
            )

            eval_script_content = f'''#!/bin/bash
# Evaluation script for best design found by GHS
# Design: {best_design.design_string}
# Score: {best_score:.4f}
# Grammar codes: {ghs.evaluator.design_to_grammar_codes(best_design, placement_mode=best_mode)}
# Hand group: {best_group.value}

echo "Evaluating best design from Graph Heuristic Search..."
echo "Design: {best_design.design_string}"
echo "Score: {best_score:.4f}"
echo "Hand group: {best_group.value}"
echo "Checkpoint: {best_checkpoint}"

python_exe="${{CODESIGN_ISAAC_PYTHON:-{ghs.evaluator.isaac_python}}}"

export CODESIGN_HAND_USD_PATH_0="{best_usd_for_script}"
export CODESIGN_HAND_GROUP="{best_group.value}"
export CODESIGN_JOINT_PRESET="{preset_for_best}"
export CODESIGN_GHS_EVALUATION="1"

VIDEO_DIR="{video_dir}"
VIDEO_NAME="{video_name}"

mkdir -p "$VIDEO_DIR"

$python_exe {args.play_script} \\
    --task={args.task} \\
    --num_envs=16 \\
    --checkpoint={best_checkpoint} \\
    --run_name="ghs_best_design_eval_rerun" \\
    --video \\
    --video_length {args.video_length} \\
    --video_folder "$VIDEO_DIR" \\
    --video_name "$VIDEO_NAME" \\
    --headless \\
    --use_unexported

echo "Evaluation complete!"
echo "Final best-design video saved to: $VIDEO_DIR/$VIDEO_NAME.mp4"
'''

            eval_script_path = os.path.join(results_dir, "evaluate_best_design.sh")
            with open(eval_script_path, 'w') as f:
                f.write(eval_script_content)
            os.chmod(eval_script_path, 0o755)

            print(f"Evaluation script created: {eval_script_path}")
            print(f"Best design will use checkpoint: {best_checkpoint}")

        else:
            print("Could not generate final URDF for best design")
    
    # Generate final URDF for worst design (for analysis)
    if ghs.worst_design is not None:
        print("\nGenerating final URDF for worst design (for analysis)...")
        worst_design, worst_group, worst_mode = _resolve_build_context(ghs.worst_design, label="worst")
        if worst_group is not None:
            worst_urdf = ghs.evaluator.generate_hand_urdf(worst_design, group=worst_group)
        else:
            worst_urdf = ghs.evaluator.generate_hand_urdf(worst_design, placement_mode=worst_mode)
        
        if worst_urdf:
            # Copy to results directory
            worst_urdf_dest = os.path.join(results_dir, "worst_design_hand.urdf")
            shutil.copy2(worst_urdf, worst_urdf_dest)
            
            # Save assets for the worst design
            worst_design_id = Path(worst_urdf).stem
            worst_candidates = [os.path.dirname(worst_urdf)]

            ghs.evaluator._copy_assets_for_design(
                worst_candidates, results_dir, worst_design_id
            )
            print(f"Worst design URDF saved to: {worst_urdf_dest}")
            print(f"Worst design score: {ghs.worst_reward:.4f}")
        else:
            print("Could not generate final URDF for worst design")
    
    print(f"\n{'='*60}")
    print("GRAPH HEURISTIC SEARCH COMPLETE")
    print(f"{'='*60}")
    
    if best_design is not None:
        print(f"Best design: {best_design.design_string}")
        print(f"Best {args.focus_metric} score: {best_score:.4f}")
        # 1) Final printout
        print(f"Grammar codes: {ghs.evaluator.design_to_grammar_codes(best_design, placement_mode=best_mode)}")
        print(f"Results saved to: {results_dir}")
        
        if 'final_urdf' in locals() and final_urdf:
            print(f"Best design URDF: {final_urdf_dest}")
            print(f"Test best design: {eval_script_path}")
    
    if ghs.worst_design is not None:
        print(f"\nWorst design: {ghs.worst_design.design_string}")
        print(f"Worst {args.focus_metric} score: {ghs.worst_reward:.4f}")
        if 'worst_urdf' in locals() and worst_urdf:
            print(f"Worst design URDF: {worst_urdf_dest}")
    
    # --- W&B ARTIFACTS & FINISH ---
    if wandb_run and wandb:
        try:
            # 1. Log final results artifact
            artifact = wandb.Artifact(name=f"ghs_results_{int(time.time())}", type="results")
            if os.path.isdir(results_dir):
                artifact.add_dir(results_dir)
            iter_csv = os.path.join(args.output_dir, "final_results", "iteration_group_stats.csv")
            iter_jsonl = os.path.join(args.output_dir, "final_results", "iteration_group_stats.jsonl")
            if os.path.exists(iter_csv):  artifact.add_file(iter_csv)
            if os.path.exists(iter_jsonl): artifact.add_file(iter_jsonl)
            # upload cycle-level stats
            cycle_csv = os.path.join(args.output_dir, "final_results", "cycle_group_stats.csv")
            cycle_jsonl = os.path.join(args.output_dir, "final_results", "cycle_group_stats.jsonl")
            if os.path.exists(cycle_csv):
                artifact.add_file(cycle_csv)
            if os.path.exists(cycle_jsonl):
                artifact.add_file(cycle_jsonl)
            
            wandb_run.log_artifact(artifact)
            
            # 2. Log GHS checkpoints
            checkpoint_dir = os.path.join(args.output_dir, "ghs_checkpoint")
            if os.path.isdir(checkpoint_dir):
                checkpoint_artifact = wandb.Artifact(
                    name=f"ghs_checkpoint_{int(time.time())}", 
                    type="model"
                )
                checkpoint_artifact.add_dir(checkpoint_dir)
                wandb_run.log_artifact(checkpoint_artifact)
                print(f"✅ Uploaded GHS checkpoints to W&B")
            
            # 3. Log complete output directory (optional - can be large)
            # Uncomment if you want full iteration history
            # full_output_artifact = wandb.Artifact(
            #     name=f"ghs_full_output_{int(time.time())}", 
            #     type="dataset"
            # )
            # full_output_artifact.add_dir(args.output_dir)
            # wandb_run.log_artifact(full_output_artifact)
            # print(f"✅ Uploaded full output directory to W&B")
            
        except Exception as e:
            print(f"[WARN] Failed to log W&B artifact: {e}")

        wandb_run.finish()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
