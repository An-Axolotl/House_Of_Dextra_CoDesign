#!/usr/bin/env python3
"""
Logging/printing helpers for CodesignPPOEvaluator.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import time
from typing import Dict, List, Optional, TYPE_CHECKING

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None

if TYPE_CHECKING:
    from hand_groups import HandGroup
    from graph_heuristic_search import DesignGraph


class EvaluatorLoggingMixin:
    def print_hand_design_performance_summary(self) -> None:
        """Print a summary of all hand design performances from recent evaluations."""
        if not hasattr(self, "last_hand_design_metrics") or not self.last_hand_design_metrics:
            print("    [hand-summary] No hand design metrics available")
            return

        print("\n    [hand-summary] Performance Summary from Recent Evaluations:")
        print(f"    {'='*60}")

        # Aggregate metrics across all recent episodes.
        all_designs = {}
        for _, hand_metrics in self.last_hand_design_metrics.items():
            for design_id, metrics in hand_metrics.items():
                if design_id not in all_designs:
                    all_designs[design_id] = {
                        "total_reward": 0.0,
                        "total_episodes": 0,
                        "total_successes": 0,
                        "avg_rewards": [],
                        "success_rates": [],
                    }

                design_data = all_designs[design_id]
                design_data["total_reward"] += metrics.get("total_reward", 0.0)
                design_data["total_episodes"] += metrics.get("total_episodes", 0)
                design_data["total_successes"] += metrics.get("success_count", 0)
                design_data["avg_rewards"].append(metrics.get("average_reward", 0.0))
                design_data["success_rates"].append(metrics.get("success_rate", 0.0))

        # Sort by average performance.
        sorted_designs = sorted(
            all_designs.items(),
            key=lambda x: sum(x[1]["avg_rewards"]) / len(x[1]["avg_rewards"]) if x[1]["avg_rewards"] else 0,
            reverse=True,
        )

        for rank, (design_id, data) in enumerate(sorted_designs, 1):
            avg_reward = sum(data["avg_rewards"]) / len(data["avg_rewards"]) if data["avg_rewards"] else 0.0
            avg_success_rate = (
                sum(data["success_rates"]) / len(data["success_rates"]) if data["success_rates"] else 0.0
            )

            print(f"    {rank}. {design_id}")
            print(f"       Average Reward: {avg_reward:.3f}")
            print(f"       Success Rate: {avg_success_rate:.1%}")
            print(f"       Total Episodes: {data['total_episodes']}")
            print(f"       Evaluations: {len(data['avg_rewards'])}")

    def _log_cycle_stats(self, cycle_num: int):
        """Log aggregated statistics for each group at the end of a cycle."""
        if not self.current_cycle_group_data:
            return

        # Store count before resetting.
        num_groups_logged = len(self.current_cycle_group_data)

        for group_name, data in self.current_cycle_group_data.items():
            scores = data.get("scores", [])
            rewards = data.get("rewards", [])

            avg_score = self._safe_mean(scores)
            avg_reward = self._safe_mean(rewards)
            best_score = self._safe_max(scores)
            best_reward = self._safe_max(rewards)
            worst_score = self._safe_min(scores)
            worst_reward = self._safe_min(rewards)

            row = {
                "cycle_num": cycle_num,
                "group": group_name,
                "avg_score": avg_score,
                "avg_reward": avg_reward,
                "best_score": best_score,
                "best_reward": best_reward,
                "worst_score": worst_score,
                "worst_reward": worst_reward,
                "n_designs": len(scores),
                "time_s": int(time.time()),
            }
            self.cycle_stats.append(row)

            # CSV (append header once).
            header = [
                "cycle_num",
                "group",
                "avg_score",
                "avg_reward",
                "best_score",
                "best_reward",
                "worst_score",
                "worst_reward",
                "n_designs",
                "time_s",
            ]
            file_exists = os.path.exists(self.cycle_stats_path_csv)
            with open(self.cycle_stats_path_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            # JSONL for flexible analysis.
            with open(self.cycle_stats_path_jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")

            # Log to wandb with proper step metric.
            if self.wandb_run and wandb:
                scalars = {
                    "cycle/index": cycle_num,
                    f"cycle/{group_name}/avg_score": avg_score,
                    f"cycle/{group_name}/avg_reward": avg_reward,
                    f"cycle/{group_name}/best_score": best_score,
                    f"cycle/{group_name}/best_reward": best_reward,
                    f"cycle/{group_name}/worst_score": worst_score,
                    f"cycle/{group_name}/worst_reward": worst_reward,
                    f"cycle/{group_name}/n_designs": len(scores),
                }
                scalars = {k: v for k, v in scalars.items() if v is not None}
                self.wandb_run.log(scalars)

                # Also log histograms per group.
                if scores:
                    self.wandb_run.log(
                        {
                            "cycle/index": cycle_num,
                            f"cycle/{group_name}/scores_hist": wandb.Histogram(scores),
                        }
                    )
                if rewards:
                    self.wandb_run.log(
                        {
                            "cycle/index": cycle_num,
                            f"cycle/{group_name}/rewards_hist": wandb.Histogram(rewards),
                        }
                    )

        # Increment cycle counter after logging all groups.
        self._cycle_count += 1

        # Reset current cycle data after logging.
        self.current_cycle_group_data = {}

        print(
            f"\n✅ Logged cycle {cycle_num} statistics for {num_groups_logged} groups "
            f"(wandb step: {self._cycle_count - 1})"
        )

    def _wandb_log_iteration_payload(
        self,
        *,
        iter_no: int,
        group: "HandGroup",
        scores: List[float],
        rewards: List[float],
        codes_by_idx: Dict[int, str],
        mode_by_idx: Dict[int, str],
        usd_paths: List[Optional[str]],
        research_data: List[dict],
    ) -> None:
        if not self.wandb_run or not wandb:
            return

        valid_scores = [s for s in scores if isinstance(s, (int, float)) and s > -999.0]
        valid_rewards = [r for r in rewards if isinstance(r, (int, float))]

        # Per-design table.
        cols = [
            "iteration_idx",
            "group",
            "design_idx",
            "codes",
            "placement_mode",
            "training_score",
            "average_reward",
            "turns_per_second",
            "rps_from_omega",
            "raw_omega_z_deg_per_sec",
            "usd_path",
        ]
        table = wandb.Table(columns=cols)
        for di, score in enumerate(scores):
            raw = research_data[di] if di < len(research_data) else {}
            table.add_data(
                iter_no,
                group.value,
                di,
                codes_by_idx.get(di, ""),
                mode_by_idx.get(di, ""),
                score,
                raw.get("average_reward", None),
                raw.get("turns_per_second", None),
                raw.get("rps_from_omega", None),
                raw.get("raw_omega_z_deg_per_sec", None),
                usd_paths[di] if di < len(usd_paths) else None,
            )

        payload = {
            "iter/index": iter_no,
            "iter/group": group.value,
            "iter/scores_hist": wandb.Histogram(valid_scores) if valid_scores else None,
            "iter/rewards_hist": wandb.Histogram(valid_rewards) if valid_rewards else None,
            "iter/n_candidates": len(scores),
            "iter/table": table,
        }
        self.wandb_run.log(payload)

    def _upload_videos_to_wandb(
        self,
        video_dir: str,
        iteration_id: str,
        scores: List[float],
        codes_by_idx: Dict[int, str],
        group: "HandGroup",
    ) -> None:
        """Upload the iteration video to wandb with metadata, organized by cycle and group."""
        del codes_by_idx  # Kept in signature for call-site compatibility.
        if not self.wandb_run or not wandb:
            return

        video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

        if not video_files:
            print(f"[WARN] No videos found in {video_dir}")
            return

        # Extract cycle number from video_dir path.
        cycle_match = re.search(r"cycle_(\d+)", video_dir)
        cycle_num = int(cycle_match.group(1)) if cycle_match else 0

        print(f"[INFO] Uploading {len(video_files)} video(s) for {group.value} (cycle {cycle_num}) to wandb...")

        # Upload each video (should typically be just one per iteration).
        for video_path in video_files:
            video_name = os.path.basename(video_path)

            # Get aggregate stats for this iteration.
            valid_scores = [s for s in scores if isinstance(s, (int, float)) and s > -999.0]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            best_score = max(valid_scores) if valid_scores else 0.0
            worst_score = min(valid_scores) if valid_scores else 0.0

            # Create caption with iteration-level metadata.
            caption = f"Cycle {cycle_num} | {group.value} | {iteration_id}"
            caption += f" | Avg: {avg_score:.3f}, Best: {best_score:.3f}, Worst: {worst_score:.3f}"
            caption += f" | {len(scores)} designs"

            # Upload video with group and cycle organized key.
            try:
                self.wandb_run.log(
                    {
                        f"videos/cycle_{cycle_num:03d}/{group.value}/{video_name}": wandb.Video(
                            video_path,
                            caption=caption,
                            fps=30,
                            format="mp4",
                        ),
                        "iter/index": self._iter_idx,
                    }
                )
                print(f"✅ Uploaded video: {video_name}")
            except Exception as exc:
                print(f"[WARN] Failed to upload video {video_name}: {exc}")

        print(f"✅ Completed video upload for {group.value} in cycle {cycle_num}")

    def _log_iteration_stats(
        self,
        *,
        iteration_idx: int,
        iteration_id: str,
        group: "HandGroup",
        avg_score: Optional[float],
        avg_reward: Optional[float],
        n_candidates: int,
        best_score_of_iter: Optional[float],
        best_reward_of_iter: Optional[float],
        best_score_so_far: Optional[float],
        best_reward_so_far: Optional[float],
        worst_score_of_iter: Optional[float] = None,
        worst_reward_of_iter: Optional[float] = None,
        worst_score_so_far: Optional[float] = None,
        worst_reward_so_far: Optional[float] = None,
    ):
        row = {
            "iteration_idx": iteration_idx,
            "iteration_id": iteration_id,
            "group": group.value,
            "avg_score": avg_score,
            "avg_reward": avg_reward,
            "best_score_of_iter": best_score_of_iter,
            "best_reward_of_iter": best_reward_of_iter,
            "best_score_so_far": best_score_so_far,
            "best_reward_so_far": best_reward_so_far,
            "worst_score_of_iter": worst_score_of_iter,
            "worst_reward_of_iter": worst_reward_of_iter,
            "worst_score_so_far": worst_score_so_far,
            "worst_reward_so_far": worst_reward_so_far,
            "n_candidates": n_candidates,
            "time_s": int(time.time()),
        }
        self.iteration_stats.append(row)

        # CSV (append header once).
        header = [
            "iteration_idx",
            "iteration_id",
            "group",
            "avg_score",
            "avg_reward",
            "best_score_of_iter",
            "best_reward_of_iter",
            "best_score_so_far",
            "best_reward_so_far",
            "worst_score_of_iter",
            "worst_reward_of_iter",
            "worst_score_so_far",
            "worst_reward_so_far",
            "n_candidates",
            "time_s",
        ]
        file_exists = os.path.exists(self.stats_path_csv)
        with open(self.stats_path_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        # JSONL for flexible analysis.
        with open(self.stats_path_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")

        if self.wandb_run and wandb:
            scalars = {
                "iter/index": iteration_idx,
                "iter/avg_score": avg_score,
                "iter/avg_reward": avg_reward,
                "iter/best_score_of_iter": best_score_of_iter,
                "iter/best_reward_of_iter": best_reward_of_iter,
                "iter/best_score_so_far": best_score_so_far,
                "iter/best_reward_so_far": best_reward_so_far,
                "iter/worst_score_of_iter": worst_score_of_iter,
                "iter/worst_reward_of_iter": worst_reward_of_iter,
                "iter/worst_score_so_far": worst_score_so_far,
                "iter/worst_reward_so_far": worst_reward_so_far,
                "iter/group_id": group.value,
            }
            scalars = {k: v for k, v in scalars.items() if v is not None}
            self.wandb_run.log(scalars)

            # Keep bests and worsts in run summary as well.
            if best_score_so_far is not None:
                self.wandb_run.summary["best_score_so_far"] = best_score_so_far
            if best_reward_so_far is not None:
                self.wandb_run.summary["best_reward_so_far"] = best_reward_so_far
            if worst_score_so_far is not None:
                self.wandb_run.summary["worst_score_so_far"] = worst_score_so_far
            if worst_reward_so_far is not None:
                self.wandb_run.summary["worst_reward_so_far"] = worst_reward_so_far

    def print_design_details(
        self,
        design: "DesignGraph",
        finger_codes: List[str],
        finger_tips: List[str],
        mode: str,
        thumb_slot: Optional[int] = None,
    ):
        """Print detailed information about a design's configuration."""
        print("    🔍 Design Details:")
        print(f"       Mode: {mode}")
        if thumb_slot is not None:
            print(f"       Thumb slot: {thumb_slot}")
        print(f"       Design string: {design.design_string}")
        print("       Fingers (1-5):")

        for i in range(1, 6):
            if i >= len(design.nodes):
                print(f"         Finger {i}: MISSING NODE")
                continue

            node = design.nodes[i]
            code = finger_codes[i - 1] if i - 1 < len(finger_codes) else ""
            tip = finger_tips[i - 1] if i - 1 < len(finger_tips) else "standard"

            servo_count = node.get("servo_count", 0)
            g1 = node.get("grammar_1_count", 0)
            g2 = node.get("grammar_2_count", 0)
            is_ghost = servo_count < 2
            is_terminal = node.get("is_terminal", False)

            status = "GHOST" if is_ghost else ("FINALIZED" if is_terminal else "ACTIVE")

            print(f"         Finger {i} [{status}]:")
            print(f"           - Servos: {servo_count}")
            print(f"           - Grammar 1: {g1}")
            print(f"           - Grammar 2: {g2}")
            print(f"           - Fingertip: {tip}")
            print(f"           - Code: '{code}'")
