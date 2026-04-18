#!/usr/bin/env python3
"""Write metadata JSON for a generated hand configuration."""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Write hand metadata JSON.")
    parser.add_argument("--metadata-file", required=True, help="Output metadata JSON path.")
    parser.add_argument("--hand-name", required=True, help="Generated hand name/id.")
    parser.add_argument(
        "--actual-finger-count",
        required=True,
        type=int,
        help="Number of active fingers on the generated palm.",
    )
    parser.add_argument("--palm-limit-lo", required=True, type=float, help="Palm joint lower limit (rad).")
    parser.add_argument("--palm-limit-hi", required=True, type=float, help="Palm joint upper limit (rad).")
    parser.add_argument(
        "--frames-json",
        required=True,
        help="Palm frame JSON path (optional at runtime; defaults used if missing).",
    )
    parser.add_argument(
        "--codes",
        nargs=5,
        required=True,
        metavar=("F1", "F2", "F3", "F4", "F5"),
        help='Five encoded finger codes (e.g. "220" or "000").',
    )
    parser.add_argument(
        "--fingertip-types",
        nargs=5,
        required=True,
        metavar=("F1", "F2", "F3", "F4", "F5"),
        help="Five fingertip type strings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    frame_data = None
    if os.path.exists(args.frames_json):
        with open(args.frames_json, "r", encoding="utf-8") as f:
            frame_data = json.load(f)

    metadata = {
        "schema_version": "1.1",
        "hand_id": args.hand_name,
        "fingers": [],
    }

    for i in range(5):
        finger_idx = i + 1
        code = args.codes[i]
        fingertip_type = args.fingertip_types[i]
        is_present = i < args.actual_finger_count and code != "000"

        finger_data = {
            "finger_idx": finger_idx,
            "present": is_present,
            "fingertip_type": fingertip_type,
        }

        if is_present:
            if frame_data and i < len(frame_data.get("frames", [])):
                frame = frame_data["frames"][i]
                pos = frame["pos"]
                quat = frame["quat"]
                finger_data["base_pose"] = {
                    "pos_palm_m": [pos["x"], pos["y"], pos["z"]],
                    "quat_wxyz": [quat["w"], quat["x"], quat["y"], quat["z"]],
                }
            else:
                finger_data["base_pose"] = {
                    "pos_palm_m": [0.0, 0.0, 0.0],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                }

            g1 = int(code[0]) if code != "000" else 0
            g2 = int(code[1]) if code != "000" else 0
            s = int(code[2]) if code != "000" else 0

            joints = []
            joints.append(
                {
                    "role": "palm",
                    "present": True,
                    "axis_xyz": [0, 0, 1],
                    "limit_lo": args.palm_limit_lo,
                    "limit_hi": args.palm_limit_hi,
                }
            )

            if s >= 3:
                joints.append(
                    {
                        "role": "base",
                        "present": True,
                        "grammar": g1,
                        "axis_xyz": [0, 1, 0],
                        "limit_lo": 0.0,
                        "limit_hi": 1.570796,
                    }
                )
            else:
                joints.append({"role": "base", "present": False})

            if s >= 2:
                joints.append(
                    {
                        "role": "middle",
                        "present": True,
                        "grammar": g2,
                        "axis_xyz": [0, 1, 0],
                        "limit_lo": 0.0,
                        "limit_hi": 1.570796,
                    }
                )
            else:
                joints.append({"role": "middle", "present": False})

            joints.append(
                {
                    "role": "end",
                    "present": True,
                    "grammar": 0,
                    "axis_xyz": [0, 1, 0],
                    "limit_lo": 0.0,
                    "limit_hi": 1.570796,
                }
            )

            finger_data["joints"] = joints
        else:
            finger_data["joints"] = [
                {"role": "palm", "present": False},
                {"role": "base", "present": False},
                {"role": "middle", "present": False},
                {"role": "end", "present": False},
            ]

        metadata["fingers"].append(finger_data)

    with open(args.metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated metadata: {args.metadata_file}")


if __name__ == "__main__":
    main()
