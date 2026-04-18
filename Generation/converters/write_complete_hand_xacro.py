#!/usr/bin/env python3
"""Compose a complete hand URDF xacro from generated palm/finger artifacts."""

import argparse
import glob
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom


def parse_args():
    parser = argparse.ArgumentParser(description="Write complete hand URDF xacro.")
    parser.add_argument("--palm-xacro", required=True, help="Generated palm xacro file.")
    parser.add_argument("--frames-json", required=True, help="Generated palm frame JSON file.")
    parser.add_argument("--out", required=True, help="Output hand xacro path.")
    parser.add_argument("--vroot", required=True, help="Per-hand build root (contains f1..f5 xacros).")
    parser.add_argument("--hand-name", required=True, help="Generated hand name.")
    parser.add_argument("--actual-finger-count", required=True, type=int, help="Number of active fingers.")
    return parser.parse_args()


def quat_to_rpy(w, x, y, z):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    args = parse_args()

    with open(args.frames_json, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    palm_tree = ET.parse(args.palm_xacro)
    palm_root = palm_tree.getroot()

    palm_link = palm_root.find(".//link[@name='palm']")
    if palm_link is None:
        raise SystemExit("No palm link found in generated palm xacro")

    palm_base_path = os.path.splitext(
        os.path.join(os.path.dirname(args.frames_json), f"{args.hand_name}_palm.obj")
    )[0]
    palm_part_files = sorted(glob.glob(f"{palm_base_path}_part*.obj"))
    if not palm_part_files:
        palm_part_files = [f"{palm_base_path}.obj"]

    for collision in list(palm_link.findall("./collision")):
        palm_link.remove(collision)

    for visual in palm_link.findall(".//visual/geometry/mesh"):
        visual_filename = os.path.basename(palm_part_files[0])
        visual.set("filename", f"meshes/{visual_filename}")

    for i, part_file in enumerate(palm_part_files):
        part_filename = os.path.basename(part_file)
        collision = ET.SubElement(palm_link, "collision", {"name": f"palm_collision_{i}"})
        collision_geom = ET.SubElement(collision, "geometry")
        ET.SubElement(collision_geom, "mesh", {"filename": f"meshes/{part_filename}"})

    robot = ET.Element("robot", {"xmlns:xacro": "http://www.ros.org/wiki/xacro", "name": "generated_hand"})

    macro_names = []
    for i in range(1, 6):
        ET.SubElement(robot, "xacro:include", {"filename": f"f{i}/random_finger.urdf.xacro"})
        finger_xacro_path = os.path.join(args.vroot, f"f{i}", "random_finger.urdf.xacro")
        with open(finger_xacro_path, "r", encoding="utf-8") as f:
            finger_content = f.read()

        match = re.search(r'<xacro:macro name="(random_finger_f\d+(?:_\w+)?)"', finger_content)
        if not match:
            raise SystemExit(f"Could not find macro name in {finger_xacro_path}")
        macro_names.append(match.group(1))

    base_link = palm_root.find(".//link[@name='base_link']")
    if base_link is not None:
        robot.append(base_link)
    else:
        base_link = ET.SubElement(robot, "link", {"name": "base_link"})
        base_inertial = ET.SubElement(base_link, "inertial")
        ET.SubElement(base_inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(base_inertial, "mass", {"value": "0.5"})
        ET.SubElement(
            base_inertial,
            "inertia",
            {"ixx": "0.01", "ixy": "0", "ixz": "0", "iyy": "0.01", "iyz": "0", "izz": "0.01"},
        )

    robot.append(palm_link)

    base_palm_joint = palm_root.find(".//joint[@name='base_to_palm']")
    if base_palm_joint is not None:
        robot.append(base_palm_joint)
    else:
        joint = ET.SubElement(robot, "joint", {"name": "base_to_palm", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": "base_link"})
        ET.SubElement(joint, "child", {"link": "palm"})
        ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    for i in range(args.actual_finger_count):
        frame = frame_data["frames"][i]
        finger_num = i + 1
        pos = frame["pos"]
        quat = frame["quat"]
        roll, pitch, yaw = quat_to_rpy(quat["w"], quat["x"], quat["y"], quat["z"])

        macro_call = ET.SubElement(robot, f"xacro:{macro_names[i]}", {"prefix": f"f{finger_num}_", "parent": "palm"})
        ET.SubElement(
            macro_call,
            "origin",
            {"xyz": f"{pos['x']:.6f} {pos['y']:.6f} {pos['z']:.6f}", "rpy": f"{roll:.6f} {pitch:.6f} {yaw:.6f}"},
        )

    for i in range(args.actual_finger_count, 5):
        finger_num = i + 1
        macro_call = ET.SubElement(robot, f"xacro:{macro_names[i]}", {"prefix": f"f{finger_num}_", "parent": "palm"})
        ET.SubElement(macro_call, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    rough = ET.tostring(robot, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    lines = pretty.split("\n")
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write("\n".join(lines))

    print(f"Generated complete hand xacro: {args.out}")


if __name__ == "__main__":
    main()
