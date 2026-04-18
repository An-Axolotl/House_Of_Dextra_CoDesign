"""Utilities for selecting CoDesign hand assets from the RL-Games CLI."""

from __future__ import annotations

import argparse
import os


def add_hand_selection_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for choosing one or more hand assets."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--hand",
        type=str,
        default=None,
        help="Single hand name, hand directory, or USD path to use.",
    )
    group.add_argument(
        "--hands",
        nargs="+",
        default=None,
        help="Multiple hand names, hand directories, or USD paths to use in parallel.",
    )
    parser.add_argument(
        "--hand-assets-dir",
        type=str,
        default=None,
        help="Asset root to scan when resolving --hand/--hands by name.",
    )


def _clear_hand_selection_env() -> None:
    for key in ("CODESIGN_HAND_NAME", "CODESIGN_HAND_NAMES", "CODESIGN_HAND_USD_PATH"):
        os.environ.pop(key, None)

    for index in range(100):
        os.environ.pop(f"CODESIGN_HAND_USD_PATH_{index}", None)


def apply_hand_selection_args(args: argparse.Namespace) -> None:
    """Apply hand-selection CLI arguments to the environment before task import."""
    hand_assets_dir = getattr(args, "hand_assets_dir", None)
    if hand_assets_dir:
        os.environ["CODESIGN_HAND_ASSETS_DIR"] = os.path.expanduser(os.path.expandvars(hand_assets_dir))

    hand = getattr(args, "hand", None)
    hands = getattr(args, "hands", None)
    if not hand and not hands:
        return

    _clear_hand_selection_env()

    if hand:
        os.environ["CODESIGN_HAND_NAME"] = hand
    elif hands:
        os.environ["CODESIGN_HAND_NAMES"] = ",".join(hands)
