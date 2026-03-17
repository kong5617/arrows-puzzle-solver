#!/usr/bin/env python3
"""Arrows Puzzle Solver — process a screenshot, generate Tasker XML."""

import argparse
import base64
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import anthropic
import cv2
import numpy as np
from PIL import Image

CLAUDE_MODEL = "claude-opus-4-6"
AXIS_TOLERANCE = 20  # px — perpendicular-axis alignment threshold (~10% of 1080px width)

DIRECTION_COLORS_BGR = {
    "right": (0, 200, 0),
    "left":  (200, 0, 0),
    "up":    (0, 0, 200),
    "down":  (0, 200, 200),
}

DIRECTION_VECTORS = {
    "right": (1, 0),
    "left":  (-1, 0),
    "up":    (0, -1),
    "down":  (0, 1),
}

# Populated in later tasks
def detect_arrows(image_path: str, api_key: str) -> list[dict]: ...
def validate_arrows(arrows: list, img_w: int, img_h: int) -> list[dict]: ...
def blocks_arrow(a: dict, b: dict) -> bool: ...
def solve_order(arrows: list[dict]) -> list[dict]: ...
def build_autoinput_action(act_idx: int, x: int, y: int) -> str: ...
def build_wait_action(act_idx: int, delay_ms: int) -> str: ...
def build_tasker_xml(task_name: str, tap_order: list[dict], delay_ms: int) -> str: ...
def draw_visualization(image_path: str, tap_order: list[dict], out_path: str) -> None: ...
def main() -> None: ...

if __name__ == "__main__":
    main()
