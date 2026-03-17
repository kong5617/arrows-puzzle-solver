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
AXIS_TOLERANCE = 20  # px — perpendicular-axis alignment threshold (~2% of 1080px width)

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

_DETECT_PROMPT = (
    "You are analyzing a mobile puzzle screenshot.\n"
    "Return ONLY a JSON array. No prose, no markdown fences.\n"
    'Each element: {"x": <int>, "y": <int>, "direction": "up"|"down"|"left"|"right"}\n'
    "x and y are the pixel coordinates of the arrow's center in the ORIGINAL image.\n"
    "Include every arrow visible in the puzzle area. Exclude UI (header, stars, hint buttons)."
)

_RETRY_PROMPT_TEMPLATE = (
    "Your previous response was not valid JSON. Return ONLY the raw JSON array with no "
    "surrounding text. Previous response: {raw}"
)

_RELEVANT_VARIABLES = (
    '&lt;StringArray sr=""&gt;'
    '&lt;_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES0&gt;%ailastbounds\n'
    'Last Bounds\n'
    'Bounds (left,top,right,bottom) of the item that the action last interacted with'
    '&lt;/_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES0&gt;'
    '&lt;_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES1&gt;%ailastcoordinates\n'
    'Last Coordinates\n'
    'Center coordinates (x,y) of the item that the action last interacted with'
    '&lt;/_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES1&gt;'
    '&lt;_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES2&gt;%err\n'
    'Error Code\n'
    'Only available if you select &amp;lt;b&amp;gt;Continue Task After Error&amp;lt;/b&amp;gt; and the action ends in error'
    '&lt;/_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES2&gt;'
    '&lt;_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES3&gt;%errmsg\n'
    'Error Message\n'
    'Only available if you select &amp;lt;b&amp;gt;Continue Task After Error&amp;lt;/b&amp;gt; and the action ends in error'
    '&lt;/_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES3&gt;'
    '&lt;/StringArray&gt;'
)

def validate_arrows(arrows: list, img_w: int, img_h: int) -> list[dict]:
    """Validate and return arrows list. Raises ValueError on bad data, SystemExit if empty."""
    if not arrows:
        print(
            "Error: No arrows detected. The model returned an empty list.\n"
            "This may mean the screenshot is not of the expected puzzle type,\n"
            "or the model incorrectly excluded all arrows as UI elements.",
            file=sys.stderr,
        )
        sys.exit(1)

    valid_directions = {"up", "down", "left", "right"}
    for i, arrow in enumerate(arrows):
        for key in ("x", "y", "direction"):
            if key not in arrow:
                raise ValueError(f"Arrow {i} missing required key '{key}': {arrow}")
        if arrow["direction"] not in valid_directions:
            raise ValueError(
                f"Unexpected direction '{arrow['direction']}' in arrow {i}. "
                f"Expected: up, down, left, right. "
                f"Try re-running or adjusting the prompt."
            )
        if not (0 <= arrow["x"] <= img_w) or not (0 <= arrow["y"] <= img_h):
            raise ValueError(
                f"Arrow {i} coordinates ({arrow['x']}, {arrow['y']}) out of bounds "
                f"for image size {img_w}x{img_h}."
            )
    return arrows


def detect_arrows(image_path: str, api_key: str, output_dir: str | None = None) -> list[dict]:
    """Call Claude Vision to detect arrows. Returns validated list of arrow dicts."""
    img = Image.open(image_path)
    img_w, img_h = img.size

    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    suffix = Path(image_path).suffix.lower().lstrip(".")
    media_type = "image/png" if suffix == "png" else "image/jpeg"

    client = anthropic.Anthropic(api_key=api_key)

    def call_api(messages):
        return client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=_DETECT_PROMPT,
            messages=messages,
        )

    initial_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": "List all arrows in this puzzle screenshot."},
            ],
        }
    ]

    response = call_api(initial_messages)
    raw = response.content[0].text.strip()

    try:
        arrows = json.loads(raw)
    except json.JSONDecodeError:
        # Retry once with the failed response for self-correction
        retry_messages = initial_messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _RETRY_PROMPT_TEMPLATE.format(raw=raw)},
        ]
        retry_response = call_api(retry_messages)
        raw = retry_response.content[0].text.strip()
        try:
            arrows = json.loads(raw)
        except json.JSONDecodeError:
            # Save error file and exit
            base = Path(image_path).stem
            err_dir = output_dir or str(Path(image_path).parent)
            err_path = os.path.join(err_dir, f"{base}_api_error.txt")
            with open(err_path, "w") as f:
                f.write(raw)
            print(f"Error: API returned invalid JSON after retry. Raw response saved to {err_path}", file=sys.stderr)
            sys.exit(1)

    return validate_arrows(arrows, img_w, img_h)


def blocks_arrow(a: dict, b: dict) -> bool:
    """Return True if arrow b is in the travel path of arrow a."""
    if a is b:
        return False
    direction = a["direction"]
    if direction == "right":
        return abs(a["y"] - b["y"]) <= AXIS_TOLERANCE and b["x"] > a["x"]
    if direction == "left":
        return abs(a["y"] - b["y"]) <= AXIS_TOLERANCE and b["x"] < a["x"]
    if direction == "up":
        return abs(a["x"] - b["x"]) <= AXIS_TOLERANCE and b["y"] < a["y"]
    if direction == "down":
        return abs(a["x"] - b["x"]) <= AXIS_TOLERANCE and b["y"] > a["y"]
    return False


def solve_order(arrows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Return (ordered_tap_list, stuck_arrows).
    ordered_tap_list: arrows in safe tap order.
    stuck_arrows: non-empty only if a cycle prevents full resolution.
    """
    remaining = list(arrows)
    ordered = []

    while remaining:
        # Recompute unblocked arrows fresh each iteration
        unblocked = [
            a for a in remaining
            if not any(blocks_arrow(a, b) for b in remaining if b is not a)
        ]
        if not unblocked:
            # No progress possible — cycle detected
            return ordered, remaining
        # Add all unblocked arrows (stable order preserves input order within batch)
        ordered.extend(unblocked)
        unblocked_ids = {id(a) for a in unblocked}
        remaining = [r for r in remaining if id(r) not in unblocked_ids]

    return ordered, []


def build_autoinput_action(act_idx: int, x: int, y: int) -> str:
    """Build an AutoInput Tap action XML string for coordinates (x, y)."""
    uid = str(uuid.uuid4())
    blurb = f"Actions To Perform: click(point,{x}\\,{y})\nNot In AutoInput: true\nNot In Tasker: true\nSeparator: ,\nCheck Millis: 1000"
    # Build parameters JSON — json.dumps will escape \ to \\ in the JSON string
    params_dict = {
        "_action": f"click(point,{x}\\,{y})",
        "_additionalOptions": {"checkMs": "1000", "separator": ",", "withCoordinates": False},
        "_whenToPerformAction": {"notInAutoInput": True, "notInTasker": True},
        "generatedValues": {},
    }
    params_json = json.dumps(params_dict, separators=(",", ":"))

    return f"""\t\t<Action sr="act{act_idx}" ve="7">
\t\t\t<code>107361459</code>
\t\t\t<Bundle sr="arg0">
\t\t\t\t<Vals sr="val">
\t\t\t\t\t<EnableDisableAccessibilityService>&lt;null&gt;</EnableDisableAccessibilityService>
\t\t\t\t\t<EnableDisableAccessibilityService-type>java.lang.String</EnableDisableAccessibilityService-type>
\t\t\t\t\t<Password>&lt;null&gt;</Password>
\t\t\t\t\t<Password-type>java.lang.String</Password-type>
\t\t\t\t\t<com.twofortyfouram.locale.intent.extra.BLURB>{blurb}</com.twofortyfouram.locale.intent.extra.BLURB>
\t\t\t\t\t<com.twofortyfouram.locale.intent.extra.BLURB-type>java.lang.String</com.twofortyfouram.locale.intent.extra.BLURB-type>
\t\t\t\t\t<net.dinglisch.android.tasker.JSON_ENCODED_KEYS>parameters</net.dinglisch.android.tasker.JSON_ENCODED_KEYS>
\t\t\t\t\t<net.dinglisch.android.tasker.JSON_ENCODED_KEYS-type>java.lang.String</net.dinglisch.android.tasker.JSON_ENCODED_KEYS-type>
\t\t\t\t\t<net.dinglisch.android.tasker.RELEVANT_VARIABLES>{_RELEVANT_VARIABLES}</net.dinglisch.android.tasker.RELEVANT_VARIABLES>
\t\t\t\t\t<net.dinglisch.android.tasker.RELEVANT_VARIABLES-type>[Ljava.lang.String;</net.dinglisch.android.tasker.RELEVANT_VARIABLES-type>
\t\t\t\t\t<net.dinglisch.android.tasker.extras.VARIABLE_REPLACE_KEYS>parameters plugininstanceid plugintypeid </net.dinglisch.android.tasker.extras.VARIABLE_REPLACE_KEYS>
\t\t\t\t\t<net.dinglisch.android.tasker.extras.VARIABLE_REPLACE_KEYS-type>java.lang.String</net.dinglisch.android.tasker.extras.VARIABLE_REPLACE_KEYS-type>
\t\t\t\t\t<net.dinglisch.android.tasker.subbundled>true</net.dinglisch.android.tasker.subbundled>
\t\t\t\t\t<net.dinglisch.android.tasker.subbundled-type>java.lang.Boolean</net.dinglisch.android.tasker.subbundled-type>
\t\t\t\t\t<parameters>{params_json}</parameters>
\t\t\t\t\t<parameters-type>java.lang.String</parameters-type>
\t\t\t\t\t<plugininstanceid>{uid}</plugininstanceid>
\t\t\t\t\t<plugininstanceid-type>java.lang.String</plugininstanceid-type>
\t\t\t\t\t<plugintypeid>com.joaomgcd.autoinput.intent.IntentActionv2</plugintypeid>
\t\t\t\t\t<plugintypeid-type>java.lang.String</plugintypeid-type>
\t\t\t\t</Vals>
\t\t\t</Bundle>
\t\t\t<Str sr="arg1" ve="3">com.joaomgcd.autoinput</Str>
\t\t\t<Str sr="arg2" ve="3">com.joaomgcd.autoinput.activity.ActivityConfigActionv2</Str>
\t\t\t<Int sr="arg3" val="60"/>
\t\t\t<Int sr="arg4" val="1"/>
\t\t</Action>"""


def build_wait_action(act_idx: int, delay_ms: int) -> str:
    return f"""\t\t<Action sr="act{act_idx}" ve="7">
\t\t\t<code>30</code>
\t\t\t<Int sr="arg0" val="{delay_ms}"/>
\t\t\t<Int sr="arg1" val="0"/>
\t\t</Action>"""


_FALLBACK_COMMENT_TEMPLATE = (
    "\t\t<!-- Fallback: Tasker Pointer Input (accessibility, Tasker 5.9+) — verify code before use\n"
    "\t\t<Action sr=\"act{idx}\" ve=\"7\">\n"
    "\t\t\t<code>993</code>\n"
    "\t\t\t<Str sr=\"arg0\" ve=\"3\">tap</Str>\n"
    "\t\t\t<Int sr=\"arg1\" val=\"{x}\"/>\n"
    "\t\t\t<Int sr=\"arg2\" val=\"{y}\"/>\n"
    "\t\t</Action>\n"
    "\t\t-->"
)


def build_tasker_xml(task_name: str, tap_order: list[dict], delay_ms: int) -> str:
    """Build a complete Tasker XML task string."""
    now_ms = int(datetime.now().timestamp() * 1000)
    actions = []
    act_idx = 0
    for arrow in tap_order:
        actions.append(build_autoinput_action(act_idx, arrow["x"], arrow["y"]))
        # Fallback comment immediately after each tap action
        actions.append(_FALLBACK_COMMENT_TEMPLATE.format(idx=act_idx, x=arrow["x"], y=arrow["y"]))
        act_idx += 1
        actions.append(build_wait_action(act_idx, delay_ms))
        act_idx += 1
    actions_xml = "\n".join(actions)
    return f"""<TaskerData sr="" dvi="1" tv="6.6.20">
\t<Task sr="task1">
\t\t<cdate>{now_ms}</cdate>
\t\t<edate>{now_ms}</edate>
\t\t<id>1</id>
\t\t<nme>{task_name}</nme>
{actions_xml}
\t</Task>
</TaskerData>"""


def draw_visualization(image_path: str, tap_order: list[dict], out_path: str) -> None: ...


def main() -> None: ...

if __name__ == "__main__":
    main()
