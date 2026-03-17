# Arrows Puzzle Solver Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI Python script that takes an Arrows puzzle screenshot, calls Claude Vision to detect arrows, solves the tap order, and outputs a Tasker-importable XML task.

**Architecture:** Single script `solve_arrows.py` with four focused helper functions (`detect_arrows`, `solve_order`, `build_tasker_xml`, `draw_visualization`) called in sequence by `main()`. All XML is built via string templates (not ElementTree) to avoid double-escaping issues with the complex AutoInput bundle format.

**Tech Stack:** Python 3.8+, `anthropic`, `Pillow`, `numpy`, `opencv-python`, `pytest`

---

## File Map

| File | Purpose |
|---|---|
| `solve_arrows.py` | CLI entry point + all pipeline logic |
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | `pytest` + `pytest-mock` |
| `tests/test_solver.py` | Tests for `solve_order()` and `blocks_arrow()` |
| `tests/test_xml.py` | Tests for `build_tasker_xml()` and `build_autoinput_action()` |
| `tests/test_detect.py` | Tests for `detect_arrows()` and `validate_arrows()` with mocked API |
| `tests/test_viz.py` | Tests for `draw_visualization()` |
| `tests/conftest.py` | Shared fixtures |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `tests/conftest.py`
- Create: `solve_arrows.py` (skeleton only)

- [ ] **Step 1: Create `requirements.txt`**

```
anthropic>=0.40.0
Pillow>=10.0
numpy>=1.24
opencv-python>=4.8
```

- [ ] **Step 2: Create `requirements-dev.txt`**

```
pytest>=7.0
pytest-mock>=3.10
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

- [ ] **Step 4: Create `tests/conftest.py`** (empty for now)

```python
# shared fixtures added in later tasks
```

- [ ] **Step 5: Create skeleton `solve_arrows.py`**

```python
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
```

- [ ] **Step 6: Verify tests directory structure and run (empty suite passes)**

```bash
pytest tests/ -v
```
Expected: `no tests ran` (or 0 passed), no errors.

- [ ] **Step 7: Commit**

```bash
git init  # if not already a repo
git add solve_arrows.py requirements.txt requirements-dev.txt tests/conftest.py
git commit -m "chore: project scaffold for arrows puzzle solver"
```

---

## Task 2: Solver Logic (TDD)

**Files:**
- Create: `tests/test_solver.py`
- Modify: `solve_arrows.py` — implement `blocks_arrow()` and `solve_order()`

The solver is pure Python with no dependencies — easiest to build first.

- [ ] **Step 1: Write failing tests for `blocks_arrow()`**

Create `tests/test_solver.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from solve_arrows import blocks_arrow, solve_order

# --- blocks_arrow tests ---

def make_arrow(x, y, direction):
    return {"x": x, "y": y, "direction": direction}

def test_blocks_right_same_row():
    a = make_arrow(100, 200, "right")
    b = make_arrow(300, 200, "right")   # same row, to the right
    assert blocks_arrow(a, b) is True

def test_does_not_block_right_left_of_shooter():
    a = make_arrow(300, 200, "right")
    b = make_arrow(100, 200, "right")   # same row, but to the LEFT of a
    assert blocks_arrow(a, b) is False

def test_blocks_left():
    a = make_arrow(300, 200, "left")
    b = make_arrow(100, 200, "up")      # to the left of a, same row
    assert blocks_arrow(a, b) is True

def test_blocks_up():
    a = make_arrow(200, 300, "up")
    b = make_arrow(200, 100, "right")   # above a, same column
    assert blocks_arrow(a, b) is True

def test_blocks_down():
    a = make_arrow(200, 100, "down")
    b = make_arrow(200, 300, "left")    # below a, same column
    assert blocks_arrow(a, b) is True

def test_tolerance_within():
    """Arrow within AXIS_TOLERANCE on perpendicular axis counts as blocking."""
    a = make_arrow(100, 200, "right")
    b = make_arrow(300, 215, "up")      # 15px off on y — within 20px tolerance
    assert blocks_arrow(a, b) is True

def test_tolerance_outside():
    """Arrow beyond AXIS_TOLERANCE does not block."""
    a = make_arrow(100, 200, "right")
    b = make_arrow(300, 225, "up")      # 25px off on y — beyond 20px tolerance
    assert blocks_arrow(a, b) is False

def test_arrow_does_not_block_itself():
    a = make_arrow(100, 200, "right")
    assert blocks_arrow(a, a) is False
```

- [ ] **Step 2: Run tests — all should FAIL**

```bash
pytest tests/test_solver.py -v
```
Expected: FAIL — `blocks_arrow` is not yet implemented (returns `...`).

- [ ] **Step 3: Implement `blocks_arrow()` in `solve_arrows.py`**

Replace the `def blocks_arrow(...)` stub:

```python
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
```

- [ ] **Step 4: Run `blocks_arrow` tests — all should PASS**

```bash
pytest tests/test_solver.py -k "block" -v
```
Expected: 7 PASS.

- [ ] **Step 5: Write failing tests for `solve_order()`**

Append to `tests/test_solver.py`:

```python
# --- solve_order tests ---

def test_single_arrow():
    arrows = [make_arrow(100, 200, "right")]
    result = solve_order(arrows)
    assert len(result) == 1
    assert result[0]["x"] == 100

def test_two_independent_arrows():
    """Two arrows on different rows — order doesn't matter but both returned."""
    a = make_arrow(100, 100, "right")
    b = make_arrow(100, 300, "right")
    result = solve_order([a, b])
    assert len(result) == 2

def test_chain_dependency():
    """Arrow at x=100 is blocked by arrow at x=300 (same row, right direction).
    Arrow at x=300 must be tapped first."""
    a = make_arrow(100, 200, "right")  # blocked by b
    b = make_arrow(300, 200, "up")     # not blocked by anything
    result = solve_order([a, b])
    assert result.index(b) < result.index(a)

def test_longer_chain():
    """a→right blocked by b, b→right blocked by c. Order must be c, b, a."""
    a = make_arrow(100, 200, "right")
    b = make_arrow(200, 200, "right")
    c = make_arrow(300, 200, "up")
    result = solve_order([a, b, c])
    # c first, then b, then a
    assert result[0] is c
    assert result[1] is b
    assert result[2] is a

def test_recomputes_per_iteration():
    """After removing a blocker, a previously-blocked arrow becomes available."""
    # a points right, blocked by b
    # b points up, not blocked by anything
    # After b is removed, a should be included
    a = make_arrow(100, 200, "right")
    b = make_arrow(300, 200, "up")
    result = solve_order([a, b])
    assert b in result
    assert a in result

def test_cycle_detection():
    """Two arrows pointing at each other — neither can exit. Returns partial order + raises."""
    a = make_arrow(100, 200, "right")   # blocked by b
    b = make_arrow(300, 200, "left")    # blocked by a
    result, stuck = solve_order([a, b])
    assert len(stuck) == 2
    assert len(result) == 0
```

Wait — the cycle detection changes the return type. We need `solve_order` to return a tuple `(ordered, stuck)` always, not just on cycles. Update the earlier tests accordingly:

Replace the simpler test cases to unpack the tuple:

```python
def test_single_arrow():
    arrows = [make_arrow(100, 200, "right")]
    ordered, stuck = solve_order(arrows)
    assert len(ordered) == 1
    assert len(stuck) == 0

def test_two_independent_arrows():
    a = make_arrow(100, 100, "right")
    b = make_arrow(100, 300, "right")
    ordered, stuck = solve_order([a, b])
    assert len(ordered) == 2
    assert len(stuck) == 0

def test_chain_dependency():
    a = make_arrow(100, 200, "right")
    b = make_arrow(300, 200, "up")
    ordered, stuck = solve_order([a, b])
    assert ordered.index(b) < ordered.index(a)

def test_longer_chain():
    a = make_arrow(100, 200, "right")
    b = make_arrow(200, 200, "right")
    c = make_arrow(300, 200, "up")
    ordered, stuck = solve_order([a, b, c])
    assert ordered[0] is c
    assert ordered[1] is b
    assert ordered[2] is a

def test_recomputes_per_iteration():
    a = make_arrow(100, 200, "right")
    b = make_arrow(300, 200, "up")
    ordered, stuck = solve_order([a, b])
    assert b in ordered
    assert a in ordered
```

- [ ] **Step 6: Run `solve_order` tests — all should FAIL**

```bash
pytest tests/test_solver.py -k "solve" -v
```
Expected: FAIL.

- [ ] **Step 7: Implement `solve_order()` in `solve_arrows.py`**

```python
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
        for a in unblocked:
            ordered.append(a)
            remaining.remove(a)

    return ordered, []
```

- [ ] **Step 8: Run all solver tests — all should PASS**

```bash
pytest tests/test_solver.py -v
```
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add solve_arrows.py tests/test_solver.py
git commit -m "feat: implement arrow collision check and topological sort solver"
```

---

## Task 3: Tasker XML Generator (TDD)

**Files:**
- Create: `tests/test_xml.py`
- Modify: `solve_arrows.py` — implement `build_autoinput_action()`, `build_wait_action()`, `build_tasker_xml()`

XML is built via string templates, not ElementTree. This avoids double-escaping the pre-escaped XML entities in the AutoInput bundle.

- [ ] **Step 1: Write failing tests**

Create `tests/test_xml.py`:

```python
import sys, os, re, json, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from solve_arrows import build_autoinput_action, build_wait_action, build_tasker_xml

def test_autoinput_action_has_correct_code():
    xml = build_autoinput_action(0, 540, 800)
    assert "<code>107361459</code>" in xml

def test_autoinput_action_coordinates_in_blurb():
    xml = build_autoinput_action(0, 540, 800)
    assert "click(point,540\\,800)" in xml   # BLURB contains single backslash

def test_autoinput_action_coordinates_in_parameters():
    xml = build_autoinput_action(0, 540, 800)
    # The parameters element contains JSON where the backslash is JSON-escaped:
    # Python string has \\, → JSON string has \\\\, → XML text has \\,
    # So in the raw XML string we look for the literal four-character sequence: "\\,"
    assert 'click(point,540\\\\,800)' in xml

def test_autoinput_action_has_plugin_package():
    xml = build_autoinput_action(0, 100, 200)
    assert "com.joaomgcd.autoinput" in xml

def test_autoinput_action_has_unique_uuids():
    xml1 = build_autoinput_action(0, 100, 200)
    xml2 = build_autoinput_action(2, 300, 400)
    # Extract UUID from each
    def extract_uuid(s):
        m = re.search(r'<plugininstanceid>([^<]+)</plugininstanceid>', s)
        return m.group(1) if m else None
    assert extract_uuid(xml1) != extract_uuid(xml2)

def test_autoinput_action_index_in_sr():
    xml = build_autoinput_action(4, 100, 200)
    assert 'sr="act4"' in xml

def test_wait_action_has_correct_code():
    xml = build_wait_action(1, 800)
    assert "<code>30</code>" in xml

def test_wait_action_has_delay():
    xml = build_wait_action(1, 1200)
    assert 'val="1200"' in xml

def test_build_tasker_xml_structure():
    arrows = [{"x": 100, "y": 200, "direction": "right", "tap_index": 1}]
    xml = build_tasker_xml("Test Task", arrows, delay_ms=500)
    assert "<TaskerData" in xml
    assert "<Task " in xml
    assert "<nme>Test Task</nme>" in xml
    assert "<code>107361459</code>" in xml
    assert "<code>30</code>" in xml

def test_build_tasker_xml_action_count():
    """N arrows → 2N actions (tap + wait per arrow)."""
    arrows = [
        {"x": 100, "y": 200, "direction": "right", "tap_index": 1},
        {"x": 300, "y": 400, "direction": "up",    "tap_index": 2},
    ]
    xml = build_tasker_xml("Test", arrows, delay_ms=800)
    assert xml.count("<code>107361459</code>") == 2
    assert xml.count("<code>30</code>") == 2

def test_build_tasker_xml_has_fallback_comment():
    """Each tap action must have a commented-out Pointer Input fallback."""
    arrows = [{"x": 100, "y": 200, "direction": "right", "tap_index": 1}]
    xml = build_tasker_xml("Test", arrows, delay_ms=800)
    assert "<!-- Fallback:" in xml
    assert "code>993<" in xml  # inside comment, angle brackets not escaped

def test_build_tasker_xml_has_timestamps():
    arrows = [{"x": 100, "y": 200, "direction": "right", "tap_index": 1}]
    xml = build_tasker_xml("Test", arrows, delay_ms=800)
    assert "<cdate>" in xml
    assert "<edate>" in xml
```

- [ ] **Step 2: Run tests — all should FAIL**

```bash
pytest tests/test_xml.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement `build_autoinput_action()` in `solve_arrows.py`**

The RELEVANT_VARIABLES value is a static pre-escaped XML string, stored as a module-level constant:

```python
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
```

Then implement `build_autoinput_action()`:

```python
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
```

- [ ] **Step 4: Implement `build_wait_action()` and `build_tasker_xml()`**

```python
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
```

- [ ] **Step 5: Run all XML tests — all should PASS**

```bash
pytest tests/test_xml.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add solve_arrows.py tests/test_xml.py
git commit -m "feat: implement Tasker XML generator with AutoInput tap actions"
```

---

## Task 4: Arrow Detection (TDD with mocked API)

**Files:**
- Create: `tests/test_detect.py`
- Modify: `solve_arrows.py` — implement `validate_arrows()` and `detect_arrows()`

- [ ] **Step 1: Write failing tests**

Create `tests/test_detect.py`:

```python
import sys, os, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from solve_arrows import validate_arrows, detect_arrows

# --- validate_arrows tests ---

def test_validate_accepts_valid_input():
    arrows = [
        {"x": 100, "y": 200, "direction": "right"},
        {"x": 300, "y": 400, "direction": "up"},
    ]
    result = validate_arrows(arrows, img_w=1080, img_h=2376)
    assert len(result) == 2

def test_validate_rejects_bad_direction():
    arrows = [{"x": 100, "y": 200, "direction": "northeast"}]
    with pytest.raises(ValueError, match="Unexpected direction"):
        validate_arrows(arrows, img_w=1080, img_h=2376)

def test_validate_rejects_missing_key():
    arrows = [{"x": 100, "direction": "right"}]  # missing y
    with pytest.raises(ValueError, match="missing required key"):
        validate_arrows(arrows, img_w=1080, img_h=2376)

def test_validate_rejects_out_of_bounds():
    arrows = [{"x": 2000, "y": 200, "direction": "right"}]  # x > img_w
    with pytest.raises(ValueError, match="out of bounds"):
        validate_arrows(arrows, img_w=1080, img_h=2376)

def test_validate_rejects_empty_list():
    with pytest.raises(SystemExit):
        validate_arrows([], img_w=1080, img_h=2376)

# --- detect_arrows tests (mocked API) ---

def test_detect_arrows_returns_parsed_list(mocker):
    """Happy path: API returns valid JSON array."""
    mock_response = mocker.MagicMock()
    mock_response.content = [mocker.MagicMock(text='[{"x":100,"y":200,"direction":"right"}]')]
    mock_client = mocker.MagicMock()
    mock_client.messages.create.return_value = mock_response
    mocker.patch("solve_arrows.anthropic.Anthropic", return_value=mock_client)

    # Need a real image file for this test — use a tiny 10x10 white PNG
    from PIL import Image
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
    Image.new("RGB", (1080, 2376), "white").save(img_path)

    result = detect_arrows(img_path, api_key="test-key")
    assert len(result) == 1
    assert result[0]["direction"] == "right"

    os.unlink(img_path)

def test_detect_arrows_retries_on_bad_json(mocker):
    """If first response is invalid JSON, retry once."""
    bad_response = mocker.MagicMock()
    bad_response.content = [mocker.MagicMock(text="Here are the arrows: [...]")]
    good_response = mocker.MagicMock()
    good_response.content = [mocker.MagicMock(text='[{"x":500,"y":600,"direction":"left"}]')]
    mock_client = mocker.MagicMock()
    mock_client.messages.create.side_effect = [bad_response, good_response]
    mocker.patch("solve_arrows.anthropic.Anthropic", return_value=mock_client)

    from PIL import Image
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
    Image.new("RGB", (1080, 2376), "white").save(img_path)

    result = detect_arrows(img_path, api_key="test-key")
    assert mock_client.messages.create.call_count == 2
    assert result[0]["direction"] == "left"
    os.unlink(img_path)

def test_detect_arrows_exits_after_two_bad_responses(mocker, tmp_path):
    """If both attempts return invalid JSON, exits and writes error file."""
    bad_response = mocker.MagicMock()
    bad_response.content = [mocker.MagicMock(text="not json at all")]
    mock_client = mocker.MagicMock()
    mock_client.messages.create.return_value = bad_response
    mocker.patch("solve_arrows.anthropic.Anthropic", return_value=mock_client)

    from PIL import Image
    img_path = str(tmp_path / "test.png")
    Image.new("RGB", (1080, 2376), "white").save(img_path)

    with pytest.raises(SystemExit):
        detect_arrows(img_path, api_key="test-key", output_dir=str(tmp_path))

    assert (tmp_path / "test_api_error.txt").exists()
```

- [ ] **Step 2: Run tests — all should FAIL**

```bash
pytest tests/test_detect.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement `validate_arrows()` in `solve_arrows.py`**

```python
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
```

- [ ] **Step 4: Implement `detect_arrows()` in `solve_arrows.py`**

```python
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
```

- [ ] **Step 5: Run all detection tests — all should PASS**

```bash
pytest tests/test_detect.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add solve_arrows.py tests/test_detect.py
git commit -m "feat: implement arrow detection with Claude Vision API and retry logic"
```

---

## Task 5: Visualization

**Files:**
- Modify: `solve_arrows.py` — implement `draw_visualization()`

Visualization is a side-effect (writes a file). Tests are minimal: check file is created and has the right dimensions.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_detect.py` (or create `tests/test_viz.py`):

```python
# tests/test_viz.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from solve_arrows import draw_visualization
from PIL import Image
import tempfile, cv2

def test_visualization_creates_output_file(tmp_path):
    img_path = str(tmp_path / "puzzle.png")
    out_path = str(tmp_path / "puzzle_detected.png")
    Image.new("RGB", (1080, 2376), "white").save(img_path)
    arrows = [
        {"x": 200, "y": 300, "direction": "right", "tap_index": 1},
        {"x": 500, "y": 700, "direction": "up",    "tap_index": 2},
    ]
    draw_visualization(img_path, arrows, out_path)
    assert os.path.exists(out_path)

def test_visualization_preserves_dimensions(tmp_path):
    img_path = str(tmp_path / "puzzle.png")
    out_path = str(tmp_path / "puzzle_detected.png")
    Image.new("RGB", (1080, 2376), "white").save(img_path)
    arrows = [{"x": 200, "y": 300, "direction": "right", "tap_index": 1}]
    draw_visualization(img_path, arrows, out_path)
    out = cv2.imread(out_path)
    assert out.shape[0] == 2376
    assert out.shape[1] == 1080
```

- [ ] **Step 2: Run tests — should FAIL**

```bash
pytest tests/test_viz.py -v
```

- [ ] **Step 3: Implement `draw_visualization()` in `solve_arrows.py`**

```python
def draw_visualization(image_path: str, tap_order: list[dict], out_path: str) -> None:
    """Draw numbered, colored overlays on image and save to out_path."""
    img = cv2.imread(image_path)
    if img is None:
        # Fall back to Pillow for unsupported formats
        pil_img = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    for arrow in tap_order:
        x, y = arrow["x"], arrow["y"]
        direction = arrow["direction"]
        idx = arrow["tap_index"]
        color = DIRECTION_COLORS_BGR[direction]
        dx, dy = DIRECTION_VECTORS[direction]

        # Circle
        cv2.circle(img, (x, y), 15, color, 3)
        # Direction indicator line
        cv2.arrowedLine(img, (x, y), (x + dx * 30, y + dy * 30), color, 2)
        # Tap order number (white, centered in circle)
        cv2.putText(img, str(idx), (x - 7, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imwrite(out_path, img)
```

- [ ] **Step 4: Run tests — should PASS**

```bash
pytest tests/test_viz.py -v
```

- [ ] **Step 5: Commit**

```bash
git add solve_arrows.py tests/test_viz.py
git commit -m "feat: implement visualization overlay for detected arrows"
```

---

## Task 6: CLI Wiring

**Files:**
- Modify: `solve_arrows.py` — implement `main()`

- [ ] **Step 1: Implement `main()` in `solve_arrows.py`**

Replace the `def main()` stub:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Arrows Puzzle Solver — generate Tasker XML from screenshot")
    parser.add_argument("screenshot", help="Path to puzzle screenshot (PNG or JPG)")
    parser.add_argument("--delay", type=int, default=800, help="Milliseconds between taps (default: 800)")
    parser.add_argument("--api-key", dest="api_key", default=None, help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--dry-run", action="store_true", help="Print tap sequence; skip writing files (API call still made)")
    args = parser.parse_args()

    # Resolve API key: CLI flag takes precedence over env var
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: No API key provided. Use --api-key or set ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)

    image_path = args.screenshot
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    base = Path(image_path).stem
    output_dir = args.output_dir or str(Path(image_path).parent)
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Detect
    print(f"Detecting arrows in {image_path} ...")
    arrows = detect_arrows(image_path, api_key, output_dir=output_dir)
    print(f"  Detected {len(arrows)} arrows.")

    # Stage 2: Solve
    print("Solving tap order...")
    ordered, stuck = solve_order(arrows)

    if stuck:
        print(f"Warning: {len(stuck)} arrows could not be ordered (cycle detected):", file=sys.stderr)
        for a in stuck:
            print(f"  ({a['x']}, {a['y']}) {a['direction']}", file=sys.stderr)

    # Attach tap_index for visualization
    for i, a in enumerate(ordered):
        a["tap_index"] = i + 1

    # Write solution JSON
    solution = {
        "total_arrows": len(arrows),
        "tap_order": [{"tap_index": a["tap_index"], "x": a["x"], "y": a["y"], "direction": a["direction"]} for a in ordered],
    }
    if stuck:
        solution["stuck_arrows"] = [{"x": a["x"], "y": a["y"], "direction": a["direction"]} for a in stuck]

    if not args.dry_run:
        json_path = os.path.join(output_dir, f"{base}_solution.json")
        with open(json_path, "w") as f:
            json.dump(solution, f, indent=2)
        print(f"  Solution JSON: {json_path}")

        # Visualization
        viz_path = os.path.join(output_dir, f"{base}_detected.png")
        draw_visualization(image_path, ordered, viz_path)
        print(f"  Visualization: {viz_path}")
    else:
        print("Tap sequence (dry-run):")
        for a in ordered:
            print(f"  {a['tap_index']}. ({a['x']}, {a['y']}) -> {a['direction']}")

    if stuck:
        print("Cannot generate Tasker XML: cycle detected in arrow dependencies.", file=sys.stderr)
        sys.exit(1)

    # Stage 3: Generate XML
    task_name = f"Solve Arrows - {base}"
    xml = build_tasker_xml(task_name, ordered, delay_ms=args.delay)

    if not args.dry_run:
        xml_path = os.path.join(output_dir, f"{base}_tasker.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml)
        print(f"  Tasker XML: {xml_path}")
    else:
        print(f"\nTasker XML would be written to: {os.path.join(output_dir, base + '_tasker.xml')}")
        return

    # Print verification instructions
    if ordered:
        sample = ordered[len(ordered) // 2]
        print(f"""
Verify tap accuracy before running the full task:
1. Note the arrow near the center of the puzzle (~tap {sample['tap_index']})
2. In Tasker, create a single-tap test action at ({sample['x']}, {sample['y']})
3. Run it and confirm it lands on that arrow
If off, check that your screenshot was taken on-device at native resolution.""")
```

- [ ] **Step 2: Run all tests to confirm nothing is broken**

```bash
pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add solve_arrows.py
git commit -m "feat: wire up CLI with full detect→solve→generate pipeline"
```

---

## Task 7: Smoke Test on Real Screenshot

Manual integration test with the actual puzzle screenshot.

- [ ] **Step 1: Run on the provided screenshot**

```bash
python solve_arrows.py path/to/arrows_screenshot.png --dry-run
```
Expected: prints detected arrow count and tap sequence. No files written.

- [ ] **Step 2: If arrow count looks reasonable (40–80 arrows for LVL 766), run for real**

```bash
python solve_arrows.py path/to/arrows_screenshot.png
```
Expected:
- `<name>_solution.json` created
- `<name>_detected.png` created (open it to verify arrows are labeled correctly)
- `<name>_tasker.xml` created

- [ ] **Step 3: Verify the XML is importable**

Copy `<name>_tasker.xml` to Android device. In Tasker:
1. Long-press the Tasks tab → Import
2. Select the XML file
3. Verify the task appears with the correct number of AutoInput tap actions

- [ ] **Step 4: Run the one-tap verification before the full task**

Per the verification instructions printed by the script: create a single-tap action for the sample coordinate and confirm it lands on an arrow.

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "docs: confirm smoke test passed on LVL 766 screenshot"
```

---

## Known Caveats

- `AXIS_TOLERANCE = 20` is tuned for ~1080px wide screenshots. If arrows are misaligned in the solution, try increasing it (e.g., `30`).
- Screenshot must be taken on-device (power+volume) for tap coordinates to match AutoInput's coordinate space.
- If Claude returns wrong arrow directions, run with `--dry-run` and open `_detected.png` to identify which arrows are mislabeled, then re-run (the model may vary between calls).
