# Arrows Puzzle Solver — Design Spec
**Date:** 2026-03-17
**Status:** Approved

## Overview

A reusable CLI Python script that processes a screenshot of the mobile "Arrows" puzzle game and produces a Tasker-importable XML task that automatically taps the arrows in the correct order to solve the puzzle.

**Puzzle rules:** Each arrow, when tapped, travels in its direction until it exits the screen. An arrow must not collide with another untapped arrow in its path. The tap order must be determined such that each arrow has a clear path to the screen edge when tapped.

---

## Architecture

Three-stage pipeline:

```
screenshot.png
      │
      ▼
 [1] Detect        Claude Vision API (CLAUDE_MODEL constant)
      │             ↳ returns JSON: [{x, y, direction}, ...]
      │
      ▼
 [2] Solve         Local topological sort
      │             ↳ returns ordered tap list
      │
      ▼
 [3] Generate      Local Tasker XML builder
                   ↳ writes <name>_tasker.xml
```

**Single entry point:** `solve_arrows.py`

---

## CLI Interface

```bash
python solve_arrows.py <screenshot.png> [--delay 800] [--api-key sk-...] [--output-dir ./out] [--dry-run]
```

| Argument | Default | Description |
|---|---|---|
| `screenshot` | required | Path to puzzle screenshot (PNG or JPG) |
| `--delay` | `800` | Milliseconds between taps in Tasker task |
| `--api-key` | env: `ANTHROPIC_API_KEY` | Anthropic API key. CLI flag takes precedence over env var. |
| `--output-dir` | same directory as input | Where to write output files |
| `--dry-run` | off | Print tap sequence to stdout; skip writing any files |

**Outputs** (prefix = input basename without extension):

| File | When written | Description |
|---|---|---|
| `<name>_tasker.xml` | Successful solve | Importable Tasker task |
| `<name>_detected.png` | Always (after detection) | Visualization: arrows numbered in tap order, colored by direction |
| `<name>_solution.json` | Always (after detection) | Tap sequence and metadata (see schema below) |
| `<name>_api_error.txt` | API failure only | Raw API response for debugging |

**File collision policy:** If any output file already exists, overwrite silently. No prompting. This enables repeated runs on the same screenshot without manual cleanup.

**`--dry-run` behavior:** The Claude API call is still made (API cost is incurred). Only file writes are skipped. The tap sequence and any errors are printed to stdout instead.

---

## Named Constants (top of script)

```python
CLAUDE_MODEL   = "claude-opus-4-6"   # Vision model for detection
AXIS_TOLERANCE = 20                   # px — arrows within this distance on the perpendicular
                                      # axis are considered aligned. ~10% of expected cell
                                      # width at 1080px screen width. Tune if needed.
```

---

## Coordinate Space & Scaling

**Key assumption:** Tasker (and AutoInput) tap at coordinates in the device's logical pixel space, which matches the screenshot's pixel dimensions when the screenshot is taken via Android's built-in screenshot mechanism (e.g., power+volume on the device itself).

**What this means in practice:**
- A screenshot taken ON the device (not captured externally) will have pixel dimensions matching the device's logical resolution
- AutoInput coordinates at `(540, 800)` will tap the pixel at `(540, 800)` in that screenshot
- Screenshots scaled or displayed at reduced size (e.g., viewed on a PC) must be passed at their ORIGINAL resolution — the script reads the full-resolution file, not the displayed version

**Verification step (documented in stdout):** After generating the XML, the script prints a verification instruction:
```
Verify tap accuracy before running the full task:
1. Note a clearly-visible arrow near the center of the puzzle
2. In Tasker, create a single-tap test action at (<x>, <y>)
3. Run it and confirm it lands on that arrow
If off, check that your screenshot was taken on-device at native resolution.
```

**No automatic scaling is applied.** If a user passes a screenshot that was already rescaled, the tap coordinates will be wrong. The script cannot detect this condition; the verification step above is the user's check.

---

## Stage 1: Arrow Detection

**Model:** value of `CLAUDE_MODEL` constant
**Input:** Screenshot encoded as base64 image
**System prompt (primary):**
```
You are analyzing a mobile puzzle screenshot.
Return ONLY a JSON array. No prose, no markdown fences.
Each element: {"x": <int>, "y": <int>, "direction": "up"|"down"|"left"|"right"}
x and y are the pixel coordinates of the arrow's center in the ORIGINAL image.
Include every arrow visible in the puzzle area. Exclude UI (header, stars, hint buttons).
```

**Retry prompt (on JSON parse failure):**
```
Your previous response was not valid JSON. Return ONLY the raw JSON array with no
surrounding text. Previous response: <raw_response_here>
```
The retry call includes the failed response as a user message for self-correction context. One retry maximum.

**Validation:** After parsing, each entry is checked for:
- Required keys: `x`, `y`, `direction`
- `direction` is exactly one of `up`, `down`, `left`, `right` — any other value (e.g., `"northeast"`) triggers a validation error with message: `"Unexpected direction '<value>' returned by model. Expected: up, down, left, right. Try re-running or adjusting the prompt."`
- `x` within `[0, image_width]`, `y` within `[0, image_height]`

**Zero-arrow guard:** If Claude returns a valid but empty array `[]`, the script exits with:
```
Error: No arrows detected. The model returned an empty list.
This may mean the screenshot is not of the expected puzzle type,
or the model incorrectly excluded all arrows as UI elements.
```

**Error handling:**
- If retry also fails to produce valid JSON → exit, save raw response to `<name>_api_error.txt`

---

## Stage 2: Solver

**Algorithm:** Iterative topological sort (`O(n²)`), suitable for puzzle sizes of ~50–100 arrows.

**Collision check:** Arrow B blocks arrow A if they are axis-aligned within `AXIS_TOLERANCE` pixels AND B is in A's travel direction:

| A's direction | Blocking condition |
|---|---|
| `right` | `abs(A.y - B.y) <= AXIS_TOLERANCE` and `B.x > A.x` |
| `left`  | `abs(A.y - B.y) <= AXIS_TOLERANCE` and `B.x < A.x` |
| `up`    | `abs(A.x - B.x) <= AXIS_TOLERANCE` and `B.y < A.y` |
| `down`  | `abs(A.x - B.x) <= AXIS_TOLERANCE` and `B.y > A.y` |

Note: the check finds the nearest blocker implicitly — any arrow in the path counts as a blocker regardless of distance.

**Tap order derivation (per-iteration recomputation required):**
1. Compute blockers for every remaining arrow against every other remaining arrow (recomputed fresh each iteration)
2. Find all arrows with zero remaining blockers → add to tap order
3. Remove those arrows from the active set
4. Repeat until active set is empty

Step 1 must be recomputed each iteration from the current active set, not precomputed upfront. This ensures that removing arrow B correctly unblocks arrows that were only blocked by B.

**Cycle detection:** If an iteration finds no unblocked arrow but arrows remain, the puzzle has an unresolvable cycle (likely a detection error). The script:
- Prints a warning listing stuck arrow positions and directions
- Writes them to `<name>_solution.json` under key `"stuck_arrows"`
- Exits without producing Tasker XML

---

## Stage 3: Tasker XML Generator

### Task structure

```
Task name: "Solve Arrows - <basename>"   (basename = input filename without extension or path)
  act0:  AutoInput Tap (x₁, y₁)
  act1:  Wait <delay>ms
  act2:  AutoInput Tap (x₂, y₂)
  act3:  Wait <delay>ms
  ...
```

### AutoInput tap action XML (primary)

Confirmed from a real Tasker 6.6.20 + AutoInput export. Action code is `107361459`.

Tap coordinates appear in two places: the human-readable `BLURB` string and the `parameters` JSON field. In the JSON, the comma between X and Y is escaped as `\\,`. A fresh `uuid4` is generated for each `plugininstanceid`.

```xml
<Action sr="actN" ve="7">
    <code>107361459</code>
    <Bundle sr="arg0">
        <Vals sr="val">
            <EnableDisableAccessibilityService>&lt;null&gt;</EnableDisableAccessibilityService>
            <EnableDisableAccessibilityService-type>java.lang.String</EnableDisableAccessibilityService-type>
            <Password>&lt;null&gt;</Password>
            <Password-type>java.lang.String</Password-type>
            <com.twofortyfouram.locale.intent.extra.BLURB>Actions To Perform: click(point,X\,Y)
Not In AutoInput: true
Not In Tasker: true
Separator: ,
Check Millis: 1000</com.twofortyfouram.locale.intent.extra.BLURB>
            <com.twofortyfouram.locale.intent.extra.BLURB-type>java.lang.String</com.twofortyfouram.locale.intent.extra.BLURB-type>
            <net.dinglisch.android.tasker.JSON_ENCODED_KEYS>parameters</net.dinglisch.android.tasker.JSON_ENCODED_KEYS>
            <net.dinglisch.android.tasker.JSON_ENCODED_KEYS-type>java.lang.String</net.dinglisch.android.tasker.JSON_ENCODED_KEYS-type>
            <net.dinglisch.android.tasker.RELEVANT_VARIABLES>&lt;StringArray sr=""&gt;&lt;_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES0&gt;%ailastbounds
Last Bounds
Bounds (left,top,right,bottom) of the item that the action last interacted with&lt;/_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES0&gt;&lt;_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES1&gt;%ailastcoordinates
Last Coordinates
Center coordinates (x,y) of the item that the action last interacted with&lt;/_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES1&gt;&lt;_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES2&gt;%err
Error Code
Only available if you select &amp;lt;b&amp;gt;Continue Task After Error&amp;lt;/b&amp;gt; and the action ends in error&lt;/_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES2&gt;&lt;_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES3&gt;%errmsg
Error Message
Only available if you select &amp;lt;b&amp;gt;Continue Task After Error&amp;lt;/b&amp;gt; and the action ends in error&lt;/_array_net.dinglisch.android.tasker.RELEVANT_VARIABLES3&gt;&lt;/StringArray&gt;</net.dinglisch.android.tasker.RELEVANT_VARIABLES>
            <net.dinglisch.android.tasker.RELEVANT_VARIABLES-type>[Ljava.lang.String;</net.dinglisch.android.tasker.RELEVANT_VARIABLES-type>
            <net.dinglisch.android.tasker.extras.VARIABLE_REPLACE_KEYS>parameters plugininstanceid plugintypeid </net.dinglisch.android.tasker.extras.VARIABLE_REPLACE_KEYS>
            <net.dinglisch.android.tasker.extras.VARIABLE_REPLACE_KEYS-type>java.lang.String</net.dinglisch.android.tasker.extras.VARIABLE_REPLACE_KEYS-type>
            <net.dinglisch.android.tasker.subbundled>true</net.dinglisch.android.tasker.subbundled>
            <net.dinglisch.android.tasker.subbundled-type>java.lang.Boolean</net.dinglisch.android.tasker.subbundled-type>
            <parameters>{"_action":"click(point,X\\,Y)","_additionalOptions":{"checkMs":"1000","separator":",","withCoordinates":false},"_whenToPerformAction":{"notInAutoInput":true,"notInTasker":true},"generatedValues":{}}</parameters>
            <parameters-type>java.lang.String</parameters-type>
            <plugininstanceid>GENERATE-UUID4-PER-ACTION</plugininstanceid>
            <plugininstanceid-type>java.lang.String</plugininstanceid-type>
            <plugintypeid>com.joaomgcd.autoinput.intent.IntentActionv2</plugintypeid>
            <plugintypeid-type>java.lang.String</plugintypeid-type>
        </Vals>
    </Bundle>
    <Str sr="arg1" ve="3">com.joaomgcd.autoinput</Str>
    <Str sr="arg2" ve="3">com.joaomgcd.autoinput.activity.ActivityConfigActionv2</Str>
    <Int sr="arg3" val="60"/>
    <Int sr="arg4" val="1"/>
</Action>
```

**XML escaping note:** The `parameters` element value is a JSON string. When writing with Python's `xml.etree.ElementTree`, `"` characters inside element text are auto-escaped to `&quot;`. The `\\,` in the JSON must remain as two characters (backslash + comma) in the final XML text.

### Tasker native Pointer Input (fallback, commented out in XML)

Emitted as an XML comment after each primary tap action. Action code unverified — user should test before relying on this fallback:

```xml
<!-- Fallback: Tasker Pointer Input (accessibility, Tasker 5.9+) — verify code before use
<Action sr="actN" ve="7">
    <code>993</code>
    <Str sr="arg0" ve="3">tap</Str>
    <Int sr="arg1" val="X"/>
    <Int sr="arg2" val="Y"/>
</Action>
-->

### Wait action XML

```xml
<Action sr="actN" ve="7">
    <code>30</code>
    <Int sr="arg0" val="DELAY_MS"/>
    <Int sr="arg1" val="0"/>
</Action>
```

### Full task XML wrapper

Confirmed from a real Tasker 6.6.20 export:

```xml
<TaskerData sr="" dvi="1" tv="6.6.20">
    <Task sr="task1">
        <cdate>TIMESTAMP_MS</cdate>
        <edate>TIMESTAMP_MS</edate>
        <id>1</id>
        <nme>Solve Arrows - BASENAME</nme>
        <!-- action pairs here -->
    </Task>
</TaskerData>
```

`cdate`/`edate` = Unix timestamp in milliseconds (creation/edit date). Set to `datetime.now()` in milliseconds at generation time.

---

## Visualization (`<name>_detected.png`)

A copy of the input image with the following overlaid for each detected arrow:

| Element | Spec |
|---|---|
| Circle | 15px radius, color = direction color, thickness 3px |
| Tap order number | White text, 14pt, centered within the circle |
| Direction arrow | 30px arrowedLine from center toward travel direction, color = direction color |

**Direction color map:**

| Direction | Color (BGR for OpenCV) |
|---|---|
| `right` | Green `(0, 200, 0)` |
| `left` | Blue `(200, 0, 0)` |
| `up` | Red `(0, 0, 200)` |
| `down` | Yellow `(0, 200, 200)` |

---

## `_solution.json` Schema

Success case:
```json
{
  "total_arrows": 42,
  "tap_order": [
    {"tap_index": 1, "x": 540, "y": 800, "direction": "right"},
    {"tap_index": 2, "x": 200, "y": 400, "direction": "up"}
  ]
}
```

Cycle/failure case (includes `stuck_arrows` key, `tap_order` contains arrows that were resolved before the cycle):
```json
{
  "total_arrows": 42,
  "tap_order": [...],
  "stuck_arrows": [
    {"x": 100, "y": 200, "direction": "left"},
    {"x": 105, "y": 200, "direction": "right"}
  ]
}
```

---

## Dependencies

**Python:** 3.8+

```
anthropic>=0.40.0   # Claude API client
Pillow>=10.0        # Image loading
numpy>=1.24         # Array operations
opencv-python>=4.8  # Visualization drawing
```

Install: `pip install anthropic pillow numpy opencv-python`

---

## Key Assumptions

- Puzzle arrows are always one of four cardinal directions (no diagonals)
- Screenshots must be at full device resolution (not rescaled) for tap coordinates to be accurate
- Arrow travel is a straight line with no path constraints
- A valid puzzle is always solvable; a detected cycle indicates a detection error
- `AXIS_TOLERANCE = 20px` is calibrated for ~1080px-wide screenshots; scale proportionally for other resolutions
