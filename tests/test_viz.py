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


def test_visualization_draws_on_image(tmp_path):
    """Output image must differ from all-white input — confirms drawing actually happened."""
    import numpy as np
    img_path = str(tmp_path / "puzzle.png")
    out_path = str(tmp_path / "puzzle_detected.png")
    Image.new("RGB", (1080, 2376), "white").save(img_path)
    arrows = [{"x": 200, "y": 300, "direction": "right", "tap_index": 1}]
    draw_visualization(img_path, arrows, out_path)
    out = cv2.imread(out_path)
    # At least some pixels must be non-white (drawing occurred)
    assert not np.all(out == 255)
