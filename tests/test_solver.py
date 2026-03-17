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

# --- solve_order tests (all use tuple unpacking) ---

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

def test_cycle_detection():
    """Two arrows pointing at each other — neither can exit."""
    a = make_arrow(100, 200, "right")   # blocked by b
    b = make_arrow(300, 200, "left")    # blocked by a
    result, stuck = solve_order([a, b])
    assert len(stuck) == 2
    assert len(result) == 0
