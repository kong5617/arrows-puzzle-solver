import sys, os, json, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from solve_arrows import validate_arrows, detect_arrows

# ---------------------------------------------------------------------------
# validate_arrows tests
# ---------------------------------------------------------------------------

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
    with pytest.raises(SystemExit) as exc_info:
        validate_arrows([], img_w=1080, img_h=2376)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# detect_arrows tests (mocked API) — flat-array responses
# ---------------------------------------------------------------------------

def _make_img(tmp_path, w=1080, h=2376):
    from PIL import Image
    p = str(tmp_path / "test.png")
    Image.new("RGB", (w, h), "white").save(p)
    return p


def test_detect_arrows_returns_parsed_list(mocker, tmp_path):
    """Happy path: API returns valid flat-array JSON."""
    resp_text = '[{"x":100,"y":200,"direction":"right"}]'
    mock_response = mocker.MagicMock()
    mock_response.content = [mocker.MagicMock(text=resp_text)]
    mock_client = mocker.MagicMock()
    mock_client.messages.create.return_value = mock_response
    mocker.patch("solve_arrows.anthropic.Anthropic", return_value=mock_client)

    result = detect_arrows(_make_img(tmp_path), api_key="test-key")
    assert len(result) == 1
    assert result[0]["direction"] == "right"
    assert result[0]["x"] == 100
    assert result[0]["y"] == 200


def test_detect_arrows_snaps_noisy_coords(mocker, tmp_path):
    """Arrows within AXIS_TOLERANCE of each other are snapped to the same column/row
    and assigned matching col/row indices so blocks_arrow uses exact grid matching."""
    # Two arrows both claiming to be in the same column: x=308 and x=312 (diff=4 < tol=14)
    # They should be snapped to the same x and get the same col index.
    resp_text = json.dumps([
        {"x": 308, "y": 200, "direction": "right"},
        {"x": 312, "y": 400, "direction": "left"},
        {"x": 600, "y": 300, "direction": "up"},   # clearly different column
    ])
    mock_response = mocker.MagicMock()
    mock_response.content = [mocker.MagicMock(text=resp_text)]
    mock_client = mocker.MagicMock()
    mock_client.messages.create.return_value = mock_response
    mocker.patch("solve_arrows.anthropic.Anthropic", return_value=mock_client)

    result = detect_arrows(_make_img(tmp_path), api_key="test-key")
    assert len(result) == 3
    cols = [a["col"] for a in result]
    # x=308 and x=312 are in the same column; x=600 is different
    assert result[0]["col"] == result[1]["col"], "Near-identical x should snap to same col"
    assert result[2]["col"] != result[0]["col"], "Distant x must be a different col"
    # Snapped x should be identical
    assert result[0]["x"] == result[1]["x"]


def test_detect_arrows_retries_on_bad_json(mocker, tmp_path):
    """If first response is invalid JSON, retry once."""
    bad_response = mocker.MagicMock()
    bad_response.content = [mocker.MagicMock(text="Here are the arrows: [...]")]
    good_response = mocker.MagicMock()
    good_response.content = [mocker.MagicMock(
        text='[{"x":500,"y":600,"direction":"left"}]')]
    mock_client = mocker.MagicMock()
    mock_client.messages.create.side_effect = [bad_response, good_response]
    mocker.patch("solve_arrows.anthropic.Anthropic", return_value=mock_client)

    result = detect_arrows(_make_img(tmp_path), api_key="test-key")
    assert mock_client.messages.create.call_count == 2
    assert result[0]["direction"] == "left"


def test_detect_arrows_exits_after_two_bad_responses(mocker, tmp_path):
    """If both attempts return invalid JSON, exits and writes error file."""
    bad_response = mocker.MagicMock()
    bad_response.content = [mocker.MagicMock(text="not json at all")]
    mock_client = mocker.MagicMock()
    mock_client.messages.create.return_value = bad_response
    mocker.patch("solve_arrows.anthropic.Anthropic", return_value=mock_client)

    with pytest.raises(SystemExit) as exc_info:
        detect_arrows(_make_img(tmp_path), api_key="test-key",
                      output_dir=str(tmp_path))
    assert exc_info.value.code == 1
    assert (tmp_path / "test_api_error.txt").exists()
