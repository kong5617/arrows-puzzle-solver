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
    with pytest.raises(SystemExit) as exc_info:
        validate_arrows([], img_w=1080, img_h=2376)
    assert exc_info.value.code == 1

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

    with pytest.raises(SystemExit) as exc_info:
        detect_arrows(img_path, api_key="test-key", output_dir=str(tmp_path))
    assert exc_info.value.code == 1

    assert (tmp_path / "test_api_error.txt").exists()
