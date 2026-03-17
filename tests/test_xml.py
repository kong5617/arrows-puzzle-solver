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
