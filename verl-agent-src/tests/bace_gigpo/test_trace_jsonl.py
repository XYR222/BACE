import json

from recipe.bace_gigpo.validate_trace import _jsonl


def test_jsonl_reader_preserves_unicode_line_separator_inside_json_string(tmp_path):
    path = tmp_path / "trace.jsonl"
    records = [{"observation": "left\u2028right"}, {"value": 2}]
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    assert _jsonl(path) == records
