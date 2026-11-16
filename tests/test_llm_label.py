from src.llm_label import parse_reply


def test_plain_json_is_parsed():
    spans, drops = parse_reply('{"spans": [[1, 3, "PhysicalObject"], [3, 4, "State"]]}', 5)
    assert spans == [[1, 3, "PhysicalObject"], [3, 4, "State"]]
    assert sum(drops.values()) == 0


def test_code_fence_around_json_is_removed():
    text = '```json\n{"spans": [[0, 2, "Activity"]]}\n```'
    spans, _ = parse_reply(text, 3)
    assert spans == [[0, 2, "Activity"]]


def test_empty_list_means_no_entities():
    spans, drops = parse_reply('{"spans": []}', 4)
    assert spans == [] and sum(drops.values()) == 0


def test_bad_spans_are_dropped_and_counted():
    text = ('{"spans": [[0, 9, "State"], [2, 2, "State"], [3, 1, "State"], [0, 1, "Machine"], '
            '[0, 1, "State"], [0, 1, "State"], [0, "x", "State"], [1]]}')
    spans, drops = parse_reply(text, 4)
    assert spans == [[0, 1, "State"]]
    assert drops == {"out_of_range": 1, "empty": 2, "unknown_type": 1, "duplicate": 1, "malformed": 2}


def test_nested_spans_are_kept():
    spans, _ = parse_reply('{"spans": [[1, 4, "PhysicalObject"], [1, 3, "PhysicalObject"]]}', 5)
    assert spans == [[1, 3, "PhysicalObject"], [1, 4, "PhysicalObject"]]


def test_unparseable_reply_returns_none():
    for text in ["Sure! Here are the entities:", '{"spans": "none"}', '{"entities": []}', ""]:
        spans, _ = parse_reply(text, 4)
        assert spans is None, text
