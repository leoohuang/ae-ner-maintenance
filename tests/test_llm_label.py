import pytest

from src.llm_label import assert_not_test, format_record, parse_reply, render_prompt, val_subsets


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


SPLIT = {"train_pool": [0, 1, 2], "val": list(range(10, 40)), "test": [50, 51]}


def test_test_records_are_refused():
    assert_not_test([0, 1, 10], SPLIT)
    with pytest.raises(ValueError):
        assert_not_test([0, 51], SPLIT)


def test_val_subsets_are_disjoint_fixed_and_from_val():
    ex, tune = val_subsets(SPLIT, n_examples=5, n_tune=20, seed=7)
    assert len(ex) == 5 and len(tune) == 20 and not set(ex) & set(tune)
    assert set(ex) | set(tune) <= set(SPLIT["val"])
    assert (ex, tune) == val_subsets(SPLIT, n_examples=5, n_tune=20, seed=7)


def test_record_shows_every_token_with_its_index():
    assert format_record({"tokens": ["<id>", "pump", "leaking"]}) == "0:<id> 1:pump 2:leaking"


def test_prompt_contains_examples_with_answers_and_the_record():
    ex = [{"tokens": ["pump", "leaking"], "spans": [[0, 1, "PhysicalObject"], [1, 2, "State"]]}]
    rec = {"tokens": ["replace", "seal"], "spans": [[1, 2, "PhysicalObject"]]}
    p = render_prompt("EXAMPLES:\n{examples}\nRECORD:\n{record}", rec, ex)
    assert "0:pump 1:leaking" in p and '{"spans": [[0, 1, "PhysicalObject"], [1, 2, "State"]]}' in p
    assert p.endswith("0:replace 1:seal")
    assert '"PhysicalObject"]]}' not in p.split("RECORD:")[1]    # the record's own answer is never shown
