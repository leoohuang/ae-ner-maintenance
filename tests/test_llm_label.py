import json

import pytest

from src.llm_label import (assert_not_test, format_record, label_one, parse_reply, render_prompt,
                           run_labelling, val_subsets)


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


class FakeModel:
    """依次返回给定的回复；回复是异常时就抛出，模拟网络错误。"""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def __call__(self, prompt):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r, {"model_version": "fake-1"}


def test_good_reply_needs_one_call():
    m = FakeModel(['{"spans": [[0, 1, "State"]]}'])
    res = label_one(m, "p", 2, sleep=lambda s: None)
    assert res["spans"] == [[0, 1, "State"]] and not res["failed"] and m.calls == 1


def test_unparseable_reply_is_retried_at_most_twice_then_fails():
    m = FakeModel(["oops", "still no", "nope", '{"spans": []}'])
    res = label_one(m, "p", 2, sleep=lambda s: None)
    assert res["failed"] and res["spans"] == [] and m.calls == 3 and len(res["attempts"]) == 3


def test_retry_that_parses_is_used():
    m = FakeModel(["oops", '{"spans": [[1, 2, "Activity"]]}'])
    res = label_one(m, "p", 2, sleep=lambda s: None)
    assert res["spans"] == [[1, 2, "Activity"]] and not res["failed"]


def test_network_errors_are_retried_and_not_counted_as_parse_retries():
    m = FakeModel([ConnectionError(), TimeoutError(), "oops", "oops", '{"spans": []}'])
    res = label_one(m, "p", 2, sleep=lambda s: None)
    assert res["spans"] == [] and not res["failed"] and m.calls == 5


def test_persistent_network_error_stops_the_run():
    m = FakeModel([ConnectionError()] * 10)
    with pytest.raises(ConnectionError):
        label_one(m, "p", 2, sleep=lambda s: None, max_net_retries=3)


def test_run_skips_labelled_records_and_writes_both_files(tmp_path):
    records = [{"id": i, "tokens": ["a", "b"], "spans": [[0, 1, "State"]]} for i in range(3)]
    labels, raw = tmp_path / "m.jsonl", tmp_path / "raw" / "m.jsonl"
    m = FakeModel(['{"spans": [[0, 2, "Activity"]]}'] * 2)
    run_labelling([0, 1], records, m, lambda r: "p", labels, raw, {"model": "fake"}, sleep=lambda s: None)
    m = FakeModel(['{"spans": []}'])
    run_labelling([0, 1, 2], records, m, lambda r: "p", labels, raw, {"model": "fake"}, sleep=lambda s: None)
    assert m.calls == 1                                   # records 0 and 1 were already done
    rows = [json.loads(l) for l in open(labels)]
    assert [r["id"] for r in rows] == [0, 1, 2] and rows[0]["spans"] == [[0, 2, "Activity"]]
    assert rows[0]["tokens"] == ["a", "b"]
    info = [json.loads(l) for l in open(raw)]
    assert info[2]["model"] == "fake" and info[2]["attempts"][0]["text"] == '{"spans": []}'


def test_rate_limits_are_waited_out_for_a_few_minutes():
    # new OpenRouter accounts are limited to 20 requests per minute per model
    waits = []
    m = FakeModel([ConnectionError()] * 8 + ['{"spans": []}'])
    res = label_one(m, "p", 2, sleep=waits.append)
    assert not res["failed"] and m.calls == 9
    assert max(waits) <= 60 and sum(waits) >= 180
