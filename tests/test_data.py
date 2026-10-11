import pytest
from src.data import to_spans

def test_nested_spans_are_kept():
    ents = [{"start": 0, "end": 2, "type": "PhysicalObject/A"},
            {"start": 1, "end": 2, "type": "PhysicalObject/B"}]
    assert to_spans(ents) == [(0, 2, "PhysicalObject"), (1, 2, "PhysicalObject")]

def test_only_top_level_type_is_kept():
    ents = [{"start": 0, "end": 1, "type": "State/X/Y"}]
    assert to_spans(ents) == [(0, 1, "State")]

def test_same_span_conflicting_types_is_rejected():
    ents = [{"start": 0, "end": 1, "type": "State/X"},
            {"start": 0, "end": 1, "type": "Activity/Y"}]
    with pytest.raises(AssertionError):
        to_spans(ents)
