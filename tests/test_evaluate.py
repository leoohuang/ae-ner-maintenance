from src.evaluate import evaluate
 
G = [{(0, 2, "Activity"), (2, 3, "PhysicalObject")}]
 
def test_perfect():
    assert evaluate(G, G)["micro_f1"] == 1.0
 
def test_boundary_error_is_wrong():
    pred = [{(0, 1, "Activity"), (2, 3, "PhysicalObject")}]      # 第一个实体少一个词
    assert abs(evaluate(G, pred)["micro_f1"] - 0.5) < 1e-9
 
def test_type_error_is_wrong():
    pred = [{(0, 2, "State"), (2, 3, "PhysicalObject")}]         # 范围对、类型错
    assert abs(evaluate(G, pred)["micro_f1"] - 0.5) < 1e-9
 
def test_empty_prediction():
    r = evaluate(G, [set()])
    assert r["recall"] == 0.0 and r["micro_f1"] == 0.0
 
def test_missing_inner_nested_span_is_false_negative():
    gold = [{(0, 2, "PhysicalObject"), (1, 2, "PhysicalObject")}]
    pred = [{(0, 2, "PhysicalObject")}]
    r = evaluate(gold, pred)
    assert r["precision"] == 1.0 and abs(r["recall"] - 0.5) < 1e-9
 
def test_duplicate_prediction_cannot_inflate_score():
    pred = [[(0, 2, "Activity"), (0, 2, "Activity"), (2, 3, "PhysicalObject")]]
    assert evaluate(G, pred)["micro_f1"] == 1.0
