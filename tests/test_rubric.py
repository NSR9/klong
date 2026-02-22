import json
from klong.evaluation.rubric import RubricNode, RubricTree

def test_leaf_node_creation():
    node = RubricNode(name="impl_loss", weight=0.5, criteria="Implements the loss function correctly")
    assert node.is_leaf
    assert node.name == "impl_loss"

def test_tree_from_dict():
    data = {
        "name": "root", "weight": 1.0, "criteria": "",
        "children": [
            {"name": "core", "weight": 0.4, "criteria": "",
             "children": [
                 {"name": "algorithm", "weight": 0.6, "criteria": "Implements main algorithm"},
                 {"name": "results", "weight": 0.4, "criteria": "Reproduces key results"},
             ]},
            {"name": "quality", "weight": 0.6, "criteria": "Code is clean and documented"},
        ]
    }
    tree = RubricTree.from_dict(data)
    assert len(tree.root.children) == 2
    leaves = tree.get_leaves()
    assert len(leaves) == 3

def test_tree_score_aggregation():
    data = {
        "name": "root", "weight": 1.0, "criteria": "",
        "children": [
            {"name": "a", "weight": 0.6, "criteria": "criterion A"},
            {"name": "b", "weight": 0.4, "criteria": "criterion B"},
        ]
    }
    tree = RubricTree.from_dict(data)
    scores = {"a": 0.8, "b": 0.5}
    total = tree.compute_score(scores)
    assert abs(total - 0.68) < 1e-6  # 0.6*0.8 + 0.4*0.5

def test_tree_serialization_roundtrip():
    data = {
        "name": "root", "weight": 1.0, "criteria": "",
        "children": [
            {"name": "x", "weight": 1.0, "criteria": "do X"},
        ]
    }
    tree = RubricTree.from_dict(data)
    d = tree.to_dict()
    tree2 = RubricTree.from_dict(d)
    assert tree2.root.children[0].criteria == "do X"
