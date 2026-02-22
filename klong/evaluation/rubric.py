from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RubricNode:
    name: str
    weight: float
    criteria: str
    children: list[RubricNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @classmethod
    def from_dict(cls, d: dict) -> RubricNode:
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(name=d["name"], weight=d["weight"],
                   criteria=d.get("criteria", ""), children=children)

    def to_dict(self) -> dict:
        d = {"name": self.name, "weight": self.weight, "criteria": self.criteria}
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class RubricTree:
    root: RubricNode

    @classmethod
    def from_dict(cls, d: dict) -> RubricTree:
        return cls(root=RubricNode.from_dict(d))

    def to_dict(self) -> dict:
        return self.root.to_dict()

    def get_leaves(self) -> list[RubricNode]:
        leaves = []
        def _walk(node: RubricNode):
            if node.is_leaf:
                leaves.append(node)
            for c in node.children:
                _walk(c)
        _walk(self.root)
        return leaves

    def compute_score(self, leaf_scores: dict[str, float]) -> float:
        def _score(node: RubricNode) -> float:
            if node.is_leaf:
                return leaf_scores.get(node.name, 0.0)
            total_weight = sum(c.weight for c in node.children)
            if total_weight == 0:
                return 0.0
            return sum(c.weight * _score(c) / total_weight for c in node.children)
        return _score(self.root)
