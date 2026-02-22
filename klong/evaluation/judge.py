from __future__ import annotations
import json
import anthropic
from klong.evaluation.rubric import RubricTree

JUDGE_PROMPT = """You are an expert code reviewer evaluating whether an AI agent successfully reproduced a research paper.

## Criterion
{criteria}

## Agent's Output Files
{artifacts}

## Instructions
Score how well the criterion is met on a scale from 0.0 to 1.0.
Respond with JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}
"""


class Judge:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.client = anthropic.Anthropic()

    async def _call_api(self, prompt: str):
        return self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

    async def evaluate_leaf(self, criteria: str, artifacts: dict[str, str]) -> float:
        artifacts_str = "\n".join(f"### {k}\n```\n{v[:2000]}\n```" for k, v in artifacts.items())
        prompt = JUDGE_PROMPT.format(criteria=criteria, artifacts=artifacts_str)
        response = await self._call_api(prompt)
        text = response.content[0].text
        try:
            data = json.loads(text)
            return max(0.0, min(1.0, float(data["score"])))
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0.0

    async def evaluate(self, tree: RubricTree, artifacts: dict[str, str]) -> tuple[float, dict]:
        leaves = tree.get_leaves()
        leaf_scores = {}
        for leaf in leaves:
            leaf_scores[leaf.name] = await self.evaluate_leaf(leaf.criteria, artifacts)
        total = tree.compute_score(leaf_scores)
        return total, leaf_scores
