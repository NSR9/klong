from __future__ import annotations
import json
import re
import logging
import anthropic
from klong.evaluation.rubric import RubricTree

logger = logging.getLogger(__name__)

RUBRIC_GEN_PROMPT = """You are an expert ML reviewer. Given a research paper and its official code, generate a hierarchical evaluation rubric for an AI agent attempting to reproduce this paper from scratch.

## Paper (Markdown)
{paper_markdown}

## Official Code (excerpt)
{code_excerpt}

## Instructions
Create a JSON rubric tree. Structure:
- Root has children: "core_contribution" (weight 0.4), "experimental_setup" (weight 0.3), "code_quality" (weight 0.3)
- Each branch has 2-5 leaf criteria
- Each leaf has: name, weight (within parent, summing to 1.0), criteria (clear evaluation description)

Return ONLY valid JSON matching this schema:
{{"name": "root", "weight": 1.0, "criteria": "", "children": [...]}}
"""

class RubricGenerator:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.client = anthropic.Anthropic()

    def _build_prompt(self, paper_markdown: str, code_excerpt: str) -> str:
        paper_md = paper_markdown[:30000] if len(paper_markdown) > 30000 else paper_markdown
        code_ex = code_excerpt[:10000] if len(code_excerpt) > 10000 else code_excerpt
        return RUBRIC_GEN_PROMPT.format(paper_markdown=paper_md, code_excerpt=code_ex)

    def generate(self, paper_markdown: str, code_excerpt: str = "") -> RubricTree:
        prompt = self._build_prompt(paper_markdown, code_excerpt)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse rubric JSON from response: {text[:200]}")
        return RubricTree.from_dict(data)

    def generate_and_save(self, paper_markdown: str, code_excerpt: str,
                          output_path: str) -> RubricTree:
        tree = self.generate(paper_markdown, code_excerpt)
        with open(output_path, "w") as f:
            json.dump(tree.to_dict(), f, indent=2)
        return tree
