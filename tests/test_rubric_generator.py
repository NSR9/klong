import pytest
from unittest.mock import MagicMock, patch
from klong.research_factory.rubric_generator import RubricGenerator

def test_rubric_generator_creation():
    gen = RubricGenerator(model="claude-sonnet-4-20250514")
    assert gen.model == "claude-sonnet-4-20250514"

def test_rubric_prompt_formatting():
    gen = RubricGenerator()
    prompt = gen._build_prompt("# Paper Title\nContent here", "def main(): pass")
    assert "Paper Title" in prompt
    assert "def main" in prompt

def test_rubric_prompt_truncation():
    gen = RubricGenerator()
    long_paper = "x" * 50000
    prompt = gen._build_prompt(long_paper, "code")
    # Should truncate paper to 30000 chars
    assert len(prompt) < 50000
