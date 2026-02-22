from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from klong.evaluation.judge import Judge
from klong.evaluation.rubric import RubricTree

@pytest.fixture
def sample_tree():
    return RubricTree.from_dict({
        "name": "root", "weight": 1.0, "criteria": "",
        "children": [
            {"name": "algo", "weight": 0.7, "criteria": "Implements the algorithm"},
            {"name": "test", "weight": 0.3, "criteria": "Has tests"},
        ]
    })

def test_judge_creation():
    judge = Judge(model="claude-sonnet-4-20250514")
    assert judge.model == "claude-sonnet-4-20250514"

@pytest.mark.asyncio
async def test_judge_evaluate_calls_api(sample_tree):
    judge = Judge(model="test-model")
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"score": 0.75, "reasoning": "good"}')]
    with patch.object(judge, '_call_api', new_callable=AsyncMock, return_value=mock_response):
        score = await judge.evaluate_leaf("Implements the algorithm", {"main.py": "print('hello')"})
        assert 0.0 <= score <= 1.0
