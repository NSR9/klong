"""End-to-end integration test with tiny model and mock sandbox."""
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from klong.config.settings import KLongConfig
from klong.evaluation.rubric import RubricTree
from klong.training.data.trajectory_splitter import TrajectorySplitter
from klong.training.data.trajectory_dataset import TrajectoryDataset
from klong.agent.scaffold import Agent, Trajectory, Turn

def test_full_trajectory_pipeline(tmp_path):
    """Test: create trajectory -> split -> format for SFT."""
    # 1. Create a mock trajectory
    traj = Trajectory(task_id="integration_test", paper_id="test_paper")
    for i in range(20):
        traj.add_turn(Turn(role="assistant", content=f"Action {i}: " + "x" * 100,
                          tool_calls=[{"name":"bash","arguments":{"command":"ls"}}]))
        traj.add_turn(Turn(role="user", content=f"Output {i}: " + "y" * 100))

    # 2. Save trajectory
    traj_path = tmp_path / "test_paper.json"
    traj_path.write_text(json.dumps(traj.to_dict()))

    # 3. Load as dataset
    ds = TrajectoryDataset(str(tmp_path), max_window_tokens=2000, overlap_tokens=200)
    assert len(ds) >= 1

    # 4. Check format
    item = ds[0]
    assert "text" in item
    assert "<|im_start|>" in item["text"]
    assert len(item["action_mask"]) == len(item["text"])

def test_rubric_evaluation_pipeline():
    """Test: create rubric -> compute score."""
    rubric_data = {
        "name": "root", "weight": 1.0, "criteria": "",
        "children": [
            {"name": "core", "weight": 0.7, "criteria": "",
             "children": [
                 {"name": "algo", "weight": 0.6, "criteria": "Implements algorithm"},
                 {"name": "results", "weight": 0.4, "criteria": "Reproduces results"},
             ]},
            {"name": "quality", "weight": 0.3, "criteria": "Clean code"},
        ]
    }
    tree = RubricTree.from_dict(rubric_data)
    scores = {"algo": 0.9, "results": 0.7, "quality": 0.8}
    total = tree.compute_score(scores)
    assert 0.0 < total < 1.0

def test_config_drives_everything():
    """Test: config system provides all needed parameters."""
    cfg = KLongConfig()
    assert cfg.model.name
    assert cfg.training.sft.learning_rate > 0
    assert cfg.training.rl.clip_epsilon > 0
    assert len(cfg.training.rl.stages) == 3
    assert cfg.training.rl.stages[0].timeout_minutes < cfg.training.rl.stages[-1].timeout_minutes

def test_trajectory_roundtrip():
    """Test: trajectory serialization/deserialization preserves data."""
    traj = Trajectory(task_id="roundtrip", paper_id="p1")
    traj.add_turn(Turn(role="assistant", content="Hello", tool_calls=[{"name": "bash", "arguments": {"command": "ls"}}]))
    traj.add_turn(Turn(role="user", content="file1.py file2.py"))
    traj.total_time_seconds = 42.5
    traj.final_score = 0.85

    d = traj.to_dict()
    traj2 = Trajectory.from_dict(d)
    assert traj2.task_id == "roundtrip"
    assert len(traj2.turns) == 2
    assert traj2.total_time_seconds == 42.5
    assert traj2.final_score == 0.85

def test_agent_with_mock_llm():
    """Test: agent loop works with a mock LLM that calls tools."""
    from klong.agent.tools.paper_reader import PaperReaderTool

    paper_tool = PaperReaderTool(sandbox_manager=None, paper_markdown="# Test Paper\nContent")
    agent = Agent(
        model_name="test",
        system_prompt="You are a researcher.",
        tools=[paper_tool],
        sandbox_manager=MagicMock(),
        max_turns=3,
        end_task_ban_turns=0,
        mandatory_read_turns=10,
    )

    call_count = 0
    def mock_generate(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 'Let me read the paper.\n```tool_call\n{"name": "read_paper", "arguments": {}}\n```'
        elif call_count == 2:
            return 'I have read the paper. Now finishing.\n```tool_call\n{"name": "end_task", "arguments": {}}\n```'
        return "Done."

    agent.set_generate_fn(mock_generate)
    trajectory = agent.run(
        sandbox_id="fake",
        task_description="Test task",
        paper_id="test",
        timeout_seconds=60,
    )
    assert len(trajectory.turns) >= 2
    # Agent should have read paper and ended task
    assert any("read_paper" in str(t.tool_calls) for t in trajectory.turns if t.role == "assistant")

def test_splitter_with_real_trajectory():
    """Test: splitting a realistic trajectory produces valid sub-trajectories."""
    traj = Trajectory(task_id="split_test", paper_id="p1")
    # Prefix: 2 turns of paper reading
    traj.add_turn(Turn(role="assistant", content="Reading paper..." + "a" * 200))
    traj.add_turn(Turn(role="user", content="Paper content: " + "b" * 500))
    # Body: 30 work turns
    for i in range(30):
        traj.add_turn(Turn(role="assistant", content=f"Step {i}: " + "c" * 100))
        traj.add_turn(Turn(role="user", content=f"Result {i}: " + "d" * 100))

    splitter = TrajectorySplitter(max_window_tokens=1500, overlap_tokens=200)
    subs = splitter.split(traj, prefix_turn_count=2)

    assert len(subs) > 1
    # Every sub-trajectory starts with prefix
    for sub in subs:
        assert sub.prefix_turns == 2
        assert sub.turns[0].content.startswith("Reading paper")
    # action_mask is correct length
    for sub in subs:
        assert len(sub.action_mask) == len(sub.turns)
