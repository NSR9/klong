import pytest
import json
from pathlib import Path
from klong.training.data.trajectory_dataset import TrajectoryDataset

def _make_trajectory_file(tmp_path):
    traj = {
        "task_id": "test_1", "paper_id": "p1",
        "total_time_seconds": 100, "final_score": 0.8,
        "turns": [
            {"role": "assistant", "content": "Let me read the paper first.", "tool_calls": [{"name":"read_paper","arguments":{}}], "tool_results": [], "timestamp": 0},
            {"role": "user", "content": "# Paper Title\nThis is the paper content.", "tool_calls": [], "tool_results": [{"name":"read_paper","output":"paper"}], "timestamp": 1},
            {"role": "assistant", "content": "Now I will implement the algorithm.", "tool_calls": [{"name":"bash","arguments":{"command":"echo hello"}}], "tool_results": [], "timestamp": 2},
            {"role": "user", "content": "hello", "tool_calls": [], "tool_results": [{"name":"bash","output":"hello"}], "timestamp": 3},
            {"role": "assistant", "content": "Implementation complete.", "tool_calls": [], "tool_results": [], "timestamp": 4},
        ]
    }
    p = tmp_path / "test_1.json"
    p.write_text(json.dumps(traj))
    return tmp_path

def test_dataset_loads(tmp_path):
    _make_trajectory_file(tmp_path)
    ds = TrajectoryDataset(str(tmp_path), max_window_tokens=100000)
    assert len(ds) >= 1

def test_dataset_getitem(tmp_path):
    _make_trajectory_file(tmp_path)
    ds = TrajectoryDataset(str(tmp_path), max_window_tokens=100000)
    item = ds[0]
    assert "text" in item
    assert "action_mask" in item

def test_dataset_chatml_format(tmp_path):
    _make_trajectory_file(tmp_path)
    ds = TrajectoryDataset(str(tmp_path), max_window_tokens=100000)
    item = ds[0]
    assert "<|im_start|>" in item["text"]
    assert "<|im_end|>" in item["text"]

def test_dataset_action_mask_length(tmp_path):
    _make_trajectory_file(tmp_path)
    ds = TrajectoryDataset(str(tmp_path), max_window_tokens=100000)
    item = ds[0]
    assert len(item["action_mask"]) == len(item["text"])
