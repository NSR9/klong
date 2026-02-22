import pytest
from klong.training.data.trajectory_splitter import TrajectorySplitter, SubTrajectory
from klong.agent.scaffold import Trajectory, Turn

def _make_trajectory(n_turns: int) -> Trajectory:
    traj = Trajectory(task_id="test", paper_id="p1")
    # First 3 pairs (6 turns) are paper reading (prefix)
    for i in range(3):
        traj.add_turn(Turn(role="assistant", content=f"Reading paper section {i}",
                          tool_calls=[{"name":"read_paper","arguments":{}}]))
        traj.add_turn(Turn(role="user", content=f"Paper content section {i} " * 50))
    # Remaining turns are work
    for i in range(3, n_turns):
        traj.add_turn(Turn(role="assistant", content=f"Action {i}: writing code " * 20,
                          tool_calls=[{"name":"bash","arguments":{"command":"ls"}}]))
        traj.add_turn(Turn(role="user", content=f"Output {i}: file listing " * 20))
    return traj

def test_splitter_creation():
    sp = TrajectorySplitter(max_window_tokens=1000, overlap_tokens=100)
    assert sp.max_window_tokens == 1000

def test_split_short_trajectory():
    traj = _make_trajectory(5)
    sp = TrajectorySplitter(max_window_tokens=100000, overlap_tokens=1000)
    subs = sp.split(traj, prefix_turn_count=6)
    assert len(subs) >= 1
    assert subs[0].prefix_turns == 6

def test_split_produces_overlap():
    traj = _make_trajectory(30)
    sp = TrajectorySplitter(max_window_tokens=2000, overlap_tokens=200)
    subs = sp.split(traj, prefix_turn_count=6)
    assert len(subs) > 1
    # Check overlap: end of window i overlaps start of window i+1
    for i in range(len(subs) - 1):
        assert subs[i].end_turn_idx > subs[i+1].start_turn_idx

def test_sub_trajectory_has_action_mask():
    traj = _make_trajectory(10)
    sp = TrajectorySplitter(max_window_tokens=100000, overlap_tokens=100)
    subs = sp.split(traj, prefix_turn_count=6)
    for sub in subs:
        assert len(sub.action_mask) > 0
        # action_mask should have True for assistant turns
        assert any(sub.action_mask)

def test_split_covers_full_trajectory():
    traj = _make_trajectory(20)
    sp = TrajectorySplitter(max_window_tokens=2000, overlap_tokens=200)
    subs = sp.split(traj, prefix_turn_count=6)
    # Every body turn should appear in at least one window
    body_start = 6
    covered = set()
    for sub in subs:
        for idx in range(sub.start_turn_idx, sub.end_turn_idx):
            covered.add(idx)
    for idx in range(body_start, len(traj.turns)):
        assert idx in covered, f"Turn {idx} not covered"
