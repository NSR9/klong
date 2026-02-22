import pytest
import json
from unittest.mock import MagicMock, patch
from klong.agent.scaffold import Agent, Turn, Trajectory

def test_turn_dataclass():
    t = Turn(role="assistant", content="hello", tool_calls=[], tool_results=[])
    assert t.role == "assistant"

def test_trajectory_append():
    traj = Trajectory(task_id="test", paper_id="p1")
    traj.add_turn(Turn(role="user", content="start"))
    traj.add_turn(Turn(role="assistant", content="ok", tool_calls=[{"name":"bash","args":{"command":"ls"}}]))
    assert len(traj.turns) == 2

def test_trajectory_serialization():
    traj = Trajectory(task_id="test", paper_id="p1")
    traj.add_turn(Turn(role="assistant", content="hello"))
    d = traj.to_dict()
    assert d["task_id"] == "test"
    assert len(d["turns"]) == 1

def test_trajectory_from_dict():
    traj = Trajectory(task_id="test", paper_id="p1")
    traj.add_turn(Turn(role="assistant", content="hello"))
    d = traj.to_dict()
    traj2 = Trajectory.from_dict(d)
    assert traj2.task_id == "test"
    assert len(traj2.turns) == 1
    assert traj2.turns[0].content == "hello"

def test_agent_creation():
    agent = Agent(
        model_name="test",
        system_prompt="You are a researcher.",
        tools=[],
        sandbox_manager=MagicMock(),
    )
    assert agent.system_prompt == "You are a researcher."

def test_agent_parse_tool_calls():
    agent = Agent(model_name="test", system_prompt="", tools=[], sandbox_manager=MagicMock())
    text = 'I will run a command.\n```tool_call\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
    calls = agent.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "bash"

def test_agent_parse_multiple_tool_calls():
    agent = Agent(model_name="test", system_prompt="", tools=[], sandbox_manager=MagicMock())
    text = '```tool_call\n{"name": "bash", "arguments": {"command": "ls"}}\n```\nSome text\n```tool_call\n{"name": "python", "arguments": {"code": "print(1)"}}\n```'
    calls = agent.parse_tool_calls(text)
    assert len(calls) == 2

def test_agent_parse_no_tool_calls():
    agent = Agent(model_name="test", system_prompt="", tools=[], sandbox_manager=MagicMock())
    text = "Just some plain text response with no tool calls."
    calls = agent.parse_tool_calls(text)
    assert len(calls) == 0
