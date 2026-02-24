from __future__ import annotations
import json
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from klong.agent.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are an expert ML researcher tasked with reproducing a research paper from scratch.

## Task
{task_description}

## Available Tools
{tool_descriptions}

## Tool Call Format
To use a tool, write:
```tool_call
{{"name": "<tool_name>", "arguments": {{<args>}}}}
```

## Rules
1. You MUST read the paper first using the read_paper tool.
2. Write clean, well-documented code.
3. Test your implementation thoroughly.
4. When finished, use end_task to signal completion.
"""

@dataclass
class Turn:
    role: str  # "system", "user" (observation), "assistant" (action)
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class Trajectory:
    task_id: str
    paper_id: str
    turns: list[Turn] = field(default_factory=list)
    total_time_seconds: float = 0.0
    final_score: float = 0.0

    def add_turn(self, turn: Turn):
        self.turns.append(turn)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "paper_id": self.paper_id,
            "total_time_seconds": self.total_time_seconds,
            "final_score": self.final_score,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "tool_calls": t.tool_calls,
                    "tool_results": t.tool_results,
                    "timestamp": t.timestamp,
                }
                for t in self.turns
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Trajectory:
        traj = cls(task_id=d["task_id"], paper_id=d["paper_id"],
                   total_time_seconds=d.get("total_time_seconds", 0),
                   final_score=d.get("final_score", 0))
        for td in d["turns"]:
            traj.add_turn(Turn(**td))
        return traj

class Agent:
    def __init__(self, model_name: str, system_prompt: str, tools: list[Tool],
                 sandbox_manager, max_turns: int = 200,
                 end_task_ban_turns: int = 10, mandatory_read_turns: int = 3):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in tools}
        self.sandbox_manager = sandbox_manager
        self.max_turns = max_turns
        self.end_task_ban_turns = end_task_ban_turns
        self.mandatory_read_turns = mandatory_read_turns
        self._generate_fn = None

    def set_generate_fn(self, fn):
        """Set the LLM generation function: fn(messages) -> str"""
        self._generate_fn = fn

    @staticmethod
    def _fix_json_newlines(s: str) -> str:
        """Escape literal newlines inside JSON string values."""
        result = []
        in_string = False
        escape = False
        for ch in s:
            if escape:
                result.append(ch)
                escape = False
                continue
            if ch == '\\':
                result.append(ch)
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                result.append(ch)
                continue
            if ch == '\n' and in_string:
                result.append('\\n')
                continue
            if ch == '\t' and in_string:
                result.append('\\t')
                continue
            result.append(ch)
        return ''.join(result)

    def parse_tool_calls(self, text: str) -> list[dict]:
        pattern = r'```tool_call\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        calls = []
        for m in matches:
            raw = m.strip()
            try:
                calls.append(json.loads(raw))
            except json.JSONDecodeError:
                # Try fixing unescaped newlines/tabs in JSON string values
                try:
                    fixed = self._fix_json_newlines(raw)
                    calls.append(json.loads(fixed))
                except (json.JSONDecodeError, Exception):
                    logger.warning(f"Failed to parse tool call: {raw[:100]}")
        return calls

    def run(self, sandbox_id: str, task_description: str,
            paper_id: str, timeout_seconds: int = 1800) -> Trajectory:
        trajectory = Trajectory(task_id=f"{paper_id}_run", paper_id=paper_id)
        start_time = time.time()
        done = False
        turn_count = 0
        paper_read = False
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": f"Begin the task. Your goal:\n{task_description}"})

        while not done and turn_count < self.max_turns:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.info("Timeout reached")
                break

            if not paper_read and turn_count >= self.mandatory_read_turns:
                messages.append({"role": "user",
                    "content": "WARNING: You must read the paper before proceeding. Use read_paper tool."})

            response_text = self._generate_fn(messages)
            tool_calls = self.parse_tool_calls(response_text)

            # Detect "end_task" as plain text (model often writes it without tool_call block)
            if not tool_calls and "end_task" in response_text.strip().lower():
                tool_calls = [{"name": "end_task", "arguments": {}}]

            turn = Turn(role="assistant", content=response_text, tool_calls=tool_calls)
            trajectory.add_turn(turn)
            messages.append({"role": "assistant", "content": response_text})

            if tool_calls:
                results = []
                for call in tool_calls:
                    tool_name = call.get("name", "")
                    args = call.get("arguments", {})

                    if tool_name == "end_task":
                        if turn_count < self.end_task_ban_turns:
                            results.append({"name": tool_name, "output": "ERROR: end_task banned for first turns."})
                        else:
                            done = True
                            results.append({"name": tool_name, "output": "Task ended."})
                    elif tool_name in self.tools:
                        if tool_name == "read_paper":
                            paper_read = True
                        try:
                            result = self.tools[tool_name].execute(sandbox_id, **args)
                            results.append({"name": tool_name, "output": result.output, "error": result.error})
                        except Exception as e:
                            results.append({"name": tool_name, "output": "", "error": str(e)})
                    else:
                        results.append({"name": tool_name, "output": "", "error": f"Unknown tool: {tool_name}"})

                obs_text = "\n".join(
                    f"[{r['name']}] {r['output']}" + (f"\nSTDERR: {r['error']}" if r.get('error') else "")
                    for r in results
                )
                obs_turn = Turn(role="user", content=obs_text, tool_results=results)
                trajectory.add_turn(obs_turn)
                messages.append({"role": "user", "content": obs_text})
            else:
                # No tool calls — prompt the model to use tools or end the task
                nudge = "Please use the available tools to make progress, or use end_task if you are done."
                messages.append({"role": "user", "content": nudge})
                trajectory.add_turn(Turn(role="user", content=nudge))

            turn_count += 1

        trajectory.total_time_seconds = time.time() - start_time
        return trajectory
