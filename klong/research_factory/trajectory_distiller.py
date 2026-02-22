from __future__ import annotations
import json
import logging
import asyncio
from pathlib import Path

import anthropic
from klong.agent.scaffold import Agent, Trajectory
from klong.agent.sandbox.docker_manager import SandboxManager, SandboxConfig
from klong.agent.tools.bash_tool import BashTool
from klong.agent.tools.python_tool import PythonTool
from klong.agent.tools.file_tool import WriteFileTool, ReadFileTool, SearchFilesTool
from klong.agent.tools.paper_reader import PaperReaderTool
from klong.evaluation.judge import Judge
from klong.evaluation.rubric import RubricTree

logger = logging.getLogger(__name__)

class TrajectoryDistiller:
    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 sandbox_config: SandboxConfig | None = None,
                 timeout_minutes: int = 120,
                 rejection_threshold: float = 0.3):
        self.model = model
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.timeout_minutes = timeout_minutes
        self.rejection_threshold = rejection_threshold
        self.client = anthropic.Anthropic()

    def _create_claude_generate_fn(self):
        def generate(messages: list[dict]) -> str:
            api_messages = []
            for m in messages:
                if m["role"] == "system":
                    continue
                api_messages.append({"role": m["role"], "content": m["content"]})
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=api_messages,
            )
            return response.content[0].text
        return generate

    def distill(self, paper_id: str, paper_markdown: str,
                task_description: str, rubric: RubricTree) -> Trajectory | None:
        sandbox_manager = SandboxManager(self.sandbox_config)
        sandbox_id = sandbox_manager.create()
        try:
            tools = [
                BashTool(sandbox_manager),
                PythonTool(sandbox_manager),
                WriteFileTool(sandbox_manager),
                ReadFileTool(sandbox_manager),
                SearchFilesTool(sandbox_manager),
                PaperReaderTool(sandbox_manager, paper_markdown),
            ]
            agent = Agent(
                model_name=self.model,
                system_prompt="",
                tools=tools,
                sandbox_manager=sandbox_manager,
                max_turns=200,
            )
            agent.set_generate_fn(self._create_claude_generate_fn())
            trajectory = agent.run(
                sandbox_id=sandbox_id,
                task_description=task_description,
                paper_id=paper_id,
                timeout_seconds=self.timeout_minutes * 60,
            )
            artifacts = self._collect_artifacts(sandbox_manager, sandbox_id)
            judge = Judge()
            score, leaf_scores = asyncio.run(judge.evaluate(rubric, artifacts))
            trajectory.final_score = score
            if score < self.rejection_threshold:
                logger.info(f"Rejecting trajectory for {paper_id}: score={score:.3f}")
                return None
            logger.info(f"Accepted trajectory for {paper_id}: score={score:.3f}")
            return trajectory
        finally:
            sandbox_manager.destroy(sandbox_id)

    def _collect_artifacts(self, sandbox_manager: SandboxManager,
                           sandbox_id: str) -> dict[str, str]:
        result = sandbox_manager.execute(sandbox_id,
            "find /workspace -name '*.py' -type f 2>/dev/null | head -20")
        artifacts = {}
        for path in result.stdout.strip().split("\n"):
            if path.strip():
                content = sandbox_manager.read_file(sandbox_id, path.strip())
                artifacts[path.strip()] = content
        return artifacts

    def distill_and_save(self, paper_id: str, paper_markdown: str,
                         task_description: str, rubric: RubricTree,
                         output_dir: str) -> bool:
        trajectory = self.distill(paper_id, paper_markdown, task_description, rubric)
        if trajectory is None:
            return False
        output_path = Path(output_dir) / f"{paper_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(trajectory.to_dict(), f, indent=2)
        return True
