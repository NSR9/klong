from __future__ import annotations
import logging
from typing import Optional

from klong.agent.scaffold import Agent, Trajectory
from klong.agent.sandbox.docker_manager import SandboxManager, SandboxConfig
from klong.agent.tools.bash_tool import BashTool
from klong.agent.tools.python_tool import PythonTool
from klong.agent.tools.file_tool import WriteFileTool, ReadFileTool, SearchFilesTool
from klong.agent.tools.paper_reader import PaperReaderTool

logger = logging.getLogger(__name__)

class RolloutGenerator:
    def __init__(self, model, tokenizer, sandbox_config: SandboxConfig | None = None,
                 max_new_tokens: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.max_new_tokens = max_new_tokens

    def _create_generate_fn(self):
        import torch

        def generate(messages: list[dict]) -> str:
            prompt = ""
            for m in messages:
                role = m["role"]
                content = m["content"]
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=self.tokenizer.model_max_length)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return generate

    def generate_rollout(self, paper_id: str, paper_markdown: str,
                         task_description: str, timeout_seconds: int) -> Trajectory:
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
                model_name="policy",
                system_prompt="",
                tools=tools,
                sandbox_manager=sandbox_manager,
            )
            agent.set_generate_fn(self._create_generate_fn())
            trajectory = agent.run(
                sandbox_id=sandbox_id,
                task_description=task_description,
                paper_id=paper_id,
                timeout_seconds=timeout_seconds,
            )
            return trajectory
        finally:
            sandbox_manager.destroy(sandbox_id)

    def collect_artifacts(self, sandbox_manager: SandboxManager,
                          sandbox_id: str) -> dict[str, str]:
        result = sandbox_manager.execute(sandbox_id,
            "find /workspace -name '*.py' -type f 2>/dev/null | head -20")
        artifacts = {}
        for path in result.stdout.strip().split("\n"):
            if path.strip():
                content = sandbox_manager.read_file(sandbox_id, path.strip())
                artifacts[path.strip()] = content
        return artifacts
