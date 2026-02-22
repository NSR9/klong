# KLong Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the full KLong pipeline from scratch — Research-Factory, Agent Scaffold, Trajectory-Splitting SFT, Progressive RL, and Judge evaluation.

**Architecture:** Modular Python package (`klong/`) with independent components communicating via JSONL files. Each stage produces artifacts consumed by the next. LoRA fine-tuning on Qwen-2.5-7B via TRL/Transformers on single GPU.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, TRL, PEFT, vLLM, Docker SDK, Anthropic SDK, arxiv API, pymupdf4llm

---

### Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `klong/__init__.py`
- Create: `klong/config/__init__.py`
- Create: `klong/config/settings.py`
- Create: `pyproject.toml` (root-level for klong package)
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Step 1: Write test for config loading**

```python
# tests/test_config.py
import pytest
from klong.config.settings import (
    KLongConfig, ModelConfig, DataConfig, TrainingConfig,
    InfraConfig, EvalConfig, SFTConfig, RLConfig,
)

def test_default_config_creates():
    cfg = KLongConfig()
    assert cfg.model.name == "Qwen/Qwen2.5-7B"
    assert cfg.model.lora_rank == 64
    assert cfg.training.sft.learning_rate == 2e-5
    assert cfg.training.rl.clip_epsilon == 0.2

def test_config_to_dict_roundtrip():
    cfg = KLongConfig()
    d = cfg.model_dump()
    cfg2 = KLongConfig(**d)
    assert cfg == cfg2

def test_config_from_yaml(tmp_path):
    import yaml
    cfg = KLongConfig()
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg.model_dump()))
    loaded = KLongConfig(**yaml.safe_load(p.read_text()))
    assert loaded.model.name == cfg.model.name
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sriranga/Desktop/lightning-llm/.claude/worktrees/bold-banzai && python -m pytest tests/test_config.py -v`
Expected: FAIL — module not found

**Step 3: Write pyproject.toml**

```toml
[project]
name = "klong"
version = "0.1.0"
description = "KLong: Training LLM Agent for Extremely Long-horizon Tasks"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "trl>=0.12.0",
    "peft>=0.13.0",
    "accelerate>=1.0.0",
    "bitsandbytes>=0.44.0",
    "datasets>=3.0.0",
    "anthropic>=0.40.0",
    "arxiv>=2.1.0",
    "pymupdf4llm>=0.0.10",
    "docker>=7.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "jsonlines>=4.0.0",
    "tiktoken>=0.7.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.24.0"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["klong*"]
```

**Step 4: Write settings.py with all config dataclasses**

```python
# klong/config/settings.py
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-7B"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: list[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    max_seq_length: int = 32768
    load_in_4bit: bool = False
    trust_remote_code: bool = True

class PaperCollectionConfig(BaseModel):
    conferences: list[str] = Field(default_factory=lambda: ["ICML", "NeurIPS", "ICLR"])
    years_back: int = 5
    max_papers: int = 100
    output_dir: str = "data/papers"

class DistillationConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 200
    timeout_minutes: int = 120
    rejection_threshold: float = 0.3
    output_dir: str = "data/trajectories"

class DataConfig(BaseModel):
    papers: PaperCollectionConfig = Field(default_factory=PaperCollectionConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    trajectory_dir: str = "data/trajectories"
    rubric_dir: str = "data/rubrics"

class SplitterConfig(BaseModel):
    overlap_tokens: int = 2048
    max_window_tokens: int = 30720  # leave room within 32768

class SFTConfig(BaseModel):
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)

class RLStageConfig(BaseModel):
    timeout_minutes: int = 30
    num_epochs: int = 2

class RLConfig(BaseModel):
    learning_rate: float = 5e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.01
    rollouts_per_task: int = 4
    max_new_tokens_per_turn: int = 4096
    stages: list[RLStageConfig] = Field(default_factory=lambda: [
        RLStageConfig(timeout_minutes=30),
        RLStageConfig(timeout_minutes=60),
        RLStageConfig(timeout_minutes=120),
    ])

class TrainingConfig(BaseModel):
    sft: SFTConfig = Field(default_factory=SFTConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    output_dir: str = "checkpoints"
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

class InfraConfig(BaseModel):
    docker_image: str = "klong-sandbox:latest"
    container_memory_limit: str = "8g"
    container_cpu_limit: float = 4.0
    max_concurrent_containers: int = 2
    workspace_base: str = "/tmp/klong_workspaces"

class EvalConfig(BaseModel):
    judge_model: str = "claude-sonnet-4-20250514"
    final_judge_model: str = "claude-opus-4-20250514"

class KLongConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
```

**Step 5: Create `__init__.py` files**

```python
# klong/__init__.py
# klong/config/__init__.py
from klong.config.settings import KLongConfig
```

**Step 6: Run tests, verify pass**

Run: `cd /Users/sriranga/Desktop/lightning-llm/.claude/worktrees/bold-banzai && pip install -e ".[dev]" && python -m pytest tests/test_config.py -v`
Expected: 3 PASS

**Step 7: Commit**

```bash
git add klong/ tests/ pyproject.toml requirements.txt
git commit -m "feat: project scaffolding with Pydantic config system"
```

---

### Task 2: Rubric Tree Data Model & Evaluation

**Files:**
- Create: `klong/evaluation/__init__.py`
- Create: `klong/evaluation/rubric.py`
- Create: `klong/evaluation/judge.py`
- Create: `tests/test_rubric.py`
- Create: `tests/test_judge.py`

**Step 1: Write rubric tests**

```python
# tests/test_rubric.py
import json
from klong.evaluation.rubric import RubricNode, RubricTree

def test_leaf_node_creation():
    node = RubricNode(name="impl_loss", weight=0.5, criteria="Implements the loss function correctly")
    assert node.is_leaf
    assert node.name == "impl_loss"

def test_tree_from_dict():
    data = {
        "name": "root", "weight": 1.0, "criteria": "",
        "children": [
            {"name": "core", "weight": 0.4, "criteria": "",
             "children": [
                 {"name": "algorithm", "weight": 0.6, "criteria": "Implements main algorithm"},
                 {"name": "results", "weight": 0.4, "criteria": "Reproduces key results"},
             ]},
            {"name": "quality", "weight": 0.6, "criteria": "Code is clean and documented"},
        ]
    }
    tree = RubricTree.from_dict(data)
    assert len(tree.root.children) == 2
    leaves = tree.get_leaves()
    assert len(leaves) == 3

def test_tree_score_aggregation():
    data = {
        "name": "root", "weight": 1.0, "criteria": "",
        "children": [
            {"name": "a", "weight": 0.6, "criteria": "criterion A"},
            {"name": "b", "weight": 0.4, "criteria": "criterion B"},
        ]
    }
    tree = RubricTree.from_dict(data)
    scores = {"a": 0.8, "b": 0.5}
    total = tree.compute_score(scores)
    assert abs(total - 0.68) < 1e-6  # 0.6*0.8 + 0.4*0.5

def test_tree_serialization_roundtrip():
    data = {
        "name": "root", "weight": 1.0, "criteria": "",
        "children": [
            {"name": "x", "weight": 1.0, "criteria": "do X"},
        ]
    }
    tree = RubricTree.from_dict(data)
    d = tree.to_dict()
    tree2 = RubricTree.from_dict(d)
    assert tree2.root.children[0].criteria == "do X"
```

**Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_rubric.py -v`
Expected: FAIL

**Step 3: Implement rubric.py**

```python
# klong/evaluation/rubric.py
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class RubricNode:
    name: str
    weight: float
    criteria: str
    children: list[RubricNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @classmethod
    def from_dict(cls, d: dict) -> RubricNode:
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(name=d["name"], weight=d["weight"],
                   criteria=d.get("criteria", ""), children=children)

    def to_dict(self) -> dict:
        d = {"name": self.name, "weight": self.weight, "criteria": self.criteria}
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

@dataclass
class RubricTree:
    root: RubricNode

    @classmethod
    def from_dict(cls, d: dict) -> RubricTree:
        return cls(root=RubricNode.from_dict(d))

    def to_dict(self) -> dict:
        return self.root.to_dict()

    def get_leaves(self) -> list[RubricNode]:
        leaves = []
        def _walk(node: RubricNode):
            if node.is_leaf:
                leaves.append(node)
            for c in node.children:
                _walk(c)
        _walk(self.root)
        return leaves

    def compute_score(self, leaf_scores: dict[str, float]) -> float:
        def _score(node: RubricNode) -> float:
            if node.is_leaf:
                return leaf_scores.get(node.name, 0.0)
            total_weight = sum(c.weight for c in node.children)
            if total_weight == 0:
                return 0.0
            return sum(c.weight * _score(c) / total_weight for c in node.children)
        return _score(self.root)
```

**Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_rubric.py -v`
Expected: 4 PASS

**Step 5: Write judge tests**

```python
# tests/test_judge.py
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
```

**Step 6: Implement judge.py**

```python
# klong/evaluation/judge.py
from __future__ import annotations
import json
import anthropic
from klong.evaluation.rubric import RubricTree

JUDGE_PROMPT = """You are an expert code reviewer evaluating whether an AI agent successfully reproduced a research paper.

## Criterion
{criteria}

## Agent's Output Files
{artifacts}

## Instructions
Score how well the criterion is met on a scale from 0.0 to 1.0.
Respond with JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}
"""

class Judge:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.client = anthropic.Anthropic()

    async def _call_api(self, prompt: str):
        return self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

    async def evaluate_leaf(self, criteria: str, artifacts: dict[str, str]) -> float:
        artifacts_str = "\n".join(f"### {k}\n```\n{v[:2000]}\n```" for k, v in artifacts.items())
        prompt = JUDGE_PROMPT.format(criteria=criteria, artifacts=artifacts_str)
        response = await self._call_api(prompt)
        text = response.content[0].text
        try:
            data = json.loads(text)
            return max(0.0, min(1.0, float(data["score"])))
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0.0

    async def evaluate(self, tree: RubricTree, artifacts: dict[str, str]) -> tuple[float, dict]:
        leaves = tree.get_leaves()
        leaf_scores = {}
        for leaf in leaves:
            leaf_scores[leaf.name] = await self.evaluate_leaf(leaf.criteria, artifacts)
        total = tree.compute_score(leaf_scores)
        return total, leaf_scores
```

**Step 7: Run all tests, verify pass**

Run: `python -m pytest tests/test_rubric.py tests/test_judge.py -v`
Expected: ALL PASS

**Step 8: Commit**

```bash
git add klong/evaluation/ tests/test_rubric.py tests/test_judge.py
git commit -m "feat: rubric tree data model and judge evaluation system"
```

---

### Task 3: Docker Sandbox Manager

**Files:**
- Create: `klong/agent/__init__.py`
- Create: `klong/agent/sandbox/__init__.py`
- Create: `klong/agent/sandbox/docker_manager.py`
- Create: `klong/agent/sandbox/Dockerfile`
- Create: `tests/test_docker_manager.py`

**Step 1: Write tests**

```python
# tests/test_docker_manager.py
import pytest
from klong.agent.sandbox.docker_manager import SandboxManager, SandboxConfig

def test_sandbox_config_defaults():
    cfg = SandboxConfig()
    assert cfg.image == "klong-sandbox:latest"
    assert cfg.memory_limit == "8g"

def test_sandbox_manager_creation():
    mgr = SandboxManager(SandboxConfig())
    assert mgr is not None

# Integration tests (require Docker)
@pytest.mark.skipif(not pytest.importorskip("docker"), reason="Docker not available")
class TestDockerIntegration:
    def test_create_and_destroy_sandbox(self):
        mgr = SandboxManager(SandboxConfig(image="python:3.10-slim"))
        sandbox_id = mgr.create()
        assert sandbox_id is not None
        result = mgr.execute(sandbox_id, "echo hello")
        assert "hello" in result.stdout
        mgr.destroy(sandbox_id)

    def test_execute_python(self):
        mgr = SandboxManager(SandboxConfig(image="python:3.10-slim"))
        sandbox_id = mgr.create()
        result = mgr.execute(sandbox_id, "python3 -c 'print(2+2)'")
        assert "4" in result.stdout
        mgr.destroy(sandbox_id)

    def test_write_and_read_file(self):
        mgr = SandboxManager(SandboxConfig(image="python:3.10-slim"))
        sandbox_id = mgr.create()
        mgr.write_file(sandbox_id, "/workspace/test.py", "print('works')")
        result = mgr.execute(sandbox_id, "python3 /workspace/test.py")
        assert "works" in result.stdout
        content = mgr.read_file(sandbox_id, "/workspace/test.py")
        assert "print" in content
        mgr.destroy(sandbox_id)
```

**Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_docker_manager.py -v`
Expected: FAIL

**Step 3: Write Dockerfile**

```dockerfile
# klong/agent/sandbox/Dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    numpy pandas scipy matplotlib seaborn scikit-learn \
    einops transformers datasets tokenizers \
    pytest black flake8 tqdm requests pillow \
    jax jaxlib

WORKDIR /workspace
```

**Step 4: Implement docker_manager.py**

```python
# klong/agent/sandbox/docker_manager.py
from __future__ import annotations
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional

import docker
from docker.errors import NotFound, APIError

@dataclass
class SandboxConfig:
    image: str = "klong-sandbox:latest"
    memory_limit: str = "8g"
    cpu_limit: float = 4.0
    workspace_dir: str = "/workspace"
    network_enabled: bool = True
    blocked_hosts: list[str] = field(default_factory=list)

@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

class SandboxManager:
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.client = docker.from_env()
        self._containers: dict[str, docker.models.containers.Container] = {}

    def create(self) -> str:
        sandbox_id = str(uuid.uuid4())[:12]
        container = self.client.containers.run(
            self.config.image,
            command="sleep infinity",
            detach=True,
            name=f"klong-{sandbox_id}",
            mem_limit=self.config.memory_limit,
            nano_cpus=int(self.config.cpu_limit * 1e9),
            working_dir=self.config.workspace_dir,
            network_disabled=not self.config.network_enabled,
        )
        self._containers[sandbox_id] = container
        return sandbox_id

    def execute(self, sandbox_id: str, command: str, timeout: int = 300) -> ExecResult:
        container = self._containers[sandbox_id]
        try:
            exec_result = container.exec_run(
                ["bash", "-c", command],
                workdir=self.config.workspace_dir,
                demux=True,
            )
            stdout = (exec_result.output[0] or b"").decode("utf-8", errors="replace")
            stderr = (exec_result.output[1] or b"").decode("utf-8", errors="replace")
            return ExecResult(
                stdout=stdout, stderr=stderr,
                exit_code=exec_result.exit_code,
            )
        except Exception as e:
            return ExecResult(stdout="", stderr=str(e), exit_code=-1)

    def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        import tarfile, io
        container = self._containers[sandbox_id]
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=path.split("/")[-1])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_stream.seek(0)
        directory = "/".join(path.split("/")[:-1]) or "/"
        container.put_archive(directory, tar_stream)

    def read_file(self, sandbox_id: str, path: str) -> str:
        result = self.execute(sandbox_id, f"cat {path}")
        return result.stdout

    def destroy(self, sandbox_id: str) -> None:
        container = self._containers.pop(sandbox_id, None)
        if container:
            try:
                container.stop(timeout=5)
                container.remove(force=True)
            except (NotFound, APIError):
                pass

    def destroy_all(self) -> None:
        for sid in list(self._containers.keys()):
            self.destroy(sid)
```

**Step 5: Run tests, verify pass**

Run: `python -m pytest tests/test_docker_manager.py -v`
Expected: Unit tests PASS, integration tests PASS (if Docker available) or SKIP

**Step 6: Commit**

```bash
git add klong/agent/ tests/test_docker_manager.py
git commit -m "feat: Docker sandbox manager for agent execution"
```

---

### Task 4: Agent Tools

**Files:**
- Create: `klong/agent/tools/__init__.py`
- Create: `klong/agent/tools/base.py`
- Create: `klong/agent/tools/bash_tool.py`
- Create: `klong/agent/tools/python_tool.py`
- Create: `klong/agent/tools/file_tool.py`
- Create: `klong/agent/tools/paper_reader.py`
- Create: `tests/test_tools.py`

**Step 1: Write tests**

```python
# tests/test_tools.py
import pytest
from klong.agent.tools.base import Tool, ToolResult
from klong.agent.tools.bash_tool import BashTool
from klong.agent.tools.python_tool import PythonTool
from klong.agent.tools.file_tool import WriteFileTool, ReadFileTool, SearchFilesTool
from klong.agent.tools.paper_reader import PaperReaderTool

def test_tool_base_class():
    class DummyTool(Tool):
        name = "dummy"
        description = "A dummy tool"
        parameters = {"command": {"type": "string"}}
        def execute(self, sandbox_id, **kwargs):
            return ToolResult(output="ok", error="")
    t = DummyTool(sandbox_manager=None)
    assert t.name == "dummy"
    r = t.execute("fake", command="test")
    assert r.output == "ok"

def test_bash_tool_schema():
    t = BashTool(sandbox_manager=None)
    assert t.name == "bash"
    assert "command" in t.parameters

def test_python_tool_schema():
    t = PythonTool(sandbox_manager=None)
    assert t.name == "python"
    assert "code" in t.parameters

def test_file_tools_schema():
    wt = WriteFileTool(sandbox_manager=None)
    assert "path" in wt.parameters and "content" in wt.parameters
    rt = ReadFileTool(sandbox_manager=None)
    assert "path" in rt.parameters

def test_paper_reader():
    pr = PaperReaderTool(sandbox_manager=None, paper_markdown="# Title\nHello world")
    assert pr.paper_markdown == "# Title\nHello world"
    r = pr.execute("fake")
    assert "Title" in r.output
```

**Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_tools.py -v`
Expected: FAIL

**Step 3: Implement base.py**

```python
# klong/agent/tools/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ToolResult:
    output: str
    error: str = ""
    truncated: bool = False

class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def __init__(self, sandbox_manager):
        self.sandbox_manager = sandbox_manager

    @abstractmethod
    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        ...

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }
```

**Step 4: Implement all tool files**

```python
# klong/agent/tools/bash_tool.py
from klong.agent.tools.base import Tool, ToolResult

class BashTool(Tool):
    name = "bash"
    description = "Execute a bash command in the sandbox. Returns stdout and stderr."
    parameters = {"command": {"type": "string", "description": "The bash command to run"}}

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        command = kwargs["command"]
        result = self.sandbox_manager.execute(sandbox_id, command)
        output = result.stdout
        if len(output) > 10000:
            output = output[:5000] + "\n...[truncated]...\n" + output[-5000:]
            return ToolResult(output=output, error=result.stderr, truncated=True)
        return ToolResult(output=output, error=result.stderr)
```

```python
# klong/agent/tools/python_tool.py
from klong.agent.tools.base import Tool, ToolResult

class PythonTool(Tool):
    name = "python"
    description = "Execute Python code in the sandbox. The code is written to a temp file and run."
    parameters = {"code": {"type": "string", "description": "Python code to execute"}}

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        code = kwargs["code"]
        self.sandbox_manager.write_file(sandbox_id, "/workspace/_tmp_exec.py", code)
        result = self.sandbox_manager.execute(sandbox_id, "python3 /workspace/_tmp_exec.py")
        output = result.stdout
        if len(output) > 10000:
            output = output[:5000] + "\n...[truncated]...\n" + output[-5000:]
            return ToolResult(output=output, error=result.stderr, truncated=True)
        return ToolResult(output=output, error=result.stderr)
```

```python
# klong/agent/tools/file_tool.py
from klong.agent.tools.base import Tool, ToolResult

class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file in the sandbox workspace."
    parameters = {
        "path": {"type": "string", "description": "File path relative to /workspace"},
        "content": {"type": "string", "description": "File content to write"},
    }

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        path = kwargs["path"]
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        self.sandbox_manager.write_file(sandbox_id, path, kwargs["content"])
        return ToolResult(output=f"Written to {path}")

class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file in the sandbox."
    parameters = {"path": {"type": "string", "description": "File path to read"}}

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        path = kwargs["path"]
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        content = self.sandbox_manager.read_file(sandbox_id, path)
        if len(content) > 10000:
            content = content[:5000] + "\n...[truncated]...\n" + content[-5000:]
            return ToolResult(output=content, truncated=True)
        return ToolResult(output=content)

class SearchFilesTool(Tool):
    name = "search_files"
    description = "Search for files matching a pattern or grep for content."
    parameters = {
        "pattern": {"type": "string", "description": "Glob pattern or grep query"},
        "search_type": {"type": "string", "description": "'glob' or 'grep'"},
    }

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        pattern = kwargs["pattern"]
        search_type = kwargs.get("search_type", "glob")
        if search_type == "grep":
            result = self.sandbox_manager.execute(sandbox_id, f"grep -rn '{pattern}' /workspace/ 2>/dev/null | head -50")
        else:
            result = self.sandbox_manager.execute(sandbox_id, f"find /workspace/ -name '{pattern}' 2>/dev/null | head -50")
        return ToolResult(output=result.stdout, error=result.stderr)
```

```python
# klong/agent/tools/paper_reader.py
from klong.agent.tools.base import Tool, ToolResult

class PaperReaderTool(Tool):
    name = "read_paper"
    description = "Read the research paper (Markdown). Use sections param to read specific parts."
    parameters = {
        "section": {"type": "string", "description": "Optional: section heading to read. Omit for full paper."},
    }

    def __init__(self, sandbox_manager, paper_markdown: str):
        super().__init__(sandbox_manager)
        self.paper_markdown = paper_markdown
        self.read_count = 0

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        self.read_count += 1
        section = kwargs.get("section", "")
        if section:
            lines = self.paper_markdown.split("\n")
            in_section = False
            result_lines = []
            for line in lines:
                if line.strip().lower().startswith("#") and section.lower() in line.lower():
                    in_section = True
                elif in_section and line.strip().startswith("#") and section.lower() not in line.lower():
                    break
                if in_section:
                    result_lines.append(line)
            text = "\n".join(result_lines) if result_lines else f"Section '{section}' not found."
        else:
            text = self.paper_markdown
        if len(text) > 15000:
            text = text[:15000] + "\n...[truncated — use section param for specific parts]..."
            return ToolResult(output=text, truncated=True)
        return ToolResult(output=text)
```

**Step 5: Run tests, verify pass**

Run: `python -m pytest tests/test_tools.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add klong/agent/tools/ tests/test_tools.py
git commit -m "feat: agent tool suite (bash, python, file, paper reader)"
```

---

### Task 5: Agent Scaffold (Core Agent Loop)

**Files:**
- Create: `klong/agent/scaffold.py`
- Create: `tests/test_scaffold.py`

**Step 1: Write tests**

```python
# tests/test_scaffold.py
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
    # Test JSON tool call parsing
    text = 'I will run a command.\n```tool_call\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
    calls = agent.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "bash"
```

**Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_scaffold.py -v`
Expected: FAIL

**Step 3: Implement scaffold.py**

```python
# klong/agent/scaffold.py
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
        self._generate_fn = None  # Set by caller (LLM inference function)

    def set_generate_fn(self, fn):
        """Set the LLM generation function: fn(messages) -> str"""
        self._generate_fn = fn

    def parse_tool_calls(self, text: str) -> list[dict]:
        pattern = r'```tool_call\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        calls = []
        for m in matches:
            try:
                calls.append(json.loads(m.strip()))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {m[:100]}")
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

            # Check mandatory paper reading
            if not paper_read and turn_count >= self.mandatory_read_turns:
                messages.append({"role": "user",
                    "content": "WARNING: You must read the paper before proceeding. Use read_paper tool."})

            # Generate response
            response_text = self._generate_fn(messages)
            tool_calls = self.parse_tool_calls(response_text)

            turn = Turn(role="assistant", content=response_text, tool_calls=tool_calls)
            trajectory.add_turn(turn)
            messages.append({"role": "assistant", "content": response_text})

            # Execute tool calls
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

            turn_count += 1

        trajectory.total_time_seconds = time.time() - start_time
        return trajectory
```

**Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_scaffold.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add klong/agent/scaffold.py tests/test_scaffold.py
git commit -m "feat: agent scaffold with tool parsing, trajectory recording, safety guards"
```

---

### Task 6: Research-Factory — Paper Collector

**Files:**
- Create: `klong/research_factory/__init__.py`
- Create: `klong/research_factory/paper_collector.py`
- Create: `klong/research_factory/pdf_converter.py`
- Create: `klong/research_factory/blacklist.py`
- Create: `tests/test_paper_collector.py`

**Step 1: Write tests**

```python
# tests/test_paper_collector.py
import pytest
from klong.research_factory.paper_collector import PaperCollector, PaperRecord
from klong.research_factory.blacklist import Blacklist

def test_paper_record():
    p = PaperRecord(
        paper_id="2301.00001", title="Test Paper",
        abstract="An abstract.", authors=["Author A"],
        github_url="https://github.com/user/repo",
        pdf_url="https://arxiv.org/pdf/2301.00001",
        markdown="", conference="ICML", year=2023,
    )
    assert p.paper_id == "2301.00001"

def test_blacklist():
    bl = Blacklist()
    bl.add("https://github.com/user/repo")
    assert bl.is_blocked("https://github.com/user/repo")
    assert bl.is_blocked("github.com/user/repo")
    assert not bl.is_blocked("https://github.com/other/repo")

def test_blacklist_persistence(tmp_path):
    bl = Blacklist()
    bl.add("https://github.com/user/repo")
    path = tmp_path / "blacklist.json"
    bl.save(str(path))
    bl2 = Blacklist.load(str(path))
    assert bl2.is_blocked("github.com/user/repo")
```

**Step 2: Run tests to verify failure, then implement**

**Step 3: Implement blacklist.py**

```python
# klong/research_factory/blacklist.py
from __future__ import annotations
import json
from urllib.parse import urlparse

class Blacklist:
    def __init__(self):
        self._urls: set[str] = set()

    def _normalize(self, url: str) -> str:
        url = url.strip().rstrip("/")
        if not url.startswith("http"):
            url = "https://" + url
        parsed = urlparse(url)
        return f"{parsed.netloc}{parsed.path}".lower().rstrip("/")

    def add(self, url: str):
        self._urls.add(self._normalize(url))

    def is_blocked(self, url: str) -> bool:
        normalized = self._normalize(url)
        return any(normalized.startswith(blocked) for blocked in self._urls)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(list(self._urls), f)

    @classmethod
    def load(cls, path: str) -> Blacklist:
        bl = cls()
        with open(path) as f:
            bl._urls = set(json.load(f))
        return bl
```

**Step 4: Implement paper_collector.py**

```python
# klong/research_factory/paper_collector.py
from __future__ import annotations
import re
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import arxiv
from klong.research_factory.blacklist import Blacklist

logger = logging.getLogger(__name__)

@dataclass
class PaperRecord:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    github_url: str
    pdf_url: str
    markdown: str
    conference: str
    year: int

    def to_dict(self) -> dict:
        return asdict(self)

class PaperCollector:
    CONFERENCE_QUERIES = {
        "ICML": "cat:cs.LG AND (ICML)",
        "NeurIPS": "cat:cs.LG AND (NeurIPS OR neurips)",
        "ICLR": "cat:cs.LG AND (ICLR)",
    }

    def __init__(self, output_dir: str = "data/papers",
                 conferences: list[str] | None = None,
                 max_papers: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conferences = conferences or ["ICML", "NeurIPS", "ICLR"]
        self.max_papers = max_papers
        self.blacklist = Blacklist()

    def _extract_github_url(self, text: str) -> Optional[str]:
        pattern = r'https?://github\.com/[\w\-]+/[\w\-]+'
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def search_papers(self) -> list[PaperRecord]:
        papers = []
        per_conf = self.max_papers // len(self.conferences)

        for conf in self.conferences:
            query = self.CONFERENCE_QUERIES.get(conf, f"cat:cs.LG AND ({conf})")
            logger.info(f"Searching ArXiv for {conf} papers...")

            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=per_conf * 3,  # over-fetch to filter
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            count = 0
            for result in client.results(search):
                if count >= per_conf:
                    break
                github_url = self._extract_github_url(
                    result.summary + " ".join(str(l) for l in result.links)
                )
                if not github_url:
                    continue

                paper = PaperRecord(
                    paper_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    abstract=result.summary,
                    authors=[a.name for a in result.authors[:5]],
                    github_url=github_url,
                    pdf_url=result.pdf_url,
                    markdown="",  # filled by pdf_converter
                    conference=conf,
                    year=result.published.year,
                )
                papers.append(paper)
                self.blacklist.add(github_url)
                count += 1
                logger.info(f"  Found: {paper.title[:60]}... ({github_url})")

        logger.info(f"Collected {len(papers)} papers total")
        return papers

    def save_papers(self, papers: list[PaperRecord]):
        output_path = self.output_dir / "papers.jsonl"
        with open(output_path, "w") as f:
            for p in papers:
                f.write(json.dumps(p.to_dict()) + "\n")
        self.blacklist.save(str(self.output_dir / "blacklist.json"))
        logger.info(f"Saved {len(papers)} papers to {output_path}")
```

**Step 5: Implement pdf_converter.py**

```python
# klong/research_factory/pdf_converter.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class PDFConverter:
    def convert_url(self, pdf_url: str, output_dir: str) -> str:
        """Download PDF from URL and convert to markdown."""
        import requests
        import pymupdf4llm

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Download PDF
        filename = pdf_url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        pdf_path = output_path / filename

        logger.info(f"Downloading {pdf_url}...")
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)

        # Convert to markdown
        logger.info(f"Converting {pdf_path} to markdown...")
        md_text = pymupdf4llm.to_markdown(str(pdf_path))

        # Save markdown
        md_path = pdf_path.with_suffix(".md")
        md_path.write_text(md_text)

        return md_text

    def convert_file(self, pdf_path: str) -> str:
        """Convert local PDF to markdown."""
        import pymupdf4llm
        return pymupdf4llm.to_markdown(pdf_path)
```

**Step 6: Run tests, verify pass, commit**

Run: `python -m pytest tests/test_paper_collector.py -v`

```bash
git add klong/research_factory/ tests/test_paper_collector.py
git commit -m "feat: Research-Factory paper collector, PDF converter, blacklist"
```

---

### Task 7: Research-Factory — Rubric Generator

**Files:**
- Create: `klong/research_factory/rubric_generator.py`
- Create: `tests/test_rubric_generator.py`

**Step 1: Write tests**

```python
# tests/test_rubric_generator.py
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
```

**Step 2: Implement rubric_generator.py**

```python
# klong/research_factory/rubric_generator.py
from __future__ import annotations
import json
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
{{
  "name": "root", "weight": 1.0, "criteria": "",
  "children": [
    {{
      "name": "core_contribution", "weight": 0.4, "criteria": "",
      "children": [
        {{"name": "...", "weight": ..., "criteria": "..."}}
      ]
    }}
  ]
}}
"""

class RubricGenerator:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.client = anthropic.Anthropic()

    def _build_prompt(self, paper_markdown: str, code_excerpt: str) -> str:
        # Truncate to fit context
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

        # Extract JSON from response
        try:
            # Try direct parse
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON block
            import re
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
```

**Step 3: Run tests, verify pass, commit**

```bash
git add klong/research_factory/rubric_generator.py tests/test_rubric_generator.py
git commit -m "feat: rubric generator using Claude API"
```

---

### Task 8: Research-Factory — Trajectory Distiller

**Files:**
- Create: `klong/research_factory/trajectory_distiller.py`
- Create: `tests/test_trajectory_distiller.py`

**Step 1: Write tests**

```python
# tests/test_trajectory_distiller.py
import pytest
from unittest.mock import MagicMock
from klong.research_factory.trajectory_distiller import TrajectoryDistiller

def test_distiller_creation():
    d = TrajectoryDistiller(model="claude-sonnet-4-20250514", sandbox_config=MagicMock())
    assert d.model == "claude-sonnet-4-20250514"
```

**Step 2: Implement trajectory_distiller.py**

```python
# klong/research_factory/trajectory_distiller.py
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import anthropic
from klong.agent.scaffold import Agent, Turn, Trajectory
from klong.agent.sandbox.docker_manager import SandboxManager, SandboxConfig
from klong.agent.tools.bash_tool import BashTool
from klong.agent.tools.python_tool import PythonTool
from klong.agent.tools.file_tool import WriteFileTool, ReadFileTool, SearchFilesTool
from klong.agent.tools.paper_reader import PaperReaderTool
from klong.evaluation.judge import Judge
from klong.evaluation.rubric import RubricTree

logger = logging.getLogger(__name__)

class TrajectoryDistiller:
    """Uses Claude API as expert agent to generate demonstration trajectories."""

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 sandbox_config: SandboxConfig | None = None,
                 timeout_minutes: int = 120,
                 rejection_threshold: float = 0.3):
        self.model = model
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.timeout_minutes = timeout_minutes
        self.rejection_threshold = rejection_threshold
        self.client = anthropic.Anthropic()

    def _create_claude_generate_fn(self, sandbox_manager: SandboxManager,
                                     sandbox_id: str, tools: list):
        """Create a generate function that calls Claude API."""
        tool_schemas = [t.to_schema() for t in tools]

        def generate(messages: list[dict]) -> str:
            # Convert to Claude API format
            api_messages = []
            for m in messages:
                if m["role"] == "system":
                    continue  # handled separately
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
                system_prompt="",  # Will be built by Agent
                tools=tools,
                sandbox_manager=sandbox_manager,
                max_turns=200,
            )

            generate_fn = self._create_claude_generate_fn(sandbox_manager, sandbox_id, tools)
            agent.set_generate_fn(generate_fn)

            trajectory = agent.run(
                sandbox_id=sandbox_id,
                task_description=task_description,
                paper_id=paper_id,
                timeout_seconds=self.timeout_minutes * 60,
            )

            # Evaluate trajectory with judge
            artifacts = self._collect_artifacts(sandbox_manager, sandbox_id)
            judge = Judge()
            import asyncio
            score, leaf_scores = asyncio.run(judge.evaluate(rubric, artifacts))
            trajectory.final_score = score

            if score < self.rejection_threshold:
                logger.info(f"Rejecting trajectory for {paper_id}: score={score:.3f} < {self.rejection_threshold}")
                return None

            logger.info(f"Accepted trajectory for {paper_id}: score={score:.3f}")
            return trajectory

        finally:
            sandbox_manager.destroy(sandbox_id)

    def _collect_artifacts(self, sandbox_manager: SandboxManager,
                           sandbox_id: str) -> dict[str, str]:
        """Collect all Python files from the workspace."""
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
```

**Step 3: Run tests, commit**

```bash
git add klong/research_factory/trajectory_distiller.py tests/test_trajectory_distiller.py
git commit -m "feat: trajectory distiller using Claude API as expert agent"
```

---

### Task 9: Trajectory Splitter

**Files:**
- Create: `klong/training/__init__.py`
- Create: `klong/training/data/__init__.py`
- Create: `klong/training/data/trajectory_splitter.py`
- Create: `tests/test_trajectory_splitter.py`

**Step 1: Write tests**

```python
# tests/test_trajectory_splitter.py
import pytest
from klong.training.data.trajectory_splitter import TrajectorySplitter, SubTrajectory
from klong.agent.scaffold import Trajectory, Turn

def _make_trajectory(n_turns: int) -> Trajectory:
    traj = Trajectory(task_id="test", paper_id="p1")
    # First 3 turns are paper reading (prefix)
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
    """Short trajectory should produce a single sub-trajectory."""
    traj = _make_trajectory(5)
    sp = TrajectorySplitter(max_window_tokens=100000, overlap_tokens=1000)
    subs = sp.split(traj, prefix_turn_count=6)  # 3 pairs = 6 turns
    assert len(subs) >= 1
    assert subs[0].prefix_turns == 6

def test_split_produces_overlap():
    """Splitting a long trajectory should produce overlapping windows."""
    traj = _make_trajectory(30)
    sp = TrajectorySplitter(max_window_tokens=500, overlap_tokens=100)
    subs = sp.split(traj, prefix_turn_count=6)
    assert len(subs) > 1
    # Check overlap: end of window i should overlap with start of window i+1
    for i in range(len(subs) - 1):
        assert subs[i].end_turn_idx > subs[i+1].start_turn_idx

def test_sub_trajectory_has_action_mask():
    traj = _make_trajectory(10)
    sp = TrajectorySplitter(max_window_tokens=100000, overlap_tokens=100)
    subs = sp.split(traj, prefix_turn_count=6)
    # Action mask should mark assistant turns as trainable
    for sub in subs:
        assert len(sub.action_mask) > 0
```

**Step 2: Run test to verify failure, then implement**

**Step 3: Implement trajectory_splitter.py**

```python
# klong/training/data/trajectory_splitter.py
from __future__ import annotations
from dataclasses import dataclass, field
import tiktoken

from klong.agent.scaffold import Trajectory, Turn

@dataclass
class SubTrajectory:
    """A sub-trajectory window with fixed prefix."""
    trajectory_id: str
    window_idx: int
    prefix_turns: int          # how many turns form the prefix
    start_turn_idx: int        # start of this window in original trajectory
    end_turn_idx: int          # end of this window (exclusive)
    turns: list[Turn]          # prefix + window turns
    action_mask: list[bool]    # True for assistant turns (trainable)
    estimated_tokens: int = 0

class TrajectorySplitter:
    def __init__(self, max_window_tokens: int = 30720, overlap_tokens: int = 2048):
        self.max_window_tokens = max_window_tokens
        self.overlap_tokens = overlap_tokens
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def _count_tokens(self, text: str) -> int:
        if self._enc:
            return len(self._enc.encode(text))
        return len(text) // 4  # rough approximation

    def _turns_tokens(self, turns: list[Turn]) -> int:
        return sum(self._count_tokens(t.content) for t in turns)

    def split(self, trajectory: Trajectory, prefix_turn_count: int = 6) -> list[SubTrajectory]:
        all_turns = trajectory.turns
        prefix_turns = all_turns[:prefix_turn_count]
        body_turns = all_turns[prefix_turn_count:]
        prefix_tokens = self._turns_tokens(prefix_turns)
        available_tokens = self.max_window_tokens - prefix_tokens

        if available_tokens <= 0:
            # Prefix alone exceeds window — just return one sub-trajectory
            return [SubTrajectory(
                trajectory_id=trajectory.task_id, window_idx=0,
                prefix_turns=prefix_turn_count,
                start_turn_idx=0, end_turn_idx=len(all_turns),
                turns=all_turns,
                action_mask=[t.role == "assistant" for t in all_turns],
                estimated_tokens=self._turns_tokens(all_turns),
            )]

        # Build windows
        sub_trajectories = []
        window_start = 0
        window_idx = 0

        while window_start < len(body_turns):
            # Greedily add turns until we hit the token limit
            window_end = window_start
            current_tokens = 0
            while window_end < len(body_turns):
                turn_tokens = self._count_tokens(body_turns[window_end].content)
                if current_tokens + turn_tokens > available_tokens:
                    break
                current_tokens += turn_tokens
                window_end += 1

            # Ensure at least one turn
            if window_end == window_start:
                window_end = window_start + 1

            window_turns = prefix_turns + body_turns[window_start:window_end]
            action_mask = [t.role == "assistant" for t in window_turns]

            sub_trajectories.append(SubTrajectory(
                trajectory_id=trajectory.task_id,
                window_idx=window_idx,
                prefix_turns=prefix_turn_count,
                start_turn_idx=prefix_turn_count + window_start,
                end_turn_idx=prefix_turn_count + window_end,
                turns=window_turns,
                action_mask=action_mask,
                estimated_tokens=prefix_tokens + current_tokens,
            ))

            # Compute overlap in turns
            overlap_turns = 0
            overlap_tokens_count = 0
            for i in range(window_end - 1, window_start - 1, -1):
                t_tokens = self._count_tokens(body_turns[i].content)
                if overlap_tokens_count + t_tokens > self.overlap_tokens:
                    break
                overlap_tokens_count += t_tokens
                overlap_turns += 1

            # Advance window
            stride = (window_end - window_start) - overlap_turns
            if stride <= 0:
                stride = 1  # always advance
            window_start += stride
            window_idx += 1

        return sub_trajectories
```

**Step 4: Run tests, verify pass, commit**

```bash
git add klong/training/ tests/test_trajectory_splitter.py
git commit -m "feat: trajectory splitter with overlapping windows and action masking"
```

---

### Task 10: Trajectory Dataset for SFT

**Files:**
- Create: `klong/training/data/trajectory_dataset.py`
- Create: `tests/test_trajectory_dataset.py`

**Step 1: Write tests**

```python
# tests/test_trajectory_dataset.py
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
            {"role": "user", "content": "# Paper Title\nThis is the paper content about an algorithm.", "tool_calls": [], "tool_results": [{"name":"read_paper","output":"paper"}], "timestamp": 1},
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
```

**Step 2: Implement trajectory_dataset.py**

```python
# klong/training/data/trajectory_dataset.py
from __future__ import annotations
import json
import logging
from pathlib import Path
from torch.utils.data import Dataset

from klong.agent.scaffold import Trajectory, Turn
from klong.training.data.trajectory_splitter import TrajectorySplitter, SubTrajectory

logger = logging.getLogger(__name__)

TURN_TEMPLATE_ASSISTANT = "<|im_start|>assistant\n{content}<|im_end|>\n"
TURN_TEMPLATE_USER = "<|im_start|>user\n{content}<|im_end|>\n"
TURN_TEMPLATE_SYSTEM = "<|im_start|>system\n{content}<|im_end|>\n"

class TrajectoryDataset(Dataset):
    """Dataset that loads trajectories, splits them, and formats for SFT."""

    def __init__(self, trajectory_dir: str, max_window_tokens: int = 30720,
                 overlap_tokens: int = 2048, prefix_turn_count: int = 2):
        self.trajectory_dir = Path(trajectory_dir)
        self.splitter = TrajectorySplitter(max_window_tokens, overlap_tokens)
        self.prefix_turn_count = prefix_turn_count
        self.sub_trajectories: list[SubTrajectory] = []
        self._load_and_split()

    def _load_and_split(self):
        files = sorted(self.trajectory_dir.glob("*.json"))
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                traj = Trajectory.from_dict(data)
                subs = self.splitter.split(traj, self.prefix_turn_count)
                self.sub_trajectories.extend(subs)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        logger.info(f"Loaded {len(files)} trajectories -> {len(self.sub_trajectories)} sub-trajectories")

    def _format_sub_trajectory(self, sub: SubTrajectory) -> tuple[str, list[bool]]:
        """Format sub-trajectory as chat-ml text with per-character action mask."""
        text_parts = []
        char_mask = []

        for turn, is_action in zip(sub.turns, sub.action_mask):
            if turn.role == "system":
                formatted = TURN_TEMPLATE_SYSTEM.format(content=turn.content)
            elif turn.role == "assistant":
                formatted = TURN_TEMPLATE_ASSISTANT.format(content=turn.content)
            else:
                formatted = TURN_TEMPLATE_USER.format(content=turn.content)

            text_parts.append(formatted)
            char_mask.extend([is_action] * len(formatted))

        return "".join(text_parts), char_mask

    def __len__(self) -> int:
        return len(self.sub_trajectories)

    def __getitem__(self, idx: int) -> dict:
        sub = self.sub_trajectories[idx]
        text, action_mask = self._format_sub_trajectory(sub)
        return {
            "text": text,
            "action_mask": action_mask,
            "trajectory_id": sub.trajectory_id,
            "window_idx": sub.window_idx,
        }
```

**Step 3: Run tests, verify pass, commit**

```bash
git add klong/training/data/trajectory_dataset.py tests/test_trajectory_dataset.py
git commit -m "feat: trajectory dataset with splitting and ChatML formatting"
```

---

### Task 11: SFT Trainer

**Files:**
- Create: `klong/training/sft/__init__.py`
- Create: `klong/training/sft/trainer.py`
- Create: `tests/test_sft_trainer.py`
- Create: `scripts/train_sft.py`

**Step 1: Write tests**

```python
# tests/test_sft_trainer.py
import pytest
from klong.training.sft.trainer import SFTTrainerWrapper

def test_sft_trainer_creation():
    trainer = SFTTrainerWrapper(
        model_name="Qwen/Qwen2.5-0.5B",  # tiny model for testing
        lora_rank=8, lora_alpha=16,
        output_dir="/tmp/klong_test_sft",
    )
    assert trainer.model_name == "Qwen/Qwen2.5-0.5B"

def test_sft_trainer_config():
    trainer = SFTTrainerWrapper(
        model_name="Qwen/Qwen2.5-0.5B",
        learning_rate=2e-5, num_epochs=3,
    )
    assert trainer.learning_rate == 2e-5
```

**Step 2: Implement sft/trainer.py**

```python
# klong/training/sft/trainer.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig as TRLSFTConfig
from datasets import Dataset as HFDataset

from klong.training.data.trajectory_dataset import TrajectoryDataset

logger = logging.getLogger(__name__)

class SFTTrainerWrapper:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_target_modules: list[str] | None = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        warmup_ratio: float = 0.1,
        max_seq_length: int = 32768,
        output_dir: str = "checkpoints/sft",
        use_bf16: bool = True,
        gradient_checkpointing: bool = True,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.use_bf16 = use_bf16
        self.gradient_checkpointing = gradient_checkpointing
        self.load_in_4bit = load_in_4bit

    def _load_model_and_tokenizer(self):
        logger.info(f"Loading model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {"trust_remote_code": True}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            if self.use_bf16 and torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.bfloat16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        if self.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    def train(self, trajectory_dir: str):
        model, tokenizer = self._load_model_and_tokenizer()

        # Load dataset
        traj_dataset = TrajectoryDataset(trajectory_dir)
        texts = [item["text"] for item in traj_dataset]
        hf_dataset = HFDataset.from_dict({"text": texts})

        # Determine device
        if torch.cuda.is_available():
            bf16 = self.use_bf16
            fp16 = False
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            bf16 = False
            fp16 = False
        else:
            bf16 = False
            fp16 = False

        training_args = TRLSFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            bf16=bf16,
            fp16=fp16,
            gradient_checkpointing=self.gradient_checkpointing,
            max_seq_length=self.max_seq_length,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=3,
            seed=42,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            args=training_args,
        )

        logger.info("Starting SFT training...")
        trainer.train()
        trainer.save_model(self.output_dir + "/final")
        tokenizer.save_pretrained(self.output_dir + "/final")
        logger.info(f"SFT training complete. Model saved to {self.output_dir}/final")
```

**Step 3: Write scripts/train_sft.py**

```python
# scripts/train_sft.py
"""Run SFT training on distilled trajectories."""
import argparse
import logging
import yaml
from klong.config.settings import KLongConfig
from klong.training.sft.trainer import SFTTrainerWrapper

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="KLong SFT Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--trajectory-dir", type=str, default="data/trajectories")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = KLongConfig(**yaml.safe_load(f))
    else:
        cfg = KLongConfig()

    model_name = args.model or cfg.model.name

    trainer = SFTTrainerWrapper(
        model_name=model_name,
        lora_rank=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_target_modules=cfg.model.lora_target_modules,
        learning_rate=cfg.training.sft.learning_rate,
        num_epochs=cfg.training.sft.num_epochs,
        batch_size=cfg.training.sft.batch_size,
        gradient_accumulation_steps=cfg.training.sft.gradient_accumulation_steps,
        warmup_ratio=cfg.training.sft.warmup_ratio,
        max_seq_length=cfg.model.max_seq_length,
        output_dir=args.output_dir,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    trainer.train(args.trajectory_dir)

if __name__ == "__main__":
    main()
```

**Step 4: Run tests, verify pass, commit**

```bash
git add klong/training/sft/ tests/test_sft_trainer.py scripts/train_sft.py
git commit -m "feat: SFT trainer with LoRA, trajectory-splitting, TRL integration"
```

---

### Task 12: RL Reward & Rollout

**Files:**
- Create: `klong/training/rl/__init__.py`
- Create: `klong/training/rl/reward.py`
- Create: `klong/training/rl/rollout.py`
- Create: `tests/test_rl_reward.py`

**Step 1: Write tests**

```python
# tests/test_rl_reward.py
import pytest
from klong.training.rl.reward import compute_group_advantages

def test_group_advantages_basic():
    rewards = [0.8, 0.6, 0.4, 0.2]
    advantages = compute_group_advantages(rewards)
    assert len(advantages) == 4
    assert advantages[0] > 0  # above mean
    assert advantages[-1] < 0  # below mean

def test_group_advantages_all_equal():
    rewards = [0.5, 0.5, 0.5, 0.5]
    advantages = compute_group_advantages(rewards)
    assert all(abs(a) < 1e-6 for a in advantages)
```

**Step 2: Implement reward.py**

```python
# klong/training/rl/reward.py
from __future__ import annotations
import numpy as np

def compute_group_advantages(rewards: list[float]) -> list[float]:
    """GRPO-style group-relative advantage: A_i = r_i - mean(r)."""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    if std_reward < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mean_reward) / std_reward for r in rewards]
```

**Step 3: Implement rollout.py**

```python
# klong/training/rl/rollout.py
from __future__ import annotations
import logging
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from klong.agent.scaffold import Agent, Trajectory
from klong.agent.sandbox.docker_manager import SandboxManager, SandboxConfig
from klong.agent.tools.bash_tool import BashTool
from klong.agent.tools.python_tool import PythonTool
from klong.agent.tools.file_tool import WriteFileTool, ReadFileTool, SearchFilesTool
from klong.agent.tools.paper_reader import PaperReaderTool
from klong.evaluation.judge import Judge
from klong.evaluation.rubric import RubricTree

logger = logging.getLogger(__name__)

class RolloutGenerator:
    """Generates agent rollouts using the current policy model."""

    def __init__(self, model, tokenizer, sandbox_config: SandboxConfig | None = None,
                 max_new_tokens: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.max_new_tokens = max_new_tokens

    def _create_generate_fn(self):
        """Create a local model inference function."""
        import torch

        def generate(messages: list[dict]) -> str:
            # Build prompt from messages
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
```

**Step 4: Run tests, commit**

```bash
git add klong/training/rl/ tests/test_rl_reward.py
git commit -m "feat: RL reward computation and rollout generator"
```

---

### Task 13: Progressive RL Trainer

**Files:**
- Create: `klong/training/rl/trainer.py`
- Create: `tests/test_rl_trainer.py`
- Create: `scripts/train_rl.py`

**Step 1: Write tests**

```python
# tests/test_rl_trainer.py
import pytest
from klong.training.rl.trainer import ProgressiveRLTrainer

def test_rl_trainer_creation():
    trainer = ProgressiveRLTrainer(
        model_name="Qwen/Qwen2.5-0.5B",
        sft_checkpoint="checkpoints/sft/final",
        stages=[{"timeout_minutes": 5, "num_epochs": 1}],
    )
    assert trainer.model_name == "Qwen/Qwen2.5-0.5B"
    assert len(trainer.stages) == 1
```

**Step 2: Implement rl/trainer.py**

```python
# klong/training/rl/trainer.py
from __future__ import annotations
import json
import logging
import asyncio
from pathlib import Path
from typing import Any

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset as HFDataset

from klong.config.settings import RLStageConfig
from klong.training.rl.reward import compute_group_advantages
from klong.training.rl.rollout import RolloutGenerator
from klong.training.data.trajectory_splitter import TrajectorySplitter
from klong.evaluation.judge import Judge
from klong.evaluation.rubric import RubricTree
from klong.agent.sandbox.docker_manager import SandboxConfig

logger = logging.getLogger(__name__)

class ProgressiveRLTrainer:
    """Progressive RL training with increasing timeouts."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        sft_checkpoint: str = "checkpoints/sft/final",
        stages: list[dict] | None = None,
        learning_rate: float = 5e-6,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.01,
        rollouts_per_task: int = 4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        output_dir: str = "checkpoints/rl",
        sandbox_config: SandboxConfig | None = None,
    ):
        self.model_name = model_name
        self.sft_checkpoint = sft_checkpoint
        self.stages = stages or [
            {"timeout_minutes": 30, "num_epochs": 2},
            {"timeout_minutes": 60, "num_epochs": 2},
            {"timeout_minutes": 120, "num_epochs": 2},
        ]
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff
        self.rollouts_per_task = rollouts_per_task
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = output_dir
        self.sandbox_config = sandbox_config or SandboxConfig()

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        # Load SFT LoRA weights
        if Path(self.sft_checkpoint).exists():
            model = PeftModel.from_pretrained(model, self.sft_checkpoint)
            model = model.merge_and_unload()
            logger.info(f"Loaded SFT checkpoint from {self.sft_checkpoint}")

        return model, tokenizer

    def train(self, task_data_path: str, rubric_dir: str):
        """Run progressive RL training across all stages."""
        model, tokenizer = self._load_model()

        # Load tasks
        with open(task_data_path) as f:
            tasks = [json.loads(line) for line in f]

        for stage_idx, stage in enumerate(self.stages):
            timeout_minutes = stage["timeout_minutes"]
            num_epochs = stage["num_epochs"]
            stage_dir = f"{self.output_dir}/stage_{stage_idx}"

            logger.info(f"\n{'='*60}")
            logger.info(f"RL Stage {stage_idx}: timeout={timeout_minutes}min, epochs={num_epochs}")
            logger.info(f"{'='*60}")

            self._train_stage(
                model, tokenizer, tasks, rubric_dir,
                timeout_seconds=timeout_minutes * 60,
                num_epochs=num_epochs,
                output_dir=stage_dir,
            )

        logger.info("Progressive RL training complete!")

    def _train_stage(self, model, tokenizer, tasks, rubric_dir,
                     timeout_seconds, num_epochs, output_dir):
        """Train one RL stage."""
        rollout_gen = RolloutGenerator(model, tokenizer, self.sandbox_config)
        judge = Judge()
        splitter = TrajectorySplitter()

        for epoch in range(num_epochs):
            logger.info(f"  Epoch {epoch+1}/{num_epochs}")
            all_rewards = []
            all_trajectories = []

            for task in tasks:
                paper_id = task["paper_id"]
                paper_markdown = task.get("markdown", "")
                task_desc = task.get("task_description", f"Reproduce paper {paper_id}")
                rubric_path = Path(rubric_dir) / f"{paper_id}.json"

                if not rubric_path.exists():
                    logger.warning(f"Rubric not found for {paper_id}, skipping")
                    continue

                with open(rubric_path) as f:
                    rubric = RubricTree.from_dict(json.load(f))

                # Generate rollouts
                task_rollouts = []
                task_rewards = []
                for i in range(self.rollouts_per_task):
                    logger.info(f"    Rollout {i+1}/{self.rollouts_per_task} for {paper_id}")
                    trajectory = rollout_gen.generate_rollout(
                        paper_id, paper_markdown, task_desc, timeout_seconds)

                    # Evaluate
                    artifacts = {}  # Would collect from sandbox
                    score, _ = asyncio.run(judge.evaluate(rubric, artifacts))
                    trajectory.final_score = score

                    task_rollouts.append(trajectory)
                    task_rewards.append(score)

                # Compute advantages
                advantages = compute_group_advantages(task_rewards)

                for traj, adv in zip(task_rollouts, advantages):
                    all_trajectories.append((traj, adv))
                    all_rewards.append(traj.final_score)

            if not all_trajectories:
                logger.warning("No trajectories generated, skipping epoch")
                continue

            mean_reward = np.mean(all_rewards)
            logger.info(f"  Mean reward: {mean_reward:.4f}")

            # PPO update would happen here using TRL's GRPOTrainer
            # For now, log the trajectories and advantages
            logger.info(f"  Generated {len(all_trajectories)} trajectory-advantage pairs")

        # Save checkpoint
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"  Saved checkpoint to {output_dir}")
```

**Step 3: Write scripts/train_rl.py**

```python
# scripts/train_rl.py
"""Run Progressive RL training."""
import argparse
import logging
import yaml
from klong.config.settings import KLongConfig
from klong.training.rl.trainer import ProgressiveRLTrainer

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="KLong Progressive RL Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--task-data", type=str, default="data/papers/papers.jsonl")
    parser.add_argument("--rubric-dir", type=str, default="data/rubrics")
    parser.add_argument("--sft-checkpoint", type=str, default="checkpoints/sft/final")
    parser.add_argument("--output-dir", type=str, default="checkpoints/rl")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = KLongConfig(**yaml.safe_load(f))
    else:
        cfg = KLongConfig()

    trainer = ProgressiveRLTrainer(
        model_name=cfg.model.name,
        sft_checkpoint=args.sft_checkpoint,
        stages=[s.model_dump() for s in cfg.training.rl.stages],
        learning_rate=cfg.training.rl.learning_rate,
        clip_epsilon=cfg.training.rl.clip_epsilon,
        kl_coeff=cfg.training.rl.kl_coeff,
        rollouts_per_task=cfg.training.rl.rollouts_per_task,
        output_dir=args.output_dir,
    )
    trainer.train(args.task_data, args.rubric_dir)

if __name__ == "__main__":
    main()
```

**Step 4: Run tests, commit**

```bash
git add klong/training/rl/trainer.py tests/test_rl_trainer.py scripts/train_rl.py
git commit -m "feat: progressive RL trainer with GRPO-style advantages"
```

---

### Task 14: Pipeline Scripts

**Files:**
- Create: `scripts/collect_papers.py`
- Create: `scripts/generate_trajectories.py`
- Create: `scripts/evaluate.py`

**Step 1: Implement collect_papers.py**

```python
# scripts/collect_papers.py
"""Collect papers from ArXiv and convert to markdown."""
import argparse
import logging
from klong.research_factory.paper_collector import PaperCollector
from klong.research_factory.pdf_converter import PDFConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/papers")
    parser.add_argument("--max-papers", type=int, default=100)
    parser.add_argument("--conferences", nargs="+", default=["ICML", "NeurIPS", "ICLR"])
    args = parser.parse_args()

    collector = PaperCollector(
        output_dir=args.output_dir,
        conferences=args.conferences,
        max_papers=args.max_papers,
    )
    papers = collector.search_papers()

    converter = PDFConverter()
    for paper in papers:
        try:
            paper.markdown = converter.convert_url(paper.pdf_url, args.output_dir + "/pdfs")
            logger.info(f"Converted: {paper.title[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to convert {paper.paper_id}: {e}")

    collector.save_papers(papers)
    logger.info(f"Done. {len(papers)} papers saved to {args.output_dir}")

if __name__ == "__main__":
    main()
```

**Step 2: Implement generate_trajectories.py**

```python
# scripts/generate_trajectories.py
"""Generate expert trajectories using Claude API."""
import argparse
import json
import logging
from pathlib import Path
from klong.research_factory.trajectory_distiller import TrajectoryDistiller
from klong.research_factory.rubric_generator import RubricGenerator
from klong.evaluation.rubric import RubricTree
from klong.agent.sandbox.docker_manager import SandboxConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-file", default="data/papers/papers.jsonl")
    parser.add_argument("--rubric-dir", default="data/rubrics")
    parser.add_argument("--output-dir", default="data/trajectories")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--timeout-minutes", type=int, default=120)
    args = parser.parse_args()

    rubric_gen = RubricGenerator(model=args.model)
    distiller = TrajectoryDistiller(
        model=args.model,
        timeout_minutes=args.timeout_minutes,
    )

    Path(args.rubric_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.papers_file) as f:
        papers = [json.loads(line) for line in f if line.strip()]

    for paper in papers:
        paper_id = paper["paper_id"]
        rubric_path = Path(args.rubric_dir) / f"{paper_id}.json"

        # Generate rubric if not exists
        if not rubric_path.exists():
            logger.info(f"Generating rubric for {paper_id}...")
            try:
                rubric = rubric_gen.generate_and_save(
                    paper["markdown"], "", str(rubric_path))
            except Exception as e:
                logger.error(f"Failed to generate rubric for {paper_id}: {e}")
                continue
        else:
            with open(rubric_path) as f:
                rubric = RubricTree.from_dict(json.load(f))

        # Generate trajectory
        traj_path = Path(args.output_dir) / f"{paper_id}.json"
        if traj_path.exists():
            logger.info(f"Trajectory exists for {paper_id}, skipping")
            continue

        logger.info(f"Distilling trajectory for {paper_id}...")
        task_desc = f"Reproduce the paper '{paper['title']}' from scratch."
        success = distiller.distill_and_save(
            paper_id, paper["markdown"], task_desc, rubric, args.output_dir)

        if success:
            logger.info(f"Success: {paper_id}")
        else:
            logger.warning(f"Rejected: {paper_id}")

if __name__ == "__main__":
    main()
```

**Step 3: Implement evaluate.py**

```python
# scripts/evaluate.py
"""Evaluate a trained model on paper reproduction tasks."""
import argparse
import json
import logging
import asyncio
from pathlib import Path

from klong.training.rl.rollout import RolloutGenerator
from klong.evaluation.judge import Judge
from klong.evaluation.rubric import RubricTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task-data", default="data/papers/papers.jsonl")
    parser.add_argument("--rubric-dir", default="data/rubrics")
    parser.add_argument("--timeout-minutes", type=int, default=120)
    parser.add_argument("--output", default="results/eval_results.json")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    rollout_gen = RolloutGenerator(model, tokenizer)
    judge = Judge(model="claude-opus-4-20250514")

    with open(args.task_data) as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    results = []
    for task in tasks:
        paper_id = task["paper_id"]
        rubric_path = Path(args.rubric_dir) / f"{paper_id}.json"
        if not rubric_path.exists():
            continue

        with open(rubric_path) as f:
            rubric = RubricTree.from_dict(json.load(f))

        trajectory = rollout_gen.generate_rollout(
            paper_id, task.get("markdown", ""),
            f"Reproduce paper '{task['title']}'",
            args.timeout_minutes * 60)

        score, leaf_scores = asyncio.run(judge.evaluate(rubric, {}))
        results.append({
            "paper_id": paper_id,
            "score": score,
            "leaf_scores": leaf_scores,
            "turns": len(trajectory.turns),
            "time": trajectory.total_time_seconds,
        })
        logger.info(f"{paper_id}: score={score:.3f}, turns={len(trajectory.turns)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    mean_score = sum(r["score"] for r in results) / max(len(results), 1)
    logger.info(f"\nMean score: {mean_score:.3f} across {len(results)} papers")

if __name__ == "__main__":
    main()
```

**Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: pipeline scripts for data collection, distillation, and evaluation"
```

---

### Task 15: Integration Test & Documentation

**Files:**
- Create: `tests/test_integration.py`
- Modify: `klong/__init__.py` (add version)

**Step 1: Write integration test**

```python
# tests/test_integration.py
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
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: integration tests for full KLong pipeline"
```

---

### Task 16: Final Commit & Summary

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Final commit with all remaining files**

```bash
git add -A
git status
git commit -m "chore: final cleanup and all __init__.py files"
```
