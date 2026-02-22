# AGENTS.md

Agent-oriented documentation for AI coding agents working on the KLong codebase.

## Project Overview

KLong implements the paper "Training LLM Agent for Extremely Long-horizon Tasks" (arXiv:2602.17547). It's a five-stage pipeline: paper collection → rubric generation → trajectory distillation → trajectory-splitting SFT → progressive RL. The base model is Qwen-2.5-7B with LoRA adapters, trained via TRL/PEFT on HuggingFace Transformers.

**This is a research prototype.** Several components have known bugs and incomplete implementations (documented below). The RL training loop in particular does not actually update model weights.

## Build & Run Commands

```bash
# Install
pip install -e ".[dev]"

# Run all tests (65 tests, no GPU/Docker/API keys required)
pytest tests/ -v

# Run a single test file
pytest tests/test_trajectory_splitter.py -v

# Run pipeline scripts
python scripts/collect_papers.py --output-dir data/papers --max-papers 50
python scripts/generate_trajectories.py --papers-file data/papers/papers.jsonl
python scripts/train_sft.py --trajectory-dir data/trajectories --output-dir checkpoints/sft
python scripts/train_rl.py --task-data data/papers/papers.jsonl --rubric-dir data/rubrics
python scripts/evaluate.py --model-path checkpoints/rl/stage_2 --task-data data/papers/papers.jsonl
```

## Directory Structure

```
klong/
├── config/settings.py             # 12 Pydantic config classes (KLongConfig is root)
├── research_factory/
│   ├── paper_collector.py         # PaperRecord, PaperCollector (ArXiv API)
│   ├── pdf_converter.py           # PDFConverter (pymupdf4llm, lazy import)
│   ├── rubric_generator.py        # RubricGenerator (Claude API)
│   ├── trajectory_distiller.py    # TrajectoryDistiller (Claude API + Docker)
│   └── blacklist.py               # Blacklist (JSON file persistence)
├── agent/
│   ├── scaffold.py                # Agent, Turn, Trajectory (core agent loop)
│   ├── tools/
│   │   ├── base.py                # Abstract Tool base, ToolResult dataclass
│   │   ├── bash_tool.py           # BashTool(Tool)
│   │   ├── python_tool.py         # PythonTool(Tool)
│   │   ├── file_tool.py           # WriteFileTool, ReadFileTool, SearchFilesTool
│   │   └── paper_reader.py        # PaperReaderTool(Tool)
│   └── sandbox/
│       └── docker_manager.py      # SandboxConfig, ExecResult, SandboxManager
├── training/
│   ├── data/
│   │   ├── trajectory_splitter.py # TrajectorySplitter, SubTrajectory
│   │   └── trajectory_dataset.py  # TrajectoryDataset(torch.utils.data.Dataset)
│   ├── sft/trainer.py             # SFTTrainerWrapper (LoRA + TRL)
│   └── rl/
│       ├── reward.py              # compute_group_advantages()
│       ├── rollout.py             # RolloutGenerator
│       └── trainer.py             # ProgressiveRLTrainer
└── evaluation/
    ├── rubric.py                  # RubricNode, RubricTree
    └── judge.py                   # Judge (async, Anthropic API)
```

## Architecture & Data Flow

```
ArXiv Papers ──► PaperCollector ──► PDFConverter ──► papers.jsonl
                                                        │
                                                        ▼
                                              RubricGenerator ──► rubrics/{paper_id}.json
                                                        │
                                                        ▼
                                            TrajectoryDistiller ──► trajectories/{paper_id}.json
                                                        │
                                                        ▼
                                           TrajectorySplitter ──► SubTrajectory windows
                                                        │
                                                        ▼
                                           TrajectoryDataset ──► ChatML formatted samples
                                                        │
                                                        ▼
                                           SFTTrainerWrapper ──► checkpoints/sft/
                                                        │
                                                        ▼
                                       ProgressiveRLTrainer ──► checkpoints/rl/stage_{0,1,2}
                                                        │
                                                        ▼
                                                  Judge + RubricTree ──► scores
```

## Key Classes and Their Relationships

### Config System (`klong/config/settings.py`)
- `KLongConfig` is the root — contains `ModelConfig`, `DataConfig`, `TrainingConfig`, `InfraConfig`, `EvalConfig`
- `DataConfig` nests `PaperCollectionConfig` and `DistillationConfig`
- `TrainingConfig` nests `SFTConfig` (which nests `SplitterConfig`) and `RLConfig` (which nests `RLStageConfig[]`)
- All classes are `pydantic.BaseModel` with default values
- Scripts load config from YAML via `KLongConfig(**yaml.safe_load(f))`

### Agent System (`klong/agent/`)
- `Agent` takes a `model_name`, `system_prompt`, list of `Tool` instances, and a `SandboxManager`
- Call `agent.set_generate_fn(fn)` to inject the LLM — `fn(messages: list[dict]) -> str`
- `Agent.run()` returns a `Trajectory` (list of `Turn` objects)
- Tool calls are parsed via regex: `` ```tool_call\n{json}\n``` ``
- Tools execute inside Docker containers via `SandboxManager`
- `scaffold.py` has `SYSTEM_PROMPT_TEMPLATE` with `{task_description}` and `{tool_descriptions}` placeholders

### Tool System (`klong/agent/tools/`)
- All tools extend `Tool` (abstract base in `base.py`)
- Required methods: `name` (property), `description` (property), `parameters` (property → JSON schema dict), `execute(sandbox_id, **kwargs) -> ToolResult`
- `ToolResult` has `output: str`, `error: str | None`, `truncated: bool`
- Output truncation limits: 10K chars (bash, python, file tools), 15K chars (paper_reader)
- `Tool.to_schema()` returns OpenAI-style function calling schema

### Trajectory Splitting (`klong/training/data/trajectory_splitter.py`)
- Core algorithm from the paper — handles trajectories longer than context window
- `TrajectorySplitter(max_window_tokens=30720, overlap_tokens=2048)`
- `split(trajectory, prefix_turn_count=6)` → `list[SubTrajectory]`
- Fixed prefix (first N turns, typically paper-reading) prepended to every window
- Overlap is calculated backwards from window end
- Token counting: tiktoken `cl100k_base` with fallback to `len(text) // 4`

### Training (`klong/training/`)
- `TrajectoryDataset` loads JSON trajectory files, splits via `TrajectorySplitter`, formats as ChatML
- ChatML format: `<|im_start|>role\ncontent<|im_end|>\n`
- `action_mask` marks assistant tokens (1) vs observation tokens (0) — used for selective loss
- `SFTTrainerWrapper` uses PEFT LoRA + TRL `SFTTrainer`
- `ProgressiveRLTrainer` runs 3 stages with increasing timeouts
- `compute_group_advantages(rewards)` implements GRPO: `(r - mean) / std`

### Evaluation (`klong/evaluation/`)
- `RubricNode`: dataclass with `name`, `weight`, `criteria`, `children`
- `RubricTree`: wrapper with `from_dict()`/`to_dict()`, `get_leaves()`, `compute_score()`
- `Judge`: async class using Anthropic API, scores each leaf criterion 0.0-1.0
- Score aggregation: bottom-up weighted average through the rubric tree

## Known Bugs (Critical)

These are implementation gaps that must be fixed before the pipeline is functional:

### 1. RL Trainer Does Not Update Weights
**File:** `klong/training/rl/trainer.py`, `_train_stage()` method (lines 94-149)
**Problem:** The method collects trajectories and computes advantages but never performs gradient updates. After the inner loops, it just saves the unmodified model.
**Fix needed:** Add PPO-clip loss computation and `optimizer.step()` using the collected `(trajectory, advantage)` pairs.

### 2. Evaluate Script Passes Empty Artifacts to Judge
**File:** `scripts/evaluate.py`, line 53
**Problem:** `judge.evaluate(rubric, {})` always passes an empty dict. The judge needs actual agent output artifacts to score against rubric criteria. Result: all evaluations score ~0.0.
**Fix needed:** Collect sandbox artifacts (generated files, test results) from the rollout trajectory and pass them to the judge.

### 3. Docker Timeout Not Enforced
**File:** `klong/agent/sandbox/docker_manager.py`
**Problem:** `SandboxManager.execute()` accepts a `timeout` parameter but never passes it to `container.exec_run()`. Commands can run forever.
**Fix needed:** Pass `timeout` to the Docker API's `exec_run()` call, or implement manual timeout with threading.

### 4. ExecResult.timed_out Never Set
**File:** `klong/agent/sandbox/docker_manager.py`
**Problem:** `ExecResult` has a `timed_out: bool` field that defaults to `False` and is never set to `True` anywhere. Callers checking this field get incorrect information.
**Fix needed:** Set `timed_out = True` when a command exceeds its timeout.

## Known Bugs (Non-Critical)

### 5. Action Mask is Per-Character, Not Per-Token
**File:** `klong/training/data/trajectory_dataset.py`
**Problem:** The `action_mask` is built at the character level (one bool per character in the ChatML string), but tokenization produces a different number of tokens. The mask and token IDs have misaligned lengths after tokenization.
**Fix needed:** Build the mask at the token level by tokenizing assistant vs non-assistant segments separately, or align the character mask to token boundaries post-tokenization.

### 6. RolloutGenerator Uses Empty System Prompt
**File:** `klong/training/rl/rollout.py`
**Problem:** `RolloutGenerator.generate_rollout()` creates an `Agent` with `system_prompt=""`. The agent will lack the structured system prompt that tells it how to use tools.
**Fix needed:** Construct the system prompt using `SYSTEM_PROMPT_TEMPLATE` from `scaffold.py`.

### 7. PaperCollectionConfig.years_back Unused
**File:** `klong/config/settings.py` and `klong/research_factory/paper_collector.py`
**Problem:** `years_back` is defined in config but `PaperCollector` doesn't use it for date-range filtering.

### 8. blocked_hosts Not Implemented
**File:** `klong/agent/sandbox/docker_manager.py`
**Problem:** `SandboxConfig.blocked_hosts` is defined but the sandbox manager doesn't configure network rules to block those hosts.

### 9. max_concurrent_containers Not Enforced
**File:** `klong/config/settings.py` → `InfraConfig.max_concurrent_containers`
**Problem:** This limit exists in config but `SandboxManager` doesn't track or limit concurrent containers.

## Hardcoded Values

| Value | Location | Description |
|-------|----------|-------------|
| `10000` | `bash_tool.py`, `python_tool.py`, `file_tool.py` | Output truncation limit (chars) |
| `15000` | `paper_reader.py` | Paper content truncation limit (chars) |
| `30000` | `rubric_generator.py` | Paper markdown truncation for Claude API |
| `10000` | `rubric_generator.py` | Code truncation for Claude API |
| `0.3` | `trajectory_distiller.py` | Rejection sampling threshold |
| `6` | `trajectory_splitter.py` | Default `prefix_turn_count` |
| `200` | `scaffold.py` | Default `max_turns` |
| `10` | `scaffold.py` | Default `end_task_ban_turns` |
| `3` | `scaffold.py` | Default `mandatory_read_turns` |
| `0.7` / `0.9` | `rollout.py` | Generation temperature / top_p |
| `cl100k_base` | `trajectory_splitter.py` | tiktoken encoding (not Qwen's actual tokenizer) |

## Code Style & Conventions

- **Python 3.10+** — uses `X | Y` union syntax, `list[str]` generics
- **Pydantic v2** `BaseModel` for all config classes
- **Dataclasses** for data structures (`Turn`, `Trajectory`, `SubTrajectory`, `RubricNode`, etc.)
- **Lazy imports** for heavy dependencies (`pymupdf4llm`, `tiktoken`, `docker`, `torch`) — modules that import these guard with try/except or import inside methods
- **Async** — `Judge.evaluate()` and `Judge.evaluate_leaf()` are async. Callers use `asyncio.run()`
- **Logging** — every module uses `logging.getLogger(__name__)`
- **No type: ignore** — the codebase avoids type suppression
- **Tests** — all in `tests/`, use `unittest.mock.patch` and `MagicMock` extensively. No real Docker, GPU, or API calls in tests.

## Testing Conventions

- Test files mirror source structure: `test_scaffold.py` tests `agent/scaffold.py`, etc.
- Fixtures use `@patch` decorators to mock external dependencies
- All API clients (Anthropic, Docker, HuggingFace) are mocked
- Tests are synchronous even for async code (mock the async calls)
- `test_integration.py` contains 6 end-to-end pipeline tests with everything mocked
- **No test requires network, GPU, Docker, or API keys**

To add a new test:
```python
# tests/test_new_module.py
from unittest.mock import patch, MagicMock
import pytest

def test_feature():
    with patch("klong.module.ExternalDep") as mock_dep:
        mock_dep.return_value.method.return_value = "result"
        # test your code
        assert result == expected
```

## Dependencies

Core runtime:
- `torch>=2.1.0`, `transformers>=4.40.0`, `trl>=0.12.0`, `peft>=0.13.0`
- `accelerate>=1.0.0`, `bitsandbytes>=0.44.0`, `datasets>=3.0.0`
- `anthropic>=0.40.0` (Claude API client)
- `arxiv>=2.1.0` (ArXiv API), `pymupdf4llm>=0.0.10` (PDF conversion)
- `docker>=7.0.0`, `pydantic>=2.0.0`, `tiktoken>=0.7.0`

Dev: `pytest>=8.0.0`, `pytest-asyncio>=0.24.0`

## Limitations

1. **Single-GPU only** — no distributed training, no FSDP/DeepSpeed integration
2. **Sequential rollouts** — RL generates rollouts one at a time (no parallel sandbox execution)
3. **No checkpoint resumption** — SFT and RL scripts don't support resuming from interruption
4. **tiktoken vs actual tokenizer** — trajectory splitting uses `cl100k_base` encoding, not Qwen's actual tokenizer, so token counts are approximate
5. **No Dockerfile provided** — `klong/agent/sandbox/Dockerfile` is referenced but gitignored (`.gitignore` contains `klong/agent/sandbox/Dockerfile.local`)
6. **No wandb/tensorboard logging** — training metrics are only printed to stdout
7. **Claude API costs** — trajectory distillation and evaluation make many Claude API calls; no cost tracking or budgeting built in
8. **No data validation** — pipeline scripts assume well-formed JSONL files without schema validation

## Common Patterns for Agents

### Adding a New Tool
1. Create `klong/agent/tools/new_tool.py`
2. Subclass `Tool` from `base.py`
3. Implement `name`, `description`, `parameters` properties and `execute()` method
4. Add the tool instance to the `tools` list when constructing `Agent`
5. Add tests in `tests/test_tools.py`

### Adding a New Config Section
1. Add a new `BaseModel` subclass in `klong/config/settings.py`
2. Add it as a field on the appropriate parent config class
3. Add tests in `tests/test_config.py`

### Modifying the Agent Loop
- The main loop is `Agent.run()` in `klong/agent/scaffold.py`
- Tool call parsing uses regex — modify `parse_tool_calls()` for different formats
- The `mandatory_read_turns` and `end_task_ban_turns` parameters control agent behavior constraints

### Working with Trajectories
- `Trajectory.to_dict()` / `Trajectory.from_dict()` for serialization
- Trajectories are stored as JSON files with turns list
- Each `Turn` has: `role` (system/user/assistant), `content`, `tool_calls`, `tool_results`, `timestamp`
