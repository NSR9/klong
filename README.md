# KLong

**Training LLM Agents for Extremely Long-Horizon Tasks**

A from-scratch implementation of the [KLong paper](https://arxiv.org/abs/2602.17547) — a pipeline that trains language model agents to tackle research paper reproduction through trajectory-splitting SFT and progressive reinforcement learning.

## What This Does

KLong takes ML research papers, generates expert "trajectories" (step-by-step reproduction attempts using an LLM agent with coding tools), then trains a smaller model to replicate that behavior. The pipeline has five stages:

1. **Paper Collection** — Scrape ArXiv for ML papers with GitHub repos
2. **Rubric & Trajectory Generation** — Use Claude API to generate evaluation rubrics and expert reproduction trajectories
3. **Supervised Fine-Tuning (SFT)** — Train Qwen-2.5-7B on trajectories using a sliding-window splitting algorithm (handles trajectories that exceed context length)
4. **Progressive Reinforcement Learning** — Three-stage RL with increasing timeouts (30→60→120 min), using GRPO-style group-relative advantages
5. **Evaluation** — LLM-as-judge scoring against hierarchical rubric trees

## Prerequisites

- **Python 3.10+**
- **Docker** — for sandboxed agent execution
- **ANTHROPIC_API_KEY** — for rubric generation, trajectory distillation, and evaluation
- **GPU** — recommended for training (CUDA or Apple MPS). CPU works for testing.

## Installation

```bash
# Clone the repo
git clone https://github.com/NSR9/klong.git
cd klong

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package and dependencies
pip install -e ".[dev]"

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"
```

### Docker Setup

Build the sandbox image that agents run inside:

```bash
docker build -t klong-sandbox:latest klong/agent/sandbox/
```

> The Dockerfile expects Ubuntu 22.04 with Python 3.10 and common ML packages pre-installed.

## Usage

### Full Pipeline (End to End)

Run each stage sequentially:

```bash
# 1. Collect papers from ArXiv
python scripts/collect_papers.py \
  --output-dir data/papers \
  --max-papers 50 \
  --conferences ICML NeurIPS ICLR

# 2. Generate rubrics and expert trajectories
python scripts/generate_trajectories.py \
  --papers-file data/papers/papers.jsonl \
  --rubric-dir data/rubrics \
  --output-dir data/trajectories \
  --model claude-sonnet-4-20250514

# 3. Run SFT on the trajectories
python scripts/train_sft.py \
  --trajectory-dir data/trajectories \
  --output-dir checkpoints/sft

# 4. Run progressive RL
python scripts/train_rl.py \
  --task-data data/papers/papers.jsonl \
  --rubric-dir data/rubrics \
  --sft-checkpoint checkpoints/sft/final \
  --output-dir checkpoints/rl

# 5. Evaluate the trained model
python scripts/evaluate.py \
  --model-path checkpoints/rl/stage_2 \
  --task-data data/papers/papers.jsonl \
  --rubric-dir data/rubrics \
  --output results/eval_results.json
```

### Using a Config File

Instead of CLI arguments, you can use a YAML config:

```yaml
# config.yaml
model:
  name: "Qwen/Qwen2.5-7B"
  lora_rank: 64
  lora_alpha: 128
  max_seq_length: 32768
  load_in_4bit: true

data:
  papers:
    conferences: ["ICML", "NeurIPS", "ICLR"]
    max_papers: 50
  distillation:
    model: "claude-sonnet-4-20250514"
    timeout_minutes: 120

training:
  sft:
    learning_rate: 2e-5
    num_epochs: 3
    batch_size: 1
    gradient_accumulation_steps: 8
  rl:
    learning_rate: 5e-6
    rollouts_per_task: 4
    stages:
      - timeout_minutes: 30
        num_epochs: 2
      - timeout_minutes: 60
        num_epochs: 2
      - timeout_minutes: 120
        num_epochs: 2
  gradient_checkpointing: true

infra:
  docker_image: "klong-sandbox:latest"
  container_memory_limit: "8g"
```

Then pass it to any script:

```bash
python scripts/train_sft.py --config config.yaml
python scripts/train_rl.py --config config.yaml
```

### Running Individual Components

Each module can be used independently:

```python
# Use the trajectory splitter standalone
from klong.training.data.trajectory_splitter import TrajectorySplitter
from klong.agent.scaffold import Trajectory

splitter = TrajectorySplitter(max_window_tokens=30720, overlap_tokens=2048)
sub_trajectories = splitter.split(trajectory, prefix_turn_count=6)

# Use the rubric system
from klong.evaluation.rubric import RubricTree

rubric = RubricTree.from_dict(rubric_dict)
score = rubric.compute_score(leaf_scores={"criterion_1": 0.8, "criterion_2": 0.6})

# Use the judge
from klong.evaluation.judge import Judge

judge = Judge(model="claude-sonnet-4-20250514")
score, breakdown = await judge.evaluate(rubric, artifacts)
```

## Project Structure

```
klong/
├── config/settings.py             # Pydantic config (12 nested config classes)
├── research_factory/
│   ├── paper_collector.py         # ArXiv paper search and metadata
│   ├── pdf_converter.py           # PDF → Markdown via pymupdf4llm
│   ├── rubric_generator.py        # Claude API rubric generation
│   ├── trajectory_distiller.py    # Claude API expert agent distillation
│   └── blacklist.py               # GitHub URL blacklisting
├── agent/
│   ├── scaffold.py                # Agent loop, Turn, Trajectory
│   ├── tools/
│   │   ├── base.py                # Abstract Tool + ToolResult
│   │   ├── bash_tool.py           # Shell command execution
│   │   ├── python_tool.py         # Python script execution
│   │   ├── file_tool.py           # File read/write/search
│   │   └── paper_reader.py        # Paper content access
│   └── sandbox/
│       └── docker_manager.py      # Docker container lifecycle
├── training/
│   ├── data/
│   │   ├── trajectory_splitter.py # Overlapping sliding-window splitting
│   │   └── trajectory_dataset.py  # PyTorch Dataset with ChatML formatting
│   ├── sft/trainer.py             # LoRA + TRL SFT training
│   └── rl/
│       ├── reward.py              # GRPO group-relative advantages
│       ├── rollout.py             # Agent rollout generation
│       └── trainer.py             # 3-stage progressive RL
└── evaluation/
    ├── rubric.py                  # RubricNode tree with weighted scoring
    └── judge.py                   # LLM-as-judge evaluation

scripts/                           # CLI entry points for each pipeline stage
tests/                             # 65 tests across 15 test files
docs/plans/                        # Design documents
```

## Architecture Overview

```
ArXiv Papers ──► Research-Factory ──► Expert Trajectories ──► SFT ──► Progressive RL
                      │                                                     │
                 Rubric Trees ──────────────────────► Judge ◄──── Agent Rollouts
```

**Key algorithms:**

- **Trajectory Splitting**: Long trajectories are split into overlapping windows that fit within the model's context length. A fixed prefix (initial paper-reading turns) is prepended to every window. Only assistant turns are trained on (observations are masked).

- **Progressive RL (GRPO)**: The model generates N rollouts per task in Docker sandboxes, a judge scores each one, and group-relative advantages drive PPO-clip updates. Three stages with increasing timeouts let the model learn incrementally harder behaviors.

## Training Defaults

| Parameter | SFT | RL |
|-----------|-----|-----|
| Learning rate | 2e-5 | 5e-6 |
| Batch size | 1 | 1 |
| Grad accumulation | 8 | 4 |
| LoRA rank | 64 | — |
| LoRA alpha | 128 | — |
| Max seq length | 32768 | — |
| Clip epsilon | — | 0.2 |
| KL coefficient | — | 0.01 |
| Rollouts per task | — | 4 |

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_trajectory_splitter.py -v

# Run with coverage (if installed)
pytest tests/ --cov=klong
```

All 65 tests use mocking to avoid requiring Docker, GPU, or API keys.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes (for data gen & eval) | Anthropic API key for Claude |
| `HF_TOKEN` | Optional | HuggingFace token for gated models |

## Citation

This is an implementation of:

```bibtex
@article{klong2025,
  title={KLong: Training LLM Agent for Extremely Long-horizon Tasks},
  author={...},
  journal={arXiv preprint arXiv:2602.17547},
  year={2025}
}
```

## License

This project is provided as-is for research and educational purposes.
