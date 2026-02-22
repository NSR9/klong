# KLong Implementation Design

**Date:** 2026-02-21
**Paper:** KLong: Training LLM Agent for Extremely Long-horizon Tasks (arXiv:2602.17547)
**Goal:** Full from-scratch implementation of the KLong pipeline at practical scale

## Decisions

| Decision | Choice |
|----------|--------|
| Scope | Full pipeline (Research-Factory + Agent + SFT + RL + Judge) |
| Base model | Qwen-2.5-7B with LoRA |
| Compute | Single GPU (MacBook MPS or remote CUDA) |
| Sandbox | Docker containers |
| Data source | ArXiv papers + Claude API for distillation |
| Framework | TRL + HuggingFace Transformers |
| Architecture | Modular pipeline with shared config |

## Project Structure

```
klong/
├── config/
│   └── settings.py              # Pydantic-based config
├── research_factory/
│   ├── paper_collector.py       # ArXiv paper scraping & filtering
│   ├── pdf_converter.py         # PDF -> Markdown conversion
│   ├── rubric_generator.py      # Claude API rubric generation
│   ├── trajectory_distiller.py  # Claude API trajectory distillation
│   └── blacklist.py             # GitHub repo blacklisting
├── agent/
│   ├── scaffold.py              # Main agent loop (observe -> think -> act)
│   ├── tools/
│   │   ├── bash_tool.py         # Bash command execution
│   │   ├── python_tool.py       # Python script execution
│   │   ├── file_tool.py         # File read/write/search
│   │   └── paper_reader.py      # Paper reading & tracking
│   └── sandbox/
│       ├── docker_manager.py    # Docker container lifecycle
│       └── Dockerfile           # Pre-built research environment
├── training/
│   ├── data/
│   │   ├── trajectory_dataset.py    # Dataset for full & split trajectories
│   │   └── trajectory_splitter.py   # Overlapping window splitting algorithm
│   ├── sft/
│   │   └── trainer.py           # Trajectory-splitting SFT with TRL
│   └── rl/
│       ├── trainer.py           # Progressive RL (GRPO) with TRL
│       ├── reward.py            # Judge-based reward computation
│       └── rollout.py           # Agent rollout generation
├── evaluation/
│   ├── judge.py                 # Judge model evaluation
│   └── rubric.py                # Rubric tree parsing & scoring
├── scripts/
│   ├── collect_papers.py        # Run paper collection
│   ├── generate_trajectories.py # Run distillation
│   ├── train_sft.py             # Run SFT stage
│   ├── train_rl.py              # Run progressive RL
│   └── evaluate.py              # Run evaluation
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Component Designs

### 1. Research-Factory Pipeline

**Stage 1 - Paper Collection:**
- Query ArXiv API for ICML/NeurIPS/ICLR papers (past 5 years)
- Filter: has GitHub repo, Python/PyTorch code, clear experimental methodology
- Download PDFs, convert to Markdown via pymupdf4llm
- Store as JSONL: `{paper_id, title, markdown, github_url, metadata}`
- Add GitHub URLs to blacklist

**Stage 2 - Rubric Generation:**
- Call Claude API with paper markdown + official code
- Generate hierarchical rubric tree (JSON):
  - Root branches: core_contribution (0.4), experimental_setup (0.3), code_quality (0.3)
  - Each branch has weighted leaf criteria with evaluation descriptions
- Store rubrics alongside paper data

**Stage 3 - Trajectory Distillation:**
- For each paper+rubric, spin up Docker sandbox
- Run Claude API as expert agent with tool access
- Record full trajectory: [(state, action), ...]
- Rejection sampling: evaluate with judge, keep score > threshold
- Target: 50-100 papers, ~30-50 high-quality trajectories

### 2. Agent Scaffold

**Agent loop:**
```
while not done and turns < max_turns:
    1. Construct prompt: system_prompt + conversation_history
    2. Call LLM (Qwen-2.5-7B)
    3. Parse response -> extract tool calls
    4. Execute tool calls in sandbox
    5. Append (observation, action) to trajectory
    6. Check termination: time limit, end_task, error threshold
```

**Tools:** bash, python, write_file, read_file, search_files, read_paper, end_task

**Sandbox:** Docker container per rollout with:
- Ubuntu 22.04 + Python 3.10 + 80 ML packages
- CPU/memory limits, timeout enforcement
- Network access (blacklisted repos blocked)

**Scaffolding optimizations:**
1. Mandatory paper reading in first N turns
2. Context-length error handling (summarize old observations)
3. File-reading progress tracking
4. Prompt caching
5. Early end_task ban (first M turns)

### 3. Trajectory-Splitting SFT

**Splitting algorithm:**
1. Extract fixed prefix p (task spec + paper content from initial turns)
2. Compute available window: L_available = L_max - len(p)
3. Sliding window with overlap:
   - stride = L_available - overlap
   - For each window position, create sub-trajectory: [p, window_content]
4. Train only on action tokens (mask observation tokens)

**Loss:** L_SFT = -sum_i sum_t log P_theta(a_t | tau^(i)_<t)

**Two-phase SFT:**
- Phase 1 (optional): General instruction SFT on coding/math data
- Phase 2: Trajectory SFT on split distilled trajectories

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA targets | q,k,v,o,gate,up,down_proj |
| Learning rate | 2e-5 |
| Batch size | 1 (grad accum 8) |
| Epochs | 3 |
| Warmup ratio | 0.1 |
| Max seq length | 32768 |
| Overlap | 2048 tokens |
| Gradient checkpointing | Yes |

### 4. Progressive Reinforcement Learning

**3 stages with increasing timeouts:**

| Stage | Timeout | Focus |
|-------|---------|-------|
| RL-Stage1 | 30 min | Basic tool use, paper reading, initial code |
| RL-Stage2 | 60 min | Debugging, iterative improvement |
| RL-Stage3 | 120 min | Full reproduction, comprehensive testing |

**Algorithm: GRPO-style (PPO-clip + group-relative advantage)**

For each stage m:
1. Generate n=4 rollouts per task in Docker sandboxes
2. Evaluate with judge: Q = Judge(output, rubric) in [0, 1]
3. Split rollout trajectories (same splitting as SFT)
4. Compute group-relative advantage: A_t = Q_t - mean(Q across all rollouts)
5. PPO-clip update: L = -mean(min(r*A, clip(r, 1-eps, 1+eps)*A)) + beta*KL

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Rollouts per task | 4 |
| Clip epsilon | 0.2 |
| KL coefficient | 0.01 |
| Learning rate | 5e-6 |
| Batch size | 1 (grad accum 4) |
| Reference model | Frozen SFT checkpoint |
| Max new tokens/turn | 4096 |

### 5. Evaluation & Judge

**Judge model:** Claude API (Sonnet for training rewards, Opus for final eval)

**Evaluation flow:**
1. Receive agent output artifacts
2. Load rubric tree
3. For each leaf criterion, prompt judge for 0-1 score
4. Aggregate bottom-up through tree using weights
5. Final score = weighted average

**Rubric tree structure:**
```python
@dataclass
class RubricNode:
    name: str
    weight: float
    criteria: str
    children: list[RubricNode]
```

**Metrics:** per-paper score, average score, per-category breakdown, agent behavior stats (turns, time, tool usage)

## Data Flow

```
ArXiv Papers -> Research-Factory -> Trajectories -> Trajectory-Split SFT -> Progressive RL
                    |                                                            |
              Rubric Trees --------------------------> Judge Evaluation <-- Agent Rollouts
```

## Key Technical Risks

1. **LoRA + RL stability**: RL on LoRA adapters can be unstable. Mitigation: conservative KL, low learning rate.
2. **Single-GPU rollout throughput**: Sequential rollouts are slow. Mitigation: shorter timeouts, fewer rollouts per task.
3. **Claude API costs**: Distillation + evaluation is expensive. Mitigation: start with small paper set, cache aggressively.
4. **Docker on Mac**: Docker Desktop on macOS has performance overhead. Mitigation: design for remote CUDA machine as primary.
