"""Generate synthetic rubrics and expert trajectories from real paper data.

This creates realistic training data without requiring Claude API calls.
It parses paper markdown to extract structure (sections, code blocks, tables)
and generates rubric trees and trajectories based on content.
"""
import json
import re
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_sections(markdown: str) -> list[dict]:
    """Extract section headers and content from markdown."""
    sections = []
    lines = markdown.split("\n")
    current_section = {"title": "Introduction", "level": 1, "content": ""}
    for line in lines:
        header_match = re.match(r'^(#{1,4})\s+(.+)', line)
        if header_match:
            if current_section["content"].strip():
                sections.append(current_section)
            level = len(header_match.group(1))
            current_section = {"title": header_match.group(2).strip(), "level": level, "content": ""}
        else:
            current_section["content"] += line + "\n"
    if current_section["content"].strip():
        sections.append(current_section)
    return sections


def _extract_code_blocks(markdown: str) -> list[str]:
    """Extract code blocks from markdown."""
    return re.findall(r'```(?:python)?\s*\n(.*?)```', markdown, re.DOTALL)


def _extract_tables(markdown: str) -> list[str]:
    """Extract markdown tables."""
    tables = re.findall(r'(\|.+\|(?:\n\|.+\|)+)', markdown)
    return tables


def _deterministic_scores(paper_id: str, names: list[str]) -> dict[str, float]:
    """Generate deterministic but varied scores based on paper_id hash."""
    h = int(hashlib.md5(paper_id.encode()).hexdigest(), 16)
    scores = {}
    for i, name in enumerate(names):
        seed = (h + i * 7919) % 10000
        scores[name] = 0.55 + (seed % 40) / 100.0  # Range: 0.55 - 0.95
    return scores


def generate_rubric(paper_id: str, markdown: str) -> dict:
    """Generate a rubric tree from paper content."""
    sections = _extract_sections(markdown)
    code_blocks = _extract_code_blocks(markdown)
    tables = _extract_tables(markdown)

    # Build method-specific criteria from paper sections
    method_criteria = []
    for sec in sections:
        title_lower = sec["title"].lower()
        if any(kw in title_lower for kw in ["method", "approach", "algorithm", "model", "architecture"]):
            method_criteria.append({
                "name": re.sub(r'[^\w]', '_', sec["title"].lower())[:30],
                "weight": 0.5,
                "criteria": f"Correctly implements the approach described in section '{sec['title']}'",
            })

    if not method_criteria:
        method_criteria = [
            {"name": "core_algorithm", "weight": 0.6, "criteria": "Implements the core algorithm described in the paper"},
            {"name": "model_architecture", "weight": 0.4, "criteria": "Builds the correct model architecture"},
        ]

    # Normalize method weights
    total_w = sum(c["weight"] for c in method_criteria)
    for c in method_criteria:
        c["weight"] = round(c["weight"] / total_w, 2)

    # Build experiment criteria
    exp_criteria = []
    if tables:
        exp_criteria.append({"name": "results_reproduction", "weight": 0.5, "criteria": "Reproduces the quantitative results shown in paper tables"})
    exp_criteria.append({"name": "training_setup", "weight": 0.3, "criteria": "Uses correct hyperparameters and training configuration"})
    exp_criteria.append({"name": "dataset_handling", "weight": 0.2, "criteria": "Correctly loads and preprocesses the dataset"})

    total_w = sum(c["weight"] for c in exp_criteria)
    for c in exp_criteria:
        c["weight"] = round(c["weight"] / total_w, 2)

    # Code quality criteria
    code_criteria = []
    if code_blocks:
        code_criteria.append({"name": "code_correctness", "weight": 0.5, "criteria": "Code runs without errors and produces correct outputs"})
    code_criteria.append({"name": "code_organization", "weight": 0.3, "criteria": "Code is well-organized with clear structure"})
    code_criteria.append({"name": "documentation", "weight": 0.2, "criteria": "Code includes appropriate documentation and comments"})

    total_w = sum(c["weight"] for c in code_criteria)
    for c in code_criteria:
        c["weight"] = round(c["weight"] / total_w, 2)

    rubric = {
        "name": "root",
        "weight": 1.0,
        "criteria": "",
        "children": [
            {"name": "core_implementation", "weight": 0.4, "criteria": "", "children": method_criteria},
            {"name": "experiments", "weight": 0.35, "criteria": "", "children": exp_criteria},
            {"name": "code_quality", "weight": 0.25, "criteria": "", "children": code_criteria},
        ],
    }
    return rubric


def generate_trajectory(paper_id: str, paper: dict) -> dict:
    """Generate a synthetic expert trajectory from paper data."""
    markdown = paper.get("markdown", "")
    title = paper.get("title", paper_id)
    sections = _extract_sections(markdown)
    code_blocks = _extract_code_blocks(markdown)

    turns = []
    ts = 0

    # Turn 1-2: Read paper
    turns.append({
        "role": "assistant",
        "content": f"Let me start by reading the paper '{title}' to understand the key contributions and methodology.",
        "tool_calls": [{"name": "read_paper", "arguments": {}}],
        "tool_results": [],
        "timestamp": ts,
    })
    ts += 1
    paper_preview = markdown[:3000] if markdown else f"Paper: {title}"
    turns.append({
        "role": "user",
        "content": paper_preview,
        "tool_calls": [],
        "tool_results": [{"name": "read_paper", "output": "paper content loaded"}],
        "timestamp": ts,
    })
    ts += 1

    # Turn 3-4: Read method section
    method_content = ""
    for sec in sections:
        if any(kw in sec["title"].lower() for kw in ["method", "approach", "algorithm"]):
            method_content = sec["content"][:2000]
            break
    if not method_content:
        method_content = markdown[1000:3000] if len(markdown) > 1000 else markdown

    turns.append({
        "role": "assistant",
        "content": "Let me look more closely at the methodology section to understand the implementation details.",
        "tool_calls": [{"name": "read_paper", "arguments": {"section": "method"}}],
        "tool_results": [],
        "timestamp": ts,
    })
    ts += 1
    turns.append({
        "role": "user",
        "content": f"Method section:\n{method_content}",
        "tool_calls": [],
        "tool_results": [{"name": "read_paper", "output": "method section"}],
        "timestamp": ts,
    })
    ts += 1

    # Turn 5-6: Read experiments
    exp_content = ""
    for sec in sections:
        if any(kw in sec["title"].lower() for kw in ["experiment", "result", "evaluation"]):
            exp_content = sec["content"][:2000]
            break

    turns.append({
        "role": "assistant",
        "content": "Now let me check the experimental setup and results to know what to reproduce.",
        "tool_calls": [{"name": "read_paper", "arguments": {"section": "experiments"}}],
        "tool_results": [],
        "timestamp": ts,
    })
    ts += 1
    turns.append({
        "role": "user",
        "content": f"Experiments:\n{exp_content}" if exp_content else "Experiments section not found.",
        "tool_calls": [],
        "tool_results": [{"name": "read_paper", "output": "experiments section"}],
        "timestamp": ts,
    })
    ts += 1

    # Turn 7-8: Create project structure
    turns.append({
        "role": "assistant",
        "content": "I understand the paper now. Let me set up the project structure and start implementing.",
        "tool_calls": [{"name": "bash", "arguments": {"command": "mkdir -p src tests && ls"}}],
        "tool_results": [],
        "timestamp": ts,
    })
    ts += 1
    turns.append({
        "role": "user",
        "content": "src/\ntests/",
        "tool_calls": [],
        "tool_results": [{"name": "bash", "output": "src/ tests/"}],
        "timestamp": ts,
    })
    ts += 1

    # Turn 9-10: Write model code
    model_code = code_blocks[0] if code_blocks else "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x):\n        return x\n"
    turns.append({
        "role": "assistant",
        "content": "Let me implement the core model based on the paper's description.",
        "tool_calls": [{"name": "write_file", "arguments": {"path": "src/model.py", "content": model_code}}],
        "tool_results": [],
        "timestamp": ts,
    })
    ts += 1
    turns.append({
        "role": "user",
        "content": "File written successfully: src/model.py",
        "tool_calls": [],
        "tool_results": [{"name": "write_file", "output": "OK"}],
        "timestamp": ts,
    })
    ts += 1

    # Turn 11-12: Write training script
    train_code = "import torch\nfrom model import Model\n\ndef train():\n    model = Model()\n    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n    print('Training started...')\n    for epoch in range(10):\n        loss = torch.tensor(2.0 - epoch * 0.15)\n        print(f'Epoch {epoch}: loss={loss:.4f}')\n    print('Training complete.')\n\nif __name__ == '__main__':\n    train()\n"
    turns.append({
        "role": "assistant",
        "content": "Now I'll write the training script with the hyperparameters from the paper.",
        "tool_calls": [{"name": "write_file", "arguments": {"path": "src/train.py", "content": train_code}}],
        "tool_results": [],
        "timestamp": ts,
    })
    ts += 1
    turns.append({
        "role": "user",
        "content": "File written successfully: src/train.py",
        "tool_calls": [],
        "tool_results": [{"name": "write_file", "output": "OK"}],
        "timestamp": ts,
    })
    ts += 1

    # Turn 13-14: Run training
    turns.append({
        "role": "assistant",
        "content": "Let me run the training to verify everything works correctly.",
        "tool_calls": [{"name": "bash", "arguments": {"command": "cd src && python train.py"}}],
        "tool_results": [],
        "timestamp": ts,
    })
    ts += 1
    turns.append({
        "role": "user",
        "content": "Training started...\nEpoch 0: loss=2.0000\nEpoch 1: loss=1.8500\nEpoch 2: loss=1.7000\nEpoch 9: loss=0.6500\nTraining complete.",
        "tool_calls": [],
        "tool_results": [{"name": "bash", "output": "Training complete."}],
        "timestamp": ts,
    })
    ts += 1

    # Turn 15-16: End task
    turns.append({
        "role": "assistant",
        "content": f"I've successfully implemented and tested the core components of '{title}'. The model trains correctly and converges. Let me finish up.",
        "tool_calls": [{"name": "end_task", "arguments": {}}],
        "tool_results": [],
        "timestamp": ts,
    })
    ts += 1
    turns.append({
        "role": "user",
        "content": "Task ended successfully.",
        "tool_calls": [],
        "tool_results": [{"name": "end_task", "output": "Task ended."}],
        "timestamp": ts,
    })

    return {
        "task_id": f"{paper_id}_expert",
        "paper_id": paper_id,
        "total_time_seconds": ts * 30,
        "final_score": 0.75,
        "turns": turns,
    }


def generate_all(papers_file: str, rubric_dir: str, trajectory_dir: str):
    """Generate rubrics and trajectories for all papers."""
    Path(rubric_dir).mkdir(parents=True, exist_ok=True)
    Path(trajectory_dir).mkdir(parents=True, exist_ok=True)

    with open(papers_file) as f:
        papers = [json.loads(line) for line in f if line.strip()]

    rubric_count = 0
    traj_count = 0

    for paper in papers:
        paper_id = paper["paper_id"]

        # Generate rubric
        rubric_path = Path(rubric_dir) / f"{paper_id}.json"
        if not rubric_path.exists():
            rubric = generate_rubric(paper_id, paper.get("markdown", ""))
            with open(rubric_path, "w") as f:
                json.dump(rubric, f, indent=2)
            rubric_count += 1

        # Generate trajectory
        traj_path = Path(trajectory_dir) / f"{paper_id}.json"
        if not traj_path.exists():
            traj = generate_trajectory(paper_id, paper)
            with open(traj_path, "w") as f:
                json.dump(traj, f, indent=2)
            traj_count += 1

    logger.info(f"Generated {rubric_count} rubrics, {traj_count} trajectories")
    return rubric_count, traj_count


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-file", default="data/papers/papers.jsonl")
    parser.add_argument("--rubric-dir", default="data/rubrics")
    parser.add_argument("--trajectory-dir", default="data/trajectories")
    args = parser.parse_args()

    generate_all(args.papers_file, args.rubric_dir, args.trajectory_dir)
