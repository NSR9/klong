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
    import os
    import torch

    if torch.cuda.is_available():
        dtype, device = torch.bfloat16, "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        dtype, device = torch.float32, "mps"
    else:
        dtype, device = torch.float32, "cpu"

    logger.info(f"Loading model on {device} with dtype {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype)
    if device != "cpu":
        model = model.to(device)

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

        with open(rubric_path) as f2:
            rubric = RubricTree.from_dict(json.load(f2))

        trajectory = rollout_gen.generate_rollout(
            paper_id, task.get("markdown", ""),
            f"Reproduce paper '{task['title']}'",
            args.timeout_minutes * 60)

        score, leaf_scores = asyncio.run(judge.evaluate(rubric, {}))
        results.append({
            "paper_id": paper_id, "score": score,
            "leaf_scores": leaf_scores, "turns": len(trajectory.turns),
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
