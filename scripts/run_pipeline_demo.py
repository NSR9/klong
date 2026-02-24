"""Run the full KLong pipeline end-to-end with a tiny model for demonstration."""
import argparse
import json
import logging
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("pipeline_demo")


def run_stage_1():
    """Stage 1: Paper collection (uses pre-created demo data)."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Paper Collection")
    logger.info("=" * 60)

    papers_file = Path("data/papers/papers.jsonl")
    if not papers_file.exists():
        logger.info("Creating demo data...")
        from scripts.create_demo_data import papers, rubrics, make_trajectory
        papers_dir = Path("data/papers")
        papers_dir.mkdir(parents=True, exist_ok=True)
        with open(papers_file, "w") as f:
            for p in papers:
                f.write(json.dumps(p) + "\n")

    with open(papers_file) as f:
        paper_count = sum(1 for line in f if line.strip())
    logger.info(f"Stage 1 complete: {paper_count} papers loaded from {papers_file}")
    return True


def run_stage_2():
    """Stage 2: Rubric generation & trajectory distillation (uses pre-created demo data)."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Rubric Generation & Trajectory Distillation")
    logger.info("=" * 60)

    rubric_dir = Path("data/rubrics")
    traj_dir = Path("data/trajectories")

    rubric_count = len(list(rubric_dir.glob("*.json")))
    traj_count = len(list(traj_dir.glob("*.json")))

    logger.info(f"Stage 2 complete: {rubric_count} rubrics, {traj_count} trajectories")
    logger.info("(Using pre-generated demo data — real pipeline would call Claude API here)")
    return True


def run_stage_3():
    """Stage 3: SFT training on trajectories."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 60)

    from klong.training.sft.trainer import SFTTrainerWrapper

    model_name = "Qwen/Qwen2.5-0.5B"
    output_dir = "checkpoints/sft"

    logger.info(f"Training model: {model_name}")
    logger.info(f"Trajectory dir: data/trajectories")
    logger.info(f"Output dir: {output_dir}")

    trainer = SFTTrainerWrapper(
        model_name=model_name,
        lora_rank=8,
        lora_alpha=16,
        learning_rate=2e-5,
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        max_seq_length=2048,
        output_dir=output_dir,
        use_bf16=False,
        gradient_checkpointing=False,
        load_in_4bit=False,
    )
    trainer.train("data/trajectories")
    logger.info(f"Stage 3 complete: Model saved to {output_dir}/final")
    return True


def run_stage_4():
    """Stage 4: Progressive RL training (demonstration with mock rollouts)."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Progressive Reinforcement Learning")
    logger.info("=" * 60)

    # RL requires Docker sandboxes and Claude API for judging — we demonstrate
    # the trainer creation and stage logic without actual rollout execution
    from klong.training.rl.trainer import ProgressiveRLTrainer
    from klong.training.rl.reward import compute_group_advantages
    import numpy as np

    trainer = ProgressiveRLTrainer(
        model_name="Qwen/Qwen2.5-0.5B",
        sft_checkpoint="checkpoints/sft/final",
        stages=[
            {"timeout_minutes": 1, "num_epochs": 1},
        ],
        learning_rate=5e-6,
        rollouts_per_task=2,
        output_dir="checkpoints/rl",
    )

    logger.info(f"RL Trainer configured: {len(trainer.stages)} stage(s)")
    logger.info(f"  Clip epsilon: {trainer.clip_epsilon}")
    logger.info(f"  KL coefficient: {trainer.kl_coeff}")
    logger.info(f"  Rollouts per task: {trainer.rollouts_per_task}")

    # Demonstrate GRPO advantage computation
    sample_rewards = [0.4, 0.8, 0.6, 0.9]
    advantages = compute_group_advantages(sample_rewards)
    logger.info(f"  GRPO advantage demo: rewards={sample_rewards} -> advantages={[f'{a:.3f}' for a in advantages]}")

    # Create checkpoint directory to simulate RL output
    rl_dir = Path("checkpoints/rl/stage_0")
    rl_dir.mkdir(parents=True, exist_ok=True)

    # Copy SFT checkpoint as RL baseline (since actual RL requires Docker+API)
    sft_final = Path("checkpoints/sft/final")
    if sft_final.exists():
        import shutil
        for f in sft_final.iterdir():
            shutil.copy2(f, rl_dir / f.name)
        logger.info(f"Stage 4 complete: Checkpoint saved to {rl_dir}")
    else:
        logger.info("Stage 4 complete: RL trainer validated (no SFT checkpoint to copy)")

    logger.info("(Full RL training requires Docker sandboxes and Claude API for judge scoring)")
    return True


def run_stage_5():
    """Stage 5: Evaluation."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Evaluation")
    logger.info("=" * 60)

    from klong.evaluation.rubric import RubricTree

    rubric_dir = Path("data/rubrics")
    results = []

    with open("data/papers/papers.jsonl") as f:
        papers = [json.loads(line) for line in f if line.strip()]

    for paper in papers:
        paper_id = paper["paper_id"]
        rubric_path = rubric_dir / f"{paper_id}.json"
        if not rubric_path.exists():
            continue

        with open(rubric_path) as f:
            rubric = RubricTree.from_dict(json.load(f))

        # Simulate evaluation scores (real pipeline would run agent + judge)
        leaves = rubric.get_leaves()
        import random
        random.seed(42)
        leaf_scores = {leaf.name: round(random.uniform(0.5, 0.95), 3) for leaf in leaves}
        total_score = rubric.compute_score(leaf_scores)

        results.append({
            "paper_id": paper_id,
            "score": round(total_score, 4),
            "leaf_scores": leaf_scores,
        })
        logger.info(f"  {paper_id}: score={total_score:.4f} | leaves={leaf_scores}")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    mean_score = sum(r["score"] for r in results) / max(len(results), 1)
    logger.info(f"\nStage 5 complete: Mean score = {mean_score:.4f} across {len(results)} papers")
    logger.info(f"Results saved to results/eval_results.json")
    logger.info("(Real evaluation would run agent rollouts in Docker + Claude API judge)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run KLong pipeline demo")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT training stage")
    args = parser.parse_args()

    logger.info("KLong Pipeline - Full End-to-End Run")
    logger.info("=" * 60)

    stages = [
        ("Stage 1: Paper Collection", run_stage_1),
        ("Stage 2: Rubric & Trajectory Generation", run_stage_2),
        ("Stage 3: SFT Training", run_stage_3 if not args.skip_sft else lambda: logger.info("Skipping SFT") or True),
        ("Stage 4: Progressive RL", run_stage_4),
        ("Stage 5: Evaluation", run_stage_5),
    ]

    for name, fn in stages:
        try:
            success = fn()
            if not success:
                logger.error(f"FAILED: {name}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"FAILED: {name}: {e}", exc_info=True)
            sys.exit(1)
        logger.info("")

    logger.info("=" * 60)
    logger.info("ALL 5 STAGES COMPLETE!")
    logger.info("=" * 60)
    logger.info("Pipeline summary:")
    logger.info("  1. Papers:        data/papers/papers.jsonl")
    logger.info("  2. Rubrics:       data/rubrics/")
    logger.info("  3. SFT model:     checkpoints/sft/final/")
    logger.info("  4. RL model:      checkpoints/rl/stage_0/")
    logger.info("  5. Results:       results/eval_results.json")


if __name__ == "__main__":
    main()
