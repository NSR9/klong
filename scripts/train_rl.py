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
