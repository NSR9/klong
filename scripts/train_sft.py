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
