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

from klong.config.settings import RLStageConfig
from klong.training.rl.reward import compute_group_advantages
from klong.training.rl.rollout import RolloutGenerator
from klong.training.data.trajectory_splitter import TrajectorySplitter
from klong.evaluation.judge import Judge
from klong.evaluation.rubric import RubricTree
from klong.agent.sandbox.docker_manager import SandboxConfig

logger = logging.getLogger(__name__)

class ProgressiveRLTrainer:
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
        if Path(self.sft_checkpoint).exists():
            model = PeftModel.from_pretrained(model, self.sft_checkpoint)
            model = model.merge_and_unload()
            logger.info(f"Loaded SFT checkpoint from {self.sft_checkpoint}")

        return model, tokenizer

    def train(self, task_data_path: str, rubric_dir: str):
        model, tokenizer = self._load_model()

        with open(task_data_path) as f:
            tasks = [json.loads(line) for line in f if line.strip()]

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

                task_rollouts = []
                task_rewards = []
                for i in range(self.rollouts_per_task):
                    logger.info(f"    Rollout {i+1}/{self.rollouts_per_task} for {paper_id}")
                    trajectory = rollout_gen.generate_rollout(
                        paper_id, paper_markdown, task_desc, timeout_seconds)

                    artifacts = {}
                    score, _ = asyncio.run(judge.evaluate(rubric, artifacts))
                    trajectory.final_score = score

                    task_rollouts.append(trajectory)
                    task_rewards.append(score)

                advantages = compute_group_advantages(task_rewards)

                for traj, adv in zip(task_rollouts, advantages):
                    all_trajectories.append((traj, adv))
                    all_rewards.append(traj.final_score)

            if not all_trajectories:
                logger.warning("No trajectories generated, skipping epoch")
                continue

            mean_reward = np.mean(all_rewards)
            logger.info(f"  Mean reward: {mean_reward:.4f}")
            logger.info(f"  Generated {len(all_trajectories)} trajectory-advantage pairs")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"  Saved checkpoint to {output_dir}")
