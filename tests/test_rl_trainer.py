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

def test_rl_trainer_defaults():
    trainer = ProgressiveRLTrainer()
    assert trainer.model_name == "Qwen/Qwen2.5-7B"
    assert len(trainer.stages) == 3
    assert trainer.stages[0]["timeout_minutes"] == 30
    assert trainer.stages[1]["timeout_minutes"] == 60
    assert trainer.stages[2]["timeout_minutes"] == 120
    assert trainer.clip_epsilon == 0.2
    assert trainer.kl_coeff == 0.01
    assert trainer.rollouts_per_task == 4

def test_rl_trainer_custom_stages():
    trainer = ProgressiveRLTrainer(
        stages=[
            {"timeout_minutes": 10, "num_epochs": 1},
            {"timeout_minutes": 20, "num_epochs": 2},
        ]
    )
    assert len(trainer.stages) == 2
