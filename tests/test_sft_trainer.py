import pytest
from klong.training.sft.trainer import SFTTrainerWrapper

def test_sft_trainer_creation():
    trainer = SFTTrainerWrapper(
        model_name="Qwen/Qwen2.5-0.5B",
        lora_rank=8, lora_alpha=16,
        output_dir="/tmp/klong_test_sft",
    )
    assert trainer.model_name == "Qwen/Qwen2.5-0.5B"

def test_sft_trainer_config():
    trainer = SFTTrainerWrapper(
        model_name="Qwen/Qwen2.5-0.5B",
        learning_rate=2e-5, num_epochs=3,
    )
    assert trainer.learning_rate == 2e-5
    assert trainer.num_epochs == 3

def test_sft_trainer_defaults():
    trainer = SFTTrainerWrapper()
    assert trainer.model_name == "Qwen/Qwen2.5-7B"
    assert trainer.lora_rank == 64
    assert trainer.lora_alpha == 128
    assert trainer.gradient_checkpointing == True
    assert trainer.batch_size == 1
    assert trainer.gradient_accumulation_steps == 8
