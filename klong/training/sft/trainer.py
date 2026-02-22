from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig as TRLSFTConfig
from datasets import Dataset as HFDataset

from klong.training.data.trajectory_dataset import TrajectoryDataset

logger = logging.getLogger(__name__)

class SFTTrainerWrapper:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_target_modules: list[str] | None = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        warmup_ratio: float = 0.1,
        max_seq_length: int = 32768,
        output_dir: str = "checkpoints/sft",
        use_bf16: bool = True,
        gradient_checkpointing: bool = True,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.use_bf16 = use_bf16
        self.gradient_checkpointing = gradient_checkpointing
        self.load_in_4bit = load_in_4bit

    def _load_model_and_tokenizer(self):
        logger.info(f"Loading model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {"trust_remote_code": True}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            if self.use_bf16 and torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.bfloat16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        if self.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    def train(self, trajectory_dir: str):
        model, tokenizer = self._load_model_and_tokenizer()

        traj_dataset = TrajectoryDataset(trajectory_dir)
        texts = [item["text"] for item in traj_dataset]
        hf_dataset = HFDataset.from_dict({"text": texts})

        if torch.cuda.is_available():
            bf16 = self.use_bf16
            fp16 = False
        else:
            bf16 = False
            fp16 = False

        training_args = TRLSFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            bf16=bf16,
            fp16=fp16,
            gradient_checkpointing=self.gradient_checkpointing,
            max_seq_length=self.max_seq_length,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=3,
            seed=42,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            args=training_args,
        )

        logger.info("Starting SFT training...")
        trainer.train()
        trainer.save_model(self.output_dir + "/final")
        tokenizer.save_pretrained(self.output_dir + "/final")
        logger.info(f"SFT training complete. Model saved to {self.output_dir}/final")
