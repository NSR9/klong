from __future__ import annotations
import os
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


def _detect_device():
    """Detect best available device and appropriate dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps", torch.float32
    return "cpu", torch.float32


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
        device_type, dtype = _detect_device()
        logger.info(f"Loading model: {self.model_name} (device={device_type}, dtype={dtype})")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {"trust_remote_code": True}

        # 4-bit quantization only works on CUDA
        if self.load_in_4bit and device_type == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_4bit:
            logger.warning(f"4-bit quantization not supported on {device_type}, using float32")

        kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        if device_type == "mps":
            model = model.to("mps")

        if self.load_in_4bit and device_type == "cuda":
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

    def _build_masked_dataset(self, traj_dataset, tokenizer):
        """Pre-tokenize trajectories with action masking.

        Converts per-character action_mask to per-token labels:
        - Assistant tokens → label = token_id (trained on)
        - Observation/user tokens → label = -100 (ignored by loss)
        """
        from transformers import DataCollatorForSeq2Seq

        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        total_tokens = 0
        action_tokens = 0

        for item in traj_dataset:
            text = item["text"]
            char_mask = item["action_mask"]

            enc = tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                return_offsets_mapping=True,
            )

            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]

            # Map character-level mask → token-level labels
            labels = []
            for tid, (start, end) in zip(input_ids, offsets):
                if start == end:
                    # Special token (BOS/EOS/PAD) — don't train
                    labels.append(-100)
                elif start < len(char_mask) and char_mask[start]:
                    # This token starts in an action (assistant) region
                    labels.append(tid)
                    action_tokens += 1
                else:
                    # Observation token — mask from loss
                    labels.append(-100)
                total_tokens += 1

            all_input_ids.append(input_ids)
            all_attention_masks.append(enc["attention_mask"])
            all_labels.append(labels)

        action_pct = (action_tokens / total_tokens * 100) if total_tokens else 0
        logger.info(
            f"Action masking: {action_pct:.1f}% of tokens are trainable "
            f"({action_tokens}/{total_tokens})"
        )

        hf_dataset = HFDataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        })

        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
        )

        return hf_dataset, collator

    def train(self, trajectory_dir: str, use_action_mask: bool = True):
        device_type, _ = _detect_device()
        model, tokenizer = self._load_model_and_tokenizer()

        traj_dataset = TrajectoryDataset(trajectory_dir)

        bf16 = self.use_bf16 and device_type == "cuda"
        fp16 = False

        if use_action_mask:
            # Paper-faithful: only train on assistant turns
            hf_dataset, collator = self._build_masked_dataset(traj_dataset, tokenizer)

            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                learning_rate=self.learning_rate,
                warmup_ratio=self.warmup_ratio,
                bf16=bf16,
                fp16=fp16,
                gradient_checkpointing=self.gradient_checkpointing,
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=3,
                seed=42,
                use_cpu=(device_type == "cpu"),
            )

            from transformers import Trainer
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=hf_dataset,
                args=training_args,
                data_collator=collator,
            )
        else:
            # Original: train on all tokens (no masking)
            texts = [item["text"] for item in traj_dataset]
            hf_dataset = HFDataset.from_dict({"text": texts})

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
                max_length=self.max_seq_length,
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=3,
                seed=42,
                dataset_text_field="text",
                use_cpu=(device_type == "cpu"),
            )

            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=hf_dataset,
                args=training_args,
            )

        logger.info(f"Starting SFT training on {device_type} "
                     f"(action_mask={'ON' if use_action_mask else 'OFF'})...")
        trainer.train()
        trainer.save_model(self.output_dir + "/final")
        tokenizer.save_pretrained(self.output_dir + "/final")
        logger.info(f"SFT training complete. Model saved to {self.output_dir}/final")
