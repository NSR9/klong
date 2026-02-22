from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-7B"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: list[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    max_seq_length: int = 32768
    load_in_4bit: bool = False
    trust_remote_code: bool = True


class PaperCollectionConfig(BaseModel):
    conferences: list[str] = Field(default_factory=lambda: ["ICML", "NeurIPS", "ICLR"])
    years_back: int = 5
    max_papers: int = 100
    output_dir: str = "data/papers"


class DistillationConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 200
    timeout_minutes: int = 120
    rejection_threshold: float = 0.3
    output_dir: str = "data/trajectories"


class DataConfig(BaseModel):
    papers: PaperCollectionConfig = Field(default_factory=PaperCollectionConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    trajectory_dir: str = "data/trajectories"
    rubric_dir: str = "data/rubrics"


class SplitterConfig(BaseModel):
    overlap_tokens: int = 2048
    max_window_tokens: int = 30720


class SFTConfig(BaseModel):
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)


class RLStageConfig(BaseModel):
    timeout_minutes: int = 30
    num_epochs: int = 2


class RLConfig(BaseModel):
    learning_rate: float = 5e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.01
    rollouts_per_task: int = 4
    max_new_tokens_per_turn: int = 4096
    stages: list[RLStageConfig] = Field(default_factory=lambda: [
        RLStageConfig(timeout_minutes=30),
        RLStageConfig(timeout_minutes=60),
        RLStageConfig(timeout_minutes=120),
    ])


class TrainingConfig(BaseModel):
    sft: SFTConfig = Field(default_factory=SFTConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    output_dir: str = "checkpoints"
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42


class InfraConfig(BaseModel):
    docker_image: str = "klong-sandbox:latest"
    container_memory_limit: str = "8g"
    container_cpu_limit: float = 4.0
    max_concurrent_containers: int = 2
    workspace_base: str = "/tmp/klong_workspaces"


class EvalConfig(BaseModel):
    judge_model: str = "claude-sonnet-4-20250514"
    final_judge_model: str = "claude-opus-4-20250514"


class KLongConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
