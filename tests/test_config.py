import pytest
from klong.config.settings import (
    KLongConfig, ModelConfig, DataConfig, TrainingConfig,
    InfraConfig, EvalConfig, SFTConfig, RLConfig,
)

def test_default_config_creates():
    cfg = KLongConfig()
    assert cfg.model.name == "Qwen/Qwen2.5-7B"
    assert cfg.model.lora_rank == 64
    assert cfg.training.sft.learning_rate == 2e-5
    assert cfg.training.rl.clip_epsilon == 0.2

def test_config_to_dict_roundtrip():
    cfg = KLongConfig()
    d = cfg.model_dump()
    cfg2 = KLongConfig(**d)
    assert cfg == cfg2

def test_config_from_yaml(tmp_path):
    import yaml
    cfg = KLongConfig()
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg.model_dump()))
    loaded = KLongConfig(**yaml.safe_load(p.read_text()))
    assert loaded.model.name == cfg.model.name
