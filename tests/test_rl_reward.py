import pytest
from klong.training.rl.reward import compute_group_advantages
from klong.training.rl.rollout import RolloutGenerator

def test_group_advantages_basic():
    rewards = [0.8, 0.6, 0.4, 0.2]
    advantages = compute_group_advantages(rewards)
    assert len(advantages) == 4
    assert advantages[0] > 0  # above mean
    assert advantages[-1] < 0  # below mean

def test_group_advantages_all_equal():
    rewards = [0.5, 0.5, 0.5, 0.5]
    advantages = compute_group_advantages(rewards)
    assert all(abs(a) < 1e-6 for a in advantages)

def test_group_advantages_normalized():
    rewards = [1.0, 0.0]
    advantages = compute_group_advantages(rewards)
    # Should be normalized (mean=0.5, std=0.5)
    assert advantages[0] > 0
    assert advantages[1] < 0
    assert abs(advantages[0] + advantages[1]) < 1e-6

def test_rollout_generator_creation():
    gen = RolloutGenerator(model=None, tokenizer=None)
    assert gen.max_new_tokens == 4096
