from __future__ import annotations
import numpy as np

def compute_group_advantages(rewards: list[float]) -> list[float]:
    """GRPO-style group-relative advantage: A_i = (r_i - mean(r)) / std(r)."""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    if std_reward < 1e-8:
        return [0.0] * len(rewards)
    return [float((r - mean_reward) / std_reward) for r in rewards]
