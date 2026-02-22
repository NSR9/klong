import pytest
from unittest.mock import MagicMock
from klong.research_factory.trajectory_distiller import TrajectoryDistiller

def test_distiller_creation():
    d = TrajectoryDistiller(model="claude-sonnet-4-20250514", sandbox_config=MagicMock())
    assert d.model == "claude-sonnet-4-20250514"

def test_distiller_default_config():
    d = TrajectoryDistiller()
    assert d.timeout_minutes == 120
    assert d.rejection_threshold == 0.3
