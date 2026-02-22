import pytest
from klong.agent.sandbox.docker_manager import SandboxManager, SandboxConfig, ExecResult

def test_sandbox_config_defaults():
    cfg = SandboxConfig()
    assert cfg.image == "klong-sandbox:latest"
    assert cfg.memory_limit == "8g"

def test_sandbox_manager_creation():
    mgr = SandboxManager(SandboxConfig())
    assert mgr is not None

def test_exec_result_dataclass():
    r = ExecResult(stdout="hello", stderr="", exit_code=0)
    assert r.stdout == "hello"
    assert not r.timed_out
