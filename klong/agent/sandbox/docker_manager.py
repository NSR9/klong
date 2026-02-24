from __future__ import annotations
import uuid
import time
import io
import tarfile
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SandboxConfig:
    image: str = "klong-sandbox:latest"
    memory_limit: str = "8g"
    cpu_limit: float = 4.0
    workspace_dir: str = "/workspace"
    network_enabled: bool = True
    blocked_hosts: list[str] = field(default_factory=list)


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class SandboxManager:
    def __init__(self, config: SandboxConfig):
        self.config = config
        self._client = None  # lazy init
        self._containers: dict[str, object] = {}

    @property
    def client(self):
        if self._client is None:
            import docker
            self._client = docker.from_env()
        return self._client

    def create(self) -> str:
        sandbox_id = str(uuid.uuid4())[:12]
        container = self.client.containers.run(
            self.config.image,
            command="sleep infinity",
            detach=True,
            name=f"klong-{sandbox_id}",
            mem_limit=self.config.memory_limit,
            nano_cpus=int(self.config.cpu_limit * 1e9),
            working_dir=self.config.workspace_dir,
            network_disabled=not self.config.network_enabled,
        )
        self._containers[sandbox_id] = container
        return sandbox_id

    def execute(self, sandbox_id: str, command: str, timeout: int = 300) -> ExecResult:
        container = self._containers[sandbox_id]
        result_holder: list[ExecResult] = []

        def _run():
            try:
                exec_result = container.exec_run(
                    ["bash", "-c", command],
                    workdir=self.config.workspace_dir,
                    demux=True,
                )
                stdout = (exec_result.output[0] or b"").decode("utf-8", errors="replace")
                stderr = (exec_result.output[1] or b"").decode("utf-8", errors="replace")
                result_holder.append(ExecResult(
                    stdout=stdout, stderr=stderr, exit_code=exec_result.exit_code))
            except Exception as e:
                result_holder.append(ExecResult(stdout="", stderr=str(e), exit_code=-1))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Kill the running command by executing kill inside the container
            try:
                container.exec_run(["bash", "-c", "kill -9 -1"], detach=True)
            except Exception:
                pass
            return ExecResult(
                stdout="", stderr=f"Command timed out after {timeout}s",
                exit_code=-1, timed_out=True)

        if result_holder:
            return result_holder[0]
        return ExecResult(stdout="", stderr="Unknown error", exit_code=-1)

    def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        container = self._containers[sandbox_id]
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=path.split("/")[-1])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_stream.seek(0)
        directory = "/".join(path.split("/")[:-1]) or "/"
        container.put_archive(directory, tar_stream)

    def read_file(self, sandbox_id: str, path: str) -> str:
        result = self.execute(sandbox_id, f"cat {path}")
        return result.stdout

    def destroy(self, sandbox_id: str) -> None:
        container = self._containers.pop(sandbox_id, None)
        if container:
            try:
                container.stop(timeout=5)
                container.remove(force=True)
            except Exception:
                pass

    def destroy_all(self) -> None:
        for sid in list(self._containers.keys()):
            self.destroy(sid)
