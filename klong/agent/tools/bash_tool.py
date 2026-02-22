from klong.agent.tools.base import Tool, ToolResult

class BashTool(Tool):
    name = "bash"
    description = "Execute a bash command in the sandbox. Returns stdout and stderr."
    parameters = {"command": {"type": "string", "description": "The bash command to run"}}

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        command = kwargs["command"]
        result = self.sandbox_manager.execute(sandbox_id, command)
        output = result.stdout
        if len(output) > 10000:
            output = output[:5000] + "\n...[truncated]...\n" + output[-5000:]
            return ToolResult(output=output, error=result.stderr, truncated=True)
        return ToolResult(output=output, error=result.stderr)
