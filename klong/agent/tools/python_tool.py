from klong.agent.tools.base import Tool, ToolResult

class PythonTool(Tool):
    name = "python"
    description = "Execute Python code in the sandbox. The code is written to a temp file and run."
    parameters = {"code": {"type": "string", "description": "Python code to execute"}}

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        code = kwargs["code"]
        self.sandbox_manager.write_file(sandbox_id, "/workspace/_tmp_exec.py", code)
        result = self.sandbox_manager.execute(sandbox_id, "python3 /workspace/_tmp_exec.py")
        output = result.stdout
        if len(output) > 10000:
            output = output[:5000] + "\n...[truncated]...\n" + output[-5000:]
            return ToolResult(output=output, error=result.stderr, truncated=True)
        return ToolResult(output=output, error=result.stderr)
