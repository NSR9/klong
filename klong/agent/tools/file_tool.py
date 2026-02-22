from klong.agent.tools.base import Tool, ToolResult

class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file in the sandbox workspace."
    parameters = {
        "path": {"type": "string", "description": "File path relative to /workspace"},
        "content": {"type": "string", "description": "File content to write"},
    }

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        path = kwargs["path"]
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        self.sandbox_manager.write_file(sandbox_id, path, kwargs["content"])
        return ToolResult(output=f"Written to {path}")

class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file in the sandbox."
    parameters = {"path": {"type": "string", "description": "File path to read"}}

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        path = kwargs["path"]
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        content = self.sandbox_manager.read_file(sandbox_id, path)
        if len(content) > 10000:
            content = content[:5000] + "\n...[truncated]...\n" + content[-5000:]
            return ToolResult(output=content, truncated=True)
        return ToolResult(output=content)

class SearchFilesTool(Tool):
    name = "search_files"
    description = "Search for files matching a pattern or grep for content."
    parameters = {
        "pattern": {"type": "string", "description": "Glob pattern or grep query"},
        "search_type": {"type": "string", "description": "'glob' or 'grep'"},
    }

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        pattern = kwargs["pattern"]
        search_type = kwargs.get("search_type", "glob")
        if search_type == "grep":
            result = self.sandbox_manager.execute(sandbox_id, f"grep -rn '{pattern}' /workspace/ 2>/dev/null | head -50")
        else:
            result = self.sandbox_manager.execute(sandbox_id, f"find /workspace/ -name '{pattern}' 2>/dev/null | head -50")
        return ToolResult(output=result.stdout, error=result.stderr)
