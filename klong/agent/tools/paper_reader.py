from klong.agent.tools.base import Tool, ToolResult

class PaperReaderTool(Tool):
    name = "read_paper"
    description = "Read the research paper (Markdown). Use section param to read specific parts."
    parameters = {
        "section": {"type": "string", "description": "Optional: section heading to read. Omit for full paper."},
    }

    def __init__(self, sandbox_manager, paper_markdown: str):
        super().__init__(sandbox_manager)
        self.paper_markdown = paper_markdown
        self.read_count = 0

    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        self.read_count += 1
        section = kwargs.get("section", "")
        if section:
            lines = self.paper_markdown.split("\n")
            in_section = False
            result_lines = []
            for line in lines:
                if line.strip().lower().startswith("#") and section.lower() in line.lower():
                    in_section = True
                elif in_section and line.strip().startswith("#") and section.lower() not in line.lower():
                    break
                if in_section:
                    result_lines.append(line)
            text = "\n".join(result_lines) if result_lines else f"Section '{section}' not found."
        else:
            text = self.paper_markdown
        if len(text) > 15000:
            text = text[:15000] + "\n...[truncated - use section param for specific parts]..."
            return ToolResult(output=text, truncated=True)
        return ToolResult(output=text)
