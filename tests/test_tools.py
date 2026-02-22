import pytest
from klong.agent.tools.base import Tool, ToolResult
from klong.agent.tools.bash_tool import BashTool
from klong.agent.tools.python_tool import PythonTool
from klong.agent.tools.file_tool import WriteFileTool, ReadFileTool, SearchFilesTool
from klong.agent.tools.paper_reader import PaperReaderTool

def test_tool_base_class():
    class DummyTool(Tool):
        name = "dummy"
        description = "A dummy tool"
        parameters = {"command": {"type": "string"}}
        def execute(self, sandbox_id, **kwargs):
            return ToolResult(output="ok", error="")
    t = DummyTool(sandbox_manager=None)
    assert t.name == "dummy"
    r = t.execute("fake", command="test")
    assert r.output == "ok"

def test_tool_to_schema():
    class DummyTool(Tool):
        name = "dummy"
        description = "A dummy"
        parameters = {"x": {"type": "string"}}
        def execute(self, sandbox_id, **kwargs):
            return ToolResult(output="")
    t = DummyTool(sandbox_manager=None)
    schema = t.to_schema()
    assert schema["name"] == "dummy"
    assert "x" in schema["parameters"]["properties"]

def test_bash_tool_schema():
    t = BashTool(sandbox_manager=None)
    assert t.name == "bash"
    assert "command" in t.parameters

def test_python_tool_schema():
    t = PythonTool(sandbox_manager=None)
    assert t.name == "python"
    assert "code" in t.parameters

def test_file_tools_schema():
    wt = WriteFileTool(sandbox_manager=None)
    assert "path" in wt.parameters and "content" in wt.parameters
    rt = ReadFileTool(sandbox_manager=None)
    assert "path" in rt.parameters
    st = SearchFilesTool(sandbox_manager=None)
    assert "pattern" in st.parameters

def test_paper_reader():
    pr = PaperReaderTool(sandbox_manager=None, paper_markdown="# Title\nHello world")
    assert pr.paper_markdown == "# Title\nHello world"
    r = pr.execute("fake")
    assert "Title" in r.output

def test_paper_reader_section():
    md = "# Introduction\nSome intro text\n# Methods\nSome methods text\n# Results\nResults here"
    pr = PaperReaderTool(sandbox_manager=None, paper_markdown=md)
    r = pr.execute("fake", section="Methods")
    assert "methods text" in r.output.lower()

def test_paper_reader_tracks_reads():
    pr = PaperReaderTool(sandbox_manager=None, paper_markdown="# Paper\nContent")
    pr.execute("fake")
    pr.execute("fake")
    assert pr.read_count == 2
