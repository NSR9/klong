import pytest
from klong.research_factory.paper_collector import PaperCollector, PaperRecord
from klong.research_factory.blacklist import Blacklist
from klong.research_factory.pdf_converter import PDFConverter

def test_paper_record():
    p = PaperRecord(
        paper_id="2301.00001", title="Test Paper",
        abstract="An abstract.", authors=["Author A"],
        github_url="https://github.com/user/repo",
        pdf_url="https://arxiv.org/pdf/2301.00001",
        markdown="", conference="ICML", year=2023,
    )
    assert p.paper_id == "2301.00001"
    d = p.to_dict()
    assert d["title"] == "Test Paper"

def test_blacklist():
    bl = Blacklist()
    bl.add("https://github.com/user/repo")
    assert bl.is_blocked("https://github.com/user/repo")
    assert bl.is_blocked("github.com/user/repo")
    assert not bl.is_blocked("https://github.com/other/repo")

def test_blacklist_persistence(tmp_path):
    bl = Blacklist()
    bl.add("https://github.com/user/repo")
    path = tmp_path / "blacklist.json"
    bl.save(str(path))
    bl2 = Blacklist.load(str(path))
    assert bl2.is_blocked("github.com/user/repo")

def test_paper_collector_creation():
    collector = PaperCollector(output_dir="/tmp/test_papers", max_papers=10)
    assert collector.max_papers == 10

def test_extract_github_url():
    collector = PaperCollector()
    url = collector._extract_github_url("Code available at https://github.com/user/awesome-project and more text")
    assert url == "https://github.com/user/awesome-project"

def test_extract_github_url_none():
    collector = PaperCollector()
    url = collector._extract_github_url("No github link here")
    assert url is None

def test_pdf_converter_creation():
    converter = PDFConverter()
    assert converter is not None
