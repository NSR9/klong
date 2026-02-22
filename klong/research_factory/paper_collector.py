from __future__ import annotations
import re
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import arxiv
from klong.research_factory.blacklist import Blacklist

logger = logging.getLogger(__name__)

@dataclass
class PaperRecord:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    github_url: str
    pdf_url: str
    markdown: str
    conference: str
    year: int

    def to_dict(self) -> dict:
        return asdict(self)

class PaperCollector:
    CONFERENCE_QUERIES = {
        "ICML": "cat:cs.LG AND (ICML)",
        "NeurIPS": "cat:cs.LG AND (NeurIPS OR neurips)",
        "ICLR": "cat:cs.LG AND (ICLR)",
    }

    def __init__(self, output_dir: str = "data/papers",
                 conferences: list[str] | None = None,
                 max_papers: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conferences = conferences or ["ICML", "NeurIPS", "ICLR"]
        self.max_papers = max_papers
        self.blacklist = Blacklist()

    def _extract_github_url(self, text: str) -> Optional[str]:
        pattern = r'https?://github\.com/[\w\-]+/[\w\-]+'
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def search_papers(self) -> list[PaperRecord]:
        papers = []
        per_conf = self.max_papers // len(self.conferences)

        for conf in self.conferences:
            query = self.CONFERENCE_QUERIES.get(conf, f"cat:cs.LG AND ({conf})")
            logger.info(f"Searching ArXiv for {conf} papers...")

            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=per_conf * 3,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            count = 0
            for result in client.results(search):
                if count >= per_conf:
                    break
                github_url = self._extract_github_url(
                    result.summary + " ".join(str(l) for l in result.links)
                )
                if not github_url:
                    continue

                paper = PaperRecord(
                    paper_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    abstract=result.summary,
                    authors=[a.name for a in result.authors[:5]],
                    github_url=github_url,
                    pdf_url=result.pdf_url,
                    markdown="",
                    conference=conf,
                    year=result.published.year,
                )
                papers.append(paper)
                self.blacklist.add(github_url)
                count += 1
                logger.info(f"  Found: {paper.title[:60]}... ({github_url})")

        logger.info(f"Collected {len(papers)} papers total")
        return papers

    def save_papers(self, papers: list[PaperRecord]):
        output_path = self.output_dir / "papers.jsonl"
        with open(output_path, "w") as f:
            for p in papers:
                f.write(json.dumps(p.to_dict()) + "\n")
        self.blacklist.save(str(self.output_dir / "blacklist.json"))
        logger.info(f"Saved {len(papers)} papers to {output_path}")
