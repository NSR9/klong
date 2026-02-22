from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFConverter:
    def convert_url(self, pdf_url: str, output_dir: str) -> str:
        import requests
        import pymupdf4llm

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = pdf_url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        pdf_path = output_path / filename

        logger.info(f"Downloading {pdf_url}...")
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)

        logger.info(f"Converting {pdf_path} to markdown...")
        md_text = pymupdf4llm.to_markdown(str(pdf_path))

        md_path = pdf_path.with_suffix(".md")
        md_path.write_text(md_text)

        return md_text

    def convert_file(self, pdf_path: str) -> str:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(pdf_path)
