"""Collect papers from ArXiv and convert to markdown."""
import argparse
import logging
from klong.research_factory.paper_collector import PaperCollector
from klong.research_factory.pdf_converter import PDFConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/papers")
    parser.add_argument("--max-papers", type=int, default=100)
    parser.add_argument("--conferences", nargs="+", default=["ICML", "NeurIPS", "ICLR"])
    args = parser.parse_args()

    collector = PaperCollector(
        output_dir=args.output_dir,
        conferences=args.conferences,
        max_papers=args.max_papers,
    )
    papers = collector.search_papers()

    converter = PDFConverter()
    for paper in papers:
        try:
            paper.markdown = converter.convert_url(paper.pdf_url, args.output_dir + "/pdfs")
            logger.info(f"Converted: {paper.title[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to convert {paper.paper_id}: {e}")

    collector.save_papers(papers)
    logger.info(f"Done. {len(papers)} papers saved to {args.output_dir}")

if __name__ == "__main__":
    main()
