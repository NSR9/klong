"""Generate expert trajectories using Claude API."""
import argparse
import json
import logging
from pathlib import Path
from klong.research_factory.trajectory_distiller import TrajectoryDistiller
from klong.research_factory.rubric_generator import RubricGenerator
from klong.evaluation.rubric import RubricTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-file", default="data/papers/papers.jsonl")
    parser.add_argument("--rubric-dir", default="data/rubrics")
    parser.add_argument("--output-dir", default="data/trajectories")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--timeout-minutes", type=int, default=120)
    args = parser.parse_args()

    rubric_gen = RubricGenerator(model=args.model)
    distiller = TrajectoryDistiller(model=args.model, timeout_minutes=args.timeout_minutes)

    Path(args.rubric_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.papers_file) as f:
        papers = [json.loads(line) for line in f if line.strip()]

    for paper in papers:
        paper_id = paper["paper_id"]
        rubric_path = Path(args.rubric_dir) / f"{paper_id}.json"

        if not rubric_path.exists():
            logger.info(f"Generating rubric for {paper_id}...")
            try:
                rubric = rubric_gen.generate_and_save(paper["markdown"], "", str(rubric_path))
            except Exception as e:
                logger.error(f"Failed to generate rubric for {paper_id}: {e}")
                continue
        else:
            with open(rubric_path) as f2:
                rubric = RubricTree.from_dict(json.load(f2))

        traj_path = Path(args.output_dir) / f"{paper_id}.json"
        if traj_path.exists():
            logger.info(f"Trajectory exists for {paper_id}, skipping")
            continue

        logger.info(f"Distilling trajectory for {paper_id}...")
        task_desc = f"Reproduce the paper '{paper['title']}' from scratch."
        success = distiller.distill_and_save(
            paper_id, paper["markdown"], task_desc, rubric, args.output_dir)
        logger.info(f"{'Success' if success else 'Rejected'}: {paper_id}")

if __name__ == "__main__":
    main()
