"""
Evaluate a sentence embedding model on SemRel2024-rw (Kinyarwanda STS).

Reports Spearman ρ and Pearson r against the 222-pair test set.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_semrel(semrel_dir: str):
    test_path = Path(semrel_dir) / "test.jsonl"
    if not test_path.exists():
        # Try HuggingFace datasets
        try:
            from datasets import load_dataset
            ds = load_dataset("SemRel/SemRel2024", "rw", split="test")
            sents1 = [r["sentence1"] for r in ds]
            sents2 = [r["sentence2"] for r in ds]
            scores = [float(r["label"]) / 4.0 for r in ds]
            return sents1, sents2, scores
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find SemRel2024 at {semrel_dir}. "
                f"Download from https://huggingface.co/datasets/SemRel/SemRel2024"
            ) from e

    sents1, sents2, scores = [], [], []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sents1.append(row["sentence1"])
            sents2.append(row["sentence2"])
            scores.append(float(row["label"]) / 4.0)
    return sents1, sents2, scores


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


def evaluate_model(model_path: str, semrel_dir: str, batch_size: int = 64) -> dict:
    logger.info(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path)

    logger.info("Loading SemRel2024-rw...")
    sents1, sents2, gold_scores = load_semrel(semrel_dir)
    logger.info(f"  {len(sents1)} test pairs")

    emb1 = model.encode(sents1, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False)
    emb2 = model.encode(sents2, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False)

    pred_scores = cosine_similarity(emb1, emb2)
    gold = np.array(gold_scores)

    spearman = spearmanr(gold, pred_scores).correlation
    pearson  = pearsonr(gold, pred_scores)[0]

    results = {
        "model": model_path,
        "n_pairs": len(sents1),
        "spearman": round(float(spearman), 4),
        "pearson":  round(float(pearson), 4),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="SentenceTransformer model path or HuggingFace name")
    parser.add_argument("--semrel_dir", default="data/semrel2024_rw",
                        help="Directory containing SemRel2024 test.jsonl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", default=None,
                        help="Optional JSON file to write results to")
    args = parser.parse_args()

    results = evaluate_model(args.model, args.semrel_dir, args.batch_size)

    print(f"\n{'='*50}")
    print(f"Model:    {results['model']}")
    print(f"Pairs:    {results['n_pairs']}")
    print(f"Spearman: {results['spearman']:.4f}")
    print(f"Pearson:  {results['pearson']:.4f}")
    print(f"{'='*50}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
