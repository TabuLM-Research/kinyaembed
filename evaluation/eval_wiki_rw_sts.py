"""
Evaluate on Wiki-RW-STS: 300 fresh Kinyarwanda Wikipedia sentence pairs
(contamination-free benchmark introduced in the KinyaEmbed paper).
"""

import argparse
import json
import logging

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_wiki_rw_sts(path: str):
    sents1, sents2, scores = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sents1.append(row["sent1"])
            sents2.append(row["sent2"])
            scores.append(float(row["score"]))
    return sents1, sents2, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", default="data/wiki_rw_sts_pairs.jsonl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    sents1, sents2, gold = load_wiki_rw_sts(args.pairs)
    logger.info(f"Loaded {len(sents1)} Wiki-RW-STS pairs")

    emb1 = model.encode(sents1, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    emb2 = model.encode(sents2, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)

    pred = (emb1 * emb2).sum(axis=1)  # cosine (already normalized)
    gold_arr = np.array(gold)

    spearman = spearmanr(gold_arr, pred).correlation

    # AUC: high-similarity pairs (score > 0.6) as positive class
    binary = (gold_arr > 0.6).astype(int)
    auc = roc_auc_score(binary, pred) if binary.sum() > 0 else None

    results = {
        "model": args.model,
        "n_pairs": len(sents1),
        "spearman": round(float(spearman), 4),
        "auc": round(float(auc), 4) if auc else None,
    }

    print(f"\n{'='*50}")
    print(f"Model:    {results['model']}")
    print(f"Pairs:    {results['n_pairs']}")
    print(f"Spearman: {results['spearman']:.4f}")
    if results["auc"]:
        print(f"AUC:      {results['auc']:.4f}")
    print(f"{'='*50}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
