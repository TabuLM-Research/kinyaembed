"""
Downstream evaluation: Information Retrieval, Document Clustering,
and Zero-Shot Classification on Kinyarwanda Wikipedia articles.

Uses wiki_rw_corpus.jsonl (300 articles, 8 categories).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TOPIC_LABELS = [
    "amateka",        # history
    "ikoranabuhanga", # science/technology
    "isi-akarere",    # geography
    "imikino",        # sports
    "politiki",       # politics
    "idini",          # religion
    "ubuhinzi",       # agriculture
    "ubuzima",        # health
]

TOPIC_EN = {
    "amateka": "history",
    "ikoranabuhanga": "science and technology",
    "isi-akarere": "geography and places",
    "imikino": "sports and games",
    "politiki": "politics and government",
    "idini": "religion and culture",
    "ubuhinzi": "agriculture and food",
    "ubuzima": "health and medicine",
}


def load_corpus(corpus_path: str) -> tuple[list[str], list[str], list[str | None]]:
    """Returns (titles, bodies, gt_labels)."""
    titles, bodies, labels = [], [], []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            titles.append(row["title"])
            bodies.append(row.get("body", row.get("text", "")))
            labels.append(row.get("category", row.get("label", None)))
    return titles, bodies, labels


def eval_ir(title_embs: np.ndarray, body_embs: np.ndarray,
            k_values: list[int] = (1, 5, 10)) -> dict:
    """Title → body retrieval: for each title, find nearest body."""
    t = title_embs / np.linalg.norm(title_embs, axis=1, keepdims=True)
    b = body_embs  / np.linalg.norm(body_embs,  axis=1, keepdims=True)
    sim = t @ b.T     # (N, N)
    gold = np.arange(len(t))

    results = {}
    for k in k_values:
        top_k = np.argsort(-sim, axis=1)[:, :k]
        hits = sum(gold[i] in top_k[i] for i in range(len(gold)))
        results[f"P@{k}"] = round(hits / len(gold), 4)

    # MRR@10
    top10 = np.argsort(-sim, axis=1)[:, :10]
    mrr = 0.0
    for i in range(len(gold)):
        for rank, j in enumerate(top10[i]):
            if j == gold[i]:
                mrr += 1.0 / (rank + 1)
                break
    results["MRR@10"] = round(mrr / len(gold), 4)
    return results


def eval_clustering(embs: np.ndarray, k: int = 8) -> dict:
    """K-means clustering quality."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embs)
    sil = silhouette_score(embs, labels)
    db  = davies_bouldin_score(embs, labels)
    return {
        "kmeans_k": k,
        "silhouette": round(float(sil), 4),
        "davies_bouldin": round(float(db), 4),
    }


def eval_classification(body_embs: np.ndarray, gt_labels: list[str | None],
                        model: SentenceTransformer) -> dict:
    """Zero-shot: cosine to topic prototypes."""
    topic_prompts = [TOPIC_EN[t] for t in TOPIC_LABELS]
    proto_embs = model.encode(topic_prompts, normalize_embeddings=True)
    body_norm  = body_embs / np.linalg.norm(body_embs, axis=1, keepdims=True)
    sim = body_norm @ proto_embs.T  # (N, n_topics)

    labeled_mask = [l is not None and l in TOPIC_LABELS for l in gt_labels]
    if not any(labeled_mask):
        logger.warning("No labeled articles found for classification eval.")
        return {"cls_acc_top1": None, "cls_acc_top3": None, "n_labeled": 0}

    correct1, correct3, total = 0, 0, 0
    for i, (mask, label) in enumerate(zip(labeled_mask, gt_labels)):
        if not mask:
            continue
        gold_idx = TOPIC_LABELS.index(label)
        top1 = sim[i].argmax()
        top3 = np.argsort(-sim[i])[:3]
        if top1 == gold_idx:
            correct1 += 1
        if gold_idx in top3:
            correct3 += 1
        total += 1

    return {
        "cls_acc_top1": round(correct1 / total, 4),
        "cls_acc_top3": round(correct3 / total, 4),
        "n_labeled": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="data/wiki_rw_corpus.jsonl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k_clusters", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    logger.info(f"Loading corpus from {args.corpus}")
    titles, bodies, gt_labels = load_corpus(args.corpus)
    logger.info(f"  {len(titles)} articles loaded")

    logger.info("Encoding titles and bodies...")
    title_embs = model.encode(titles, batch_size=args.batch_size, show_progress_bar=True,
                               normalize_embeddings=False)
    body_embs  = model.encode(bodies, batch_size=args.batch_size, show_progress_bar=True,
                               normalize_embeddings=False)

    results = {"model": args.model}

    logger.info("Evaluating IR (title → body retrieval)...")
    ir = eval_ir(title_embs, body_embs)
    results.update(ir)

    logger.info("Evaluating clustering...")
    body_norm = body_embs / np.linalg.norm(body_embs, axis=1, keepdims=True)
    clust = eval_clustering(body_norm, k=args.k_clusters)
    results.update(clust)

    logger.info("Evaluating zero-shot classification...")
    cls = eval_classification(body_embs, gt_labels, model)
    results.update(cls)

    print(f"\n{'='*55}")
    print(f"Model: {results['model']}")
    print(f"IR  — P@1: {ir.get('P@1', 'N/A')}  P@5: {ir.get('P@5', 'N/A')}  MRR@10: {ir.get('MRR@10', 'N/A')}")
    print(f"Clust — Silhouette: {clust['silhouette']}  Davies-Bouldin: {clust['davies_bouldin']}")
    print(f"Cls   — Top-1: {cls['cls_acc_top1']}  Top-3: {cls['cls_acc_top3']}  (n={cls['n_labeled']})")
    print(f"{'='*55}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
