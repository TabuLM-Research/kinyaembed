"""
Stage 4: KinyaEmbed KinyaCOMET Fine-Tuning
Continues from Stage 3 (step22A) and trains on high-quality
Kinyarwanda–English sentence pairs from KinyaCOMET (score ≥ 0.8).

This is the final training stage. Output checkpoint: step23A
"""

import argparse
import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_kinyacomet_pairs(path: str, min_score: float = 0.8) -> list[InputExample]:
    """Load filtered KinyaCOMET pairs with quality score >= min_score."""
    examples = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            score = float(row.get("score", row.get("quality_score", 0.0)))
            if score >= min_score:
                en = row.get("en", row.get("source", "")).strip()
                rw = row.get("rw", row.get("translation", "")).strip()
                if en and rw:
                    examples.append(InputExample(texts=[en, rw]))
            else:
                skipped += 1
    logger.info(f"  {len(examples)} pairs kept (score ≥ {min_score}), {skipped} filtered out")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", required=True,
                        help="Stage 3 checkpoint (step22A)")
    parser.add_argument("--kinyacomet_pairs", required=True,
                        help="Path to kinycomet_pairs.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--semrel_dir", default=None)
    parser.add_argument("--min_score", type=float, default=0.8,
                        help="Minimum KinyaCOMET quality score to include a pair")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Smaller batch for fine dataset")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Lower LR for fine-tuning on small dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Stage 3 checkpoint: {args.init_model}")
    model = SentenceTransformer(args.init_model)
    model.max_seq_length = args.max_seq_len

    logger.info(f"Loading KinyaCOMET pairs from {args.kinyacomet_pairs}")
    examples = load_kinyacomet_pairs(args.kinyacomet_pairs, args.min_score)

    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    evaluator = None
    if args.semrel_dir:
        from train_stage1_gazette import load_semrel_evaluator
        evaluator = load_semrel_evaluator(args.semrel_dir)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=str(output_dir / "step23A"),
        show_progress_bar=True,
    )

    logger.info(f"Stage 4 complete. Checkpoint: {output_dir}/step23A")


if __name__ == "__main__":
    main()
