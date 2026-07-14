"""
Stage 2: KinyaEmbed MNLI Triplet Training
Continues from a Stage 1 checkpoint and fine-tunes on machine-translated
MultiNLI (anchor, positive, negative) triplets in Kinyarwanda.

Output checkpoint: v12
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


def load_triplets(path: str) -> list[InputExample]:
    """Load anchor/positive/negative triplets from JSONL."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # MNRL with triplets: include the negative as a third text
            examples.append(InputExample(
                texts=[row["anchor"], row["positive"], row["negative"]]
            ))
    return examples


def load_pairs(path: str) -> list[InputExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(InputExample(texts=[row["sent1"], row["sent2"]]))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", required=True,
                        help="Stage 1 checkpoint path (e.g. .../sc35)")
    parser.add_argument("--triplets", required=True,
                        help="Path to MNLI triplets JSONL (mnli_kin_triplets.jsonl)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save the v12 checkpoint")
    parser.add_argument("--semrel_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Stage 1 checkpoint: {args.init_model}")
    model = SentenceTransformer(args.init_model)
    model.max_seq_length = args.max_seq_len

    logger.info(f"Loading MNLI triplets from {args.triplets}")
    examples = load_triplets(args.triplets)
    logger.info(f"  {len(examples)} triplets loaded")

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
        output_path=str(output_dir / "v12"),
        show_progress_bar=True,
    )

    logger.info(f"Stage 2 complete. Checkpoint: {output_dir}/v12")


if __name__ == "__main__":
    main()
