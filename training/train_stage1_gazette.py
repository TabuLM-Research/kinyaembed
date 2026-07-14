"""
Stage 1: KinyaEmbed Gazette Paraphrase Training
Fine-tunes KinyaBERT-large on monolingual Kinyarwanda paraphrase pairs
from the Official Gazette of Rwanda using MNRL.

Saves checkpoints at three training scale milestones: sc30, sc35, sc40.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pairs(path: str) -> list[InputExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(InputExample(texts=[row["sent1"], row["sent2"]]))
    return examples


def load_semrel_evaluator(semrel_dir: str):
    pairs, scores = [], []
    test_path = Path(semrel_dir) / "test.jsonl"
    if not test_path.exists():
        return None
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pairs.append([row["sentence1"], row["sentence2"]])
            scores.append(float(row["label"]) / 4.0)  # SemRel uses 0-4 scale
    return EmbeddingSimilarityEvaluator(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        scores,
        name="semrel2024-rw",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="jean-paul/KinyaBERTlarge",
                        help="HuggingFace model name or local path")
    parser.add_argument("--pairs", required=True,
                        help="Path to gazette paraphrase pairs JSONL")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--semrel_dir", default=None,
                        help="SemRel2024 directory for evaluation during training")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--scale_checkpoints", nargs="+", type=int,
                        default=[30, 35, 40],
                        help="Save checkpoints at these % of training progress")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_len

    logger.info(f"Loading gazette paraphrase pairs from {args.pairs}")
    examples = load_pairs(args.pairs)
    logger.info(f"  {len(examples)} pairs loaded")

    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    evaluator = load_semrel_evaluator(args.semrel_dir) if args.semrel_dir else None

    total_steps = len(train_loader) * 1  # 1 epoch for Stage 1
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Compute absolute step counts for scale checkpoints
    checkpoint_steps = {
        int(s / 100 * total_steps): f"sc{s}"
        for s in args.scale_checkpoints
    }

    logger.info(f"Training for {total_steps} steps, warmup={warmup_steps}")
    logger.info(f"Checkpoint milestones: {checkpoint_steps}")

    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        evaluator=evaluator,
        epochs=1,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=str(output_dir / "sc40"),  # Final epoch = sc40
        checkpoint_path=str(output_dir),
        checkpoint_save_steps=min(checkpoint_steps.keys()),
        show_progress_bar=True,
    )

    # Rename checkpoints to sc30 / sc35 / sc40 names
    for step, name in checkpoint_steps.items():
        ck_path = output_dir / str(step)
        dest = output_dir / name
        if ck_path.exists() and not dest.exists():
            ck_path.rename(dest)
            logger.info(f"  Checkpoint {step} → {name}")

    logger.info("Stage 1 complete.")


if __name__ == "__main__":
    main()
