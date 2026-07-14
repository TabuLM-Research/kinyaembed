"""
Stage 3: KinyaEmbed OPUS-100 Cross-Lingual Training
Continues from Stage 2 (v12) and trains on English–Kinyarwanda translation
pairs from OPUS-100, aligning the two language embedding spaces.

Output checkpoint: step22A
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_opus_pairs(opus_dir: str = None) -> list[InputExample]:
    """Load English–Kinyarwanda parallel pairs from OPUS-100."""
    if opus_dir and Path(opus_dir).exists():
        logger.info(f"Loading OPUS-100 from local: {opus_dir}")
        dataset = load_from_disk(opus_dir)
        split = dataset["train"]
    else:
        logger.info("Downloading OPUS-100 en-rw from HuggingFace...")
        dataset = load_dataset("Helsinki-NLP/opus-100", "en-rw")
        split = dataset["train"]

    examples = []
    for row in split:
        en = row["translation"]["en"].strip()
        rw = row["translation"]["rw"].strip()
        if en and rw:
            examples.append(InputExample(texts=[en, rw]))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", required=True,
                        help="Stage 2 checkpoint path (v12)")
    parser.add_argument("--opus_dir", default=None,
                        help="Local OPUS-100 en-rw dataset directory (optional)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--semrel_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Stage 2 checkpoint: {args.init_model}")
    model = SentenceTransformer(args.init_model)
    model.max_seq_length = args.max_seq_len

    examples = load_opus_pairs(args.opus_dir)
    logger.info(f"  {len(examples)} EN–RW pairs loaded")

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
        output_path=str(output_dir / "step22A"),
        show_progress_bar=True,
    )

    logger.info(f"Stage 3 complete. Checkpoint: {output_dir}/step22A")


if __name__ == "__main__":
    main()
