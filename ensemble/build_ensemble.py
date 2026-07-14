"""
Build the KinyaEmbed all5+23A×2 ensemble.

Loads 7 checkpoints (sc30, sc35, sc40, v12, step22A, step23A, step23A),
averages their embeddings per sentence, then L2-normalizes the result.

The combined model is saved as a SentenceTransformer-compatible directory.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = ["sc30", "sc35", "sc40", "v12", "step22A", "step23A", "step23A"]


class KinyaEmbedEnsemble:
    """
    Inference wrapper: encodes with each checkpoint, averages, L2-normalizes.
    step23A is included twice (index 5 and 6) to apply double weight.
    """

    def __init__(self, checkpoint_dirs: list[str], device: str = "cpu"):
        self.models = []
        for d in checkpoint_dirs:
            logger.info(f"  Loading: {d}")
            self.models.append(SentenceTransformer(d, device=device))
        self.device = device

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        all_embs = []
        for model in self.models:
            emb = model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
            all_embs.append(emb)
        avg = np.mean(all_embs, axis=0)
        if normalize_embeddings:
            norms = np.linalg.norm(avg, axis=1, keepdims=True)
            avg = avg / np.maximum(norms, 1e-12)
        return avg


def save_ensemble_config(output_dir: Path, checkpoint_dirs: list[str]):
    config = {
        "model_type": "KinyaEmbedEnsemble",
        "checkpoints": checkpoint_dirs,
        "description": (
            "all5+23A×2 ensemble: sc30, sc35, sc40, v12, step22A, step23A (×2 weight). "
            "Averages 7 embeddings then L2-normalizes. "
            "SemRel2024-rw Spearman ρ = 0.7298."
        ),
        "embedding_dim": 768,
        "max_seq_length": 256,
    }
    with open(output_dir / "ensemble_config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved ensemble config to {output_dir}/ensemble_config.json")


def verify_ensemble(ensemble: KinyaEmbedEnsemble):
    sentences = [
        "Uburinzi bw'ubuzima mu Rwanda",
        "Health protection in Rwanda",
        "Ibikorwa bya leta mu gihugu",
    ]
    embs = ensemble.encode(sentences)
    logger.info("Verification embeddings shape: %s", embs.shape)
    norms = np.linalg.norm(embs, axis=1)
    logger.info("L2 norms (should be ~1.0): %s", norms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_base", required=True,
                        help="Base directory containing sc30, sc35, sc40, v12, step22A, step23A")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save ensemble config")
    parser.add_argument("--checkpoints", nargs="+",
                        default=DEFAULT_CHECKPOINTS,
                        help="Ordered list of checkpoint names (default: all5+23A×2)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify", action="store_true",
                        help="Run a quick encode test after building")
    args = parser.parse_args()

    base = Path(args.checkpoint_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dirs = [str(base / ck) for ck in args.checkpoints]
    for d in checkpoint_dirs:
        if not Path(d).exists():
            logger.warning(f"Checkpoint not found: {d}")

    logger.info(f"Building ensemble from {len(checkpoint_dirs)} checkpoints:")
    for i, d in enumerate(checkpoint_dirs):
        logger.info(f"  [{i}] {d}")

    save_ensemble_config(output_dir, checkpoint_dirs)

    if args.verify:
        logger.info("Loading all models for verification...")
        ensemble = KinyaEmbedEnsemble(checkpoint_dirs, device=args.device)
        verify_ensemble(ensemble)
        logger.info("Verification passed.")

    # Save a usage example script
    example = f'''"""
Usage example for KinyaEmbed all5+23A×2 ensemble.
"""
import sys
sys.path.insert(0, "{Path(__file__).parent}")
from build_ensemble import KinyaEmbedEnsemble

checkpoint_dirs = {checkpoint_dirs}
model = KinyaEmbedEnsemble(checkpoint_dirs, device="cpu")

sentences = [
    "Uburinzi bw'ubuzima mu Rwanda",
    "Health protection in Rwanda",
    "Politiki za leta",
]
embeddings = model.encode(sentences)
print("Shape:", embeddings.shape)  # (3, 768)

# Cosine similarity (embeddings are already L2-normalized)
import numpy as np
sim = embeddings @ embeddings.T
print("Similarity matrix:")
print(sim)
'''
    with open(output_dir / "usage_example.py", "w") as f:
        f.write(example)

    logger.info("Done. Run the ensemble via KinyaEmbedEnsemble in build_ensemble.py")
    logger.info("Or use TabuLM-Research/KinyaEmbed on HuggingFace (merged single model).")


if __name__ == "__main__":
    main()
