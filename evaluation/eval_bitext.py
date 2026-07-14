"""
Evaluate a sentence embedding model on bitext mining tasks.

Benchmarks:
  - OPUS-100 en-rw test split (P@1 averaged both directions)
  - FLORES-200 eng_Latn <-> kin_Latn devtest (P@1)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def precision_at_1(src_embs: np.ndarray, tgt_embs: np.ndarray) -> float:
    """P@1: for each source embedding, check if nearest target is the correct one."""
    src = src_embs / np.linalg.norm(src_embs, axis=1, keepdims=True)
    tgt = tgt_embs / np.linalg.norm(tgt_embs, axis=1, keepdims=True)
    sim = src @ tgt.T          # (N, N) cosine sim matrix
    preds = sim.argmax(axis=1)  # each row's argmax
    gold  = np.arange(len(src))
    return float((preds == gold).mean())


def eval_opus(model: SentenceTransformer, opus_dir: str | None, batch_size: int) -> dict:
    if opus_dir and Path(opus_dir).exists():
        from datasets import load_from_disk
        ds = load_from_disk(opus_dir)["test"]
    else:
        from datasets import load_dataset
        logger.info("Downloading OPUS-100 en-rw from HuggingFace...")
        ds = load_dataset("Helsinki-NLP/opus-100", "en-rw")["test"]

    en_sents = [r["translation"]["en"] for r in ds]
    rw_sents = [r["translation"]["rw"] for r in ds]
    logger.info(f"OPUS-100 test: {len(en_sents)} pairs")

    en_embs = model.encode(en_sents, batch_size=batch_size, show_progress_bar=True)
    rw_embs = model.encode(rw_sents, batch_size=batch_size, show_progress_bar=True)

    p1_en2rw = precision_at_1(en_embs, rw_embs)
    p1_rw2en = precision_at_1(rw_embs, en_embs)
    p1_avg   = (p1_en2rw + p1_rw2en) / 2

    return {
        "opus_p1_en2rw": round(p1_en2rw, 4),
        "opus_p1_rw2en": round(p1_rw2en, 4),
        "opus_p1_avg":   round(p1_avg, 4),
    }


def eval_flores(model: SentenceTransformer, batch_size: int) -> dict:
    from datasets import load_dataset
    logger.info("Loading FLORES-200 eng_Latn<->kin_Latn devtest...")
    ds = load_dataset("facebook/flores", "eng_Latn-kin_Latn", split="devtest")

    en_sents = [r["sentence_eng_Latn"] for r in ds]
    rw_sents = [r["sentence_kin_Latn"] for r in ds]
    logger.info(f"FLORES-200 devtest: {len(en_sents)} pairs")

    en_embs = model.encode(en_sents, batch_size=batch_size, show_progress_bar=True)
    rw_embs = model.encode(rw_sents, batch_size=batch_size, show_progress_bar=True)

    p1_en2rw = precision_at_1(en_embs, rw_embs)
    p1_rw2en = precision_at_1(rw_embs, en_embs)
    p1_avg   = (p1_en2rw + p1_rw2en) / 2

    return {
        "flores_p1_en2rw": round(p1_en2rw, 4),
        "flores_p1_rw2en": round(p1_rw2en, 4),
        "flores_p1_avg":   round(p1_avg, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--opus_dir", default=None,
                        help="Local OPUS-100 en-rw dataset directory")
    parser.add_argument("--eval_opus", action="store_true", default=True)
    parser.add_argument("--eval_flores", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model = SentenceTransformer(args.model)

    results = {"model": args.model}

    if args.eval_opus:
        results.update(eval_opus(model, args.opus_dir, args.batch_size))

    if args.eval_flores:
        results.update(eval_flores(model, args.batch_size))

    print(f"\n{'='*50}")
    print(f"Model: {results['model']}")
    if "opus_p1_avg" in results:
        print(f"OPUS-100 P@1:    {results['opus_p1_avg']:.4f}  "
              f"(en→rw {results['opus_p1_en2rw']:.4f}, rw→en {results['opus_p1_rw2en']:.4f})")
    if "flores_p1_avg" in results:
        print(f"FLORES-200 P@1:  {results['flores_p1_avg']:.4f}  "
              f"(en→rw {results['flores_p1_en2rw']:.4f}, rw→en {results['flores_p1_rw2en']:.4f})")
    print(f"{'='*50}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
