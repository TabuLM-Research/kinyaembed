# KinyaEmbed: Contrastive Sentence Embeddings for Kinyarwanda

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-TabuLM--Research/KinyaEmbed-yellow)](https://huggingface.co/TabuLM-Research/KinyaEmbed)

**KinyaEmbed** is the first dedicated sentence embedding model for Kinyarwanda, a Bantu language spoken by over 12 million people in Rwanda. It achieves Spearman ρ = **0.7298** on SemRel2024-rw — 20.9% above the best multilingual baseline (mE5-large) and 41.0% above OpenAI `text-embedding-3-large`.

> Paper: *KinyaEmbed: Contrastive Sentence Embeddings for Kinyarwanda via Multi-Stage Curriculum Training*  
> Anonymous Submission — AAAI 2027 AI for Social Impact Track

---

## Key Results

| Model | SemRel STS ↑ | Wiki-RW-STS ↑ | Silhouette ↑ |
|---|---|---|---|
| **KinyaEmbed (ours)** | **0.7298** | **0.6005** | **0.2146** |
| mE5-large | 0.6039 | 0.5337 | 0.0794 |
| AfriE5-instruct | 0.6037 | 0.5391 | 0.1104 |
| mE5-large-instruct | 0.5975 | 0.5531 | 0.1073 |
| BGE-M3 | 0.5523 | 0.4877 | 0.1086 |
| OpenAI text-emb-3-large | 0.5175 | 0.5319 | 0.0846 |
| LaBSE | 0.4535 | 0.2197 | 0.1882 |

---

## Quick Start

```python
from ensemble.build_ensemble import KinyaEmbedEnsemble

# Download checkpoints from HuggingFace first:
# huggingface-cli download TabuLM-Research/KinyaEmbed --local-dir ./kinyaembed_checkpoints

checkpoint_dirs = [
    "kinyaembed_checkpoints/sc30",
    "kinyaembed_checkpoints/sc35",
    "kinyaembed_checkpoints/sc40",
    "kinyaembed_checkpoints/v12",
    "kinyaembed_checkpoints/step22A",
    "kinyaembed_checkpoints/step23A",
    "kinyaembed_checkpoints/step23A",   # double-weighted
]

model = KinyaEmbedEnsemble(checkpoint_dirs, device="cpu")

sentences = [
    "Uburinzi bw'ubuzima mu Rwanda",   # Health protection in Rwanda
    "Health protection in Rwanda",
    "Politiki za leta",                 # Government policy
]
embeddings = model.encode(sentences)   # shape (3, 768), L2-normalized
print(embeddings @ embeddings.T)       # cosine similarity matrix
```

---

## Repository Structure

```
kinyaembed/
├── training/                  # Stage-by-stage training scripts
│   ├── train_stage1_gazette.py    # Stage 1: Gazette paraphrases → sc30/sc35/sc40
│   ├── train_stage2_mnli.py       # Stage 2: MNLI triplets → v12
│   ├── train_stage3_opus.py       # Stage 3: OPUS-100 cross-lingual → step22A
│   └── train_stage4_kinyacomet.py # Stage 4: KinyaCOMET fine-tuning → step23A
├── ensemble/
│   └── build_ensemble.py      # all5+23A×2 ensemble construction & inference
├── evaluation/
│   ├── eval_sts.py            # SemRel2024-rw Spearman ρ
│   ├── eval_wiki_rw_sts.py    # Wiki-RW-STS (our benchmark)
│   ├── eval_bitext.py         # OPUS-100 + FLORES-200 P@1
│   └── eval_downstream.py     # IR, clustering, classification
├── data/
│   ├── kinycomet_pairs.jsonl      # 2,936 filtered KinyaCOMET pairs (score ≥ 0.8)
│   ├── wiki_rw_sts_pairs.jsonl    # Wiki-RW-STS: 300 held-out evaluation pairs
│   ├── wiki_rw_corpus.jsonl       # 300 Wikipedia articles for downstream eval
│   └── mnli_kin_triplets.jsonl    # Machine-translated MNLI triplets (Kinyarwanda)
├── results/                   # Evaluation result JSONs
├── scripts/                   # Figure generation and misc scripts
└── paper/                     # LaTeX source and figures
    ├── kinyaembed_aaai27.tex
    ├── kinyaembed.bib
    └── figures/
```

---

## Training Pipeline

KinyaEmbed uses a 4-stage curriculum with MultipleNegativesRankingLoss (MNRL):

```
KinyaBERT-large (pretrained)
    │
    ▼ Stage 1 — Gazette paraphrases (monolingual)
    ├── sc30 ──┐
    ├── sc35 ──┤
    └── sc40 ──┤
               │
    ▼ Stage 2 — MNLI triplets (machine-translated)
    └── v12 ───┤
               │
    ▼ Stage 3 — OPUS-100 EN↔RW (cross-lingual)
    └── step22A┤
               │
    ▼ Stage 4 — KinyaCOMET (2,936 human pairs, score ≥ 0.8)
    └── step23A┤ (×2 weight)
               │
    ▼ Ensemble: average 7 embeddings → L2 normalize
    └── KinyaEmbed all5+23A×2  (STS = 0.7298)
```

### Running Training

```bash
# Install dependencies
pip install -r requirements.txt

# Stage 1: Gazette paraphrases
python training/train_stage1_gazette.py \
  --model jean-paul/KinyaBERTlarge \
  --pairs data/umuganda_paraphrase_pairs.jsonl \
  --output_dir checkpoints/stage1

# Stage 2: MNLI triplets
python training/train_stage2_mnli.py \
  --init_model checkpoints/stage1/sc35 \
  --triplets data/mnli_kin_triplets.jsonl \
  --output_dir checkpoints/stage2

# Stage 3: OPUS-100 cross-lingual
python training/train_stage3_opus.py \
  --init_model checkpoints/stage2/v12 \
  --output_dir checkpoints/stage3

# Stage 4: KinyaCOMET fine-tuning
python training/train_stage4_kinyacomet.py \
  --init_model checkpoints/stage3/step22A \
  --kinyacomet_pairs data/kinycomet_pairs.jsonl \
  --output_dir checkpoints/stage4

# Build ensemble
python ensemble/build_ensemble.py \
  --checkpoint_base checkpoints \
  --output_dir kinyaembed_ensemble \
  --verify
```

---

## Evaluation

```bash
# SemRel2024-rw STS
python evaluation/eval_sts.py \
  --model kinyaembed_ensemble \
  --semrel_dir data/semrel2024_rw

# Wiki-RW-STS (our benchmark — no training contamination)
python evaluation/eval_wiki_rw_sts.py \
  --model kinyaembed_ensemble \
  --pairs data/wiki_rw_sts_pairs.jsonl

# Bitext mining (OPUS-100 + FLORES-200)
python evaluation/eval_bitext.py \
  --model kinyaembed_ensemble \
  --eval_opus --eval_flores

# Downstream: IR, clustering, classification
python evaluation/eval_downstream.py \
  --model kinyaembed_ensemble \
  --corpus data/wiki_rw_corpus.jsonl
```

---

## Data

| File | Description | Size |
|------|-------------|------|
| `kinycomet_pairs.jsonl` | KinyaCOMET pairs filtered at quality ≥ 0.8 | 933K |
| `wiki_rw_sts_pairs.jsonl` | Wiki-RW-STS: 300 held-out pairs at 3 similarity levels | 112K |
| `wiki_rw_corpus.jsonl` | 300 Kinyarwanda Wikipedia articles (8 topics) | 493K |
| `mnli_kin_triplets.jsonl` | Machine-translated MultiNLI triplets in Kinyarwanda | 171K |

Large corpora (Gazette paraphrase pairs, OPUS-100) are available via download:
- Gazette pairs: contact authors
- OPUS-100 en-rw: `datasets.load_dataset("Helsinki-NLP/opus-100", "en-rw")`
- FLORES-200: `datasets.load_dataset("facebook/flores", "eng_Latn-kin_Latn")`

---

## Citation

```bibtex
@inproceedings{kinyaembed2027,
  title     = {{KinyaEmbed}: Contrastive Sentence Embeddings for {K}inyarwanda
               via Multi-Stage Curriculum Training},
  author    = {Anonymous},
  booktitle = {Proceedings of the 41st AAAI Conference on Artificial Intelligence
               (AI for Social Impact Track)},
  year      = {2027},
}
```

---

## License

Apache License 2.0. Training data licenses:
- Gazette of Rwanda: public domain (government publication)
- OPUS-100: CC BY 4.0
- KinyaCOMET: see [original repo](https://github.com/anzeyimana/KinyaCOMET)
- Wiki-RW-STS: CC BY-SA 4.0 (derived from Kinyarwanda Wikipedia)
