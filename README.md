# KinyaEmbed: Contrastive Sentence Embeddings for Kinyarwanda via Multi-Stage Curriculum Training

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20%26%20Data-yellow)](https://huggingface.co/TabuLM-Research/KinyaEmbed)

**Ireddi Rakshitha** (Software Engineer, Barclays) · **Devavarapu Yashwanth** (Software Engineer, Barclays) · **Pierre Ntakirutimana** (Research Associate, Carnegie Mellon University)

> **arXiv preprint · August 2026**

---

## Overview

**KinyaEmbed** is the first dedicated sentence embedding model for Kinyarwanda, trained via a four-stage curriculum on top of KinyaBERT-large using MultipleNegativesRankingLoss (MNRL).

| Stage | Data | Checkpoints |
|---|---|---|
| 1 — Gazette Paraphrases | Official Gazette of Rwanda | `sc30`, `sc35`, `sc40` |
| 2 — MNLI Triplets | 715 NLLB-translated NLI triplets | `v12` |
| 3 — OPUS-100 Alignment | English–Kinyarwanda pairs | `step22A` |
| 4 — KinyaCOMET | 2,936 human-annotated pairs (score ≥ 0.8) | `step23A` |

Final ensemble: **all5+23A×2** (7 checkpoints, `step23A` double-weighted).

---

## Results

| Model | SemRel2024-rw ρ | Wiki-RW-STS ρ | Clustering Sil. |
|---|---|---|---|
| LaBSE | 0.4535 | 0.2197 | 0.1882 |
| mE5-large | 0.6039 | 0.5337 | 0.0794 |
| AfriE5-instruct | 0.6037 | 0.5391 | 0.1104 |
| OpenAI text-emb-3-large | 0.5175 | 0.5319 | 0.0846 |
| **KinyaEmbed (ours)** | **0.7298** | **0.6005** | **0.2146** |

- **+20.9%** over mE5-large on SemRel2024-rw STS
- **+8.6%** over mE5-large-instruct on Wiki-RW-STS (contamination-free)
- Best document clustering across all 7 models

---

## Repository Contents

```
kinyaembed-arxiv/
├── main.tex                  ← arXiv-ready LaTeX source
├── kinyaembed_refs.bib       ← BibTeX references
├── kinyaembed_pipeline.pdf   ← Pipeline figure
├── tsne_comparison.pdf       ← t-SNE figure
└── README.md
```

## Compile

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Model, Data & Benchmark

All checkpoints, 2,936 KinyaCOMET filtered pairs, and Wiki-RW-STS benchmark:

**[huggingface.co/TabuLM-Research/KinyaEmbed](https://huggingface.co/TabuLM-Research/KinyaEmbed)**

---

## Citation

```bibtex
@article{ireddi2026kinyaembed,
  title   = {{KinyaEmbed}: Contrastive Sentence Embeddings for {Kinyarwanda}
             via Multi-Stage Curriculum Training},
  author  = {Ireddi, Rakshitha and Devavarapu, Yashwanth and Ntakirutimana, Pierre},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Code: MIT · KinyaCOMET filtered pairs: CC-BY 4.0 · Wiki-RW-STS: CC-BY-SA 4.0
