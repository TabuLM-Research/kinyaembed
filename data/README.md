# KinyaEmbed Data Files

## Included in this repository

### `kinycomet_pairs.jsonl`
2,936 high-quality Kinyarwanda–English sentence pairs filtered from KinyaCOMET
(quality score ≥ 0.8). Used in Stage 4 training.

Format: `{"en": "...", "rw": "...", "score": 0.85}`

### `wiki_rw_sts_pairs.jsonl`
**Wiki-RW-STS**: 300 Kinyarwanda Wikipedia sentence pairs at three similarity levels
(high ≈ 0.85, medium ≈ 0.50, low ≈ 0.10). Contamination-free held-out benchmark.

Format: `{"sent1": "...", "sent2": "...", "score": 0.85, "level": "high"}`

### `wiki_rw_corpus.jsonl`
300 Kinyarwanda Wikipedia articles across 8 topic categories, used for downstream
evaluation (IR, clustering, classification).

Format: `{"title": "...", "body": "...", "category": "amateka"}`

Categories:
- `amateka` (history), `ikoranabuhanga` (science/technology), `isi-akarere` (geography),
  `imikino` (sports), `politiki` (politics), `idini` (religion),
  `ubuhinzi` (agriculture), `ubuzima` (health)

### `mnli_kin_triplets.jsonl`
Machine-translated MultiNLI triplets in Kinyarwanda for Stage 2 training.

Format: `{"anchor": "...", "positive": "...", "negative": "..."}`

## Not included (download separately)

| Resource | How to get |
|----------|-----------|
| Gazette paraphrase pairs | Contact authors — `umuganda_paraphrase_pairs.jsonl` |
| OPUS-100 en-rw | `load_dataset("Helsinki-NLP/opus-100", "en-rw")` |
| FLORES-200 en-rw | `load_dataset("facebook/flores", "eng_Latn-kin_Latn")` |
| SemRel2024-rw | `load_dataset("SemRel/SemRel2024", "rw")` |
| KinyaBERT-large | `jean-paul/KinyaBERTlarge` on HuggingFace |
