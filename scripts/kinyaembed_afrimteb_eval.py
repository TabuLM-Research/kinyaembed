"""
KinyaEmbed AfriMTEB-style evaluation — standalone (no mteb library needed).

Tasks:
  1. STS              — SemRel2024-rw Spearman cosine (test, n=222)
  2. Pair Classification — SemRel2024-rw binary AUC (threshold 0.5)
  3. Bitext Mining    — OPUS-100 en-rw P@1 (test, n=2000)

Models:
  A. kinyaembed_sc35        (best single)
  B. kinyaembed_ensemble    (all5 equal weight)
  C. AfriE5-Large-instruct  (baseline)
"""

import json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
import torch

DATA_DIR  = Path('/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/kinyaembed')
CACHE_DIR = DATA_DIR / 'eval_cache'
CACHE_DIR.mkdir(exist_ok=True)

MODELS = {
    'kinyaembed_sc35':   str(DATA_DIR / 'kinyaembed_scale35/best'),
    'afrie5_instruct':   'McGill-NLP/AfriE5-Large-instruct',
}
ENSEMBLE_PATHS = [
    str(DATA_DIR / 'kinyaembed_scale30/best'),
    str(DATA_DIR / 'kinyaembed_scale35/best'),
    str(DATA_DIR / 'kinyaembed_scale40/best'),
    str(DATA_DIR / 'kinyaembed_mnli_v12/best'),
    str(DATA_DIR / 'kinyaembed_simcse_v9/best'),
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# ─── helpers ──────────────────────────────────────────────────────────────────

def encode(model, texts, bs=64):
    return model.encode(texts, batch_size=bs, normalize_embeddings=True,
                        show_progress_bar=False)

def ensemble_encode(paths, texts, bs=32):
    embs = []
    for p in paths:
        m = SentenceTransformer(p, device=device)
        embs.append(encode(m, texts, bs))
        del m; torch.cuda.empty_cache()
    avg = np.mean(embs, axis=0)
    norms = np.linalg.norm(avg, axis=1, keepdims=True)
    return avg / np.maximum(norms, 1e-9)

def load_semrel():
    ds = load_from_disk(str(DATA_DIR / 'semrel2024_rw'))
    split = ds['test']
    cols = split.column_names
    s1c = 'sentence1' if 'sentence1' in cols else 'text1'
    s2c = 'sentence2' if 'sentence2' in cols else 'text2'
    sc  = 'label'     if 'label'     in cols else 'score'
    return list(split[s1c]), list(split[s2c]), [float(x) for x in split[sc]]

# ─── Task 1: STS ──────────────────────────────────────────────────────────────

def eval_sts(emb1, emb2, gold):
    cos = (np.array(emb1) * np.array(emb2)).sum(1)
    sp, _ = spearmanr(cos, gold)
    return round(float(sp), 4)

# ─── Task 2: Pair Classification (binary AUC on SemRel) ───────────────────────

def eval_pair_cls(emb1, emb2, gold):
    cos  = (np.array(emb1) * np.array(emb2)).sum(1)
    labels = (np.array(gold) >= 0.5).astype(int)   # similar vs. not
    auc = roc_auc_score(labels, cos)
    ap  = average_precision_score(labels, cos)
    return {'auc': round(float(auc), 4), 'ap': round(float(ap), 4)}

# ─── Task 3: Bitext Mining (OPUS-100 en-rw, P@1) ─────────────────────────────

def load_opus():
    ds = load_dataset('Helsinki-NLP/opus-100', 'en-rw',
                      split='test', cache_dir=str(CACHE_DIR))
    # Filter out empty/garbage lines
    en_texts, rw_texts = [], []
    for item in ds:
        en = item['translation']['en'].strip()
        rw = item['translation']['rw'].strip()
        if en and rw and len(en) > 3 and len(rw) > 3:
            en_texts.append(en)
            rw_texts.append(rw)
    print(f"  OPUS-100 en-rw: {len(en_texts)} valid pairs (of 2000 total)")
    return en_texts, rw_texts

def eval_bitext(eng_emb, kin_emb):
    n = len(eng_emb)
    sim = eng_emb @ kin_emb.T           # (n, n) cosine sim matrix
    nn_e2k = sim.argmax(axis=1)
    nn_k2e = sim.argmax(axis=0)
    p1_e2k = float(np.mean(nn_e2k == np.arange(n)))
    p1_k2e = float(np.mean(nn_k2e == np.arange(n)))
    avg = (p1_e2k + p1_k2e) / 2
    return {'p1_en2rw': round(p1_e2k, 4),
            'p1_rw2en': round(p1_k2e, 4),
            'avg_p1':   round(avg, 4)}

# ─── Load shared data once ────────────────────────────────────────────────────

print("Loading SemRel2024-rw test set...")
s1, s2, gold = load_semrel()
print(f"  {len(s1)} pairs loaded\n")

print("Loading OPUS-100 en-rw test set...")
en_texts, rw_texts = load_opus()
print()

results = {}

# ─── Single models ────────────────────────────────────────────────────────────

for model_name, model_path in MODELS.items():
    print(f"{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    m = SentenceTransformer(model_path, device=device)

    # STS + pair cls on SemRel
    e1 = encode(m, s1)
    e2 = encode(m, s2)
    sts_r    = eval_sts(e1, e2, gold)
    paircls  = eval_pair_cls(e1, e2, gold)
    print(f"  STS        Spearman = {sts_r:.4f}")
    print(f"  PairCls    AUC={paircls['auc']:.4f}  AP={paircls['ap']:.4f}")

    # Bitext mining on OPUS-100
    en_emb = encode(m, en_texts, bs=64)
    rw_emb = encode(m, rw_texts, bs=64)
    bitext = eval_bitext(en_emb, rw_emb)
    print(f"  Bitext P@1 en->rw={bitext['p1_en2rw']:.4f}  rw->en={bitext['p1_rw2en']:.4f}  avg={bitext['avg_p1']:.4f}")

    del m; torch.cuda.empty_cache()
    results[model_name] = {'sts': sts_r, 'pair_cls': paircls, 'bitext': bitext}

# ─── Ensemble ─────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"MODEL: kinyaembed_ensemble_all5")
print(f"{'='*60}")

e1 = ensemble_encode(ENSEMBLE_PATHS, s1)
e2 = ensemble_encode(ENSEMBLE_PATHS, s2)
sts_r   = eval_sts(e1, e2, gold)
paircls = eval_pair_cls(e1, e2, gold)
print(f"  STS        Spearman = {sts_r:.4f}")
print(f"  PairCls    AUC={paircls['auc']:.4f}  AP={paircls['ap']:.4f}")

en_emb = ensemble_encode(ENSEMBLE_PATHS, en_texts, bs=32)
rw_emb = ensemble_encode(ENSEMBLE_PATHS, rw_texts, bs=32)
bitext = eval_bitext(en_emb, rw_emb)
print(f"  Bitext P@1 en->rw={bitext['p1_en2rw']:.4f}  rw->en={bitext['p1_rw2en']:.4f}  avg={bitext['avg_p1']:.4f}")

results['kinyaembed_ensemble'] = {'sts': sts_r, 'pair_cls': paircls, 'bitext': bitext}

# ─── Save & print summary ─────────────────────────────────────────────────────

out = DATA_DIR / 'kinyaembed_afrimteb_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")

print("\n" + "="*75)
print("FINAL SUMMARY")
print("="*75)
print(f"{'Model':<28} {'STS':>7} {'PairAUC':>9} {'PairAP':>8} {'Bitext':>8}")
print("-"*75)
for mn, r in results.items():
    sts = f"{r['sts']:.4f}"
    auc = f"{r['pair_cls']['auc']:.4f}"
    ap  = f"{r['pair_cls']['ap']:.4f}"
    bt  = f"{r['bitext']['avg_p1']:.4f}"
    print(f"  {mn:<26} {sts:>7} {auc:>9} {ap:>8} {bt:>8}")
