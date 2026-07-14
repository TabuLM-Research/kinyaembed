"""
1. Compute SemRel STS for mE5-large-instruct and BGE-M3.
2. Generate t-SNE comparison figure (KinyaEmbed vs LaBSE, side-by-side).
3. Generate pipeline architecture figure.
Save all output to kinyaembed/ directory.
"""
import json, sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

BASE = Path('/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/kinyaembed')
HF_CACHE = str(BASE / 'eval_cache')

# ── Load SemRel test set ──────────────────────────────────────────────────────
ds = load_from_disk(str(BASE / 'semrel2024_rw'))['test']
cols = ds.column_names
s1c = 'sentence1' if 'sentence1' in cols else 'text1'
s2c = 'sentence2' if 'sentence2' in cols else 'text2'
sc  = 'label'     if 'label'     in cols else 'score'
s1 = list(ds[s1c]); s2 = list(ds[s2c]); gold = [float(x) for x in ds[sc]]
print(f'SemRel: {len(s1)} pairs')

missing_models = {
    'mE5-large-instruct': 'intfloat/multilingual-e5-large-instruct',
    'BGE-M3':             'BAAI/bge-m3',
}
INST = 'Instruct: Given a pair of sentences, compute their semantic relatedness.\nQuery: '
new_sts = {}
for name, hf_id in missing_models.items():
    print(f'Evaluating {name}...')
    m = SentenceTransformer(hf_id, device='cpu', cache_folder=HF_CACHE)
    if 'instruct' in name.lower():
        e1 = m.encode([INST+t for t in s1], normalize_embeddings=True, show_progress_bar=False)
        e2 = m.encode([INST+t for t in s2], normalize_embeddings=True, show_progress_bar=False)
    else:
        e1 = m.encode(s1, normalize_embeddings=True, show_progress_bar=False)
        e2 = m.encode(s2, normalize_embeddings=True, show_progress_bar=False)
    cos = (e1 * e2).sum(1)
    rho, _ = spearmanr(cos, gold)
    new_sts[name] = round(float(rho), 4)
    print(f'  {name}: Spearman={rho:.4f}')
    del m

# Also compute FLORES for the same models
from datasets import load_dataset
fl = load_dataset('mteb/FloresBitextMining', split='devtest', cache_dir=HF_CACHE)
en_fl  = [s.strip() for s in fl['eng_Latn'] if s and s.strip()]
kin_fl = [s.strip() for s in fl['kin_Latn'] if s and s.strip()]
n = min(len(en_fl), len(kin_fl)); en_fl, kin_fl = en_fl[:n], kin_fl[:n]
print(f'FLORES: {n} pairs')

def p1(ea, eb):
    n = len(ea); sim = ea @ eb.T
    return float((np.mean(sim.argmax(1)==np.arange(n)) +
                  np.mean(sim.argmax(0)==np.arange(n))) / 2)

new_flores = {}
for name, hf_id in missing_models.items():
    print(f'FLORES eval {name}...')
    m = SentenceTransformer(hf_id, device='cpu', cache_folder=HF_CACHE)
    if 'instruct' in name.lower():
        ee = m.encode([INST+t for t in en_fl], normalize_embeddings=True, show_progress_bar=False)
        ek = m.encode([INST+t for t in kin_fl], normalize_embeddings=True, show_progress_bar=False)
    else:
        ee = m.encode(en_fl, normalize_embeddings=True, show_progress_bar=False)
        ek = m.encode(kin_fl, normalize_embeddings=True, show_progress_bar=False)
    new_flores[name] = round(p1(ee, ek), 4)
    print(f'  {name}: FLORES P@1={new_flores[name]:.4f}')
    del m

# Save combined score file
out = {'new_sts_semrel': new_sts, 'new_flores': new_flores}
with open(BASE / 'missing_scores.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved missing_scores.json')
print(json.dumps(out, indent=2))

# ── Generate t-SNE comparison figure ─────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TOPIC_COLORS = {
    'history':   '#E63946', 'science':   '#457B9D', 'geography': '#2A9D8F',
    'sports':    '#E9C46A', 'politics':  '#8B5CF6', 'religion':  '#F4A261',
    'arts':      '#6BCB77', 'economics': '#4D96FF', 'other':     '#94A3B8',
}

def load_tsne(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    coords = np.array(d['coords'])
    labels = d.get('labels', ['other'] * len(coords))
    return coords, labels

ke_c, ke_l = load_tsne(BASE / 'downstream_plots/tsne_kinyaembed.json')
lb_c, lb_l = load_tsne(BASE / 'downstream_plots/tsne_labse.json')

# -- pick comparison models for a 2×2 grid --
me5_c,  me5_l  = load_tsne(BASE / 'downstream_plots/tsne_me5_large_instruct.json')
bge_c,  bge_l  = load_tsne(BASE / 'downstream_plots/tsne_bge_m3.json')

panels = [
    (ke_c,  ke_l,  'KinyaEmbed (ours)\nSilhouette: 0.2146'),
    (lb_c,  lb_l,  'LaBSE\nSilhouette: 0.1882'),
    (me5_c, me5_l, 'mE5-large-instruct\nSilhouette: 0.1073'),
    (bge_c, bge_l, 'BGE-M3\nSilhouette: 0.1086'),
]

fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), facecolor='white')
fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.84, bottom=0.18)

all_labels = []
for ax, (coords, labels, title) in zip(axes, panels):
    xs, ys = coords[:,0], coords[:,1]
    for lbl in dict.fromkeys(labels):
        mask = np.array([l == lbl for l in labels])
        ax.scatter(xs[mask], ys[mask], c=TOPIC_COLORS.get(lbl,'#94A3B8'),
                   s=18, alpha=0.85, linewidths=0, zorder=2)
        if lbl not in all_labels: all_labels.append(lbl)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[['top','right','bottom','left']].set_visible(False)
    ax.set_facecolor('#F8F9FA')
    ax.set_title(title, fontsize=8, fontweight='bold', pad=5, color='#1E293B')
    if ax == axes[0]:
        for spine in ax.spines.values():
            pass
        # Bold border for our model
        for side in ['top','right','bottom','left']:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color('#7C3AED')
            ax.spines[side].set_linewidth(1.5)

patches = [mpatches.Patch(color=TOPIC_COLORS.get(l,'#94A3B8'), label=l.capitalize())
           for l in all_labels if l in TOPIC_COLORS]
fig.legend(handles=patches, loc='lower center', ncol=len(patches),
           fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0),
           handlelength=0.9, handleheight=0.8, columnspacing=0.8)

fig.text(0.5, 0.95, 'Topic Clusters: 300 Kinyarwanda Wikipedia Articles (t-SNE Projections)',
         ha='center', fontsize=9, fontweight='bold', color='#1E293B')

for fmt in ['png', 'pdf']:
    out_path = BASE / f'figures/tsne_comparison.{fmt}'
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out_path}')

print('DONE')
