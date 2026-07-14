"""
Generate 4-panel t-SNE comparison figure for KinyaEmbed paper.
Colors: K-means cluster assignments (what Silhouette score measures).
Panels: KinyaEmbed | LaBSE | mE5-large-instruct | BGE-M3
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

HERE = Path(__file__).parent

CLUSTER_COLORS = [
    '#E63946', '#457B9D', '#2A9D8F', '#E9C46A',
    '#8B5CF6', '#F4A261', '#6BCB77', '#4D96FF',
    '#94A3B8', '#F72585',
]

PANELS = [
    ('tsne_kinyaembed.json',        'KinyaEmbed (ours)', 0.2146, True),
    ('tsne_labse.json',              'LaBSE',             0.1882, False),
    ('tsne_me5_large_instruct.json', 'mE5-large-instruct',0.1073, False),
    ('tsne_bge_m3.json',             'BGE-M3',            0.1086, False),
]

def load_tsne(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    coords  = np.array(data['coords'])           # (N, 2)
    clusters = data.get('clusters',              # integer cluster id
                        list(range(len(coords))))
    return coords, np.array(clusters)

def plot_panel(ax, coords, clusters, title, silhouette, highlight):
    xs, ys = coords[:, 0], coords[:, 1]
    for cid in sorted(set(clusters)):
        mask = clusters == cid
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax.scatter(xs[mask], ys[mask], c=color, s=16, alpha=0.78,
                   linewidths=0, zorder=2)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('#F8F9FA')

    weight = 'bold' if highlight else 'normal'
    color  = '#1D3557' if highlight else '#374151'
    ax.set_title(f'{title}\nSilhouette: {silhouette:.4f}',
                 fontsize=9, fontweight=weight, color=color, pad=6)

    if highlight:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_color('#457B9D')

def main():
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), facecolor='white')
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.84, bottom=0.10)

    for ax, (fname, title, sil, hl) in zip(axes, PANELS):
        path = HERE / fname
        if not path.exists():
            ax.text(0.5, 0.5, f'Missing:\n{fname}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7, color='red')
            ax.set_xticks([]); ax.set_yticks([])
            continue
        coords, clusters = load_tsne(path)
        plot_panel(ax, coords, clusters, title, sil, hl)

    fig.text(0.5, 0.01,
             '300 Kinyarwanda Wikipedia articles · K-means clusters (K=10) · t-SNE projection  '
             '· Higher Silhouette = better-separated clusters',
             ha='center', fontsize=7, color='#64748B')

    for ext in ('png', 'pdf'):
        out = HERE / f'tsne_comparison.{ext}'
        fig.savefig(out, dpi=300 if ext == 'png' else 150,
                    bbox_inches='tight', facecolor='white')
        print(f'Saved: {out}')

if __name__ == '__main__':
    main()
