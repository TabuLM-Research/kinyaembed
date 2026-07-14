"""
Generate KinyaEmbed pipeline architecture figure using matplotlib.
Output: kinyaembed_pipeline.pdf and .png  (copy to Figures/ for LaTeX)

Run: python generate_pipeline_figure.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    'bg':        '#F8FAFC',
    'data_face': '#F0FDF4',
    'data_edge': '#4ADE80',
    'data_txt':  '#15803D',
    'ck_face':   '#FFFBEB',
    'ck_edge':   '#FCD34D',
    'ck_txt':    '#92400E',
    'ck23_face': '#FFF7ED',
    'ck23_edge': '#FB923C',
    'ck23_txt':  '#9A3412',
    'ens_face':  '#F5F3FF',
    'ens_edge':  '#A78BFA',
    'ens_txt':   '#5B21B6',
    'out_face':  '#E0F2FE',
    'out_edge':  '#38BDF8',
    'out_txt':   '#075985',
    'bb_face':   '#EFF6FF',
    'bb_edge':   '#93C5FD',
    'bb_txt':    '#1D4ED8',
    'arrow':     '#94A3B8',
    'bus':       '#A78BFA',
    'muted':     '#64748B',
}

def fbox(ax, x, y, w, h, fc, ec, lw=1.2, rad=0.015, ls='-'):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={rad}',
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=ls,
        transform=ax.transData, clip_on=False
    )
    ax.add_patch(box)
    return box

def txt(ax, x, y, s, size=7, color='#1E293B', weight='normal',
        ha='center', va='center', family='sans-serif'):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, fontfamily=family)

def arrow(ax, x0, y0, x1, y1, color='#94A3B8', lw=1.0, hw=0.006, hl=0.012):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f'->', color=color,
                                lw=lw, mutation_scale=10))


def main():
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis('off')

    # ── Backbone banner ────────────────────────────────────────────────────────
    fbox(ax, 0.01, 0.88, 0.60, 0.10,
         fc=C['bb_face'], ec=C['bb_edge'], lw=1.5)
    txt(ax, 0.31, 0.95, 'KinyaBERT-large', size=9, weight='bold', color=C['bb_txt'])
    txt(ax, 0.31, 0.91, 'Pretrained encoder · 12 layers · 768-dim · Mean Pooling · L2 Normalize',
        size=6.5, color=C['bb_txt'])

    # Down arrow from backbone
    ax.annotate('', xy=(0.31, 0.87), xytext=(0.31, 0.88),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.0))

    # ── Dashed training group ──────────────────────────────────────────────────
    fbox(ax, 0.01, 0.01, 0.60, 0.86,
         fc='none', ec=C['muted'], lw=1.0, rad=0.01, ls=(0, (6, 3)))
    txt(ax, 0.01, 0.875, 'SEQUENTIAL TRAINING  ·  MultipleNegativesRankingLoss (MNRL)',
        size=6, color=C['muted'], ha='left')

    # ── Stage helper parameters ────────────────────────────────────────────────
    # 4 stages stacked top to bottom within [0.04, 0.59] in x, [0.02..0.85] in y
    stage_defs = [
        # (label, data_lines, ck_name, ck_lines, is_kinyacomet, y_top)
        ('Stage 1', ['Gazette Paraphrases', 'Kinyarwanda Legal Corpus', 'monolingual pairs'],
         ['sc30', 'sc35', 'sc40'], ['3 checkpoints (scale 30/35/40)'], False, 0.84),
        ('Stage 2', ['MNLI Triplets', 'Machine-translated rw', 'anchor · pos · neg'],
         ['v12'], ['MNLI fine-tuned'], False, 0.64),
        ('Stage 3', ['OPUS-100 en↔rw', 'Cross-lingual parallel', 'English ↔ Kinyarwanda'],
         ['step22A'], ['OPUS cross-lingual checkpoint'], False, 0.44),
        ('Stage 4', ['KinyaCOMET', '2,936 human pairs (score ≥ 0.8)', 'Novel en–rw annotation'],
         ['step23A'], ['KinyaCOMET checkpoint  ·  ×2 weight'], True, 0.24),
    ]

    stage_h = 0.175   # height of each stage row
    data_x, data_w   = 0.03, 0.22
    mnrl_x            = 0.27
    ck_x, ck_w       = 0.30, 0.27

    # Checkpoint right-edge x (where bus spurs connect)
    bus_x = 0.615

    ck_centers_y = []  # collect y-centers of checkpoint boxes for bus

    for (stage_lbl, data_lines, ck_names, ck_lines, is_kc, y_top) in stage_defs:
        y_bot = y_top - stage_h
        y_mid = (y_top + y_bot) / 2

        # Stage pill
        fbox(ax, data_x, y_top - 0.025, 0.055, 0.02,
             fc=C['ck_face'] if not is_kc else '#FFF7ED',
             ec=C['ck_edge'] if not is_kc else '#FB923C', lw=1.0)
        txt(ax, data_x + 0.027, y_top - 0.015, stage_lbl,
            size=6, weight='bold',
            color=C['ck_txt'] if not is_kc else C['ck23_txt'])

        # Data box
        fbox(ax, data_x, y_bot + 0.01, data_w, stage_h - 0.04,
             fc=C['data_face'] if not is_kc else '#FEFCE8',
             ec=C['data_edge'] if not is_kc else '#FDE047', lw=1.4)
        txt(ax, data_x + data_w/2, y_mid + 0.04, data_lines[0],
            size=7.5, weight='bold', color=C['data_txt'])
        for i, dl in enumerate(data_lines[1:]):
            txt(ax, data_x + data_w/2, y_mid + 0.01 - i*0.028, dl,
                size=6.5, color=C['data_txt'])

        # MNRL label + arrow
        txt(ax, mnrl_x, y_mid + 0.025, 'MNRL', size=6, color=C['muted'])
        ax.annotate('', xy=(ck_x + 0.005, y_mid), xytext=(mnrl_x - 0.01, y_mid),
                    arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.0,
                                    mutation_scale=8))

        # Checkpoint box(es)
        ck_face = C['ck23_face'] if is_kc else C['ck_face']
        ck_ec   = C['ck23_edge'] if is_kc else C['ck_edge']
        ck_txt_c= C['ck23_txt']  if is_kc else C['ck_txt']
        lw      = 1.8 if is_kc else 1.4

        if len(ck_names) == 3:
            # 3 small boxes side by side
            sub_w = (ck_w - 0.02) / 3
            for i, name in enumerate(ck_names):
                sx = ck_x + i * (sub_w + 0.01)
                fbox(ax, sx, y_bot + 0.02, sub_w, stage_h - 0.05,
                     fc=ck_face, ec=ck_ec, lw=lw)
                txt(ax, sx + sub_w/2, y_mid + 0.01, name,
                    size=7.5, weight='bold', color=ck_txt_c, family='monospace')
                txt(ax, sx + sub_w/2, y_mid - 0.025, f'scale {name[2:]}',
                    size=5.5, color=ck_txt_c)
            # spur from right edge of sc40 to bus
            spur_x = ck_x + 3*(sub_w + 0.01) - 0.01
            ax.plot([spur_x, bus_x], [y_mid, y_mid],
                    color=C['bus'], lw=1.0, zorder=3)
        else:
            # Single wide box
            fbox(ax, ck_x, y_bot + 0.02, ck_w, stage_h - 0.05,
                 fc=ck_face, ec=ck_ec, lw=lw)
            txt(ax, ck_x + ck_w/2, y_mid + 0.025, ck_names[0],
                size=8.5, weight='bold', color=ck_txt_c, family='monospace')
            txt(ax, ck_x + ck_w/2, y_mid - 0.018, ck_lines[0],
                size=6, color=ck_txt_c)
            if is_kc:
                fbox(ax, ck_x + ck_w/2 - 0.055, y_mid - 0.06, 0.11, 0.022,
                     fc='#FED7AA', ec='#FB923C', lw=1.0)
                txt(ax, ck_x + ck_w/2, y_mid - 0.049, '× 2 weight in ensemble',
                    size=5.5, color=C['ck23_txt'])
            # spur from right edge to bus
            ax.plot([ck_x + ck_w, bus_x], [y_mid, y_mid],
                    color=C['bus'] if not is_kc else '#FB923C',
                    lw=1.0 if not is_kc else 1.5, zorder=3)

        ck_centers_y.append(y_mid)

    # ── Vertical bus ──────────────────────────────────────────────────────────
    y_top_bus = max(ck_centers_y)
    y_bot_bus = min(ck_centers_y)
    ax.plot([bus_x, bus_x], [y_bot_bus, y_top_bus],
            color=C['bus'], lw=1.4, zorder=3)

    # Arrow from bus midpoint to ensemble
    bus_mid_y = (y_top_bus + y_bot_bus) / 2
    ens_x = 0.64
    ax.annotate('', xy=(ens_x, bus_mid_y), xytext=(bus_x + 0.003, bus_mid_y),
                arrowprops=dict(arrowstyle='->', color=C['bus'], lw=1.3,
                                mutation_scale=9))
    txt(ax, (bus_x + ens_x)/2, bus_mid_y + 0.025, 'avg',
        size=6, color=C['ens_txt'])

    # ── Ensemble box ──────────────────────────────────────────────────────────
    ens_y0, ens_w, ens_h = 0.12, 0.22, 0.77
    fbox(ax, ens_x, ens_y0, ens_w, ens_h, fc=C['ens_face'], ec=C['ens_edge'], lw=1.8)
    txt(ax, ens_x + ens_w/2, ens_y0 + ens_h - 0.05, 'Ensemble',
        size=9, weight='bold', color=C['ens_txt'])
    txt(ax, ens_x + ens_w/2, ens_y0 + ens_h - 0.09, 'all5 + step23A×2',
        size=6.5, color=C['ens_txt'], family='monospace')

    # Divider
    ax.plot([ens_x + 0.01, ens_x + ens_w - 0.01], [ens_y0 + ens_h - 0.11]*2,
            color=C['ens_edge'], lw=0.8, alpha=0.6)

    # 6 chips inside ensemble
    chips = ['sc30', 'sc35', 'sc40', 'v12', 'step22A', 'step23A×2']
    chip_colors = [(C['ck_face'], C['ck_edge'], C['ck_txt'])] * 5 + \
                  [(C['ck23_face'], C['ck23_edge'], C['ck23_txt'])]
    cw, ch = 0.09, 0.062
    cols = 2
    for i, (chip, (cf, ce, ct)) in enumerate(zip(chips, chip_colors)):
        row, col = divmod(i, cols)
        cx = ens_x + 0.015 + col * (cw + 0.015)
        cy = ens_y0 + ens_h - 0.19 - row * (ch + 0.015)
        fbox(ax, cx, cy, cw, ch, fc=cf, ec=ce, lw=1.0)
        txt(ax, cx + cw/2, cy + ch/2, chip, size=6.5, weight='bold',
            color=ct, family='monospace')

    # Operation labels
    ax.plot([ens_x + 0.01, ens_x + ens_w - 0.01],
            [ens_y0 + 0.23]*2, color=C['ens_edge'], lw=0.8, alpha=0.6)
    txt(ax, ens_x + ens_w/2, ens_y0 + 0.19, 'Sum 7 embeddings', size=6.5, color=C['ens_sub'] if 'ens_sub' in C else C['ens_txt'])
    txt(ax, ens_x + ens_w/2, ens_y0 + 0.155, 'Divide by 7', size=6.5, color=C['ens_txt'])
    txt(ax, ens_x + ens_w/2, ens_y0 + 0.12, 'L2 Normalize', size=6.5, color=C['ens_txt'])

    # ── Output box ────────────────────────────────────────────────────────────
    out_x, out_y, out_w, out_h = ens_x, 0.01, ens_w, 0.10
    fbox(ax, out_x, out_y, out_w, out_h, fc=C['out_face'], ec=C['out_edge'], lw=1.8)
    txt(ax, out_x + out_w/2, out_y + out_h/2 + 0.015, '768-dim Sentence Embedding',
        size=7.5, weight='bold', color=C['out_txt'])
    txt(ax, out_x + out_w/2, out_y + out_h/2 - 0.02, 'Cosine-ready · Kinyarwanda + cross-lingual',
        size=6, color=C['out_txt'])

    # Arrow from ensemble to output
    ax.annotate('', xy=(out_x + out_w/2, out_y + out_h),
                xytext=(ens_x + ens_w/2, ens_y0),
                arrowprops=dict(arrowstyle='->', color=C['out_edge'], lw=1.3,
                                mutation_scale=9))

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.tight_layout(pad=0.2)

    png_path = HERE / 'kinyaembed_pipeline.png'
    pdf_path = HERE / 'kinyaembed_pipeline.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == '__main__':
    main()
