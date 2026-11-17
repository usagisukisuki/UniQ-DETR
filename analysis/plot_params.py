"""Parameter count of the original RT-DETRv2 R50 (closed set, 20 known classes, configs/e0_closed.yml) vs. UniQ-DETR
(configs/uniq_detr.yml), split into backbone / encoder / decoder, with the number of parameters UniQ-DETR adds.

  PYTHONPATH=analysis uv run python analysis/plot_params.py
"""
import collections
import os
import re
import statistics
import sys

sys.path.insert(0, os.getcwd())
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import owod.register  # noqa: F401,E402
from plot_tasks import BG, COLORS, GRID, INK, INK2, MUTED  # noqa: E402
from src.core import YAMLConfig  # noqa: E402

PARTS = ['backbone', 'encoder', 'decoder']
PART_LABEL = {'backbone': 'Backbone', 'encoder': 'Hybrid encoder', 'decoder': 'Decoder + heads'}
PART_COLOR = {'backbone': '#86b6ef', 'encoder': '#3987e5', 'decoder': '#1c5cab'}   # sequential blue steps
ADDED = [('dec_unk_head', 'Unknown head'), ('dec_primary_head', 'Primary head'), ('dec_quality_head', 'Quality head'),
         ('dec_nc_head', 'NC head')]


def load_fps():
    """median fp32 FPS over the 3 rounds of scripts/run_bench_fps_all.sh (same runs as plot_acc_fps.py)"""
    runs, k = collections.defaultdict(list), None
    for r in (1, 2, 3):
        for line in open(f'outputs/logs/fps_all_settings_r{r}.log'):
            if line.startswith('=='):
                k = line.split()[1]
            m = re.search(r'amp=False: .* ([\d.]+) FPS', line)
            if m:
                runs[k].append(float(m.group(1)))
    return {k: statistics.median(v) for k, v in runs.items()}


def counts(cfg):
    return {k: v.numel() for k, v in YAMLConfig(cfg).model.named_parameters()}


def main():
    base, ours = counts('configs/e0_closed.yml'), counts('configs/uniq_detr.yml')
    models = [('RT-DETRv2 (R50)', base), ('UniQ-DETR (R50)', ours)]
    added = collections.OrderedDict()
    for key, label in ADDED:
        added[label] = sum(v for k, v in ours.items() if f'.{key}.' in k)
    added['Unknown class column\n(class heads + denoising embed)'] = (
        sum(v for k, v in ours.items() if k in base) - sum(base.values()))
    total_add = sum(ours.values()) - sum(base.values())
    assert total_add == sum(added.values()), (total_add, added)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 3.6), facecolor=BG, sharey=True,
                                  gridspec_kw={'width_ratios': [3, 2], 'wspace': 0.08})
    for y, (name, p) in enumerate(models):
        left = 0
        for part in PARTS:
            n = sum(v for k, v in p.items() if k.split('.')[0] == part) / 1e6
            ax.barh(y, n, left=left, height=0.72, color=PART_COLOR[part], edgecolor=BG, linewidth=2,
                    label=PART_LABEL[part] if y == 0 else None)
            ax.text(left + n / 2, y, f'{n:.2f}M', ha='center', va='center', fontsize=10.5, color='#ffffff',
                    fontweight='bold')
            left += n
        ax.text(left + 0.6, y, f'{sum(p.values()) / 1e6:.3f}M', ha='left', va='center', fontsize=13, color=INK,
                fontweight='bold')
    ax.annotate(f'+{total_add:,}, +{total_add / sum(base.values()) * 100:.3f}%', (left + 0.6, 1),
                textcoords='offset points', xytext=(0, -20), ha='left', va='center', fontsize=10.5, color=COLORS[0],
                fontweight='bold')
    ax.set_yticks([0, 1], [m[0] for m in models], fontsize=11.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 58)
    ax.set_xlabel('Parameters [M]', fontsize=10.5, color=INK2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=False, fontsize=10, labelcolor=INK2)

    fps = load_fps()
    fv = [fps['rtdetrv2'], fps['uniq_detr']]
    ax2.barh([0, 1], fv, height=0.72, color=['#b4b3ad', COLORS[0]], edgecolor=BG, linewidth=2)
    for y, v in enumerate(fv):
        ax2.text(v + 2, y, f'{v:.1f}', ha='left', va='center', fontsize=13, color=INK, fontweight='bold')
    d = fv[1] - fv[0]
    ax2.annotate(f'{d:+.1f} FPS, {d / fv[0] * 100:+.1f}%', (fv[1] + 2, 1), textcoords='offset points',
                 xytext=(0, -20), ha='left', va='center', fontsize=10.5, color=COLORS[0], fontweight='bold')
    ax2.set_xlim(0, 150)
    ax2.set_xlabel('FPS', fontsize=10.5, color=INK2)

    for a in (ax, ax2):
        a.set_facecolor(BG)
        a.grid(axis='x', color=GRID, lw=1)
        a.set_axisbelow(True)
        for sp in ('top', 'right', 'left'):
            a.spines[sp].set_visible(False)
        a.spines['bottom'].set_color(MUTED)
        a.tick_params(length=0, colors=INK2)
    fig.text(0.5, 0.985, 'RT-DETRv2 vs UniQ-DETR (model size and speed)', fontsize=16, fontweight='bold', color=INK, ha='center',
             va='top')
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.3)
    out = 'analysis/figs/params_rtdetrv2_vs_uniq.png'
    fig.savefig(out, dpi=220, facecolor=BG, bbox_inches='tight', pad_inches=0.08)
    fig.savefig(out.replace('.png', '.pdf'), facecolor=BG, bbox_inches='tight', pad_inches=0.08)
    print(sum(base.values()), sum(ours.values()), dict(added), fps)
    print('saved', out)


if __name__ == '__main__':
    main()
