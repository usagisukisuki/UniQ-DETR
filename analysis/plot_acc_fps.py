"""Accuracy vs. speed on Task 1: each method at its reported setting, FPS = median of 3 runs of
scripts/run_bench_fps_all.sh (fp32, batch 1, 640x640, model + post-processing, RTX 4090)."""
import re
import statistics

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

from plot_tasks import ARROW_UP, BG, COLORS, GRID, INK, INK2, M, MARKERS, MUTED, row

FPS_KEY = ['uniq_detr', 'prob_rt_nms', 'owdetr_rt_topk', 'cat_rt_nms', 'randbox_rt', 'orthogonaldet_rt']   # order of M


# label placement (dx, dy, ha) in points; default: left of the point, right for slow methods and OW-DETR
LABEL_OFF = {('full_K_AP50', 1): (-10, 14, 'right'), ('full_K_AP50', 3): (-12, -12, 'right'),
             ('full_U_AP', 3): (-10, 14, 'right'), ('full_U_AP', 1): (-12, -14, 'right')}


def label_off(key, i, x):
    if (key, i) in LABEL_OFF:
        return LABEL_OFF[key, i]
    return (12, 0, 'left') if x < 30 or i == 2 else (-12, 0, 'right')


def load_fps():
    runs = {}
    for r in (1, 2, 3):
        k = None
        for line in open(f'outputs/logs/fps_all_settings_r{r}.log'):
            if line.startswith('=='):
                k = line.split()[1]
            m = re.search(r'amp=False: .* ([\d.]+) FPS', line)
            if m:
                runs.setdefault(k, []).append(float(m.group(1)))
    return {k: statistics.median(v) for k, v in runs.items()}


def figure(panels, suptitle, out):
    fps = load_fps()
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), facecolor=BG,
                             gridspec_kw={'wspace': 0.18})
    for ax, (title, ylabel, key, nd) in zip(axes, panels):
        ax.set_facecolor(BG)
        ys = [row(dirs[0], sets[0])[key] for _, dirs, sets in M]
        xs = [fps[k] for k in FPS_KEY]
        ax.axvspan(30, 300, color=GRID, alpha=0.45, lw=0, zorder=0)   # real-time region
        ax.text(31.5, 0.985, 'real-time (≥30 FPS)', transform=ax.get_xaxis_transform(), fontsize=9.5, color=MUTED,
                va='top')
        for i, (name, _, _) in reversed(list(enumerate(M))):
            main = i == 0
            ax.scatter(xs[i], ys[i], s=230 if main else 110, marker=MARKERS[i], color=COLORS[i], edgecolor=BG,
                       linewidth=1.8, zorder=4 if main else 3, label=name)
            dx, dy, ha = label_off(key, i, xs[i])
            ax.annotate(f'{name.replace(" (ours)", "")}\n{ys[i]:.{nd}f} · {xs[i]:.0f} FPS', (xs[i], ys[i]),
                        textcoords='offset points', xytext=(dx, dy), ha=ha, va='center',
                        fontsize=10.5 if main else 9.5, color=COLORS[0] if main else INK2,
                        fontweight='bold' if main else 'normal', zorder=5,
                        path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
        t_ = ax.set_title(title, loc='center', fontsize=14, color=INK, fontweight='bold', pad=6)
        ax.annotate('↑', (1, 0.5), xycoords=t_, xytext=(5, 0), textcoords='offset points', ha='left', va='center',
                    fontsize=14, fontweight='bold', color=ARROW_UP)
        ax.set_xscale('log')
        ax.set_xlim(8, 300)
        ax.xaxis.set_major_locator(FixedLocator([10, 20, 30, 50, 100, 200]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(['10', '20', '30', '50', '100', '200'])
        ax.set_xlabel('FPS (log scale, fp32, batch 1, RTX 4090) →', fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        lo, hi = min(ys), max(ys)
        ax.set_ylim(lo - (hi - lo) * 0.15, hi + (hi - lo) * 0.15)
        ax.grid(color=GRID, lw=1)
        ax.set_axisbelow(True)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color(MUTED)
        ax.tick_params(length=0, colors=INK2)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=6, frameon=False,
               fontsize=10.5, labelcolor=INK2, markerscale=0.8)
    fig.text(0.5, 0.975, suptitle, fontsize=16, fontweight='bold', color=INK, ha='center', va='top')
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.2)
    fig.savefig(out, dpi=220, facecolor=BG)
    fig.savefig(out.replace('.png', '.pdf'), facecolor=BG)
    print('saved', out)


if __name__ == '__main__':
    figure([('Unknown recall', 'U-Recall (%)', 'full_U_R50', 1), ('Known-class mAP', 'mAP@0.5 (%)', 'full_K_AP50', 1)],
           'Accuracy vs. speed on M-OWODB Task 1 (RT-DETRv2)', 'analysis/figs/acc_fps_urecall_map.png')
    figure([('Unknown AP', 'U-AP@[.5:.95] (%)', 'full_U_AP', 2), ('Known AP', 'K-AP@[.5:.95] (%)', 'full_K_AP', 1)],
           'Accuracy vs. speed on M-OWODB Task 1 (RT-DETRv2, Hosoya et al. metrics)',
           'analysis/figs/acc_fps_kap_uap.png')
