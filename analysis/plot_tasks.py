"""U-Recall / Known mAP (AP50) and Hosoya et al. K-AP / U-AP over Tasks 1-4 for all methods, full test, reported settings (from sweep_ap.json)."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

# (label, ckpt dirs T1..T4, reported setting (nms_iou, unk_weight) per task)
M = [('UniQ-DETR (ours)', ['outputs/g4_8'] + [f'outputs/uniq_t{t}/ft' for t in (2, 3, 4)], [(None, 12.)] * 4),
     ('PROB', ['outputs/e2_obj'] + [f'outputs/prob_rt_t{t}/ft' for t in (2, 3, 4)], [(0.7, 0.7)] * 4),
     ('OW-DETR', ['outputs/b_owdetr_rt'] + [f'outputs/owdetr_rt_t{t}/ft' for t in (2, 3, 4)], [(None, 32.)] * 4),
     ('CAT', ['outputs/b_cat_rt'] + [f'outputs/cat_rt_t{t}/ft' for t in (2, 3, 4)], [(0.7, 16.)] * 4),
     ('RandBox', ['outputs/b_randbox_rt'] + [f'outputs/randbox_rt_t{t}/ft' for t in (2, 3, 4)],
      [(0.5, 256.)] + [(0.5, 32.)] * 3),
     ('OrthogonalDet', ['outputs/b_orthogonaldet_rt'] + [f'outputs/orthogonaldet_rt_t{t}/ft' for t in (2, 3, 4)],
      [(0.5, 8.)] + [(0.5, 4.)] * 3)]
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # dataviz default categorical, fixed order
MARKERS = ['o', 's', '^', 'D', 'v', 'P']


def row(d, s):
    for r in json.load(open(d + '/sweep_ap.json')):
        if (r.get('nms_iou'), r['unk_weight']) == s:
            return r
    raise KeyError(d, s)


for f in ('Regular', 'Bold'):
    fm.fontManager.addfont(f'/usr/share/fonts/truetype/lato/Lato-{f}.ttf')
INK, INK2, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#8a8984', '#e9e8e4', '#ffffff'
ARROW_UP, ARROW_DOWN = '#e34948', '#2a78d6'   # higher / lower is better
plt.rcParams.update({'font.family': 'Lato', 'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.size': 0, 'ytick.major.size': 0})
glow = [pe.Stroke(linewidth=7, foreground=BG), pe.Normal()]


def figure(panels, suptitle, out):
    """panels: [(title, ylabel, sweep key, tasks, decimals)] -> one row of line charts, UniQ-DETR highlighted"""
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), facecolor=BG,
                             gridspec_kw={'width_ratios': [len(p[3]) for p in panels], 'wspace': 0.18})
    for ax, (title, ylabel, key, tasks, nd, *low) in zip(axes, panels):
        low = bool(low and low[0])   # lower is better (K2U)
        ax.set_facecolor(BG)
        vals = {name: [row(dirs[t - 1], sets[t - 1])[key] for t in tasks] for name, dirs, sets in M}
        ours = vals[M[0][0]]
        best = [(min if low else max)(v[i] for n, v in vals.items() if n != M[0][0]) for i in range(len(tasks))]
        ax.fill_between(tasks, best, ours, where=[(o <= b) if low else (o >= b) for o, b in zip(ours, best)], interpolate=True,
                        color=COLORS[0], alpha=0.10, lw=0, zorder=1)
        for i, (name, _, _) in reversed(list(enumerate(M))):
            main = i == 0
            ax.plot(tasks, vals[name], color=COLORS[i], marker=MARKERS[i], ms=9 if main else 6, lw=3.2 if main else 1.6,
                    alpha=1 if main else 0.85, zorder=4 if main else 2, label=name, markeredgecolor=BG,
                    markeredgewidth=1.6, path_effects=glow if main else None, solid_capstyle='round')
        for t, o, b in zip(tasks, ours, best):
            ax.annotate(f'{o:.{nd}f}', (t, o), textcoords='offset points', xytext=(0, 12), ha='center', fontsize=11.5,
                        fontweight='bold', color=COLORS[0], zorder=5,
                        path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
        t_ = ax.set_title(title, loc='center', fontsize=14, color=INK, fontweight='bold', pad=6)
        ax.annotate('↓' if low else '↑', (1, 0.5), xycoords=t_, xytext=(5, 0), textcoords='offset points', ha='left',
                    va='center', fontsize=14, fontweight='bold', color=ARROW_DOWN if low else ARROW_UP)   # better direction
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_xticks(tasks, [f'Task {t}' for t in tasks], fontsize=11.5, color=INK)
        for t, o, b in zip(tasks, ours, best):   # gain over the best baseline, under each tick
            d = o - b
            ax.annotate(f'{d:+.{nd}f} vs best', (t, 0), xycoords=('data', 'axes fraction'), textcoords='offset points',
                        xytext=(0, -30), ha='center', va='top', fontsize=10,
                        color=COLORS[0] if (d < 0 if low else d > 0) else MUTED,
                        fontweight='bold' if (d < 0 if low else d > 0) else 'normal')
        ax.set_xlim(tasks[0] - 0.25, tasks[-1] + 0.35)
        lo, hi = min(min(v) for v in vals.values()), max(max(v) for v in vals.values())
        ax.set_ylim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.12)
        ax.grid(axis='y', color=GRID, lw=1)
        ax.set_axisbelow(True)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color(MUTED)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=6, frameon=False,
               fontsize=10.5, handlelength=2.2, columnspacing=1.6, labelcolor=INK2)
    fig.text(0.5, 0.975, suptitle, fontsize=16, fontweight='bold', color=INK,
             ha='center', va='top')
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22)
    fig.savefig(out, dpi=220, facecolor=BG)
    fig.savefig(out.replace('.png', '.pdf'), facecolor=BG)
    print('saved', out)


if __name__ == '__main__':
    figure([('Unknown recall', 'U-Recall (%)', 'full_U_R50', [1, 2, 3], 1),
            ('Known-class mAP', 'mAP@0.5 (%)', 'full_K_AP50', [1, 2, 3, 4], 1)],
           'UniQ-DETR vs. open-world baselines on M-OWODB (RT-DETRv2)', 'analysis/figs/tasks_urecall_map.png')
    # Hosoya et al. (IJCV 2025): AP over IoU 0.50:0.95 (COCO 101-point) for known and unknown
    figure([('Unknown AP', 'U-AP@[.5:.95] (%)', 'full_U_AP', [1, 2, 3], 2),
            ('Known AP', 'K-AP@[.5:.95] (%)', 'full_K_AP', [1, 2, 3, 4], 1)],
           'UniQ-DETR vs. open-world baselines on M-OWODB (RT-DETRv2, Hosoya et al. metrics)',
           'analysis/figs/tasks_kap_uap.png')
