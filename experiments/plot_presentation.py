"""Export presentation figures from the 15 saved factorial runs; never trains.

Run from the repository: .venv\\Scripts\\python.exe experiments/plot_presentation.py
Writes PNG, SVG and numerical evidence to results/figures/presentation/.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_factorial import PREFIX, PREDICTORS, forgetting, ci

OUT = ROOT / 'results/figures/presentation'
SOURCE = ROOT / 'results/raw/phase5_factorial'
SEEDS = (1, 2, 3)
ORDER = ('M1', 'M2', 'M3', 'M4', 'M4noA')
FOCUS = ('M3', 'M4', 'M4noA')
COLORS = {'M1': '#a7adb4', 'M2': '#68727e', 'M3': '#16806a',
          'M4': '#1768aa', 'M4noA': '#bc6528'}
MARKERS = {'M3': '^', 'M4': 'o', 'M4noA': 's'}
SOURCE_NOTE = 'Source: saved phase5_factorial runs | Tiny ImageNet, 10 x 20 classes | width 128, 50 epochs/task | seeds 1, 2, 3'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 13, 'axes.labelsize': 11,
                     'legend.fontsize': 10, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})


def style(ax, axis='y'):
    ax.grid(axis=axis, color='#e3e6e9', linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[['left', 'bottom']].set_color('#a3aab1')


def finish(fig, stem, title, note):
    fig.suptitle(title, x=.07, y=.97, ha='left', fontsize=18, weight='bold')
    fig.text(.07, .055, note, fontsize=9, color='#424b55', va='bottom')
    fig.text(.07, .025, SOURCE_NOTE, fontsize=8, color='#5d6670')
    fig.savefig(OUT / f'{stem}.png', dpi=240, facecolor='white')
    fig.savefig(OUT / f'{stem}.svg', facecolor='white')
    plt.close(fig)


def load():
    records, matrices, provenance = {}, {}, []
    for cell in ORDER:
        records[cell], matrices[cell] = {}, {}
        for seed in SEEDS:
            path = SOURCE / f'{PREFIX[cell]}_fac_w128_t10_e50_s{seed}_results.json'
            data = json.loads(path.read_text())
            assert data['seed'] == seed and data['base_width'] == 128
            assert data['num_tasks'] == 10 and data['epochs_per_task'] == 50
            records[cell][seed] = data
            provenance.append({'path': path.relative_to(ROOT).as_posix(),
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        for predictor, key in PREDICTORS[cell].items():
            arr = np.full((3, 10, 10), np.nan)
            for si, seed in enumerate(SEEDS):
                rows = records[cell][seed][key]
                assert len(rows) == 10
                for t, row in enumerate(rows):
                    assert len(row) == t + 1
                    assert all(0 <= v <= 1 for v in row)
                    arr[si, t, :t + 1] = 100 * np.asarray(row)
            matrices[cell][predictor] = arr
    return records, matrices, provenance


def final(matrices, cell, predictor=None):
    predictor = predictor or ('head' if cell in ('M1', 'M2') else 'hier')
    return matrices[cell][predictor][:, -1].mean(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, matrices, provenance = load()
    evidence = {'source_files': provenance, 'seeds': SEEDS,
                'uncertainty': 'SD is sample SD across three optimization seeds; paired CIs use t(df=2), unadjusted.',
                'scope': 'One class-order setup. M1/M2 head; M3/M4/M4noA hierarchical unless explicitly labeled.',
                'metrics': {}, 'curves': {}, 'contrasts': {}, 'predictors': {}, 'heatmaps': {}}

    # 1. All configurations, with individual seeds instead of hidden variability.
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=.10, right=.97, bottom=.24, top=.82)
    labels = ['M1\nCNN baseline\nHead', 'M2\nCNN + Online EWC\nHead',
              'M3\nMemory pipeline\nHierarchical', 'M4\nFull configuration\nHierarchical',
              'M4noA\nAttention/fusion bypassed\nHierarchical']
    for x, cell in enumerate(ORDER):
        values = final(matrices, cell)
        ax.bar(x, values.mean(), width=.63, color=COLORS[cell], alpha=.88)
        ax.errorbar(x, values.mean(), yerr=values.std(ddof=1), color='#20262c', capsize=5, lw=1.3)
        ax.scatter(x + np.array([-.13, 0, .13]), values, facecolor='white', edgecolor='#20262c', s=32, zorder=4)
        ax.text(x, values.mean() + 1.05, f'{values.mean():.2f}%', ha='center', weight='bold')
        evidence['metrics'][cell] = {'final_accuracy_pct_per_seed': values.tolist()}
    ax.set(xticks=range(5), xticklabels=labels, ylabel='Final average accuracy (%)', ylim=(0, 29))
    style(ax)
    finish(fig, '01_final_accuracy', 'M3 leads overall; M4 outperforms its no-attention variant',
           'Bars: seed means; whiskers: +/-1 sample SD; dots: individual seeds. Baseline comparisons include different prediction/training pipelines.')

    # 2. Seen-task accuracy is distinct from old-task retention.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.subplots_adjust(left=.07, right=.98, bottom=.20, top=.79, wspace=.23)
    for cell in FOCUS:
        a = matrices[cell]['hier']
        seen = np.stack([a[:, t, :t + 1].mean(1) for t in range(10)], axis=1)
        old = np.stack([a[:, t, :t].mean(1) for t in range(1, 10)], axis=1)
        evidence['curves'][cell] = {'seen_accuracy_pct_per_seed': seen.tolist(), 'old_accuracy_pct_per_seed_tasks2to10': old.tolist()}
        for ax, x, values in [(axes[0], np.arange(1, 11), seen), (axes[1], np.arange(2, 11), old)]:
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(x, mean, marker=MARKERS[cell], color=COLORS[cell], label=cell, lw=2, ms=5)
            ax.fill_between(x, mean - sd, mean + sd, color=COLORS[cell], alpha=.13)
    for ax in axes:
        style(ax); ax.set(xlabel='Number of tasks learned', ylabel='Hierarchical accuracy (%)', ylim=(0, 85))
        ax.set_xticks(range(1 if ax is axes[0] else 2, 11))
    axes[0].set_title('All seen tasks, including the newest task')
    axes[1].set_title('Earlier tasks only, excluding the newest task')
    axes[0].legend(frameon=False)
    finish(fig, '02_task_progression', 'Accuracy declines as the recognition problem expands',
           'Lines: three-seed means; bands: +/-1 sample SD. Declining seen-task accuracy reflects both forgetting and an expanding class set.')

    # 3. Learning and retention must accompany the forgetting metric.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={'width_ratios': [1.65, 1]})
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.78, wspace=.25)
    for k, cell in enumerate(FOCUS):
        a = matrices[cell]['hier']
        vals = np.column_stack([np.diagonal(a, axis1=1, axis2=2).mean(1), a[:, -1, :9].mean(1), a[:, -1, 9]])
        f = np.array([100 * forgetting(records[cell][s][PREDICTORS[cell]['hier']]) for s in SEEDS])
        evidence['metrics'][cell].update({'learning_accuracy_pct_per_seed': vals[:, 0].tolist(),
            'final_old_accuracy_pct_per_seed': vals[:, 1].tolist(), 'final_newest_accuracy_pct_per_seed': vals[:, 2].tolist(),
            'forgetting_pp_per_seed': f.tolist()})
        xx = np.arange(3) + (k - 1) * .25
        axes[0].bar(xx, vals.mean(0), width=.23, color=COLORS[cell], label=cell,
                    yerr=vals.std(0, ddof=1), capsize=3, error_kw={'lw': 1})
        for x, v, sd in zip(xx, vals.mean(0), vals.std(0, ddof=1)):
            axes[0].text(x, v + sd + .8, f'{v:.1f}', ha='center', fontsize=9)
        axes[1].bar(k, f.mean(), width=.6, color=COLORS[cell], yerr=f.std(ddof=1), capsize=4)
        axes[1].text(k, f.mean() + f.std(ddof=1) + .6, f'{f.mean():.2f}', ha='center', fontsize=11)
    axes[0].set(xticks=range(3), xticklabels=['Immediately after\nlearning each task', 'Final old tasks\n(Tasks 1-9)', 'Final newest task\n(Task 10)'], ylabel='Hierarchical accuracy (%)', ylim=(0, 52), title='Acquisition and final recognition (higher is better)')
    axes[0].legend(frameon=False, ncol=3, loc='upper right')
    axes[1].set(xticks=range(3), xticklabels=FOCUS, ylabel='Average forgetting (percentage points)', ylim=(0, 28), title='Forgetting (lower is better)')
    for ax in axes: style(ax)
    finish(fig, '03_learning_and_forgetting', 'Lower forgetting does not necessarily mean better recognition',
           'Bars: means; whiskers: +/-1 sample SD. M4noA has lower forgetting, but also weaker acquisition and lower final accuracy.')

    # 4. Paired differences preserve seed pairing; do not treat 10 tasks as replicates.
    specs = [('M4 - M4noA | hierarchical', 'M4', 'M4noA', 'hier'),
             ('M4 - M4noA | head only', 'M4', 'M4noA', 'head'),
             ('M4 - M3 | hierarchical', 'M4', 'M3', 'hier'),
             ('M2 - M1 | head only', 'M2', 'M1', 'head')]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(left=.30, right=.89, bottom=.21, top=.80)
    for y, (label, c1, c2, pred) in enumerate(specs):
        values = final(matrices, c1, pred) - final(matrices, c2, pred)
        mean, sd, lo, hi = ci(values)
        evidence['contrasts'][label] = {'difference_pp_per_seed': values.tolist(), 'mean_pp': mean, 'sd_pp': sd, 'ci95_pp': [lo, hi]}
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], color=COLORS['M4'] if mean > 0 else '#626c76', fmt='D', ms=7, capsize=5, lw=2, zorder=3)
        ax.scatter(values, y + np.array([-.15, -.11, -.07]), c='#727b83', s=22, marker='o', alpha=.8)
        ax.text(3.5, y, f'{mean:+.2f}\n[{lo:+.2f}, {hi:+.2f}]', va='center', fontsize=10)
    ax.axvline(0, color='#22272d', ls='--', lw=1)
    ax.set(yticks=range(4), yticklabels=[s[0] for s in specs], xlabel='Final accuracy difference (percentage points)', xlim=(-3.5, 3.4), ylim=(3.55, -.55))
    ax.text(3.5, -.45, 'Mean / 95% CI', fontsize=10, weight='bold')
    style(ax, 'x')
    finish(fig, '04_paired_comparisons', 'The attention/fusion gain depends on the reported predictor',
           'Diamonds: paired-seed means; whiskers: 95% t intervals (df=2); small dots: individual differences. Three seeds; no multiplicity correction.')

    # 5. Same scale for absolute matrices; symmetric scale for signed differences.
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.7))
    fig.subplots_adjust(left=.055, right=.97, bottom=.23, top=.77, wspace=.40)
    m4 = matrices['M4']['hier'].mean(0)
    noa = matrices['M4noA']['hier'].mean(0)
    for i, (ax, values, title) in enumerate(zip(axes, [m4, noa, m4 - noa], ['M4: hierarchical accuracy', 'M4noA: hierarchical accuracy', 'M4 minus M4noA'])):
        cmap = plt.get_cmap('Blues' if i < 2 else 'RdBu').copy()
        cmap.set_bad('#eeeeee')
        limit = max(10, np.ceil(np.nanmax(np.abs(m4 - noa)) / 5) * 5)
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0 if i < 2 else -limit, vmax=100 if i < 2 else limit)
        for r in range(10):
            for c in range(r + 1):
                v = values[r, c]
                white = v > 60 if i < 2 else abs(v) > .65 * limit
                label = f'{v:.0f}' if i < 2 else ('0.0' if abs(v) < .05 else f'{v:+.1f}')
                ax.text(c, r, label, ha='center', va='center', fontsize=7.5, color='white' if white else '#17202a')
        ax.set(title=title, xlabel='Evaluated task', ylabel='After learning task', xticks=range(10), yticks=range(10), xticklabels=range(1, 11), yticklabels=range(1, 11))
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=.17, fraction=.05, label='Accuracy (%)' if i < 2 else 'Difference (percentage points)')
    evidence['heatmaps'] = {'M4_mean_pct': [[None if np.isnan(v) else float(v) for v in row] for row in m4],
                           'M4noA_mean_pct': [[None if np.isnan(v) else float(v) for v in row] for row in noa],
                           'final_task_difference_pp': (m4[-1] - noa[-1]).tolist()}
    finish(fig, '05_task_heatmaps', 'Task-level results show where the M4 advantage appears',
           'Each cell: mean across three seeds; displayed labels rounded. Gray: not yet learned. Difference panel: positive values favor M4.')

    # 6. Predictor choice is part of the evaluation, not an architectural constant.
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=.09, right=.97, bottom=.20, top=.79)
    preds = ('head', 'ncm', 'ens', 'hier')
    for k, cell in enumerate(FOCUS):
        values = np.column_stack([final(matrices, cell, p) for p in preds])
        evidence['predictors'][cell] = {p: values[:, j].tolist() for j, p in enumerate(preds)}
        xx = np.arange(4) + (k - 1) * .25
        ax.bar(xx, values.mean(0), width=.23, color=COLORS[cell], label=cell,
               yerr=values.std(0, ddof=1), capsize=3, error_kw={'lw': 1})
        for x, v, sd in zip(xx, values.mean(0), values.std(0, ddof=1)):
            ax.text(x, v + sd + .3, f'{v:.2f}', ha='center', fontsize=9)
    ax.set(xticks=range(4), xticklabels=['Classifier head', 'Prototype similarity', 'Ensemble', 'Hierarchical'], ylabel='Final average accuracy (%)', ylim=(0, 29))
    ax.legend(frameon=False, ncol=3, loc='upper left'); style(ax)
    finish(fig, '06_predictor_comparison', 'Combining head and prototype scores improves the reported mean',
           'Bars: three-seed means; whiskers: +/-1 sample SD. Prototype scores use maximum similarity across ten prototypes per class.')

    assert abs(final(matrices, 'M4').mean() - 22.2333333333) < 1e-6
    assert abs(final(matrices, 'M4noA').mean() - 20.84) < 1e-6
    (OUT / 'plot_data.json').write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'figures': 6, 'formats': ['PNG (240 dpi)', 'SVG'],
                      'contrasts': evidence['contrasts'],
                      'final_task_M4_minus_M4noA_pp': evidence['heatmaps']['final_task_difference_pp']}, indent=2))


if __name__ == '__main__':
    main()
