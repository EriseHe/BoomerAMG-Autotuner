"""Module 2's fixed-w=1 denominator, with failed comparisons explicit."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'readout'
audit = json.loads((OUT / 'audit.json').read_text())
assert all(audit['checks'].values())
summary = json.loads((ROOT / 'run/analysis/summary.json').read_text())
rows = []
for seed in range(1, 7):
    rr = {r['method']: r for r in summary['per_seed'] if r['seed'] == seed}
    row = {'seed': seed, 'fixed1_final_failures': 100 - rr['reference']['completed']}
    for method in ('periodic', 'rl'):
        eligible = rr['reference']['completed'] == rr[method]['completed'] == 100
        row[method + '_final_failures'] = 100 - rr[method]['completed']
        row[method + '_all_success_eligible'] = eligible
        for field, label in [('native_continuation_sec', 'solver'),
                             ('inclusive_continuation_sec', 'inclusive')]:
            value = 100 * (1 - rr[method][field] / rr['reference'][field])
            row[method + '_' + label + '_measured_cost_reduction_pct'] = value
            row[method + '_' + label + '_strict_module2_reduction_pct'] = value if eligible else None
    rows.append(row)
with (OUT / 'reduction_vs_fixed1.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
(OUT / 'reduction_vs_fixed1.json').write_text(json.dumps({
    'rows': rows,
    'denominator': 'Fixed w=1 on each bandit-selected hierarchy.',
    'metric': 'Sum of per-problem repetition-averaged continuation times, including recovery and controller computation; common primary setup excluded.',
    'qualification': 'All 100 inputs are retained in measured-cost ratios. Comparisons with any final failure are marked and excluded from strict Module 2 all-success ratios.',
}, indent=2) + '\n')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif', 'font.size': 8.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65})
fig, ax = plt.subplots(figsize=(7.4, 3.35))
for method, label, marker, offset, face in [
        ('periodic', 'Period-two (2.85, 1.10)', 's', -.1, 'white'),
        ('rl', 'Frozen LSTDQ', 'D', .1, '#222222')]:
    xx = [r[method + '_inclusive_measured_cost_reduction_pct'] for r in rows]
    yy = [r['seed'] + offset for r in rows]
    ax.scatter(xx, yy, marker=marker, s=30, edgecolors='#222222', facecolors=face,
               linewidths=.9, label=label, zorder=3)
    for r, x, y in zip(rows, xx, yy):
        flag = '' if r[method + '_all_success_eligible'] else '†'
        ax.annotate(f'{x:+.1f}{flag}', (x, y), xytext=(7, 0), textcoords='offset points',
                    fontsize=8, va='center', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .2})
ax.axvline(0, color='#777777', linewidth=.7)
ax.set(ylim=(6.55, .45), yticks=range(1, 7),
       yticklabels=[f'Seed {s}' for s in range(1, 7)], xlim=(-165, 77),
       xlabel=r'Total continuation-time reduction relative to $w=1$ (%)')
ax.set_xticks([-150, -100, -50, 0, 50])
ax.set_title(r'Variable-flow $60^3$: six Module 3 models', fontsize=10, pad=10)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2, frameon=False, fontsize=8)
fig.text(.5, .015, '† Comparator or reference has unfinished solves; see failure counts.',
         ha='center', fontsize=7.5)
fig.subplots_adjust(left=.1, right=.98, bottom=.26, top=.88)
fig.savefig(OUT / 'reduction_vs_fixed1.png', dpi=230, bbox_inches='tight', pad_inches=.05)
plt.close(fig)
print(json.dumps(rows, indent=2))
