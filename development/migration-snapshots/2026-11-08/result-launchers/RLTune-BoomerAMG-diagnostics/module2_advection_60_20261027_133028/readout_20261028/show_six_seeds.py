"""Display the completed diffusion-advection study without rerunning solvers."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

REPO = Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG')
sys.path.insert(0, str(REPO))
from experiments.paper_final.plot_policy import read_study, paired_gains
from experiments.paper_final.common.figure_style import matched_figures
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
RUN = OUT.parent / 'run'
summary, seeds, cases, costs, success, weights, sources = read_study(RUN)
native = paired_gains(costs, success, 0)
inclusive = paired_gains(costs, success, 1)
rows = []
for k, seed in enumerate(seeds):
    rows.append({
        'seed': seed,
        'periodic_solver_reduction_pct': float(native[k, 3]),
        'rl_solver_reduction_pct': float(native[k, 4]),
        'periodic_inclusive_continuation_reduction_pct': float(inclusive[k, 3]),
        'rl_inclusive_continuation_reduction_pct': float(inclusive[k, 4]),
        'rl_extra_inclusive_continuation_time_vs_periodic_pct':
            float(100 * (costs[k, 4, :, 1].sum() / costs[k, 3, :, 1].sum() - 1)),
    })
with (OUT / 'six_seed_comparison.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)

# Reuse the repository renderer, exporting only its savings panel.
figures = matched_figures(inclusive, weights, cases, 0, seeds,
                         r'Diffusion-advection $60^3$')
fig = dict(figures)['matched_policy_savings']
ax = fig.axes[0]
for collection in ax.collections:
    if isinstance(collection, PathCollection):
        collection.set_edgecolor('#222222')
for legend in list(fig.legends):
    legend.remove()
if ax.get_legend() is not None:
    ax.get_legend().remove()
ax.set_xlabel(r'Total continuation-time reduction relative to $w=1$ (%)')
ax.set_title(r'Diffusion-advection $60^3$: six frozen seeds', fontsize=9, pad=11)
fig.savefig(OUT / 'total_continuation_savings.png', dpi=240,
            bbox_inches='tight', pad_inches=.05)
for _, other in figures:
    plt.close(other)
(OUT / 'provenance.json').write_text(json.dumps({
    'source_run': str(RUN),
    'source_sha256': sources,
    'seeds': seeds,
    'cases_per_seed': len(cases),
    'timing_repetitions': summary['protocol']['repetitions_per_case'],
    'metric': 'inclusive_continuation_sec',
    'definition': 'Native solve plus recovery setup/solve plus controller computation; common primary setup excluded.',
    'aggregation': 'Average repetitions within problem, sum problems within seed, then compute paired reduction against fixed w=1.',
    'figure': 'Circles: all six seed results. Diamonds and whiskers: mean and sample standard deviation.',
    'rows': rows,
}, indent=2) + '\n')
print(json.dumps(rows, indent=2))
