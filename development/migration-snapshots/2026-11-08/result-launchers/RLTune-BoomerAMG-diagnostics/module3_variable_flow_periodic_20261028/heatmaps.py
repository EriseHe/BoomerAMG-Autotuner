"""Module 2-style action heatmaps for the completed variable-flow test."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run'
OUT = ROOT / 'readout/heatmaps'
MAIN = Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG')
sys.path.insert(0, str(MAIN))
from experiments.paper_final.common.figure_style import WEIGHT_CMAP

NAMES = ('fixed_1', 'periodic', 'rl_frozen')
LABELS = (r'Fixed $w=1$', 'Period-two (2.85, 1.10)', 'Frozen LSTDQ')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    marker = json.loads((RUN / 'raw/test/complete.json').read_text())
    assert (RUN / 'complete.json').exists()
    seeds = protocol['training_seeds']
    cap = protocol['max_cycles']
    case_ids = np.arange(protocol['test_cases'])
    weights = np.full((len(seeds), len(NAMES), len(case_ids), cap), np.nan)
    seen = set()
    hashes = {}
    for rel, expected in marker['raw_sha256'].items():
        path = RUN / rel
        assert sha(path) == expected
        hashes[str(path)] = expected
        for line in path.open():
            row = json.loads(line)
            if row['repeat'] != 0:
                continue
            key = (row['seed'], row['policy'], row['case_id'])
            assert key not in seen
            seen.add(key)
            o = row['outcome']
            primary = (o.get('primary_attempts') or [o])[0]
            cycles = int(primary.get('cycles', o.get('primary_cycles', o.get('iterations', 0))))
            actions = primary.get('cycle_actions', o.get('cycle_actions', o.get('cycle_action_values', [])))
            if not actions and cycles:
                actions = o.get('cycle_action_values', o.get('cycle_actions', []))
            assert len(actions) == cycles and 0 <= cycles <= cap
            assert all(np.isfinite(actions)) and all(1 <= w <= 3 for w in actions)
            spec = protocol['policies'][row['policy']]
            if spec['kind'] in ('fixed', 'periodic'):
                np.testing.assert_allclose(actions, np.resize(spec['pattern'], cycles), atol=1e-12, rtol=0)
            weights[seeds.index(row['seed']), NAMES.index(row['policy']), row['case_id'], :cycles] = actions
    assert len(seen) == 6 * 3 * 100
    orders = np.array([np.lexsort((case_ids, -np.isfinite(w[0]).sum(axis=1))) for w in weights])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'recorded_weights.npz', weights=weights, seeds=seeds,
                        policies=NAMES, case_ids=case_ids, case_order=orders)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65})
    ticks = [0, 10, 20, 30, 40, 49]

    def panel(ax, si, mi):
        im = ax.imshow(weights[si, mi, orders[si], :].T, origin='upper',
            interpolation='nearest', aspect='auto', extent=(.5, 100.5, 49.5, -.5),
            cmap=WEIGHT_CMAP, vmin=1, vmax=3, rasterized=True)
        ax.set(xticks=[1, 25, 50, 75, 100], yticks=ticks)
        ax.tick_params(length=3, width=.65)
        return im

    fig, axes = plt.subplots(6, 3, figsize=(9.0, 10.7), sharex=True, sharey=True)
    for si, seed in enumerate(seeds):
        for mi, label in enumerate(LABELS):
            im = panel(axes[si, mi], si, mi)
            if si == 0:
                axes[si, mi].set_title(label, loc='left', pad=7)
        axes[si, 0].set_ylabel(f'Seed {seed}\nCycle index $k$', labelpad=7)
    for ax in axes[-1]:
        ax.set_xlabel('Test problem rank')
    fig.subplots_adjust(left=.10, right=.895, top=.935, bottom=.070, wspace=.13, hspace=.20)
    cax = fig.add_axes([.925, .070, .013, .865])
    fig.colorbar(im, cax=cax, label='Relaxation weight', ticks=[1, 1.5, 2, 2.5, 3])
    fig.suptitle(r'Variable-flow $60^3$: recorded relaxation weights for all six seeds',
                 fontsize=10.5, y=.977)
    fig.text(.5, .016, 'First timing repetition; primary attempts only. White cells follow termination.\n'
             'Within each seed, all panels share the order of decreasing fixed-w=1 primary cycle count.',
             ha='center', fontsize=7.5)
    fig.savefig(OUT / 'all_six_seeds.png', dpi=190, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)

    for si, seed in enumerate(seeds):
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.55), sharex=True, sharey=True)
        for mi, label in enumerate(LABELS):
            im = panel(axes[mi], si, mi)
            axes[mi].set_title(label, loc='left', pad=6)
            axes[mi].set_xlabel('Test problem rank')
        axes[0].set_ylabel(r'Cycle index $k$')
        fig.subplots_adjust(left=.075, right=.895, top=.78, bottom=.19, wspace=.13)
        cax = fig.add_axes([.925, .19, .014, .59])
        fig.colorbar(im, cax=cax, label='Relaxation weight', ticks=[1, 1.5, 2, 2.5, 3])
        fig.suptitle(rf'Variable-flow $60^3$: matched-policy weights, seed {seed}', fontsize=9.5, y=.97)
        fig.savefig(OUT / f'seed_{seed}.png', dpi=220, bbox_inches='tight', pad_inches=.05)
        plt.close(fig)
    (OUT / 'provenance.json').write_text(json.dumps({
        'run': str(RUN), 'raw_sha256': hashes, 'seeds': seeds, 'policies': NAMES,
        'repetition_0_based': 0, 'primary_only': True, 'fallback_cycles_omitted': True,
        'row_index': 'k=0 means the action for the first cycle; k=49 is the fiftieth cycle.',
        'ordering': 'Within each seed, sort by decreasing fixed-w=1 primary cycle count, then case ID.',
        'cmap_source': str(MAIN / 'experiments/paper_final/common/figure_style.py'),
        'cmap_source_sha256': sha(MAIN / 'experiments/paper_final/common/figure_style.py'),
        'cases_checked': len(seen), 'weights_are_recorded_not_averaged': True,
    }, indent=2) + '\n')
    print(OUT / 'all_six_seeds.png')


if __name__ == '__main__':
    main()
