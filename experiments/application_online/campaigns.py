"""Application scenarios declared before the 1,000-system experiments."""
import numpy as np


def scenarios(application, seed):
    if application == 'spe10':
        cases = [dict(viscosity_ratio=ratio, schedule=schedule)
                 for ratio in (1., 2., 5., 10., 20.)
                 for schedule in ('constant', 'pulsed')]
    elif application == 'heat':
        regimes = ((.5, .01, .1), (.5, .1, .1), (.1, .1, .1),
                   (.5, .1, .05), (.5, .1, .2))
        cases = [dict(kappa=k, alpha=a, dt=dt, hot_amplitude=amplitude)
                 for k, a, dt in regimes for amplitude in (1., 4.)]
    else:
        raise ValueError(application)
    order = np.random.default_rng(seed + 991).permutation(len(cases))
    return [dict(scenario=i+1, **cases[int(j)]) for i, j in enumerate(order)]
