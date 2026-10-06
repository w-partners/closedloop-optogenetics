"""Regenerate fig1_state_dependence.png with the revised (honest) title."""
import h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(nwb_path):
    f = h5py.File(nwb_path, 'r')
    sp = f['stimulus/presentation']; u = f['units']
    st_all = u['spike_times'][:]; sti = u['spike_times_index'][:]
    bounds = np.concatenate([[0], sti]); nu = len(bounds) - 1
    pulses = []
    for k in sorted(sp.keys()):
        g = sp[k]; d = g['data'][:]; ts = g['timestamps'][:]
        on = d > d.max() * 0.5; e = np.diff(on.astype(np.int8))
        st = np.where(e == 1)[0] + 1; en = np.where(e == -1)[0] + 1
        if len(en) and len(st) and en[0] < st[0]:
            en = en[1:]
        n = min(len(st), len(en))
        for i in range(n):
            pulses.append(float(ts[st[i]]))
    pulses.sort(); T = np.array(pulses)
    base = np.zeros(len(T)); ev = np.zeros(len(T))
    for ui in range(nu):
        s = st_all[bounds[ui]:bounds[ui + 1]]
        base += np.searchsorted(s, T) - np.searchsorted(s, T - 0.2)
        ev += np.searchsorted(s, T + 0.1) - np.searchsorted(s, T)
    f.close()
    return base, ev, nu

rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.0), sharey=False)
fig.suptitle('Figure 1. Recent network activity predicts optogenetic response magnitude',
             fontsize=13)

for ax, (path, lbl) in zip(axes, [('data/ogen1.nwb', 'sess9'),
                                   ('data/ogen2.nwb', 'sess12')]):
    base, ev, nu = load(path)
    n = len(base)
    r = np.corrcoef(base, ev)[0, 1]
    idx = rng.choice(n, size=8000, replace=False)
    ax.scatter(base[idx], ev[idx], s=6, color='#7fb8e8', alpha=0.35, rasterized=True)
    # binned means (quantile bins)
    qs = np.quantile(base, np.linspace(0, 1, 21))
    qs[0] -= 1e-6; qs[-1] += 1e-6
    dig = np.digitize(base, qs) - 1
    bx = [base[dig == b].mean() for b in range(20)]
    by = [ev[dig == b].mean() for b in range(20)]
    ax.plot(bx, by, color='#e8445c', marker='o', markersize=5, linewidth=1.5)
    ax.set_xlabel(f'Baseline spikes (200 ms pre-pulse, {nu} units)')
    ax.set_title(f'{lbl}: r = {r:.3f}, n = {n:,} pulses', loc='left', fontsize=11)
    print(lbl, f'r={r:.3f}', f'n={n}', f'units={nu}')

axes[0].set_ylabel('Evoked spikes (100 ms post-pulse)')
axes[1].set_ylabel('Evoked spikes (100 ms post-pulse)')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('fig1_state_dependence.png', dpi=110)
print('saved fig1_state_dependence.png')
