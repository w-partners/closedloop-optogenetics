"""
Threshold-sensitivity analysis: sweep the closed-loop policy threshold across
percentiles and trace the efficiency vs stimulation-rate trade-off curve.
Addresses: "why 75th percentile? cherry-picked?"
Method mirrors the manuscript: threshold defined on train half (first 50% of
pulses), evaluated on test half (second 50%) using recorded outcomes.
"""
import h5py, numpy as np, json

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
    return base, ev

if __name__ == '__main__':
    out = []
    for p, lbl in [('data/ogen1.nwb', 'sess9'), ('data/ogen2.nwb', 'sess12')]:
        base, ev = load(p)
        n = len(base)
        tr, te = np.arange(n // 2), np.arange(n // 2, n)
        btr, bte = base[tr], base[te]
        ete = ev[te]
        mean_all = ete.mean()
        rows = []
        for pct in [50, 60, 70, 75, 80, 90]:
            thr = np.quantile(btr, pct / 100.0)
            m = bte >= thr
            nm = int(m.sum())
            if nm == 0:
                rows.append(dict(pct=pct, select_frac=0.0, gain=None, saving=None))
                continue
            gain = float(ete[m].mean() / mean_all - 1)
            # light saving: pulses needed for matched total activation
            saving = float(1 - nm / (ete[m].sum() / mean_all))
            rows.append(dict(pct=pct, threshold=float(thr),
                             select_frac=round(nm / len(te), 3),
                             gain_per_pulse=round(gain, 3),
                             light_saving=round(saving, 3)))
        out.append(dict(label=lbl, n_pulses=n, thresholds=rows))
        print(lbl)
        for r in rows:
            print(f"  p{r['pct']}: selfrac={r['select_frac']} gain={r['gain_per_pulse']} save={r['light_saving']}")
    json.dump(out, open('data/threshold_sweep.json', 'w'), indent=2)
    print('saved data/threshold_sweep.json')
