"""
Drift-robustness analyses (fully specified, reproducible):
1. Detrended correlation: remove slow (>60 s) fluctuations by 60-s block
   demeaning (blocks aligned to first pulse), correlate residuals.
2. Block analysis: split each session into 20 equal-count blocks (~5 min each),
   Pearson r per block; report minimum and sign consistency.
"""
import h5py, numpy as np, json

def load(nwb_path):
    f = h5py.File(nwb_path, 'r')
    sp = f['stimulus/presentation']; u = f['units']
    st_all = u['spike_times'][:]; sti = u['spike_times_index'][:]
    bounds = np.concatenate([[0], sti]); nu = len(bounds) - 1
    T = []
    for k in sorted(sp.keys()):
        g = sp[k]; d = g['data'][:]; ts = g['timestamps'][:]
        on = d > d.max() * 0.5; e = np.diff(on.astype(np.int8))
        st = np.where(e == 1)[0] + 1; en = np.where(e == -1)[0] + 1
        if len(en) and len(st) and en[0] < st[0]:
            en = en[1:]
        n = min(len(st), len(en))
        for i in range(n):
            T.append(float(ts[st[i]]))
    T = np.array(sorted(T)); n = len(T)
    base = np.zeros(n); ev = np.zeros(n)
    for ui in range(nu):
        s = st_all[bounds[ui]:bounds[ui + 1]]
        base += np.searchsorted(s, T) - np.searchsorted(s, T - 0.2)
        ev += np.searchsorted(s, T + 0.1) - np.searchsorted(s, T)
    f.close()
    return T, base, ev

if __name__ == '__main__':
    out = []
    for p, lbl in [('data/ogen1.nwb', 'sess9'), ('data/ogen2.nwb', 'sess12')]:
        T, base, ev = load(p)
        n = len(T)
        # 1) 60-s block demeaning
        blk = ((T - T.min()) // 60).astype(int)
        rb = np.zeros(n); re = np.zeros(n)
        for b in np.unique(blk):
            m = blk == b
            rb[m] = base[m] - base[m].mean()
            re[m] = ev[m] - ev[m].mean()
        r_det = float(np.corrcoef(rb, re)[0, 1])
        # 2) 20 equal-count blocks
        idx = np.array_split(np.arange(n), 20)
        rs = [float(np.corrcoef(base[ii], ev[ii])[0, 1]) for ii in idx]
        out.append(dict(label=lbl, n_pulses=n,
                        detrended_r=round(r_det, 3),
                        block_r_min=round(min(rs), 3),
                        block_r_all_positive=all(r > 0 for r in rs),
                        n_blocks=len(rs)))
        print(lbl, 'detrended r =', round(r_det, 3),
              '| block min =', round(min(rs), 3),
              '| all positive:', all(r > 0 for r in rs))
    json.dump(out, open('data/drift_robustness.json', 'w'), indent=2)
    print('saved data/drift_robustness.json')
