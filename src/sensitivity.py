"""
Baseline-window sensitivity analysis for the closed-loop optogenetics paper.
Reviewer-driven robustness check: does the baseline->evoked prediction survive
when the baseline window is shortened, and when pulses whose baseline window
overlaps the previous pulse are excluded?

Windows tested: [-50,0), [-100,0), [-200,0), [-300,0) ms
Non-overlap subset: pulses with inter-pulse interval >= 300 ms
  (previous pulse evoked window [T_prev, T_prev+100ms) does not touch
   baseline window [T-200ms, T))
Also: GLM with previous-pulse evoked response as extra covariate.
"""
import h5py, numpy as np, json

def load_pulses(nwb_path):
    f = h5py.File(nwb_path, 'r')
    sp = f['stimulus/presentation']
    u = f['units']
    st_all = u['spike_times'][:]; sti = u['spike_times_index'][:]
    bounds = np.concatenate([[0], sti]); n_units = len(bounds) - 1
    pulses = []
    for k in sorted(sp.keys()):
        g = sp[k]
        d = g['data'][:]; ts = g['timestamps'][:]
        on = d > d.max() * 0.5
        edges = np.diff(on.astype(np.int8))
        starts = np.where(edges == 1)[0] + 1; ends = np.where(edges == -1)[0] + 1
        if len(ends) and len(starts) and ends[0] < starts[0]:
            ends = ends[1:]
        n = min(len(starts), len(ends))
        for i in range(n):
            pulses.append((float(ts[starts[i]]), k))
    pulses.sort()
    T = np.array([t for t, _ in pulses]); sites = np.array([s for _, s in pulses])
    return f, st_all, bounds, n_units, T, sites

def counts_in_window(st_all, bounds, n_units, T, lo, hi):
    """spike counts per pulse in [T+lo, T+hi), summed over units; also per-unit matrix"""
    mat = np.zeros((len(T), n_units))
    for ui in range(n_units):
        s = st_all[bounds[ui]:bounds[ui + 1]]
        mat[:, ui] = np.searchsorted(s, T + hi) - np.searchsorted(s, T + lo)
    return mat

def fit_poisson_glm(Xtr, ytr, Xte, yte, iters=50):
    beta = np.zeros(Xtr.shape[1])
    for _ in range(iters):
        lam = np.exp(Xtr @ beta) + 1e-9
        z = Xtr @ beta + (ytr - lam) / lam
        XtW = Xtr.T * lam
        bnew = np.linalg.solve(XtW @ Xtr + 1e-6 * np.eye(Xtr.shape[1]), XtW @ z)
        if np.abs(bnew - beta).max() < 1e-8:
            beta = bnew; break
        beta = bnew
    lam_te = np.exp(Xte @ beta)
    dev = lambda y, l: 2 * np.sum(y * np.log((y + 1e-9) / (l + 1e-9)) - (y - l))
    expl = 1 - dev(yte, lam_te) / dev(yte, np.full_like(yte, ytr.mean()))
    return beta, float(expl)

def analyze_session(nwb_path, label):
    f, st_all, bounds, n_units, T, sites = load_pulses(nwb_path)
    n = len(T)
    ipi = np.diff(T, prepend=T[0] - 1.0)  # first pulse gets large IPI
    usites = sorted(set(sites)); sidx = np.array([usites.index(s) for s in sites])
    evoked = counts_in_window(st_all, bounds, n_units, T, 0.0, 0.1).sum(axis=1)
    out = {'label': label, 'n_pulses': n, 'n_units': n_units,
           'median_ipi_ms': round(float(np.median(np.diff(T)) * 1000), 1)}
    # 1) window-width sensitivity
    for w_ms in [50, 100, 200, 300]:
        w = w_ms / 1000.0
        base = counts_in_window(st_all, bounds, n_units, T, -w, 0.0).sum(axis=1)
        r = float(np.corrcoef(base, evoked)[0, 1])
        tr, te = np.arange(n // 2), np.arange(n // 2, n)
        def design(b, s):
            return np.column_stack([np.ones(len(b)), np.log1p(b)] +
                                   [(s == j).astype(float) for j in range(1, len(usites))])
        Xtr, Xte = design(base[tr], sidx[tr]), design(base[te], sidx[te])
        beta, expl = fit_poisson_glm(Xtr, evoked[tr], Xte, evoked[te])
        thr = np.quantile(base[tr], .75); m = base[te] >= thr
        gain = float(evoked[te][m].mean() / evoked[te].mean() - 1)
        save = float(1 - m.sum() / (evoked[te][m].sum() / evoked[te].mean()))
        out[f'w{w_ms}'] = dict(corr=round(r, 3), beta_base=round(float(beta[1]), 3),
                               dev_explained=round(expl, 3),
                               gain_per_pulse=round(gain, 3),
                               light_saving=round(save, 3),
                               test_select_frac=round(float(m.mean()), 3))
    # 2) non-overlapping subset (IPI >= 300 ms) with w=200 ms
    mask = ipi >= 0.3
    nm = int(mask.sum())
    base_m = counts_in_window(st_all, bounds, n_units, T[mask], -0.2, 0.0).sum(axis=1)
    evm = evoked[mask]; sm = sidx[mask]
    r_m = float(np.corrcoef(base_m, evm)[0, 1]) if nm > 10 else float('nan')
    # train/test within subset (use GLOBAL site encoding for consistency)
    trm, tem = np.arange(nm // 2), np.arange(nm // 2, nm)
    def design2(b, s):
        return np.column_stack([np.ones(len(b)), np.log1p(b)] +
                               [(s == j).astype(float) for j in range(1, len(usites))])
    beta_m, expl_m = fit_poisson_glm(design2(base_m[trm], sm[trm]), evm[trm],
                                     design2(base_m[tem], sm[tem]), evm[tem])
    out['nonoverlap_ipi300'] = dict(n=nm, frac=round(nm / n, 3),
                                    corr=round(r_m, 3),
                                    beta_base=round(float(beta_m[1]), 3),
                                    dev_explained=round(expl_m, 3))
    # 3) previous-pulse evoked as extra covariate (w=200 ms), full sample
    base200 = counts_in_window(st_all, bounds, n_units, T, -0.2, 0.0).sum(axis=1)
    prev_ev = np.concatenate([[0.0], evoked[:-1]])
    r_bprev = float(np.corrcoef(base200, prev_ev)[0, 1])
    tr, te = np.arange(n // 2), np.arange(n // 2, n)
    def design3(b, p, s):
        return np.column_stack([np.ones(len(b)), np.log1p(b), np.log1p(p)] +
                               [(s == j).astype(float) for j in range(1, len(usites))])
    Xtr3 = design3(base200[tr], prev_ev[tr], sidx[tr])
    Xte3 = design3(base200[te], prev_ev[te], sidx[te])
    beta3, expl3 = fit_poisson_glm(Xtr3, evoked[tr], Xte3, evoked[te])
    out['prev_pulse_covariate'] = dict(corr_base_prevEvoked=round(r_bprev, 3),
                                       beta_base=round(float(beta3[1]), 3),
                                       beta_prevEvoked=round(float(beta3[2]), 3),
                                       dev_explained=round(expl3, 3))
    f.close()
    return out

if __name__ == '__main__':
    import sys
    res = [analyze_session(p, l) for p, l in
           [('data/ogen1.nwb', 'sess9'), ('data/ogen2.nwb', 'sess12')]]
    json.dump(res, open('data/sensitivity.json', 'w'), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
