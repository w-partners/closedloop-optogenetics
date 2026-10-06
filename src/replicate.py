"""
AI BioLab — 상태 의존성 재현 분석 파이프라인 (재사용 가능)
입력: DANDI 000568 no-raw-data NWB (ogen 세션)
출력: 상관계수, GLM 결과, 정책 평가
"""
import h5py, numpy as np, json, sys

def analyze(nwb_path, label):
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
    n = len(T)
    print(f'[{label}] pulses={n}, units={n_units}')

    evoked = np.zeros((n, n_units)); base = np.zeros((n, n_units))
    for ui in range(n_units):
        s = st_all[bounds[ui]:bounds[ui + 1]]
        evoked[:, ui] = np.searchsorted(s, T + 0.1) - np.searchsorted(s, T)
        base[:, ui] = np.searchsorted(s, T) - np.searchsorted(s, T - 0.2)
    e_sum, b_sum = evoked.sum(axis=1), base.sum(axis=1)
    r = float(np.corrcoef(b_sum, e_sum)[0, 1])

    # GLM (train/test split)
    usites = sorted(set(sites)); sidx = np.array([usites.index(s) for s in sites])
    tr, te = np.arange(n // 2), np.arange(n // 2, n)
    def design(b, s):
        return np.column_stack([np.ones(len(b)), np.log1p(b)] +
                               [(s == j).astype(float) for j in range(1, len(usites))])
    Xtr, Xte = design(b_sum[tr], sidx[tr]), design(b_sum[te], sidx[te])
    ytr, yte = e_sum[tr], e_sum[te]
    beta = np.zeros(Xtr.shape[1])
    for _ in range(50):
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
    # 정책 평가
    thr = np.quantile(b_sum[tr], .75); m = b_sum[te] >= thr
    gain = float(yte[m].mean() / yte.mean() - 1)
    save = float(1 - m.sum() / (yte[m].sum() / yte.mean()))
    out = dict(label=label, n_pulses=n, n_units=n_units, corr=round(r, 3),
               beta_base=round(float(beta[1]), 3), dev_explained=round(float(expl), 3),
               gain_per_pulse=round(gain, 3), light_saving=round(save, 3))
    print(json.dumps(out, ensure_ascii=False))
    f.close()
    return out

if __name__ == '__main__':
    results = [analyze(p, l) for p, l in
               [('data/ogen1.nwb', 'sess9'), ('data/ogen2.nwb', 'sess12')]]
    json.dump(results, open('data/replication.json', 'w'), ensure_ascii=False, indent=2)
    print('saved data/replication.json')
