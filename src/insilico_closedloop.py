"""
AI BioLab — In Silico Closed-loop Optogenetics Experiment (Step 5, simplified PoC) v2
가상 망막 회로 + WILD-like closed-loop 컨트롤러 vs 개루프(고정 자극) 비교.

단위: mV 기반 LIF. dV = (-(V - E_L) + I_tot) * dt/tau_m
- 표적 30세포: 고감도 옵신 발현 (빛만으로 발화 가능)
- 비표적 170세포: 빛 산란 8%만 수신
- 이벤트: 표적 중 15세포에 30ms 강한 drive (치료 개입이 필요한 순간을 가정)
"""
import numpy as np

DT, T = 0.1, 4000.0
STEPS = int(T / DT)
N, N_TGT = 200, 30
RNG = np.random.default_rng(7)

TAU_M, E_L, V_TH, V_RESET, T_REF = 20.0, -70.0, -50.0, -65.0, 2.0
TAU_ON, TAU_OFF = 2.0, 15.0
G_OPS = 35.0          # 정규화 광전류 게인: 전개시 ~30mV drive
SCATTER = 0.08
PULSE_MS, DETECT_LAT_MS, LOCKOUT_MS = 5.0, 1.0, 50.0

def run(noise_mv: float, closed: bool):
    steps = STEPS
    V = np.full(N, E_L); ref = np.zeros(N); O = np.zeros(N)
    spikes = np.zeros((steps, N), dtype=np.bool_)
    light = np.zeros(steps)

    evt_onsets = RNG.choice(np.arange(300, steps - 400, dtype=int), size=8, replace=False)
    evt_cells = [RNG.choice(N_TGT, size=15, replace=False) for _ in evt_onsets]
    drive = np.zeros((steps, N))
    for on, cells in zip(evt_onsets, evt_cells):
        drive[on:on + int(30 / DT), cells] = 30.0

    stim_at = list(range(int(250 / DT), steps, int(250 / DT))) if not closed else []
    stim_queue, lockout_until = [], -1
    win = int(20 / DT)
    recent = np.zeros(win, dtype=int)
    pulses, covered = 0, 0
    light_on_until = -1
    evt_cell_hit = []  # 커버된 이벤트에서 버스트 세포 중 자극 후 발화 비율

    for t in range(steps):
        I = RNG.normal(9.0, noise_mv, N) + drive[t]
        is_on = t < light_on_until
        light[t] = 1.0 if is_on else 0.0
        inten = np.full(N, SCATTER * light[t]); inten[:N_TGT] = light[t]
        tau = np.where(inten > 0, TAU_ON, TAU_OFF)
        O += (inten - O) * (DT / tau)
        I += G_OPS * O * (0.0 - V) / 70.0
        ref = np.maximum(ref - DT, 0.0)
        V = np.where(ref <= 0, V + (-(V - E_L) + I) * (DT / TAU_M), V_RESET)
        spk = (V >= V_TH) & (ref <= 0)
        spikes[t] = spk
        V = np.where(spk, V_RESET, V); ref = np.where(spk, T_REF, ref)

        recent = np.roll(recent, -1); recent[-1] = spk[:N_TGT].sum()
        if closed and t > win and t >= lockout_until and recent.sum() >= 8:
            stim_queue.append(t + int(DETECT_LAT_MS / DT))
            lockout_until = t + int(LOCKOUT_MS / DT)
        if not closed and stim_at and t == stim_at[0]:
            stim_at.pop(0); light_on_until = t + int(PULSE_MS / DT); pulses += 1
        while stim_queue and stim_queue[0] <= t:
            stim_queue.pop(0); light_on_until = t + int(PULSE_MS / DT); pulses += 1

    pulse_steps = np.where(np.diff(light.astype(int)) == 1)[0]
    tgt_act, off_act = [], []
    for ps in pulse_steps:
        w = slice(ps, min(ps + int(25 / DT), steps))
        tgt_act.append(spikes[w, :N_TGT].any(axis=0).sum())
        off_act.append(spikes[w, N_TGT:].any(axis=0).sum())
    for on, cells in zip(evt_onsets, evt_cells):
        lo, hi = on, on + int(40 / DT)
        hit = [ps for ps in pulse_steps if lo <= ps <= hi]
        if hit:
            covered += 1
            ps = hit[0]
            w = slice(ps, min(ps + int(30 / DT), steps))
            fired = spikes[w][:, cells].any(axis=0).sum()
            evt_cell_hit.append(fired / len(cells))
    return dict(
        pulses=pulses,
        light_ms=round(float(light.sum() * DT), 1),
        tgt_cells_per_pulse=round(float(np.mean(tgt_act)), 1) if tgt_act else 0.0,
        off_cells_per_pulse=round(float(np.mean(off_act)), 2) if off_act else 0.0,
        event_coverage=f"{covered}/8",
        evt_target_hit_rate=round(float(np.mean(evt_cell_hit)), 2) if evt_cell_hit else 0.0,
    )

if __name__ == "__main__":
    import json
    out = {}
    for noise in (3.0, 6.0, 10.0):
        for mode in ("closed", "open"):
            key = f"noise{int(noise)}_{mode}"
            out[key] = run(noise, closed=(mode == "closed"))
            print(key, json.dumps(out[key], ensure_ascii=False))
    with open("results.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved results.json")
