"""Regenerate hybrid_closed_vs_open.png with English labels (was Korean)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load('data/hybrid_sim.npz')
pc, ec = d['pc'], d['ec']   # closed-loop: times (s), evoked spikes
po, eo = d['po'], d['eo']   # open-loop

mc, mo = ec.mean(), eo.mean()
print(f'closed mean={mc:.2f} (n={len(ec)}), open mean={mo:.2f} (n={len(eo)})')

fig, axes = plt.subplots(2, 1, figsize=(13.2, 6.6), sharex=True)
fig.suptitle('Hybrid in silico experiment: evoked spikes per pulse (GLM fitted on real data)',
             fontsize=13)

ax = axes[0]
ax.scatter(pc, ec, s=10, color='#7fd8d0', alpha=0.8)
ax.axhline(mc, color='#e0a800', linestyle='--', linewidth=1.2,
           label=f'mean = {mc:.1f} spikes/pulse')
ax.set_ylabel('Evoked spikes per pulse')
ax.set_title('Closed-loop (state-dependent stimulation)', loc='left', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(3, 32)

ax = axes[1]
ax.scatter(po, eo, s=10, color='#7fd8d0', alpha=0.8)
ax.axhline(mo, color='#e0a800', linestyle='--', linewidth=1.2,
           label=f'mean = {mo:.1f} spikes/pulse')
ax.set_ylabel('Evoked spikes per pulse')
ax.set_xlabel('Simulated time (s)')
ax.set_title('Open-loop (stimulate every candidate)', loc='left', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(3, 32)
ax.set_xlim(0, 1200)
ax.set_xticks([0, 300, 600, 900, 1200])

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('hybrid_closed_vs_open.png', dpi=110)
print('saved hybrid_closed_vs_open.png')
