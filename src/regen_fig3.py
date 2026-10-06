"""Figure 3: threshold-sensitivity trade-off curve.
X: fraction of test pulses selected (stimulation rate)
Y: gain per pulse (%). Points labeled by train-set percentile."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open('data/threshold_sweep.json'))

fig, ax = plt.subplots(figsize=(9.5, 5.6))
colors = {'sess9': '#e8445c', 'sess12': '#2a7fbf'}
for s in d:
    lbl = s['label']
    xs = [r['select_frac'] for r in s['thresholds']]
    ys = [r['gain_per_pulse'] * 100 for r in s['thresholds']]
    pcts = [r['pct'] for r in s['thresholds']]
    ax.plot(xs, ys, color=colors[lbl], marker='o', markersize=6,
            linewidth=1.8, label=lbl)
    for x, y, p in zip(xs, ys, pcts):
        ax.annotate(f'{p}th', (x, y), textcoords='offset points',
                    xytext=(6, 6), fontsize=8, color='#333')
    # highlight the 75th-percentile operating point
    i75 = pcts.index(75)
    ax.scatter([xs[i75]], [ys[i75]], s=110, facecolors='none',
               edgecolors=colors[lbl], linewidths=2)

ax.set_xlabel('Fraction of test pulses selected (stimulation rate)')
ax.set_ylabel('Gain in spikes per pulse (%)')
ax.set_title('Figure 2. Policy threshold trade-off: efficiency vs. stimulation rate',
             fontsize=12)
ax.legend(title='Session', fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('fig2_threshold_tradeoff.png', dpi=110)
print('saved fig2_threshold_tradeoff.png')
