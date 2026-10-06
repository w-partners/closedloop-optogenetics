# Recent network activity predicts optogenetic response magnitude: data-driven design of efficient closed-loop photostimulation in silico

Dongchul Suh (AI BioLab Insights, Independent Research Laboratory, Seogwipo, Jeju, Republic of Korea)

## Abstract

Closed-loop optogenetics promises precise neural control by delivering light only when needed, but designing efficient stimulation policies requires knowing what makes a pulse effective. Here we re-analyzed a public dataset of pulsed optogenetic stimulation in mouse hippocampus (DANDI Archive dandiset 000568; 92,397 pulses across two sessions from one animal, 57 and 47 simultaneously recorded units). We found that the magnitude of the evoked response depends strongly on recent network activity (Pearson r = 0.45 and 0.43) but not on inter-pulse interval. The effect persisted across baseline window widths (50–300 ms) and in the subset of pulses with no window overlap (r = 0.48 and 0.45). A Poisson generalized linear model using pre-stimulus baseline activity predicted evoked spike counts on held-out data (27.8% and 19.2% deviance explained). An offline closed-loop policy that stimulates only when baseline activity meets or exceeds its 75th percentile increased spikes per pulse by 15–29% and reduced total light delivery by 13–23% for matched total activation, with gains positive across policy thresholds from the 50th to the 90th percentile. Embedding the fitted response model in a virtual closed-loop experiment reproduced the efficiency gain (+12.2% spikes per pulse, −10.9% light). All analyses used only public data with fully open code and fixed random seeds. Limitations include a single animal, a dim-light stimulation paradigm, and a hippocampal (non-retinal) preparation. We propose activity-dependent timing — stimulating when the circuit is receptive — as a simple, data-driven principle for closed-loop photostimulation design.

## Reproduction

1. Download the two NWB sessions from DANDI dandiset 000568 into data/
   as ogen1.nwb and ogen2.nwb.
2. pip install -r requirements.txt
3. python src/replicate.py            → data/replication.json (main results)
4. python src/sensitivity.py          → data/sensitivity.json (baseline-window robustness)
5. python src/threshold_sweep.py      → data/threshold_sweep.json (policy threshold trade-off)
6. python src/drift_robustness.py     → data/drift_robustness.json (drift robustness)
7. python src/regen_fig1.py src/regen_fig2.py src/regen_fig3.py → figures/

Software: Python 3.12.3, NumPy 1.26.4, SciPy 1.11.4. Analyses are deterministic;
figure subsampling uses a fixed seed (NumPy default_rng(0)).

## Data provenance

Raw NWB files (ogen1.nwb, ogen2.nwb, ~70 MB total) are from DANDI Archive
dandiset 000568 ("Probing subthreshold dynamics of hippocampal neurons by
pulsed optogenetics", https://dandiarchive.org/dandiset/000568).
Download via the DANDI API; see src/replicate.py for the expected filenames.

## Repository layout

- `src/` — analysis code
- `data/` — computed artifacts (small; raw data via provenance above)
- `figures/` — manuscript figures
- `manuscript/` — manuscript HTML + PDF

## License

Code: MIT. Manuscript: CC-BY 4.0.

## Citation

If you use this work, please cite the preprint (DOI to be added upon posting).
