# Pre-stimulus network state predicts optogenetic response magnitude: data-driven design of efficient closed-loop photostimulation in silico

Dongchul Suh (AI BioLab Insights, Independent Research Laboratory, Seoul, Republic of Korea)

## Abstract

Closed-loop optogenetic stimulation promises precise neural control with
minimal light delivery, but the principles for timing light pulses remain
largely empirical. Here we take a fully data-driven, in silico approach...


## Reproduction

1. Download the two NWB sessions from DANDI dandiset 000568 into data/
   as ogen1.nwb and ogen2.nwb.
2. pip install -r requirements.txt
3. python src/replicate.py  → data/replication.json


## Data provenance

Raw NWB files (ogen1.nwb, ogen2.nwb, ~70 MB total) are from DANDI Archive
dandiset 000568 ("Probing subthreshold dynamics of hippocampal neurons by
pulsed optogenetics", https://dandiarchive.org/dandiset/000568).
Download via the DANDI API; see src/replicate.py for the expected filenames.


## Repository layout

- `src/` — analysis code
- `data/` — computed artifacts (small; raw data via provenance above)
- `figures/` — manuscript figures
- `manuscript/` — manuscript PDF

## License

Code: MIT. Manuscript: CC-BY 4.0.

## Citation

If you use this work, please cite the preprint (DOI to be added upon posting).
