# IVDA — Boundary-Aware V-Detector with Statistically Guided Termination

Reference implementation for the paper *"Compiling Compact Negative-Selection Detectors
into Spiking Neural Networks for Energy-Efficient, Low False-Alarm Intrusion Detection"*.
Every number reported in the paper's Section 4 and in the canonical-split evaluation of
Section 5 is produced by the scripts here, and the exact outputs used in the paper are
committed under [`results/`](results/), so any value can be checked without re-running
anything.

Release v1.1.0 adds the spiking compilation, the spike-encoding and self-tolerance
experiments (Sections 3.7–3.9 and 4.12–4.16) and the canonical KDDTrain+/KDDTest+
evaluation (Section 5). The detector generator and every v1.0.0 result are unchanged.

## What is implemented

- **`ivda/detector.py`** — the V-detector generator, with two independently togglable
  mechanisms: statistically guided termination (a one-sided z-test on the Monte Carlo
  coverage proportion; generation stops when H₀: C ≤ C_target is rejected) and
  boundary-aware detector construction (a candidate is accepted if it covers enough
  previously uncovered probes, or if its nearest self sample is a boundary self point).
  A fixed common radius can also be forced, which reduces it to the classical
  fixed-radius NSA baseline.
- **`ivda/multiclass.py`** — the one-vs-rest multi-class wrapper: detector set D_k treats
  class k as self, and prediction is ŷ = argmin_k Φ_k(x) with nearest-self tie-breaking.
- **`data_pipeline.py`** — NSL-KDD and UNSW-NB15 preprocessing (one-hot encoding, min–max
  scaling, PCA to 15 components), matching Section 4.1 of the paper.
- **Spiking compilation** (`run_snn_*.py`) — each detector (c, r) becomes one
  integrate-and-fire neuron on the augmented input [x, ‖x‖²] with weights [2c, −1] and
  threshold ‖c‖² − r², so the network is written from the detectors without training.
  Inputs are centred on the self mean and sent as one time-to-first-spike per coordinate;
  self-tolerance is restored after quantization by shrinking each radius until it covers no
  quantized self training sample. Energy is an analytical 45 nm estimate (0.9 pJ per
  accumulate, 4.6 pJ per multiply-accumulate), not a hardware measurement.

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # Linux/macOS
```

Then place the datasets as described in [`data/README.md`](data/README.md) — they are not
redistributed here, and SHA-256 checksums of the exact files used for the paper are listed
there.

To check the committed outputs against the paper's headline numbers, without downloading a
dataset or re-running an experiment:

```
python verify_numbers.py
```

## Reproducing the paper, table by table

Run `run_ablation_nslkdd.py` first: two of the other scripts read its per-fold output to
compute paired significance tests against V3.

| Paper artefact | Script | Output |
|---|---|---|
| Table 1 (feature dimensionality) | `run_feature_dim_sweep.py` | `results/nslkdd_feature_dim.csv` |
| Table 3 (per-fold CV), Table 6 (ablation V0–V4), Section 4.9 significance, Table 5 and Figure 4 (per-class), KNN and K-means rows of Table 4 | `run_ablation_nslkdd.py` | `results/nslkdd_ablation_*.csv`, `nslkdd_significance.csv`, `nslkdd_per_class_folds.csv`, `nslkdd_baselines.csv` |
| Fixed-radius NSA row of Table 4, and its radius sweep | `run_fixed_radius_nsa.py` | `results/nslkdd_fixed_radius*.csv` |
| Random-forest and gradient-boosted-tree rows of Table 4 | `run_supervised_baselines.py` | `results/nslkdd_supervised_*.csv` |
| Figure 5 (coverage sweep) | `run_coverage_sweep.py` | `results/nslkdd_coverage_sweep.csv` |
| Section 4.10 and Figure 7 (UNSW-NB15) | `run_unsw.py` | `results/unsw_*.csv` |
| Figures 4–7 (rendered from the CSVs) | `generate_figures.py` | `figures/*.png` |

### Spiking implementation and canonical split

Run `run_snn_guard.py` first: it fits the detector sets on the same ten folds as
`run_ablation_nslkdd.py`, caches them in `results/snn_detectors_pooled.npz` (not committed,
about 2 MB) and the other pooled spiking scripts read that cache.

| Paper artefact | Script | Output |
|---|---|---|
| Table 8 (false-alarm inflation and guards), Table 9 (detector-set size) | `run_snn_guard.py` | `results/snn_guard_*.csv` |
| Table 7 (spike encodings) | `run_snn_encodings_pooled.py` | `results/snn_encodings_pooled*.csv` |
| Table 10 (UNSW-NB15); Section 4.14 exact detector at the self-tolerant radii; matched comparison with uniform shrinking (Sections 4.14 and 4.16) | `run_snn_rigor.py` | `results/snn_rigor_*.csv` |
| Section 3.7 exactness check and development runs (canonical split, unguarded pooled folds, operating-point check) | `run_snn_compile.py`, `run_snn_encodings.py`, `run_snn_pooled.py`, `run_snn_operating_point.py` | `results/snn_compile_canonical.csv`, `snn_encodings_canonical.csv`, `snn_pooled_*.csv`, `snn_operating_point.csv` |
| Section 5, canonical KDDTrain+/KDDTest+ split, unseen attack types, hybrid sweep | `run_canonical_split.py`, `run_canonical_unseen.py`, `run_canonical_hybrid.py` | `results/nslkdd_canonical_*.csv` |
| Figures 3 and 8 | `generate_snn_figures.py` | `figures/figure3_*`, `figures/figure8_*` (PNG, PDF, SVG) |

Approximate wall-clock on the 4-core laptop specified in Section 4.2: the ablation is the
long one, roughly 45–60 minutes for five variants over ten folds; the feature-dimension
sweep about 30 minutes, dominated by the 122-dimensional configuration; the coverage sweep
about 10 minutes; the rest are minutes.

## Determinism

All runs use fixed seeds (`RNG_SEED = 0`, per-fold seeds derived as `seed * 1000 + fold`),
`StratifiedKFold(10, shuffle=True, random_state=0)`, and preprocessing fitted with
`random_state=0`. Library versions are pinned in `requirements.txt`. The reported results
were produced on Python 3.14, except the fixed-radius NSA baseline and the
feature-dimension sweep, which ran on Python 3.12 with identical library versions.

## Protocol notes

Repeated here from the paper so the numbers are not misread:

- All cross-validated NSL-KDD results pool KDDTrain+ and KDDTest+ (148,517 records) and
  partition the pool into stratified folds. This is **not** the canonical
  KDDTrain+/KDDTest+ split with its deliberate train/test distribution shift. Values here
  are optimistic relative to that protocol and are not directly comparable with results
  published on it.
- The method consumes labelled non-self samples during training, for coverage probes and
  for boundary-point identification, so it is not a purely one-class method in the way the
  classical negative selection algorithm is.
- The KNN and K-means rows of Table 4 come from a single stratified 80/20 split; the
  fixed-radius NSA, random-forest and gradient-boosted-tree rows use the same ten folds as
  IVDA and are directly comparable with it.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

Archived releases are on Zenodo; v1.0.0 is
[10.5281/zenodo.21880843](https://doi.org/10.5281/zenodo.21880843), and each later release
receives its own DOI under the same record. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff). Journal details will be added on acceptance.
