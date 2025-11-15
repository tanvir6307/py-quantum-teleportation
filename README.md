# Quantum Teleportation & GHZ Secret-Sharing Simulation Suite

**Repository purpose:** code supporting simulations and analysis for a research study comparing decoherence/noise models (Markovian thermal relaxation vs. Gaussian / collective dephasing / amplitude damping) in single-qubit teleportation (Bell circuit) and 3-qubit GHZ-based secret-sharing protocols. These scripts were developed to produce the figures and data used in the associated research paper.

---

## Contents

- `Quantum Teleportation.py` — Simulation & analysis pipeline for single-qubit teleportation under competing noise models (Markovian thermal relaxation vs Gaussian phase-kicks). Generates fidelity vs time curves, heatmaps, overlays, error bands, uncertainty surfaces, FFT panels and a variety of diagnostic plots and CSV output.
- `GHZ-Based Secret Sharing.py` — Simulation & analysis pipeline for GHZ-based secret-sharing under amplitude-damping (local T1-like) and collective dephasing (Ornstein–Uhlenbeck correlated phase kicks). Produces per-parameter fidelity traces, sensitivity scans, overlays and summary tables useful for comparing noise models.
- `results/` or `results_GHZ_final/` — Default output directories (created at runtime) containing generated `.png` figures and `.csv` summary tables.

---

## Key features

- Physics-inspired Hamiltonian stepping with optional custom potential (`V_func`).
- Two families of noise models: Markovian thermal relaxation (qiskit thermal_relaxation_error) and Gaussian/collective stochastic dephasing implemented as per-step phase kicks or OU-process phases.
- Monte Carlo sampling capability for Gaussian/collective noise and configurable realizations count.
- Automatic generation of publication-ready figures: time-series, overlays, heatmaps, contour plots, error bands, Bland–Altman, FFTs, sensitivity quivers and more.
- Exports CSV summary files (`combined_results_*.csv`) that can be used for downstream plotting or statistical analysis.

---

## Requirements

Tested with Python 3.11.+ and the following packages (install with `pip`):

```bash
pip install numpy pandas matplotlib qiskit qiskit-aer scipy tqdm
```

Notes:
- `qiskit-aer` is used for density-matrix simulations (`AerSimulator`). If you cannot install `qiskit-aer`, parts of the pipeline that require the Aer density-matrix simulator will fail. You can still inspect the code for algorithms and use small local modifications for statevector-only runs.
- `scipy` is optional but improves heatmap smoothing and interpolation (the scripts check for it and fall back gracefully).

---

## Configuration & quick edits

Both files expose a `default_runconfig()` (or `RunConfig` dataclass) near the top of the file. Edit this section to:
- change the time vector `times_us` (scan resolution / total time),
- update `T2_list_us` and `lambda_list_rad_per_us` parameter grids,
- toggle Monte Carlo realizations (`mc_realizations_gaussian` / `mc_realizations_collective`),
- set the results directory (e.g., `results_dir="my_results"`),
- modify physics parameters (`HParams.omega`, `lambda_max`, `drive_freq`).
Example: to run a tiny smoke test, open either file and set:

```py
cfg = default_runconfig()
cfg.scan.times_us = np.linspace(0.0, 10.0, 11)
cfg.scan.T2_list_us = [80.0]
cfg.scan.lambda_list_rad_per_us = np.array([0.0, 0.062832])
cfg.mc_realizations_gaussian = 4  # fewer Monte Carlo samples
cfg.results_dir = "smoke_results"
```

---

## Output (what to expect)

- Per-model CSV files such as `results_<tag>_gaussian.csv`, `results_<tag>_markovian.csv` and a concatenated `combined_results_<tag>.csv` containing columns like `time_us`, `T2_us`, `lambda_max_rad_per_us`, `fidelity_mean`, `fidelity_std`.
- Figures saved under subfolders created by the scripts, including:
  - `Mean fidelity` / `heatmaps` (contour/colored heatmaps)
  - `Overlay` time-series (Gaussian vs Markovian or collective vs ampdamp)
  - `Error bands`, `FFT`, `Sensitivity`, `BlandAltman`, `CV` bubble charts, `Phase diagrams`, etc.
These are designed to be publication-ready but you may want to tweak matplotlib figure size, DPI, fonts or labels in the plotting functions to match journal requirements.

---


## Development & testing tips

- Start with small scans (shorter time arrays and fewer T2/λ values) to validate that the pipeline runs on your machine.
- Use `logging` level changes (`logging.basicConfig(level=logging.DEBUG, ...)`) for more verbose runtime diagnostics.
- The scripts check for `scipy` and `tqdm` and will fall back gracefully if missing; still, installing them improves usability.

---

