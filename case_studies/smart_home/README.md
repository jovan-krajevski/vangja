# Smart Home Ablation Study

Ablation study evaluating the impact of transfer learning hyperparameters on short time series forecasting using the smart home energy dataset.

## Setup

```bash
# From the repository root
conda activate pymc_env  # or your PyMC environment
pip install -e ".[reproducibility]"
```

## Quick Start

### 1. Run classical baselines (~1 minute)

```bash
python case_studies/smart_home/classical_baselines.py
```

Fits ARIMA, Holt-Winters, and Seasonal Naive models per-series. Results go to `results_classical/`.

### 2. Run the full ablation study (several hours)

```bash
python case_studies/smart_home/run.py
```

Results go to `metrics.csv`.

## Experiment Design

**Base model**: Temperature data (Boston, 2014–2016) trained with NUTS to learn yearly seasonality.

**Target model**: Smart home energy data (4 appliance series, ~3 months training) with transfer learning from the base model.

## Running on HPC

See `case_studies/README.md` for the full Singularity + SLURM setup guide. Quick usage:

```bash
# Submit to SLURM
sbatch case_studies/submit_ablation.slurm smart_home fast
sbatch case_studies/submit_ablation.slurm smart_home full
sbatch case_studies/submit_ablation.slurm smart_home classical
```
