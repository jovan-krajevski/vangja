# Smart Home Ablation Study

Ablation study evaluating the impact of transfer learning hyperparameters on short time series forecasting using the smart home energy dataset.

## Setup

```bash
# From the repository root
conda activate pymc_env  # or your PyMC environment
pip install -e ".[reproducibility]"
```

## Quick Start

### 1. Validate the pipeline (fast, ~2 minutes)

```bash
python case_studies/smart_home/ablation_fast.py
```

This runs a reduced hyperparameter grid (2 series, VI instead of NUTS, fewer combinations) to verify everything works. Results go to `results_fast/`.

### 2. Run classical baselines (~1 minute)

```bash
python case_studies/smart_home/classical_baselines.py
```

Fits ARIMA, Holt-Winters, and Seasonal Naive models per-series. Results go to `results_classical/`.

### 3. Run the full ablation study (several hours)

```bash
python case_studies/smart_home/ablation_full.py
```

To see the grid size without running:

```bash
python case_studies/smart_home/ablation_full.py --dry-run
```

Results go to `results/`.

## Files

| File | Description |
|------|-------------|
| `train.py` | Original training script (single configuration) |
| `ablation_fast.py` | Fast pipeline validation (reduced grid, VI, 2 series) |
| `ablation_full.py` | Full ablation study (NUTS base model, exhaustive grid) |
| `classical_baselines.py` | ARIMA, Holt-Winters, Seasonal Naive comparison |
| `ablations.md` | Original experiment design document |
| `general_ablations.md` | Reusable template for other datasets |
| `README.md` | This file |

## Experiment Design

**Base model**: Temperature data (Boston, 2014–2016) trained with NUTS to learn yearly seasonality.

**Target model**: Smart home energy data (4 appliance series, ~3 months training) with transfer learning from the base model.

### Hyperparameters Swept

| Component | Parameter | Values |
|-----------|-----------|--------|
| Base (yearly) | `series_order` | 5, 10 |
| Base (yearly) | `beta_sd` | 0.1, 1, 10 |
| Both | `scaler` | "minmax", "standard" |
| Target (yearly) | `tune_method` | "parametric", "prior_from_idata" |
| Target (all) | `pool_type` | "individual", "partial" |
| Target (yearly) | `loss_factor_for_tune` | -1, -0.5, 0, 0.5, 1 |
| Target (all, partial only) | `shrinkage_strength` | 1, 10, 100, 1000 |
| Target (constant) | constant type | None, Uniform, Beta, Normal |
| Target | extra seasonalities | None, monthly, quarterly, both |

### Baselines

- Vangja without transfer learning (with/without yearly seasonality)
- Seasonal Naive (period 7 and 30)
- ARIMA(1,1,1) with and without seasonal(1,0,1,7)
- Holt-Winters (additive, period 7 and 30)

## Outputs

Each script generates:

- **CSV results** — one row per configuration with all metrics
- **Checkpoint CSV** — incremental `results_checkpoint.csv` for crash recovery
- **Base model cache** — `base_models/` directory with MCMC/VI traces (NetCDF) and `t_scale_params` (JSON)
- **Summary plots** — RMSE bar charts ranking configurations
- **Per-hyperparameter plots** — effect of each hyperparameter on RMSE
- **Prediction plots** — predicted vs. actual for selected series

## Checkpointing and Crash Recovery

Both `ablation_fast.py` and `ablation_full.py` support automatic checkpointing. If a run is interrupted (SLURM timeout, OOM, etc.), simply re-run the same command — completed experiments will be skipped and cached base models will be loaded from disk instead of re-trained.

## Running on HPC

See `case_studies/README.md` for the full Singularity + SLURM setup guide. Quick usage:

```bash
# Submit to SLURM
sbatch case_studies/submit_ablation.slurm smart_home fast
sbatch case_studies/submit_ablation.slurm smart_home full
sbatch case_studies/submit_ablation.slurm smart_home classical
```

## Adapting to Other Datasets

See `general_ablations.md` for a reusable template. The key changes are:

1. Replace data loading functions
2. Adjust train/test split
3. Choose appropriate trend component (FlatTrend vs LinearTrend)
4. Set seasonality periods for your domain
