# Reproducing Ablation Studies on HPC Clusters

This directory contains infrastructure for running vangja ablation studies on High Performance Computing (HPC) clusters using Singularity (SyLabs) containers and the SLURM workload manager. The containerized setup ensures **bit-for-bit reproducibility** of all experiments reported in the accompanying paper.

## Directory Structure

```
case_studies/
├── README.md                 # This file
├── singularity.def           # Container definition (Miniconda + PyMC + vangja)
├── submit_ablation.slurm     # Generic SLURM batch script for any case study
├── smart_home/               # Smart home energy dataset case study
│   ├── ablation_fast.py      # Fast pipeline validation (~minutes)
│   ├── ablation_full.py      # Full hyperparameter sweep (~hours)
│   ├── classical_baselines.py
│   ├── train.py
│   ├── ablations.md
│   ├── general_ablations.md  # Reusable template for new datasets
│   └── README.md
└── <other_datasets>/         # Additional case studies follow the same layout
```

## Prerequisites

- A local machine with [Singularity / Apptainer](https://apptainer.org/) installed (for building the container image)
- Access to an HPC cluster with:
  - SLURM workload manager
  - Singularity / Apptainer container runtime
  - At least 32 GB RAM per node (for MCMC sampling)
- Basic familiarity with the Linux command line and SSH

## Step-by-Step Reproduction Guide

### 1. Build the Singularity Image (Local Machine)

Building the container image requires root privileges, which are typically unavailable on HPC login nodes. **Build the image on your local machine**, then transfer it to the cluster.

```bash
# Clone the repository
git clone https://github.com/jovan-krajevski/vangja.git
cd vangja

# Build the Singularity image (~10–15 minutes)
sudo singularity build case_studies/vangja.sif case_studies/singularity.def
```

The resulting `vangja.sif` file (~2–4 GB) is a self-contained, immutable image that bundles:

| Component | Version |
|-----------|---------|
| Python | 3.13 |
| PyMC | >= 5.27.1 |
| vangja | current (with `[reproducibility]` extras) |
| statsmodels | >= 0.14 |
| matplotlib | >= 3.7 |
| seaborn | >= 0.12 |
| kagglehub | >= 0.3 |

Verify the image locally before transferring:

```bash
singularity exec case_studies/vangja.sif python -c "
import vangja, pymc, arviz, statsmodels
print(f'vangja {vangja.__version__}')
print(f'PyMC {pymc.__version__}')
print(f'ArviZ {arviz.__version__}')
print(f'statsmodels {statsmodels.__version__}')
"
```

### 2. Transfer the Repository and Image to the HPC Cluster

```bash
# Transfer the image
scp case_studies/vangja.sif <user>@<cluster>:<work_dir>/vangja/case_studies/

# Alternatively, clone the repository on the cluster and copy only the image
ssh <user>@<cluster>
cd <work_dir>
git clone https://github.com/jovan-krajevski/vangja.git
# Then scp the .sif file into case_studies/
```

**Storage recommendations:**

| Directory | Use |
|-----------|-----|
| `$HOME` | Repository code (backed up, limited quota) |
| `$SCRATCH` or `$WORK` | Container image + results (high quota, fast I/O) |

If your cluster separates code from data storage, symlink the results directories:

```bash
mkdir -p $SCRATCH/vangja_results/smart_home
ln -s $SCRATCH/vangja_results/smart_home case_studies/smart_home/results
```

### 3. Configure Git Access on the Cluster (Optional)

If you need to pull updates or push results from the cluster:

```bash
# Option A: SSH key forwarding (recommended)
# On your local machine, add to ~/.ssh/config:
#   Host <cluster>
#       ForwardAgent yes
ssh-add ~/.ssh/id_ed25519
ssh <user>@<cluster>

# Option B: Generate a deploy key on the cluster
ssh-keygen -t ed25519 -C "hpc-cluster"
cat ~/.ssh/id_ed25519.pub
# Add this key to GitHub → Settings → SSH and GPG keys

# Verify access
git ls-remote git@github.com:jovan-krajevski/vangja.git
```

### 4. Run Ablation Studies

#### Quick Validation (Fast Mode)

Validate the full pipeline with a reduced hyperparameter grid before committing to the full run:

```bash
cd <work_dir>/vangja
sbatch case_studies/submit_ablation.slurm smart_home fast
```

This runs `ablation_fast.py` — two series, VI instead of NUTS, fewer hyperparameter combinations. Completes in minutes.

#### Full Ablation Study

```bash
sbatch case_studies/submit_ablation.slurm smart_home full
```

This runs `ablation_full.py` — four series, NUTS sampling for base models, exhaustive hyperparameter grid. Expect several hours of wall time.

#### Classical Baselines

```bash
sbatch case_studies/submit_ablation.slurm smart_home classical
```

Fits ARIMA, Holt-Winters, and Seasonal Naive models per-series. Completes in minutes.

#### Monitoring Jobs

```bash
squeue -u $USER                    # List your queued/running jobs
tail -f logs/ablation_<jobid>.out  # Follow live output
scancel <jobid>                    # Cancel a job
```

### 5. Checkpointing and Crash Recovery

Both `ablation_fast.py` and `ablation_full.py` implement automatic checkpointing to survive node failures, job timeouts, and out-of-memory conditions:

- **Base model caching**: After each NUTS/VI base model is trained, its ArviZ `InferenceData` trace and `t_scale_params` are saved to disk under `results*/base_models/`. On restart, cached base models are loaded instead of retrained.
- **Per-experiment results**: After each target model experiment completes, its metrics are appended to an incremental `results_checkpoint.csv`. On restart, completed experiments are skipped automatically.
- **Atomic writes**: Checkpoint files are written atomically (write to temp file, then rename) to prevent corruption from mid-write crashes.

If a job is interrupted, simply resubmit the same command — the script will resume from where it left off:

```bash
# Job timed out or crashed? Just resubmit:
sbatch case_studies/submit_ablation.slurm smart_home full
```

### 6. Customizing SLURM Parameters

Edit `submit_ablation.slurm` or override via `sbatch` flags:

```bash
# Request more time and memory
sbatch --time=48:00:00 --mem=64G case_studies/submit_ablation.slurm smart_home full

# Target a specific partition or node
sbatch --partition=long --nodelist=node01 case_studies/submit_ablation.slurm smart_home full
```

Key SLURM parameters to tune:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--time` | 24:00:00 | NUTS sampling can be slow; increase for large grids |
| `--cpus-per-task` | 8 | PyMC/JAX can use multiple cores for parallel chains |
| `--mem` | 32G | Increase if MCMC chains are large or many series |

### 7. Collecting Results

After all jobs complete, results are stored in the case study's output directory:

```
case_studies/smart_home/results/
├── results.csv              # Full results table (one row per configuration)
├── results_checkpoint.csv   # Incremental checkpoint (same content as results.csv)
├── summary_rmse.png         # Bar chart ranking all configurations
├── hparam_*.png             # Per-hyperparameter effect plots
├── pred_*.png               # Prediction plots for selected configurations
├── best_*.png               # Best model per series with uncertainty
└── base_models/             # Cached base model traces
    ├── base_so=5_bsd=1.0_sc=minmax.nc
    ├── base_so=5_bsd=1.0_sc=minmax_tscale.json
    └── ...
```

## Adding New Case Studies

To create an ablation study for a new dataset:

```bash
mkdir case_studies/my_dataset

# Copy the template scripts
cp case_studies/smart_home/ablation_fast.py case_studies/my_dataset/
cp case_studies/smart_home/ablation_full.py case_studies/my_dataset/
cp case_studies/smart_home/classical_baselines.py case_studies/my_dataset/
```

Then modify the data loading section, hyperparameter grid, and train/test split. See `case_studies/smart_home/general_ablations.md` for a detailed template.

Run the new case study using the same SLURM script:

```bash
sbatch case_studies/submit_ablation.slurm my_dataset fast
sbatch case_studies/submit_ablation.slurm my_dataset full
```

## Reproducibility Checklist

For submission to journals with strict reproducibility requirements (e.g., Journal of Statistical Software):

- [ ] `singularity.def` pins the base Docker image tag (`continuumio/miniconda3:24.7.1-0`)
- [ ] All Python dependencies are version-pinned in `pyproject.toml`
- [ ] Random seeds are set (`np.random.seed(42)`) before stochastic operations
- [ ] The container image is built from `singularity.def` and verified with `%test`
- [ ] All experiments use the same `vangja.sif` image
- [ ] Base model MCMC traces are cached to disk for deterministic transfer learning
- [ ] Checkpointing ensures restarts produce identical results (completed experiments are not re-run)
- [ ] Results CSV files contain all hyperparameters alongside metrics for full traceability
- [ ] The `vangja.sif` image SHA256 hash is recorded: `sha256sum case_studies/vangja.sif`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `singularity: command not found` | Load the module: `module load singularity` or `module load apptainer` |
| Permission denied building `.sif` | Build locally with `sudo`, then `scp` to cluster |
| Out of memory during NUTS | Increase `--mem` in SLURM or reduce `BASE_NUTS_CHAINS` |
| Job timeout | Increase `--time`; checkpointing will resume from last completed experiment |
| `kagglehub` download fails (no internet on compute nodes) | Run `ablation_fast.py` once on a login node to cache datasets, then submit SLURM jobs |
| Results differ across runs | Ensure the same `vangja.sif` is used and random seeds are set; MCMC is inherently stochastic but base model caching ensures deterministic transfer |
