# Reproducing Ablation Studies on HPC Clusters

This directory contains infrastructure for running vangja ablation studies on High Performance Computing (HPC) clusters using Singularity (SyLabs) containers and the SLURM workload manager. The containerized setup ensures reproducibility of all experiments reported in the accompanying paper.

## Directory Structure

```
case_studies/
├── README.md                       # This file
├── singularity.def                 # Container definition (Miniconda + PyMC + vangja)
├── singularity_gpu.def             # GPU-enabled container definition (optional)
├── submit_ablation.slurm           # Generic SLURM batch script for any case study
├── submit_ablation_gpu.slurm       # SLURM script for GPU-enabled ablation studies (optional)
├── <case_study_1>/                 # Case study
│   ├── run_vangja.py               # Full hyperparameter sweep (~hours)
│   ├── run_baselines.py            # Classical baselines (~minutes)
|   ├── run_bayesian_workflow.py    # Bayesian workflow (~minutes)
|   ├── run_prophet.py              # Prophet baseline (~minutes)
|   ├── run_timeseers.py            # TimeSeers baseline (~minutes)
│   └── results/                    # Output directory for results and plots
└── <other_case_studies>/           # Additional case studies follow the same layout
```

## Prerequisites

- A local machine with [Singularity / Apptainer](https://apptainer.org/) installed (for building the container image)
- Access to an HPC cluster with:
  - SLURM workload manager
  - Singularity / Apptainer container runtime
  - At least 16 GB RAM per node (for MCMC sampling)
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

You can run the ablation studies by submitting SLURM jobs using the provided `submit_ablation.slurm` script. This script is designed to run any of the case studies. You need to specify the case study name (e.g., `smart_home`) and the pipeline type (`vangja`, for running the Vangja ablation pipeline, `baselines`, for running the classical baselines, `workflow` for running the Bayesian workflow, or `prophet` for running the Prophet baseline).

```bash
cd <work_dir>/vangja
sbatch case_studies/submit_ablation.slurm <case_study> <pipeline>
```

As an example, to run the full Vangja ablation study for the `smart_home` case study:

```bash
cd <work_dir>/vangja
sbatch case_studies/submit_ablation.slurm smart_home vangja
```

As for running the classical baselines for the same case study:

```bash
cd <work_dir>/vangja
sbatch case_studies/submit_ablation.slurm smart_home baselines
```

Similarly, you can run the other pipelines.

#### Monitoring Jobs

```bash
squeue -u $USER                    # List your queued/running jobs
tail -f logs/ablation_<jobid>.out  # Follow live output
scancel <jobid>                    # Cancel a job
```

### 5. Checkpointing and Crash Recovery

`run_vangja.py` implements automatic checkpointing to survive node failures, job timeouts, and out-of-memory conditions. After each target model experiment completes, its metrics are appended to an incremental CSV. On restart, completed experiments are skipped automatically.

If a job is interrupted, simply resubmit the same command — the script will resume from where it left off.

### 6. Customizing SLURM Parameters

Edit `submit_ablation.slurm` or override via `sbatch` flags:

```bash
# Request more time and memory
sbatch --time=48:00:00 --mem=64G case_studies/submit_ablation.slurm <case_study> <pipeline>

# Target a specific partition or node
sbatch --partition=long --nodelist=node01 case_studies/submit_ablation.slurm <case_study> <pipeline>
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
case_studies/<case_study_1>/results/
├── vangja/                         # Vangja ablation results
    ├── metrics_<start_date>.csv    # Metrics for all configurations on a given start date in the dataset
    └── ...                         # Other Vangja-specific outputs for analysis
├── baselines/                      # Classical baselines results
    ├── metrics_<start_date>.csv    # Metrics for all baselines on a given start date in the dataset
    └── ...                         # Other baseline-specific outputs for analysis
├── workflow/                       # Bayesian workflow results
    ├── <artifact>                  # Artifacts from the Bayesian workflow
    └── ...                         # Other workflow-specific outputs for analysis
└── prophet/                        # Prophet baseline results
    ├── metrics_<start_date>.csv    # Metrics for the Prophet baseline on a given start date in the dataset
    └── ...                         # Other Prophet-specific outputs for analysis
└── timeseers/                      # TimeSeers baseline results
    ├── metrics_<start_date>.csv    # Metrics for the TimeSeers baseline on a given start date in the dataset
    └── ...                         # Other TimeSeers-specific outputs for analysis
```

## Adding New Case Studies

To create an ablation study for a new dataset:

```bash
mkdir case_studies/my_dataset

# Create the pipelines
touch case_studies/my_dataset/run_vangja.py
touch case_studies/my_dataset/run_baselines.py
touch case_studies/my_dataset/run_bayesian_workflow.py
touch case_studies/my_dataset/run_prophet.py
touch case_studies/my_dataset/run_timeseers.py
```

Then create the data loading section, hyperparameter grid, and train/test split. See the other case studies for a detailed template.

Run the new case study using the same SLURM script:

```bash
sbatch case_studies/submit_ablation.slurm my_dataset vangja
sbatch case_studies/submit_ablation.slurm my_dataset baselines
sbatch case_studies/submit_ablation.slurm my_dataset workflow
sbatch case_studies/submit_ablation.slurm my_dataset prophet
sbatch case_studies/submit_ablation.slurm my_dataset timeseers
```

## Reproducibility Checklist

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
| Results differ across runs | Ensure the same `vangja.sif` is used and random seeds are set; remember that MCMC is inherently stochastic |
