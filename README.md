# Flex-Compress

**How compressible is protein dynamics, can a compact profile of it be learned, and how much of that profile is predictable from structure and sequence?**

Flex-Compress is an empirical study on the mdCATH and ATLAS molecular dynamics datasets. Its goal is a *rich, compressed dynamics profile* of a protein: one representation covering

- per-residue flexibility amplitude,
- anisotropy,
- direction-dependent timescales,
- residue–residue coupling, resolved by timescale,
- collective, domain-like motion,
- non-equilibrium behaviour (for example, high-temperature unfolding).

The project tests whether such a profile exists, and how small it can be, before any large model is built to predict it.

Data access, archive decoding and splits come from [`DynaMode-dev`](DynaMode-dev), a **read-only git submodule tracking its `dev` branch**. Flex-Compress never edits it. Each run freezes a copy of the code it used, so later changes never alter a running study.

---

## 1. Framing

### A profile of the process, not of one trajectory

An MD trajectory has two parts:

- the protein's stochastic dynamics, which are the same in every replica;
- one particular sequence of thermal kicks, which is unique to that run.

The second part cannot be compressed, and a codec that reconstructs trajectories frame by frame spends its bits on it. The target here is the **law of the process**, given the structure and the conditions (temperature, and the starting state for non-equilibrium runs). Two consequences follow:

- **Replica invariance is the test.** Independent replicas of the same protein should map to the same profile. ATLAS has 3 replicas per protein; mdCATH has 5 per temperature.
- **Cross-replica scoring and a noise floor.** Every codec is fitted on some replicas and scored on a held-out one. The *replica floor* is the distance between the pooled raw statistics of the fit replicas and the held-out replica. A distortion ratio of 1 means "as good as the data allow". A ratio below 1 means the compressed model is closer to a new replica than the raw statistics are: it is denoising rather than memorising.

The profile can later be used *to describe a run*. That use is explicitly a **stochastic hypothesis**: a surrogate trajectory sampled to be consistent with the profile, ideally a maximum-entropy sample. It is not a claim to emulate MD. That sampler is future work; this repository is about the profile.

### The physics constrains the representation

- **No lead/lag at equilibrium.** Equilibrium MD is time-reversible, and positions do not change sign under time reversal. Onsager reciprocity therefore gives `C_ij(τ) = C_ji(τ)`: the lagged covariance is symmetric, the cross-spectral density is real, and "residue *i* moves, then *j* responds" is zero in expectation. Ordering only becomes real
  - away from equilibrium (unfolding from the native state at high temperature), or
  - conditional on a transition (A→B paths have an order that B→A paths mirror).

  The **antisymmetric part** of `C(τ)`, and whether it is reproducible across replicas, is therefore the irreversibility signal. It switches on for non-equilibrium runs inside the same representation.
- **Overdamped dynamics: relaxation times, not frequencies.** At the stored frame spacings, solvated backbone motion is overdamped. For reversible dynamics the second-order structure is a positive mixture of decaying exponentials:

  ```
  C(τ) = Σ_b exp(-τ / t_b) A_b ,      A_b ⪰ 0  (one 3N×3N amplitude matrix per relaxation band)
  ```

  This is Noé's *dynamical fingerprint*; per residue it generalises the Lipari–Szabo picture (S², τ). The diagonal 3×3 blocks of each `A_b` give:
  - amplitude (RMSF² = Σ_b tr A_b,ii),
  - anisotropy,
  - direction-dependent timescales ("x is fast, y is slow").

  The off-diagonal blocks give coupling resolved by timescale. DynaProt (ICLR 2026) predicts the `τ = 0` slice of this object (Σ_b A_b).
- **Your DCT energy measure, in these terms.** A relaxation process with time *t* puts a fraction `(2/π)·arctan(ω_c t)` of its variance below a DCT cutoff `ω_c`. Truncation energy loss is therefore a blurred reading of this spectrum. Cosines are a good measurement basis but a poor representation, and DCT phase carries no physical information at equilibrium.
- **Finite trajectories bias the estimator.** Subtracting each replica's own mean biases `Ĉ(τ)` downward when `t_b` is not much smaller than `T`, and removes non-decaying variance entirely. Instead of correcting the data, every model is fitted and scored through the **exact expected value of the estimator** for each band (`kernels.py`, O(T) per band). Fitted amplitudes are then process-level, and replicas of different lengths compare like for like.

### Local + collective ↔ residue tokens + collective tokens

The spectral codec has two parts:

- **Local:** a per-residue band-resolved 3×3 tensor, 6 numbers per band.
- **Collective:** a shared low-dimensional spatial subspace with a full PSD core per band.

This mirrors the hierarchical latent the learned model should eventually have (per-residue tokens plus a few collective/domain tokens). The classical version measures how much rate each part needs before any network is trained.

---

## 2. Questions and pass/fail criteria

All distortions are measured against a held-out replica and reported relative to the replica floor.

| | Question | Test |
|---|---|---|
| **Q1** | Is protein dynamics compressible? | Classical low-rank spectral fits reach distortion/floor ≈ 1 at small rank, per protein; the rank needed grows sublinearly with N. |
| **Q2** | Is there a shared, learnable code? | Per-residue descriptors pooled across proteins have a low-dimensional replica-reproducible subspace that transfers to held-out proteins. A learned encoder trained across proteins then reaches the floor on held-out folds at fewer bits than per-protein fits. |
| **Q3** | Is the profile predictable? | Fraction of *replica-reliable* variance explained from structure + sequence, by timescale band and temperature (R² / noise ceiling). |

---

## 3. Datasets and timescale windows

Both datasets are read from DynaMode-dev's **compressed all-backbone archives**: one `*.zarr.zip` per domain, holding N/CA/C/O coordinates in Å.

- **Alignment.** Two-pass native-CA Kabsch on the least-mobile half of residues; every replica shares the native orientation.
- **Quantisation.** cartesian12 codec at 0.01 Å. This adds white noise of 8×10⁻⁶ Å² per coordinate, which lands in the sub-frame band; the descriptor ε (10⁻⁴ Å²) sits above it.

| | ATLAS | mdCATH |
|---|---|---|
| Systems | 1,390 chains (ECOD X-level non-redundant) | 5,398 CATH domains (S20), 50–500 residues |
| Runs | 3 × 100 ns at 300 K | 5 replicas × 5 temperatures (320, 348, 379, 413, 450 K), mostly 500 ns |
| Frame spacing | 10 ps (stride-100 1 ns view also stored) | 1 ns |
| Force field | CHARMM36m / TIP3P | CHARMM22* / TIP3P |
| Resolved relaxation bands | sub-frame + 10 ps … 50 ns | sub-frame (<1 ns, aggregate) + 1 ns … 250 ns |
| Reliable lag range (≈ 2 frames … T/10) | ~20 ps – 10 ns | ~2 – 50 ns |

- **Motion faster than one frame still counts.** In mdCATH it appears as the sub-frame band: one 3×3 tensor per residue giving its total fast amplitude.
- **The datasets overlap only between ~2 and 10 ns, plus the aggregate fast amplitude.** Cross-dataset comparisons (Stage 3) must use that window and carry force-field and temperature covariates. `configs/stage01_atlas_1ns.yaml` gives ATLAS at mdCATH's 1 ns spacing.

---

## 4. Study plan

| Stage | Content | Status |
|---|---|---|
| **0** | Targets, replica noise floors and reliability; PCA and sampling diagnostics | **implemented** |
| **1** | Classical compressibility: rate–distortion of spectral, PCA and DCT codecs; rank needed vs N; cross-protein universality of per-residue descriptors | **implemented** |
| 2 | Learned codec: set encoder over replica windows → per-residue + collective tokens, structure-conditioned decoder, cross-replica targets, rate sweep | next |
| 3 | Predictability: structure + PLM embeddings → profile, scored as R² / noise ceiling by band and temperature; ENM / DynaProt / AlphaFlow / BioEmu / PLM-probe baselines; fold-level splits; mdCATH→ATLAS transfer | planned |

Stage 0–1 run on **the whole training split of each dataset**: all 1,266 ATLAS training chains (300 K) and all 4,304 mdCATH training domains at all five temperatures (21,520 targets). Validation and test proteins stay untouched for the predictability work.

### What Stage 0–1 computes (per domain × temperature)

**Statistics.** For each replica:

- lagged 3×3 blocks for every residue, on a log lag grid up to T/10;
- full lagged covariance on an evenly spaced subset of ≤128 residues;
- PC-space lagged covariance (top K ≤ 256 PCs of the fit replicas).

All statistics of a target are computed on windows of one **common length**: the shortest replica, by default. Each window is re-centred, and longer replicas use evenly spaced overlapping windows, so no frames are dropped. Fit replicas, the held-out replica, the floor and every model therefore share a single estimator expectation, even when mdCATH replica lengths differ. Without this, a short held-out replica makes models appear to beat the floor purely from length bias.

**Stage 0: reliability and diagnostics**

- Replica-vs-replica floor for every distortion metric, and correlations (log-MSF, anisotropy shape, DCCM, ACF by lag, lagged DCCM by lag). Spearman–Brown corrections give the noise ceiling.
- Reliability of each band's per-residue amplitude across replicas, and the reliable lag horizon.
- Spatial PCA counts for 50/80/90/95/99 % of variance: CA and full backbone, per replica and pooled. This tests the "40 PCs = 90 %" observation.
- **Hess cosine content** of the top PCs. Values near 1 mean undersampled diffusion, not intrinsic low dimension.
- **RMSIP** between replicas' top-10 and top-20 PC subspaces.
- Between-replica mean shift, half-trajectory drift, and native-contact fraction Q (flags unfolding).
- Antisymmetric/symmetric ratio and its replica-to-replica reproducibility (irreversibility).

**Stage 1: leave-one-replica-out rate–distortion.** Codecs:

| Codec | What it stores | Parameters / residue |
|---|---|---|
| `spectral_subspace_local` (primary) | top-R spatial subspace + PSD core per band + per-residue band tensors | 3R + B·R(R+1)/(2N) + 6B |
| `spectral_subspace` | collective part only | 3R + B·R(R+1)/(2N) |
| `spectral_band_local` | top-R (band, pattern) pairs + local | 3R + 6B |
| `pca_exp` | static PCA, one relaxation time per PC | ≈ 3R |
| `dct` | truncated DCT of each fit replica's *realisation* (DynaMode-style) | 3k × n_fit |

Distortion metrics:

- MSF (RMSF²) profile, log-MSF, anisotropy shape;
- per-residue autocorrelation, and its direction-resolved version;
- lag-0 and lagged cross-correlation;
- full lagged covariance.

`rel_max` is the worst of them. Rank needed to reach 1.0 and 1.1 × floor is reported per protein, with its log-log scaling against N (bootstrap CI). The **full-rank model's distortion/floor** checks whether the exponential-band family is adequate: ratios well above 1 point to non-stationarity or oscillation.

**Universality (Q2 precursor).** Per residue and per replica, the band tensors are fitted on a fixed per-source band grid, rotated into structure-fixed residue frames, and embedded log-Euclidean (6 numbers per band). Pooled over proteins:

- the **cross-replica covariance** isolates replica-reproducible variance (noise is independent across replicas);
- its eigenspectrum gives the number of shared dimensions needed for 80/90/95/99 % of reliable variance;
- a 2-fold protein split checks that these dimensions transfer to held-out proteins.

**Fitting.** PSD-constrained least squares over the band amplitudes, solved by ADMM with eigenvalue projection. Ridge 1e-3 picks the minimum-norm split between nearly collinear adjacent bands. Treat the fitted spectrum as smoothed: variance within a timescale window is well determined, the split between neighbouring half-decade bands is not.

Validated on synthetic OU mixtures with known spectra (`tests/`, 28 tests):

- the exact estimator kernel matches the brute-force quadratic form;
- windowed estimates of long replicas match the common-length kernel, and unequal replica lengths leave model/floor ≈ 1;
- band variance in each timescale window is recovered;
- subspace rank equal to the true collective rank reaches the floor, while per-band truncation at the same rank does not;
- irreversibility is detected only for a non-reversible process.

---

## 5. Running

The launchers default to **Isambard-AI** (project u6xq, `ssh u6xq.aip2.isambard`, root `/projects/u6xq/hew`). The Oxford HTC cluster is still supported with `FLEXC_SITE=htc`. Each site profile in `sbatch/sites/` holds that cluster's data and run paths, conda setup, per-job parallelism and Slurm requests. Every value can be overridden with an environment variable.

| | Isambard-AI (default) | HTC (`FLEXC_SITE=htc`) |
|---|---|---|
| Archives | `/projects/u6xq/hew/DATA/{ATLAS,mdCATH}/compressed_bb_zarr` | `/data/stat-opig/$USER/DATA/{ATLAS,mdCATH}/compressed_bb_zarr` |
| Runs | `/projects/u6xq/hew/flex-compress-runs` | `/data/stat-opig/$USER/DATA/flex-compress-runs` |
| Conda | `/projects/u6xq/hew/miniforge3`; first env that activates: `$FLEXC_ENV`, `$DYNAMODE_ENV`, **`dynamode-gh200`**, `flexc` | `~/hpc_env.sh`; `$FLEXC_ENV`, `$DYNAMODE_ENV`, `dynamode-v100` |
| One array task | `--nodes=1 --gpus=1`: one whole GH200 (72 Arm cores, ~115 GB, H100), 24 h | `--partition=long`, 4 CPUs, 24 GB, 24 h |
| Inside a task | up to 32 proteins in flight × 2 BLAS threads, admitted by memory (95 GB host / 80 GB GPU); linear algebra on the shared H100 | 1 target × 4 threads, numpy |
| Worker tasks per round (full / pilot) | 16 GH200s / 1 | 64 / 8 |
| Rounds before a run stops on its own | 4 × 24 h | 6 × 24 h |

Isambard-AI hands out whole superchips. Each task therefore does two things:

- **CPU:** a pool of worker processes (`worker --jobs`, longest targets first) keeps the 72 Grace cores busy with archive decoding and bookkeeping.
- **GPU:** each process sends its heavy linear algebra to the task's H100 (see *GPU acceleration* below).

The relevant limits (BriCS docs) are: `workq` partition, 24 h walltime, at most 1,000 array tasks, 256 running jobs.

### Setup (Isambard-AI, once)

```bash
ssh u6xq.aip2.isambard
cd /projects/u6xq/hew && git clone --recurse-submodules <flex-compress-remote> Flex-Compress
cd Flex-Compress && git -C DynaMode-dev checkout dev      # submodule tracks DynaMode-dev's dev branch
source /projects/u6xq/hew/miniforge3/bin/activate /projects/u6xq/hew/envs/dynamode-gh200
PYTHONPATH=src:DynaMode-dev/src python -m pytest -q      # ~1 min, synthetic data, includes torch-vs-numpy checks
sbatch sbatch/bench.sbatch                                # GPU vs CPU timing + agreement on one GH200
sbatch sbatch/bench.sbatch --throughput --devices cuda --jobs-list 8,16,32,48   # proteins/hour vs concurrency
```

The launchers activate `dynamode-gh200` from the project Miniforge automatically. It needs numpy, scipy, zarr<3, numcodecs<0.16, pyyaml, matplotlib and CUDA PyTorch, all of which DynaMode already requires. `bash sbatch/setup_env.sh` builds a separate CPU-only `flexc` env if you ever need one (`FLEXC_TORCH=1` adds CUDA PyTorch). Archives are read through `dynamode.dataloader.compressed_backbone`, which uses only numpy, scipy and zarr. If DynaMode's `trajectory_index.v2.json` exists under an archive root it is used; otherwise the requested archives are scanned directly from the login node during `configure`. For mdCATH's 4,304 training domains on Lustre, that scan takes a few minutes.

### Full training splits

```bash
bash sbatch/submit_train.sh                                   # ATLAS + mdCATH training splits, 16 GH200s each
bash sbatch/submit.sh configs/stage01_atlas.yaml              # one dataset (full split is the default)
bash sbatch/submit.sh configs/stage01_atlas.yaml pilot        # optional smoke test: 32 proteins, 1 GH200
```

Each dataset's training split becomes one frozen run that keeps going until every target is done. Nothing needs sizing by hand.

- **One shared queue.** The run has no fixed shards. Every worker task (one GH200) claims the largest target that is not finished and that no live worker holds, so all tasks stay busy until the split runs out and finish together. Claims are files under `$RUN/claims/`, created atomically. A claim whose worker stops refreshing it for 15 minutes is taken over.
- **Rounds.** Each round is a worker array of `FLEXC_SHARDS` tasks plus a finalize job that runs after it. The finalize job records progress in `$RUN/rounds.json`, submits the next round while targets remain, and aggregates what is done so far into `report/`. A run stops after `FLEXC_MAX_ROUNDS` rounds, or as soon as a round neither completes nor fails a single target (something is wrong: check `logs/` and `errors/`).
- **Walltime.** A worker starts a protein only if it is predicted to finish before the walltime. The prediction comes from the wall time per unit cost of proteins that worker has already finished. Workers therefore stop cleanly near 24 h and leave the rest of the split to the next round. Claims from a round that ended are voided by its finalize job, so nothing waits on them.
- **Coverage.** `configure` records every split domain that yields no target, with the reason: no archive, too few replicas with enough frames, or inconsistent replicas. `report/summary.json` has a `coverage` block covering split domains, eligible targets, done, failed and not started. Check progress at any time with:

```bash
PYTHONPATH=$RUN/code/src python -m flexcompress.run status --run-dir $RUN
```

- **More GPUs.** `FLEXC_SHARDS` (worker tasks per round) sets how fast a split finishes, not whether it does. Set it at submission; later rounds keep it.

`submit.sh` does five things:

1. creates `$FLEXC_RUNS_ROOT/<config>-<mode>-<timestamp>/`;
2. freezes the resolved config, the target list, the coverage record, provenance (git commits of both repos, package versions), and a code snapshot;
3. records the site choices (site, conda env, jobs × threads, memory budgets, rounds) in `site.env`, so workers, later rounds, resumes and re-aggregations match the submission (`FLEXC_*` variables set when resuming take precedence);
4. submits round 1 (`launch.sh`): the worker array;
5. submits that round's finalize job (`afterany`).

Common overrides are listed in the header of `sbatch/submit.sh`: `FLEXC_SHARDS`, `FLEXC_MAX_ROUNDS`, `FLEXC_JOBS` / `FLEXC_THREADS`, `FLEXC_MEM_BUDGET_GB` / `FLEXC_GPU_MEM_BUDGET_GB`, `FLEXC_DEVICE`, `FLEXC_MPS`, `FLEXC_TIME`, `FLEXC_RUNS_ROOT`, `FLEXC_ENV`.

### GPU acceleration

`src/flexcompress/accel.py` provides a PyTorch backend for every heavy kernel, chosen per process by `compute.device` in the config, or by `worker --device` / `FLEXC_DEVICE`:

| Setting | Meaning |
|---|---|
| `auto` | the H100 if torch sees a GPU, otherwise numpy |
| `cuda` | force the GPU |
| `numpy` | CPU reference |
| `torch-cpu` | the torch code path on the CPU (tests) |

The backend is not part of the result hash, so runs can be resumed on either. Each result records the backend it used under `compute`.

| On the GPU | Kernel |
|---|---|
| windowed lagged covariance sums | batched matmuls (`bmm`) |
| PC basis, pooled and backbone PCA, randomized SVD | `eigh` / `qr` / `svd` |
| ADMM band fits, per-residue 3×3 (most of the CPU time) | batched Jacobi rotations written as tensor ops (no cuSOLVER: its batched solver asked for 100–360 GB of workspace on real proteins) |
| ADMM band fits, collective K×K (K up to 256) | **Newton–Schulz PSD projection** (matmuls only) |
| subspace / per-band model predictions | `einsum` |
| DCT baseline | projection onto an explicit orthonormal DCT-II basis |

Archive decoding, metrics and aggregation stay on the CPU.

**Why Newton–Schulz.** For matrices larger than 32×32, batched `torch.linalg.eigh` on CUDA falls back to solving one matrix at a time ([pytorch#175585](https://github.com/pytorch/pytorch/issues/175585)). That would make the GPU no faster than the Grace cores. The projection instead uses P₊(S) = (S + S·sign(S))/2, with sign(S) from 40 Newton–Schulz steps. Eigenvalues above ~10⁻⁶‖S‖ converge, and smaller ones contribute errors no larger than themselves, which ADMM tolerates. The final solution gets one exact eigendecomposition, so it is always PSD. Mid-sized eigendecompositions on the GPU run in chunks of 4,096 matrices to bound cuSOLVER's workspace.

**Agreement.** The tests check every kernel against numpy, and the whole pipeline on `torch-cpu`. With exact projections the backends agree to rounding: ADMM takes the same number of iterations, and Stage-1 metrics differ by about 10⁻¹⁴. With Newton–Schulz, Stage-1 metrics agree to about 10⁻³ or better.

**Batched solves.** All statistics of a protein share one window length, so all folds share one kernel matrix. Per protein, Stage 1 therefore runs:

- one batched ADMM solve for the collective fits of all folds, shape (B, folds, K, K);
- one batched solve for the per-residue fits of every (fold, codec, rank), about 66 × N × B small 3×3 problems for ATLAS;
- one batched solve for the descriptor fits of all replicas.

Before, that was about 70 sequential small solves, each a long chain of tiny GPU kernels with a host synchronisation every iteration. Launches and synchronisations drop roughly 20×, and each launch carries enough work to occupy the H100. Each batch element is scaled separately, so a batched solve gives the same answer as solving each problem alone (tested across a 1000× spread of scales).

**Speed.** On CPU the torch path is already about 1.3–1.4× faster than numpy. The H100 numbers have not been measured yet:

- `sbatch sbatch/bench.sbatch` times one synthetic ATLAS-sized target on numpy and CUDA, and exits non-zero if they disagree by more than 1 %.
- `--throughput` runs the real worker over 24 synthetic proteins (N = 80–400) at several concurrencies and prints proteins/hour. Pick `FLEXC_JOBS` from that curve.

### Many proteins per GH200

Throughput per GH200 is limited by memory, not cores, so the worker schedules by memory:

- **Admission.** It estimates each protein's host and GPU footprint (decoded trajectories, backbone PCA working set, batched fit state, runtime and CUDA context). A protein starts only when its footprint fits the remaining budget.
- **Defaults.** On Isambard: up to `FLEXC_JOBS=32` in flight, `FLEXC_MEM_BUDGET_GB=95` of ~115 GB, `FLEXC_GPU_MEM_BUDGET_GB=80` of 96 GB.
- **Order.** Largest proteins go first. Small ones backfill while a large one waits. After 60 s the waiting protein's memory is reserved so it can't starve, and a protein larger than the whole budget runs alone.
- **GPU memory.** GPU estimates are multiplied by a factor (initially 4) that is recalibrated from the peak memory each finished protein actually used, because PyTorch's caching allocator and cuSOLVER workspaces add far more than the arrays themselves. A protein that still runs the GPU out of memory (including cuSOLVER failing to start for lack of memory) is not recorded as failed: the factor goes up 1.5x and the protein runs again alone on the GPU, then on the CPU if it must. A single request larger than the whole GPU goes straight to the CPU, and the factor is capped at 16. The worker log records where each GPU memory error happened.
- **Systematic failures.** A worker stops starting proteins once more than half of its first 20+ have failed, and the finalize job stops the run (`action=failing`) instead of burning through the split.
- **Crashes.** If a worker process is killed (for example for exceeding memory), the pool is rebuilt and the in-flight proteins are re-run one at a time. Only a protein that crashes on its own is reported failed, with a hint to lower `FLEXC_JOBS`.

With 32 processes on one H100, kernels from different processes time-slice by default. `FLEXC_MPS=1` starts a CUDA MPS daemon inside each job so those kernels can run concurrently, and falls back to time-slicing if it can't start. Compare the two with:

```bash
sbatch sbatch/bench.sbatch --throughput --devices cuda --jobs-list 16,32,48
FLEXC_MPS=1 sbatch sbatch/bench.sbatch --throughput --devices cuda --jobs-list 16,32,48
```

Then make the better setting the default in `sbatch/sites/isambard.sh`.

**Rough cost.**

- **CPU only:** a synthetic 300-residue ATLAS-sized target (3 × 10,000 frames) took 4.5 minutes on 4 slow cloud cores with all codecs, with a 1.4 GB peak. About 85 % of that is Stage 1, mostly ADMM eigendecompositions, which is the part the GPU backend takes over.
- **ATLAS training split (1,266 targets):** roughly 50–100 CPU-hours (about 1–2 GH200-hours), less with the H100. On 16 GH200s it should take well under an hour once the jobs start.
- **mdCATH training split (21,520 targets):** 5 folds per target at K = 128. By the same cost model the total is similar to ATLAS's, within a factor of a few. If it overruns one round, the next round continues it automatically. Each result records its `timing` and backend, and `report/summary.json` gives the timing distribution. `folds.max_folds=2` would halve Stage 1 cost, but it is not needed to fit the walltime any more.
- **Memory:** proteins are admitted against 95 GB host / 80 GB GPU budgets, as described above. The worker log prints the budgets and the largest estimated footprint at start-up.

**Failures and resuming.** A target that fails is written to `$RUN/errors/` with its traceback and listed in `summary.json`. A protein that kills its worker process even when running alone (usually out of memory) is recorded there too. Later rounds do not retry failures. Once a run has stopped, `bash sbatch/resume.sh $RUN` starts a new series of rounds that retries failures recorded before it was called and continues anything not yet done. It uses the run's frozen code and site settings, with `FLEXC_*` variables set now taking precedence (for example `FLEXC_JOBS=16` after out-of-memory kills). While jobs of the run are still queued or running it refuses, because the run continues on its own. `FLEXC_FORCE=1` overrides that; two series can run side by side safely because they share the queue.

**Interactive or debug** (one process: configure, work through every target, aggregate; use `--jobs` on a compute node):

```bash
PYTHONPATH=src:DynaMode-dev/src python -m flexcompress.run local --config configs/stage01_atlas.yaml \
    --run-dir /tmp/fc-debug --set "selection.domains=['6o2v_A','7ead_A']" --jobs 2
```

Configs read the archive locations from `DYNAMODE_ATLAS_ARCHIVE` / `DYNAMODE_MDCATH_ARCHIVE` (exported by the site profile in `submit.sh`). If these are unset, they fall back to the Isambard paths. On HTC outside `submit.sh`, export them, or run `source sbatch/sites/htc.sh` (paths only; avoid sourcing `flexc_env.sh` in an interactive shell, since it sets `set -e`). Any setting can be overridden with `--set key.sub=value`; unknown keys are rejected.

### Outputs (`$RUN/report/`)

| File | Content |
|---|---|
| `stage0.csv` | one row per target: PCA counts, cosine content, RMSIP, drift, Q, floors, replica correlations, band reliability, asymmetry |
| `stage1.csv` | one row per (target, fold, codec, rate): absolute and floor-normalised distortions |
| `stage1_fits.csv` | fit diagnostics and the relaxation spectrum (variance fraction per band) per fold |
| `summary.json` | coverage of the split (eligible, done, failed, not started; exclusions are listed in `targets.json`), grouped (source/temperature) medians and IQRs, rank-needed and N-scaling, full-model adequacy, reliability-vs-lag curves, universality |
| `figures/` | rate–distortion grid, rank needed vs N, PCA-90 vs N, reliability vs lag (by temperature), universality curve |

Reading the replica-reliability numbers: anisotropy-shape correlations use the deviation from isotropy, because raw trace-normalised tensors correlate about 0.6–0.9 even when the two replicas share no anisotropy. `acf_horizon_ps` is the last lag before the median cross-replica correlation of per-residue autocorrelation first drops below 0.5.

Per-target JSON results are in `$RUN/results/`. Per-residue descriptors (with residue names and DSSP, ready for Stage 3 probes) are in `$RUN/descriptors/`.

---

## 6. Caveats to keep in view

- **Slow processes are invisible within a replica.** Variance that does not decay within T is removed by mean subtraction. It appears instead as between-replica mean shift and low RMSIP. ATLAS's 100 ns is short for domain motions.
- **High temperature in mdCATH is non-stationary.** Use Q, drift and asymmetry to separate stationary targets before reading compressibility numbers at 413/450 K. A full-model ratio above 1 there is expected and informative.
- **Shared domains across datasets.** Some mdCATH domains overlap ATLAS chains; deduplicate (Foldseek/MMseqs) before any cross-dataset claim. DynaMode's `splits/ATLAS/test_ids_no_mdcath.txt` is a start.
- **Coupling metrics use a residue subset** (≤128) for large proteins; per-residue metrics use all residues.
- **Termini dominate MSF.** Report core-only numbers when interpreting RMSF fidelity.

---

## 7. Layout

```
configs/                    stage01_atlas.yaml, stage01_mdcath.yaml, stage01_atlas_1ns.yaml
sbatch/                     submit_train.sh (both training splits), submit.sh, resume.sh, launch.sh (one round),
                            worker.sbatch, aggregate.sbatch (finalize a round), flexc_env.sh, setup_env.sh
  sites/                    isambard.sh (default), htc.sh: paths, conda, device, parallelism, Slurm requests
  bench.sbatch              GPU vs CPU benchmark on one GH200
src/flexcompress/
  data.py                   archive discovery, splits, decoding via DynaMode-dev
  stats.py                  lag grids, exact replica-poolable lagged covariance sums
  kernels.py                expected mean-subtracted estimator for each relaxation band
  spectral.py               band grid, ADMM PSD fit, subspace / per-band truncation, local residual
  baselines.py              PCA + one timescale, truncated DCT
  metrics.py                distortion metrics
  diagnostics.py            PCA counts, cosine content, RMSIP, native contacts, asymmetry
  frames.py                 structure-fixed residue frames
  pipeline.py               per-target Stage 0, descriptors, Stage 1
  run.py                    configure / worker / status / aggregate / local CLI
  workqueue.py              shared queue: atomic claims, heartbeats, takeover, walltime-aware admission
  aggregate.py, report.py   tables, summaries, universality, figures
  synthetic.py              OU-mixture ground truth written as DynaMode archives (tests)
  accel.py                  optional PyTorch/CUDA backend for the heavy kernels
  scheduler.py              memory-aware admission of proteins into the worker pool
  bench.py                  backend timing/agreement benchmark (`run bench`)
tests/                      unit + end-to-end tests on synthetic archives
DynaMode-dev/               read-only submodule (dev branch)
```

---

## 8. References

- Noé et al. 2011, *Dynamical fingerprints for probing individual relaxation processes*, PNAS 108:4822.
- Lipari & Szabo 1982, model-free NMR relaxation, JACS 104:4546.
- Hess 2000/2002, *Similarities between principal components of protein dynamics and random diffusion*, PRE 62:8438; *Convergence of sampling in protein simulations*, PRE 65:031910.
- Bafna et al. 2026, *Learning residue level protein dynamics with multiscale Gaussians* (DynaProt), ICLR.
- Schmidt & Schmid 2019, *A conditional space–time POD formalism for intermittent and rare events*, JFM 867:R2.
- Kutz, Fu & Brunton 2016, *Multiresolution dynamic mode decomposition*, SIAM J. Appl. Dyn. Syst.
- Edwards & Storkey 2017, *Towards a neural statistician*, ICLR (set encoders for the Stage 2 codec).
- Chevyrev & Lyons 2016; Salvi et al. 2021: path signatures and signature kernels as open-ended path-space distortions for Stage 2.
- Mirarchi et al. 2024, *mdCATH*, Sci. Data 11:1299. Vander Meersche et al. 2024, *ATLAS*, NAR 52:D384.
- Phipps et al. 2026, *Spectral Diffusion for Protein Dynamics* (DynaMode), ICML GenBio workshop.
