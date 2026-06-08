# The Quotable Canon

Code and notebooks supporting the article *The Quotable Canon: Poetry as Example in the
Princeton Prosody Archive (1532–1929)* on PPA found poems: which poems are quoted in the [Princeton Prosody Archive](https://prosody.princeton.edu/), how concentrated that record is, and which poems rise or fall across the long nineteenth century.

## Notebooks

Run in order with the working directory set to this repository root:

| Notebook | Purpose |
|---|---|
| `01_corpus_construction.ipynb` | Build the analytic corpus from Passim alignments; writes `exports/*.parquet` |
| `02_reception_record_shape.ipynb` | Descriptive shape of the reception record (volume, concentration, corpus-level rates) |
| `03_trajectory_model.ipynb` | Fit a hierarchical Bayesian model of how each poem’s quoting rate changes decade by decade; validate the fit; writes `exports/model/` |
| `04_findings.ipynb` | Risers, decliners, trajectory panels, and period summaries (reads the saved model) |

You do not need to re-run **01** — `exports/*.parquet` is committed; notebooks **02** and **03** read those tables directly.

**Notebook 04 §1–§3** needs the fitted posterior in `exports/model/bambi_idata.nc`. That file is **not** in the repository (too large for GitHub). Run **03** once to generate it; the smaller model hand-offs in `exports/model/` (`compare_df.parquet`, `slope_info.parquet`, etc.) are committed and enough for **04 §2–§4** tables and period figures, but not for the full trajectory panels in §1.

With `exports/*.parquet` plus a **03** run, you can work through **02–04** without rebuilding the corpus or downloading the full Passim bundle.

## What is in this repository

- **Notebooks** (`01`–`04`) and **`src/`** Python modules
- **`exports/`** — analysis outputs (partially committed):
  - **`exports/*.parquet`** — notebook 01 hand-offs (`excerpts_df`, exposure tables, host metadata); in repo
  - **`exports/model/*.parquet`** + **`model_meta.json`** — summarized model outputs from notebook 03; in repo
  - **`exports/model/bambi_idata.nc`** — full ArviZ inference data (~1.2 GB); **not** in repo; produced by running notebook **03** (NUTS fit; often 30–90+ minutes depending on sampling settings)
  - **`exports/figures/`** — figure exports from notebooks 02/04, when present

With committed parquets and a one-time **03** run for `bambi_idata.nc`, a clone can run **02–04** without rebuilding the corpus. A small **`data/`** tree is still needed for some notebook sections — see below.

## Data you need locally

Paths are resolved by `src/paths.py`. After cloning, create the folders below and download the listed files from Zenodo.

### 1. Found Poems Dataset — [Zenodo](https://zenodo.org/records/20044417)

DOI: [10.5281/zenodo.20044417](https://doi.org/10.5281/zenodo.20044417)

Download the zip from that record and unzip into `data/ppa_found_poems/` (rename the extracted folder if needed — e.g. from `20044417` to `ppa_found_poems`):

```
data/ppa_found_poems/
  excerpts.csv.gz
  ppa_work_metadata.csv
  poem_meta.csv
```

Notebook **03** needs **`poem_meta.csv`**; notebook **02** does not. The rest of the Passim bundle is required only to **re-run notebook 01** and rebuild `exports/` from scratch.

### 2. Chadwyck reference catalogue — [Poetry Metadata on Zenodo](https://zenodo.org/records/20597001)

DOI: [10.5281/zenodo.20597001](https://doi.org/10.5281/zenodo.20597001)

Notebook **04** (literary periods) needs **`poetry_metadata.csv`**. Download it from that record and place it at:

```
data/poetry_metadata.csv
```

### 3. Local only — required for notebook 01 only

These are not on Zenodo and must be obtained separately to **rebuild the analytic corpus** from Passim alignments:

| Path | Role |
|---|---|
| `data/chadwyckhealey/` | Normalized Chadwyck-Healey poem texts |
| `data/internet_poems/` | Internet-poems metadata + normalized `.txt` files |

Full expected layout for notebooks **02–04**:

```
data/
  ppa_found_poems/
    poem_meta.csv            ← Zenodo 10.5281/zenodo.20044417 (notebook 03)
  poetry_metadata.csv        ← Zenodo 10.5281/zenodo.20597001 (notebook 04)
exports/
  *.parquet                  ← notebook 01 (in repo)
  model/
    *.parquet                ← notebook 03 summaries (in repo)
    model_meta.json          ← in repo
    bambi_idata.nc           ← notebook 03 only (not in repo — run 03)
```

Notebook **01** additionally needs the full Passim bundle (`excerpts.csv.gz`, `ppa_work_metadata.csv`) plus `data/chadwyckhealey/` and `data/internet_poems/`.

### What you can run without the large datasets

| Notebook | Requires |
|---|---|
| **01** | Full `data/` tree above |
| **02** | `exports/*.parquet` only |
| **03** | `exports/*.parquet` + `poem_meta.csv` ([Zenodo](https://doi.org/10.5281/zenodo.20044417)) |
| **04** §1–§3 | `exports/` + **`exports/model/bambi_idata.nc`** (run notebook **03**) |
| **04** (tables / periods) | committed `exports/model/*.parquet` + `exports/*.parquet` |
| **04** | above + `poetry_metadata.csv` ([Zenodo](https://doi.org/10.5281/zenodo.20597001)) |

Passim / Spark are not required to run any notebook in this repository. Notebook 01 reads pre-compiled alignments from the Found Poems Zenodo record; you only need Passim/Spark if you are rebuilding that alignment export from scratch (outside this repo).

## Environment

**Recommended:** conda (handles PyMC/Bambi reliably).

```bash
conda env create -f environment.yml
conda activate quotable-canon
python -m ipykernel install --user --name quotable-canon --display-name "quotable-canon"
```

Open Jupyter with cwd = repo root and select the **quotable-canon** kernel.

**Alternative:** `pip install -r requirements.txt` (may be finicky for the Bayesian stack).

### Key dependencies

- **corppa** — PPA excerpt loading utilities; installed from [Princeton-CDH/corppa](https://github.com/Princeton-CDH/corppa) tag `0.4` (not on PyPI)
- **polars**, **pandas**, **altair** — data and charts
- **statsmodels**, **scipy** — frequentist companion models (notebooks 02–03)
- **bambi**, **pymc**, **arviz** — hierarchical beta-binomial model (notebooks 03–04)

See [Data you need locally](#data-you-need-locally) for Zenodo downloads.

## Local code

Python modules live in `src/`:

- `paths.py` — repo/data/export path constants
- `excerpt_universe.py` — PPA host-side universe and exposure tables
- `poem_corpus.py` — reference metadata joins and temporal plausibility filter
- `ref_metadata.py` — Chadwyck + internet reference catalogue
- `exposure_charts.py` — Altair exposure ribbons
