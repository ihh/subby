# Eigensubstitution Accumulation Benchmarks

Performance benchmarks for the phylogenetic sufficient-statistics library,
measuring `LogLike()` and `Counts()` across backends, alphabet sizes,
alignment widths, and tree depths.

## Quick start

```bash
cd benchmarks
make          # runs benchmarks → generates tables → generates plots → compiles PDF
```

## Individual steps

```bash
# Run benchmarks only (saves to results/<hostname>.json)
make results

# Regenerate LaTeX tables from existing results
make tables

# Regenerate plots from existing results
make figures

# Compile PDF report
make report.pdf

# Clean generated artifacts
make clean
```

## CLI options

```bash
# Specify backends (default: jax_cpu,oracle)
python run_benchmarks.py --backends jax_cpu,oracle

# Set number of timing repetitions (default: 5)
python run_benchmarks.py --reps 10

# Dry run (print grid without timing)
python run_benchmarks.py --dry-run

# Regenerate tables only (no benchmarking)
python run_benchmarks.py --tables-only
```

## Parameter grid

| Parameter | Values | Description |
|-----------|--------|-------------|
| A | 4, 20, 64 | Alphabet size |
| C | 100, 1000, 10000 | Alignment width (columns) |
| R | 7, 15, 31 | Tree size (nodes, balanced binary) |

## Results

Results are stored as `results/<hostname>.json`, keyed by hardware ID.
Running on multiple machines produces separate JSON files; the report
and plots automatically incorporate all available results.

## Requirements

- Python 3.12+ with JAX and NumPy (e.g. `~/jax-env`)
- R with `ggplot2`, `dplyr`, `tidyr`, `jsonlite`
- `latexmk` and `pdflatex` (TeX Live or similar)
