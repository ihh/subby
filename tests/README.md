# tests/

Unit and integration tests, mirroring the `src/` directory structure.

- `test_phylo/` — Phylogenetic library tests (pruning, eigensub, outside algorithm, F81 fast path, HKY85, oracle cross-validation, golden file comparisons). Golden test data in `test_phylo/golden/`.
- `test_model/` — Model architecture tests (Mamba tower, track encoder, cross-attention, gradient flow).
- `test_data/` — Data pipeline tests (tokenization, featurization, RNA-Seq BAM processing).

Run all tests: `python -m pytest tests/`
