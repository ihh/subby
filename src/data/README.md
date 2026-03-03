# src/data/

Data pipeline for subby: MSA tokenization, phylogenetic feature extraction, and RNA-Seq preprocessing.

## Contents

- `tokenizer.py` — Tokenizes multiple sequence alignments into four parallel streams: substitution, triplet, phase, and annotation transfer.
- `featurize.py` — Extracts phylogenetic feature vectors from tokenized MSAs using the `src/phylo/` library.
- `rnaseq.py` — Extracts 6-channel RNA-Seq tensors (2 strands x coverage, donor junctions, acceptor junctions) from BAM files with Borzoi nonlinear compression. See `docs/api/rnaseq.md`.

## Tests

`tests/test_data/` — tests for tokenization, featurization, and RNA-Seq preprocessing.
