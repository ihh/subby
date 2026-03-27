#!/usr/bin/env python
"""Batch eigensub computation from MAF files with GPU acceleration via jax.vmap.

Uses an actual Newick guide tree (pruned per-block to present species) instead
of a star tree with hardcoded distances.

Two-pass approach:
1. Scan MAF file, filter/exclude species, prune guide tree per species set,
   bucket blocks by pruned tree node count
2. For each bucket: concatenate columns within species groups, then vmap
   Counts() across groups with same tree size for efficient GPU batching

Produces per-block .npz files + manifest.json.

Usage:
    # Basic (requires a Newick guide tree):
    python scripts/batch_eigensub_maf.py input.maf output_dir/ --tree hg38.100way.nh

    # Exclude primates (for leakage prevention):
    python scripts/batch_eigensub_maf.py input.maf output_dir/ --tree hg38.100way.nh \
        --exclude-species hg38,panTro4,gorGor3,ponAbe2,rheMac3

    # With validation:
    python scripts/batch_eigensub_maf.py input.maf output_dir/ --tree hg38.100way.nh --validate 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

# Ensure subby is importable (works when run from repo root)
_subby_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _subby_root not in sys.path:
    sys.path.insert(0, _subby_root)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from subby.formats import Tree, parse_newick
from subby.jax import Counts, pad_alignment, unpad_columns
from subby.jax.models import hky85_diag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COL_BIN = 128   # Column padding bin size (avoids JIT recompilation)
GPU_MEM_BUDGET = 4 * 1024**3  # 4 GB conservative budget
FEATURE_DIM = 20  # 16 (4x4 transition) + 4 (dwell)

DEFAULT_KAPPA = 4.0
DEFAULT_PI = [0.295, 0.205, 0.205, 0.295]  # A, C, G, T


# ---------------------------------------------------------------------------
# MAF block iterator
# ---------------------------------------------------------------------------

def iter_maf_blocks(maf_path: str):
    """Iterate over alignment blocks in a MAF file.

    Yields dicts with keys:
        species: list of species names (str)
        seqs: list of aligned sequences (str, same length)
        srcs: list of full source names (e.g. "hg38.chr1")
        starts: list of start positions (int)
        sizes: list of alignment sizes (int)
        strands: list of strand characters ('+' or '-')
        src_sizes: list of source sequence sizes (int)

    Supports plain text and gzip-compressed MAF files.
    """
    import gzip

    open_fn = gzip.open if maf_path.endswith('.gz') else open
    mode = 'rt' if maf_path.endswith('.gz') else 'r'

    block = None

    with open_fn(maf_path, mode) as f:
        for line in f:
            line = line.rstrip('\n\r')
            if line.startswith('a'):
                if block is not None and block['species']:
                    yield block
                block = {
                    'species': [], 'seqs': [], 'srcs': [],
                    'starts': [], 'sizes': [], 'strands': [],
                    'src_sizes': [],
                }
            elif line.startswith('s') and block is not None:
                parts = line.split()
                # s src start size strand srcSize sequence
                src = parts[1]
                start = int(parts[2])
                size = int(parts[3])
                strand = parts[4]
                src_size = int(parts[5])
                seq = parts[6]
                species = src.split('.')[0]

                block['species'].append(species)
                block['seqs'].append(seq)
                block['srcs'].append(src)
                block['starts'].append(start)
                block['sizes'].append(size)
                block['strands'].append(strand)
                block['src_sizes'].append(src_size)

        # Yield last block
        if block is not None and block['species']:
            yield block


# ---------------------------------------------------------------------------
# Sequence encoding
# ---------------------------------------------------------------------------

def _seqs_to_subby_alignment(seqs: list[str], alphabet_size: int = 4) -> np.ndarray:
    """Convert list of aligned sequence strings to subby int32 alignment.

    subby token encoding:
        0..A-1 = observed nucleotide (A=0, C=1, G=2, T=3)
        A      = ungapped unobserved (N)
        A+1    = gap (-)

    Returns:
        (R, C) int32 array
    """
    A = alphabet_size
    char_map = {
        'A': 0, 'a': 0,
        'C': 1, 'c': 1,
        'G': 2, 'g': 2,
        'T': 3, 't': 3,
        'N': A, 'n': A,
        '-': A + 1, '.': A + 1,
    }

    R = len(seqs)
    C = len(seqs[0])
    alignment = np.full((R, C), A + 1, dtype=np.int32)  # default gap
    for r, seq in enumerate(seqs):
        for c, ch in enumerate(seq):
            alignment[r, c] = char_map.get(ch, A)
    return alignment


# ---------------------------------------------------------------------------
# Tree pruning and alignment expansion
# ---------------------------------------------------------------------------

def prune_tree(tree_data: dict, keep_leaves: set[str]) -> dict | None:
    """Prune a parsed Newick tree to a subset of leaves.

    Removes absent leaves, collapses single-child internal nodes
    (summing branch lengths through them), and returns a valid
    preorder tree.

    Args:
        tree_data: dict from parse_newick (parentIndex, distanceToParent,
                   leaf_names, node_names)
        keep_leaves: set of leaf names to retain

    Returns:
        dict with same keys as tree_data, pruned to keep_leaves.
        Returns None if fewer than 2 leaves remain.
    """
    parent_index = tree_data["parentIndex"]
    dist = tree_data["distanceToParent"]
    node_names = tree_data["node_names"]
    R = len(parent_index)

    # Build children lists
    children = [[] for _ in range(R)]
    for i in range(1, R):
        children[parent_index[i]].append(i)

    # Mark which leaves to keep
    is_leaf = [len(children[i]) == 0 for i in range(R)]
    keep_node = [False] * R
    for i in range(R):
        if is_leaf[i] and node_names[i] in keep_leaves:
            keep_node[i] = True

    # Bottom-up: mark internal nodes that have >=1 kept descendant
    def mark_ancestors(node):
        if is_leaf[node]:
            return keep_node[node]
        has_kept = False
        for c in children[node]:
            if mark_ancestors(c):
                has_kept = True
        keep_node[node] = has_kept
        return has_kept

    mark_ancestors(0)  # root

    # Build pruned tree in preorder, collapsing single-child nodes
    new_parent = []
    new_dist = []
    new_names = []
    new_leaf_names = []

    def emit(old_idx, new_parent_idx, extra_dist):
        """Emit a node or collapse if single-child internal."""
        kept_children = [c for c in children[old_idx] if keep_node[c]]
        total_dist = dist[old_idx] + extra_dist

        if is_leaf[old_idx]:
            # Leaf: always emit
            new_idx = len(new_parent)
            new_parent.append(new_parent_idx)
            new_dist.append(total_dist if new_parent_idx >= 0 else 0.0)
            new_names.append(node_names[old_idx])
            new_leaf_names.append(node_names[old_idx])
            return

        if len(kept_children) == 1:
            # Single child: collapse, pass accumulated distance
            emit(kept_children[0], new_parent_idx, total_dist)
            return

        # Internal node with >=2 kept children: emit
        new_idx = len(new_parent)
        new_parent.append(new_parent_idx)
        new_dist.append(total_dist if new_parent_idx >= 0 else 0.0)
        new_names.append(node_names[old_idx])
        for c in kept_children:
            emit(c, new_idx, 0.0)

    if not keep_node[0]:
        return None

    emit(0, -1, 0.0)

    if len(new_leaf_names) < 2:
        return None

    return {
        "parentIndex": np.array(new_parent, dtype=np.int32),
        "distanceToParent": np.array(new_dist, dtype=np.float64),
        "leaf_names": new_leaf_names,
        "node_names": new_names,
    }


def _expand_alignment_to_tree(alignment: np.ndarray, species: list[str],
                               pruned_tree: dict) -> np.ndarray:
    """Map alignment rows to tree node positions.

    Args:
        alignment: (R_species, L) int32 in subby encoding
        species: list of species names matching alignment rows
        pruned_tree: dict from prune_tree with leaf_names

    Returns:
        (n_nodes, L) int32 array with internal nodes = gap token (5)
    """
    n_nodes = len(pruned_tree["parentIndex"])
    L = alignment.shape[1]
    expanded = np.full((n_nodes, L), 5, dtype=np.int32)  # gap token

    # Map species name -> alignment row
    species_to_row = {sp: i for i, sp in enumerate(species)}

    # Map tree leaf positions to alignment rows
    node_names = pruned_tree["node_names"]
    parent_index = pruned_tree["parentIndex"]
    # Leaves are nodes with no children
    children_count = np.zeros(n_nodes, dtype=int)
    for i in range(1, n_nodes):
        children_count[parent_index[i]] += 1

    for i in range(n_nodes):
        if children_count[i] == 0:  # leaf
            name = node_names[i]
            if name in species_to_row:
                expanded[i] = alignment[species_to_row[name]]

    return expanded


# ---------------------------------------------------------------------------
# Dynamic batch sizing
# ---------------------------------------------------------------------------

def _vmap_batch_size(n_nodes: int, max_cols: int) -> int:
    """Compute safe vmap batch size given tree and column dimensions.

    Each element needs ~n_nodes * max_cols * 8 bytes for the alignment,
    plus ~12x for intermediates (U, D, J, subMatrices, etc).
    """
    bytes_per_element = n_nodes * max_cols * 8 * 12
    batch = max(1, GPU_MEM_BUDGET // max(bytes_per_element, 1))
    return min(batch, 512)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _counts_to_features(counts_np: np.ndarray, L: int) -> np.ndarray:
    """Convert (4, 4, L) counts array to (L, FEATURE_DIM) features.

    Layout: 16 transition counts (flattened 4x4) + 4 dwell times = 20 dims.
    """
    counts_per_col = np.transpose(counts_np, (2, 0, 1))  # (L, 4, 4)
    dwell = np.diagonal(counts_per_col, axis1=1, axis2=2)  # (L, 4)
    trans = counts_per_col.reshape(L, 16)  # (L, 16)
    return np.concatenate([trans, dwell], axis=1)  # (L, 20)


# ---------------------------------------------------------------------------
# Validation diagnostics
# ---------------------------------------------------------------------------

def validate_block(features: np.ndarray, n_species: int, block_idx: int) -> dict:
    """Validate eigensub output against biological expectations.

    Returns dict of diagnostics.
    """
    L = features.shape[0]

    trans_matrices = features[:, :16].reshape(L, 4, 4)
    dwell_times = features[:, 16:]

    # 1. Symmetry (detailed balance)
    mean_trans = trans_matrices.mean(axis=0)
    upper = mean_trans[np.triu_indices(4, k=1)]
    lower = mean_trans[np.tril_indices(4, k=-1)]
    symmetry_ratio = (upper / np.maximum(lower, 1e-10)).mean()

    # 2. Total dwell times
    total_dwell = dwell_times.sum(axis=1).mean()

    # 3. Ti/Tv ratio
    ti_pairs = [(0, 2), (2, 0), (1, 3), (3, 1)]
    tv_pairs = [(0, 1), (1, 0), (0, 3), (3, 0),
                (2, 1), (1, 2), (2, 3), (3, 2)]
    ti_total = sum(mean_trans[i, j] for i, j in ti_pairs)
    tv_total = sum(mean_trans[i, j] for i, j in tv_pairs)
    ti_tv_ratio = ti_total / max(tv_total, 1e-10)

    # 4. Conservation signal
    off_diag_mask = ~np.eye(4, dtype=bool)
    total_counts_per_col = trans_matrices[:, off_diag_mask].sum(axis=1)
    count_cv = total_counts_per_col.std() / max(total_counts_per_col.mean(), 1e-10)

    # 5. Dominant base dwell fraction in conserved columns
    conserved = total_counts_per_col < np.percentile(total_counts_per_col, 25)
    if conserved.any():
        conserved_dwell = dwell_times[conserved]
        dominant_frac = conserved_dwell.max(axis=1) / np.maximum(
            conserved_dwell.sum(axis=1), 1e-10)
        mean_dominant_frac = dominant_frac.mean()
    else:
        mean_dominant_frac = float('nan')

    return {
        'block_idx': block_idx,
        'n_species': n_species,
        'n_cols': L,
        'mean_trans_matrix': mean_trans.tolist(),
        'symmetry_ratio': float(symmetry_ratio),
        'total_dwell_per_col': float(total_dwell),
        'ti_tv_ratio': float(ti_tv_ratio),
        'count_cv': float(count_cv),
        'conserved_dominant_frac': float(mean_dominant_frac),
        'total_counts_mean': float(total_counts_per_col.mean()),
        'total_counts_std': float(total_counts_per_col.std()),
    }


def print_diagnostics(diag: dict):
    """Print human-readable validation summary."""
    print(f"\n{'='*70}")
    print(f"Block {diag['block_idx']}: {diag['n_species']} species, "
          f"{diag['n_cols']} columns")
    print(f"{'='*70}")

    ti_tv = diag['ti_tv_ratio']
    status = "OK" if 1.5 < ti_tv < 6.0 else "WARNING"
    print(f"  Ti/Tv ratio:                {ti_tv:.3f}  [{status}]  "
          f"(expect 2-4 for vertebrate HKY85)")

    sym = diag['symmetry_ratio']
    status = "OK" if 0.5 < sym < 2.0 else "WARNING"
    print(f"  Symmetry ratio (upper/lower): {sym:.3f}  [{status}]  "
          f"(expect ~1.0 for reversible model)")

    dwell = diag['total_dwell_per_col']
    print(f"  Total dwell/col:            {dwell:.3f}")

    cv = diag['count_cv']
    dom = diag['conserved_dominant_frac']
    print(f"  Count CV across columns:    {cv:.3f}  "
          f"(>0.3 = good conservation signal)")
    print(f"  Conserved col dominant base: {dom:.3f}  "
          f"(expect >0.7 in conserved columns)")

    mat = np.array(diag['mean_trans_matrix'])
    labels = ['A', 'C', 'G', 'T']
    print(f"\n  Mean transition count matrix:")
    print(f"       {'    '.join(labels)}")
    for i, row in enumerate(mat):
        vals = '  '.join(f'{v:6.3f}' for v in row)
        print(f"    {labels[i]}  {vals}")


# ---------------------------------------------------------------------------
# Main precomputation
# ---------------------------------------------------------------------------

def precompute_maf_gpu(
    maf_path: str,
    output_dir: str,
    model,
    guide_tree: dict,
    exclude_species: set[str] | None = None,
    min_species: int = 3,
    min_cols: int = 10,
    max_cols: int = 2048,
    validate_n: int = 0,
) -> list[dict]:
    """Pre-compute eigensub features for all blocks in a MAF file.

    Uses the actual guide tree (pruned per block) instead of a star tree.

    Two-pass approach:
    1. Scan MAF, filter excluded species, prune guide tree per species set,
       bucket blocks by pruned tree node count
    2. For each bucket: concatenate columns within same-species groups,
       then vmap Counts() across groups with the same tree size

    Args:
        maf_path: path to .maf or .maf.gz file
        output_dir: directory for .npz output files
        model: subby DiagModel (e.g. from hky85_diag)
        guide_tree: parsed Newick tree dict (from parse_newick)
        exclude_species: set of species names to exclude
        min_species: minimum species per block (after exclusion)
        min_cols: minimum alignment columns
        max_cols: maximum alignment columns
        validate_n: validate first N blocks (0 to skip)

    Returns:
        list of manifest entries (one per block)
    """
    if exclude_species is None:
        exclude_species = set()

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # Cache: species_key (sorted tuple) -> pruned_tree_dict or None
    _prune_cache: dict[tuple, dict | None] = {}

    # --- Pass 1: scan, filter, prune trees, bucket by node count ---
    print("  Pass 1: scanning and bucketing blocks...")

    buckets: dict[int, list[tuple]] = defaultdict(list)
    n_total = 0
    n_skipped = 0

    for block in iter_maf_blocks(maf_path):
        all_species = block['species']
        all_seqs = block['seqs']
        L = len(all_seqs[0]) if all_seqs else 0

        if L < min_cols or L > max_cols:
            n_skipped += 1
            continue

        # Filter excluded species
        keep = [i for i, sp in enumerate(all_species) if sp not in exclude_species]
        if len(keep) < min_species:
            n_skipped += 1
            continue

        species = [all_species[i] for i in keep]
        seqs = [all_seqs[i] for i in keep]
        species_key = tuple(sorted(species))

        # Prune guide tree to this species set (cached)
        if species_key not in _prune_cache:
            _prune_cache[species_key] = prune_tree(guide_tree, set(species))
        pruned = _prune_cache[species_key]
        if pruned is None:
            n_skipped += 1
            continue

        n_nodes = len(pruned["parentIndex"])

        # Convert to subby alignment and expand to tree topology
        alignment = _seqs_to_subby_alignment(seqs)
        expanded = _expand_alignment_to_tree(alignment, species, pruned)

        rec = (n_total, species, expanded, len(all_species))
        buckets[n_nodes].append(rec)
        n_total += 1

    elapsed_scan = time.time() - t0
    total_cols = sum(rec[2].shape[1] for recs in buckets.values() for rec in recs)
    print(f"  Pass 1 done: {n_total:,} blocks, {len(buckets)} tree sizes, "
          f"{total_cols:,} cols in {elapsed_scan:.0f}s")
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} blocks (too few species/cols or excluded)")

    # --- Pass 2: vmap Counts per tree-size bucket ---
    print("  Pass 2: computing eigensub (vmap per tree size)...")

    manifest = [None] * n_total
    diagnostics = []
    cols_processed = 0
    blocks_processed = 0
    t1 = time.time()

    # Build vmapped Counts function
    def _counts_one(alignment, parent_index, dist_to_parent):
        tree = Tree(parentIndex=parent_index, distanceToParent=dist_to_parent)
        return Counts(alignment, tree, model, maxChunkSize=256, branch_mask="auto")

    _counts_batched = jax.vmap(_counts_one)

    for bucket_idx, (n_nodes, recs) in enumerate(
            sorted(buckets.items(), key=lambda kv: -len(kv[1]))):
        # Group by exact species set (sorted) for column concatenation.
        # Same species set -> same pruned tree -> can concatenate columns.
        species_groups: dict[tuple, list[int]] = defaultdict(list)
        for i, rec in enumerate(recs):
            species_groups[tuple(sorted(rec[1]))].append(i)

        # Build per-group concatenated alignments
        group_data = []
        for species_key, indices in species_groups.items():
            # Look up the cached pruned tree for this species set
            pruned = _prune_cache[species_key]
            parent_idx = pruned["parentIndex"]
            dist = pruned["distanceToParent"]

            parts = [recs[i][2] for i in indices]
            lengths = [p.shape[1] for p in parts]
            big = np.concatenate(parts, axis=1)
            group_data.append((parent_idx, dist, big, lengths, indices))

        # Sort by column count for efficient batching
        group_data.sort(key=lambda g: g[2].shape[1])

        # Process in dynamically-sized batches
        i = 0
        while i < len(group_data):
            max_cols_so_far = group_data[i][2].shape[1]
            C_padded = ((max_cols_so_far + COL_BIN - 1) // COL_BIN) * COL_BIN
            B_max = _vmap_batch_size(n_nodes, C_padded)
            B = 1
            while i + B < len(group_data) and B < B_max:
                next_cols = group_data[i + B][2].shape[1]
                next_padded = ((next_cols + COL_BIN - 1) // COL_BIN) * COL_BIN
                if next_padded > C_padded:
                    C_padded = next_padded
                    B_max = _vmap_batch_size(n_nodes, C_padded)
                    if B >= B_max:
                        break
                B += 1

            batch = group_data[i:i + B]
            actual_max = max(g[2].shape[1] for g in batch)
            C_padded = ((actual_max + COL_BIN - 1) // COL_BIN) * COL_BIN

            # Stack arrays for vmap
            aln_stack = np.full((B, n_nodes, C_padded), 5, dtype=np.int32)
            parent_stack = np.zeros((B, n_nodes), dtype=np.int32)
            dist_stack = np.zeros((B, n_nodes), dtype=np.float64)
            orig_cols = []

            for j, (pidx, dist, big, lengths, indices) in enumerate(batch):
                C = big.shape[1]
                aln_stack[j, :, :C] = big
                parent_stack[j] = pidx
                dist_stack[j] = dist
                orig_cols.append(C)

            i += B

            # Single vmapped call: (B, 4, 4, C_padded)
            all_counts = _counts_batched(
                jnp.array(aln_stack),
                jnp.array(parent_stack),
                jnp.array(dist_stack),
            )
            all_counts_np = np.array(all_counts, dtype=np.float32)

            # Unpack results
            for j, (pidx, dist, big, lengths, indices) in enumerate(batch):
                counts_group = all_counts_np[j, :, :, :orig_cols[j]]

                col_offset = 0
                for k, rec_idx in enumerate(indices):
                    L = lengths[k]
                    global_idx, species, _, n_species_orig = recs[rec_idx]
                    n_sp = len(species)
                    block_counts = counts_group[:, :, col_offset:col_offset + L]
                    col_offset += L

                    features = _counts_to_features(block_counts, L)

                    if global_idx < validate_n:
                        diag = validate_block(features, n_sp, global_idx)
                        diag['n_species_excluded'] = n_species_orig - n_sp
                        diagnostics.append(diag)
                        print_diagnostics(diag)

                    fname = f"features_{global_idx:06d}.npz"
                    fpath = os.path.join(output_dir, fname)
                    np.savez_compressed(
                        fpath,
                        features=features,
                        species=np.array(species),
                        n_species=n_sp,
                        n_species_original=n_species_orig,
                        length=L,
                    )

                    manifest[global_idx] = {
                        'file': fname,
                        'n_species': n_sp,
                        'n_species_original': n_species_orig,
                        'length': L,
                    }

                    blocks_processed += 1
                    cols_processed += L

        # Progress per bucket
        elapsed = time.time() - t1
        rate = cols_processed / elapsed if elapsed > 0 else 0
        print(f"  tree={n_nodes:3d} nodes: "
              f"{len(recs):,} blocks in {len(species_groups):,} groups, "
              f"running total: {blocks_processed:,} blocks, "
              f"{cols_processed:,} cols, {rate:.0f} cols/s")

    elapsed_total = time.time() - t0
    elapsed_compute = time.time() - t1
    rate = cols_processed / elapsed_compute if elapsed_compute > 0 else 0
    print(f"\n  Done: {blocks_processed:,} blocks, {cols_processed:,} cols "
          f"in {elapsed_total:.1f}s (compute: {elapsed_compute:.1f}s, {rate:.0f} cols/s)")

    if diagnostics:
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        ti_tvs = [d['ti_tv_ratio'] for d in diagnostics]
        syms = [d['symmetry_ratio'] for d in diagnostics]
        dwells = [d['total_dwell_per_col'] for d in diagnostics]
        n_sp = [d['n_species'] for d in diagnostics]
        print(f"  Blocks validated:      {len(diagnostics)}")
        print(f"  Species range:         {min(n_sp)}-{max(n_sp)}")
        print(f"  Ti/Tv ratio:           {np.mean(ti_tvs):.3f} "
              f"(range {min(ti_tvs):.3f}-{max(ti_tvs):.3f})")
        print(f"  Symmetry ratio:        {np.mean(syms):.3f} "
              f"(range {min(syms):.3f}-{max(syms):.3f})")
        print(f"  Dwell/col (mean):      {np.mean(dwells):.3f}")

        problems = []
        if any(d['ti_tv_ratio'] < 1.0 for d in diagnostics):
            problems.append("Ti/Tv < 1.0 in some blocks (expect >2)")
        if any(d['symmetry_ratio'] < 0.3 or d['symmetry_ratio'] > 3.0
               for d in diagnostics):
            problems.append("Symmetry ratio far from 1.0")
        if any(d['total_dwell_per_col'] < 0.1 for d in diagnostics):
            problems.append("Very low dwell times -- check tree/model")

        if problems:
            print(f"\n  WARNINGS:")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"\n  All checks PASSED")

        diag_path = os.path.join(output_dir, "validation_diagnostics.json")
        with open(diag_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"  Diagnostics saved to: {diag_path}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch eigensub computation from MAF files with GPU acceleration")
    parser.add_argument("maf_path", type=str,
                        help="Path to MAF file (.maf or .maf.gz)")
    parser.add_argument("output_dir", type=str,
                        help="Output directory for .npz files and manifest")
    parser.add_argument("--tree", type=str, required=True,
                        help="Path to Newick guide tree file (.nh/.nwk)")
    parser.add_argument("--exclude-species", type=str, default="",
                        help="Comma-separated list of species to exclude")
    parser.add_argument("--min-species", type=int, default=3,
                        help="Minimum species per block after exclusion (default: 3)")
    parser.add_argument("--min-cols", type=int, default=10,
                        help="Minimum alignment columns (default: 10)")
    parser.add_argument("--max-cols", type=int, default=2048,
                        help="Maximum alignment columns (default: 2048)")
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA,
                        help=f"HKY85 ti/tv ratio (default: {DEFAULT_KAPPA})")
    parser.add_argument("--pi", type=str, default=None,
                        help="HKY85 equilibrium freqs as comma-separated A,C,G,T "
                             f"(default: {DEFAULT_PI})")
    parser.add_argument("--validate", type=int, default=0,
                        help="Validate first N blocks (default: 0)")
    args = parser.parse_args()

    # Parse --pi
    if args.pi is not None:
        pi = [float(x) for x in args.pi.split(',')]
        assert len(pi) == 4, f"--pi must have 4 values, got {len(pi)}"
        assert abs(sum(pi) - 1.0) < 0.01, f"--pi must sum to 1, got {sum(pi)}"
    else:
        pi = DEFAULT_PI

    exclude_species = set()
    if args.exclude_species:
        exclude_species = set(args.exclude_species.split(','))

    # Load guide tree
    with open(args.tree) as f:
        tree_text = f.read()
    tree_text = ''.join(tree_text.split())  # strip whitespace for multi-line Newick
    guide_tree = parse_newick(tree_text)

    # Print config
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Guide tree: {args.tree} ({len(guide_tree['leaf_names'])} leaves)")
    print(f"HKY85 model: kappa={args.kappa}, pi={pi}")
    print(f"Column bin size: {COL_BIN}")
    if exclude_species:
        print(f"Excluding species: {sorted(exclude_species)}")
    print(f"Filters: min_species={args.min_species}, "
          f"min_cols={args.min_cols}, max_cols={args.max_cols}")
    print()

    # Build model
    pi_jnp = jnp.array(pi, dtype=jnp.float64)
    model = hky85_diag(args.kappa, pi_jnp)

    # Run
    t_start = time.time()
    manifest = precompute_maf_gpu(
        args.maf_path,
        args.output_dir,
        model,
        guide_tree=guide_tree,
        exclude_species=exclude_species,
        min_species=args.min_species,
        min_cols=args.min_cols,
        max_cols=args.max_cols,
        validate_n=args.validate,
    )

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t_start
    total_blocks = sum(1 for m in manifest if m is not None)
    total_cols = sum(m['length'] for m in manifest if m is not None)

    print(f"\n{'='*70}")
    print(f"PRECOMPUTATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output:       {args.output_dir}")
    print(f"Manifest:     {manifest_path}")
    print(f"Total blocks: {total_blocks:,}")
    print(f"Total columns:{total_cols:,}")
    print(f"Total time:   {total_time:.0f}s")
    if total_blocks > 0:
        print(f"Avg time/block: {total_time/total_blocks:.3f}s")
        print(f"Throughput: {total_cols/total_time:.0f} cols/s")


if __name__ == "__main__":
    main()
