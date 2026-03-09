"""Parsers for standard phylogenetic file formats.

Converts Newick trees, FASTA/Stockholm/MAF alignments, and plain strings
into subby's internal representation: (R, C) int32 alignment tensor +
parentIndex/distanceToParent tree arrays.

Pure Python + NumPy, no external dependencies.
"""

import re
import numpy as np
from typing import NamedTuple


class Tree(NamedTuple):
    parentIndex: np.ndarray       # (R,) int32, preorder: parentIndex[i] < i, parentIndex[0] = -1
    distanceToParent: np.ndarray  # (R,) float64


class CombinedResult(NamedTuple):
    alignment: np.ndarray  # (R, C) int32
    tree: Tree
    alphabet: list
    leaf_names: list


# ---------------------------------------------------------------------------
# Alphabet detection
# ---------------------------------------------------------------------------

_DNA = list("ACGT")
_RNA = list("ACGU")
_PROTEIN = list("ACDEFGHIKLMNPQRSTVWY")
_GAP_CHARS = {"-", "."}


def detect_alphabet(chars):
    """Auto-detect alphabet from a set of characters.

    Recognizes DNA (ACGT), RNA (ACGU), protein (20 standard amino acids),
    otherwise returns sorted unique non-gap characters.

    Args:
        chars: set of single characters found in the data

    Returns:
        list of alphabet characters (excludes gaps)
    """
    upper = {ch.upper() for ch in chars} - _GAP_CHARS
    if upper <= set(_DNA):
        return list(_DNA)
    if upper <= set(_RNA):
        return list(_RNA)
    if upper <= set(_PROTEIN):
        return list(_PROTEIN)
    return sorted(upper)


# ---------------------------------------------------------------------------
# Newick parser
# ---------------------------------------------------------------------------

def _strip_comments(s):
    """Remove Newick comments [...] outside quoted labels."""
    out = []
    depth = 0
    in_quote = False
    for ch in s:
        if ch == "'" and depth == 0:
            in_quote = not in_quote
            out.append(ch)
        elif in_quote:
            out.append(ch)
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


class _NewickTokenizer:
    """Simple tokenizer for Newick strings."""

    def __init__(self, s):
        self.s = s
        self.pos = 0

    def peek(self):
        if self.pos >= len(self.s):
            return None
        return self.s[self.pos]

    def consume(self, expected=None):
        ch = self.s[self.pos]
        if expected is not None and ch != expected:
            raise ValueError(
                f"Expected '{expected}' at position {self.pos}, got '{ch}'"
            )
        self.pos += 1
        return ch

    def read_label(self):
        """Read an unquoted or quoted label."""
        if self.pos >= len(self.s):
            return None
        if self.s[self.pos] == "'":
            return self._read_quoted()
        return self._read_unquoted()

    def _read_quoted(self):
        self.consume("'")
        parts = []
        while self.pos < len(self.s):
            ch = self.s[self.pos]
            self.pos += 1
            if ch == "'":
                # doubled quote = literal quote
                if self.pos < len(self.s) and self.s[self.pos] == "'":
                    parts.append("'")
                    self.pos += 1
                else:
                    return "".join(parts)
            else:
                parts.append(ch)
        raise ValueError("Unterminated quoted label")

    def _read_unquoted(self):
        start = self.pos
        while self.pos < len(self.s) and self.s[self.pos] not in "(),:;":
            self.pos += 1
        text = self.s[start : self.pos].strip()
        return text if text else None

    def read_branch_length(self):
        """If next char is ':', consume it and read the float."""
        if self.pos < len(self.s) and self.s[self.pos] == ":":
            self.consume(":")
            start = self.pos
            while self.pos < len(self.s) and self.s[self.pos] not in "(),;":
                self.pos += 1
            text = self.s[start : self.pos].strip()
            if not text:
                return 0.0
            return float(text)
        return None


def parse_newick(newick_str):
    """Parse a Newick tree string.

    Returns:
        dict with:
            parentIndex: (R,) int32 array, DFS preorder, root=-1
            distanceToParent: (R,) float64 array
            leaf_names: list of leaf names in tree order
            node_names: list of name|None for every node
    """
    s = _strip_comments(newick_str.strip())
    # Remove trailing semicolons
    while s.endswith(";"):
        s = s[:-1]
    s = s.strip()

    if not s:
        raise ValueError("Empty Newick string")

    tok = _NewickTokenizer(s)
    # Parse into tree nodes: list of (name, branch_length, children_indices)
    nodes = []  # [(name, dist, [child_indices])]

    def parse_node():
        """Parse a single node (leaf or internal) and return its index."""
        children = []
        if tok.peek() == "(":
            tok.consume("(")
            children.append(parse_node())
            while tok.peek() == ",":
                tok.consume(",")
                children.append(parse_node())
            tok.consume(")")

        name = tok.read_label()
        dist = tok.read_branch_length()
        if dist is None:
            dist = 0.0

        idx = len(nodes)
        nodes.append((name, dist, children))
        return idx

    root_idx = parse_node()

    # Convert to DFS preorder arrays
    parent_index = []
    distance_to_parent = []
    node_names = []
    leaf_names = []
    old_to_new = {}

    def dfs(old_idx, parent_new):
        new_idx = len(parent_index)
        old_to_new[old_idx] = new_idx
        name, dist, children = nodes[old_idx]
        parent_index.append(parent_new)
        distance_to_parent.append(dist if parent_new >= 0 else 0.0)
        node_names.append(name)
        if not children:
            leaf_names.append(name)
        for child_old in children:
            dfs(child_old, new_idx)

    dfs(root_idx, -1)

    return {
        "parentIndex": np.array(parent_index, dtype=np.int32),
        "distanceToParent": np.array(distance_to_parent, dtype=np.float64),
        "leaf_names": leaf_names,
        "node_names": node_names,
    }


# ---------------------------------------------------------------------------
# FASTA parser
# ---------------------------------------------------------------------------

def parse_fasta(text, alphabet=None):
    """Parse FASTA-formatted alignment text.

    All sequences must have equal length.

    Args:
        text: FASTA string
        alphabet: list of characters, or None for auto-detect

    Returns:
        dict with alignment (N, C) int32, leaf_names, alphabet
    """
    sequences = []
    names = []
    current_name = None
    current_seq = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_name is not None:
                sequences.append("".join(current_seq))
            # Name = first word after >
            header = line[1:].strip()
            current_name = header.split()[0] if header else ""
            names.append(current_name)
            current_seq = []
        else:
            current_seq.append(line)

    if current_name is not None:
        sequences.append("".join(current_seq))

    if not sequences:
        raise ValueError("No sequences found in FASTA input")

    # Check equal lengths
    lengths = {len(s) for s in sequences}
    if len(lengths) > 1:
        raise ValueError(
            f"Unequal sequence lengths in FASTA: {sorted(lengths)}"
        )

    C = lengths.pop()
    if C == 0:
        raise ValueError("Empty sequences in FASTA input")

    # Detect or validate alphabet
    all_chars = set()
    for s in sequences:
        all_chars.update(s.upper())
    all_chars -= _GAP_CHARS

    if alphabet is None:
        alphabet = detect_alphabet(all_chars)

    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    # Also map lowercase
    for ch, i in list(char_to_idx.items()):
        char_to_idx[ch.lower()] = i
    gap_idx = len(alphabet) + 1

    N = len(sequences)
    alignment = np.zeros((N, C), dtype=np.int32)
    for r, seq in enumerate(sequences):
        for c, ch in enumerate(seq):
            if ch in _GAP_CHARS:
                alignment[r, c] = gap_idx
            elif ch in char_to_idx:
                alignment[r, c] = char_to_idx[ch]
            elif ch.lower() in char_to_idx:
                alignment[r, c] = char_to_idx[ch.lower()]
            else:
                raise ValueError(
                    f"Unknown character '{ch}' in sequence '{names[r]}' at position {c}"
                )

    return {
        "alignment": alignment,
        "leaf_names": names,
        "alphabet": alphabet,
    }


# ---------------------------------------------------------------------------
# Stockholm parser
# ---------------------------------------------------------------------------

def parse_stockholm(text, alphabet=None):
    """Parse Stockholm-format alignment text.

    Extracts #=GF NH lines for tree, sequence lines for alignment.

    Args:
        text: Stockholm string
        alphabet: list of characters, or None for auto-detect

    Returns:
        dict with alignment (N, C) int32, leaf_names, alphabet,
        and optionally tree (parentIndex, distanceToParent) if NH found
    """
    seq_data = {}  # name -> sequence fragments
    seq_order = []
    nh_parts = []

    for line in text.splitlines():
        line = line.rstrip()
        if not line or line.startswith("# STOCKHOLM"):
            continue
        if line == "//":
            break
        if line.startswith("#=GF NH"):
            # Tree line
            nh_parts.append(line[7:].strip())
            continue
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            seq = parts[1]
            if name not in seq_data:
                seq_data[name] = []
                seq_order.append(name)
            seq_data[name].append(seq)

    if not seq_data:
        raise ValueError("No sequences found in Stockholm input")

    sequences = ["".join(seq_data[name]) for name in seq_order]
    lengths = {len(s) for s in sequences}
    if len(lengths) > 1:
        raise ValueError(
            f"Unequal sequence lengths in Stockholm: {sorted(lengths)}"
        )

    C = lengths.pop()
    all_chars = set()
    for s in sequences:
        all_chars.update(s.upper())
    all_chars -= _GAP_CHARS

    if alphabet is None:
        alphabet = detect_alphabet(all_chars)

    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    for ch, i in list(char_to_idx.items()):
        char_to_idx[ch.lower()] = i
    gap_idx = len(alphabet) + 1

    N = len(sequences)
    alignment = np.zeros((N, C), dtype=np.int32)
    for r, seq in enumerate(sequences):
        for c, ch in enumerate(seq):
            if ch in _GAP_CHARS:
                alignment[r, c] = gap_idx
            elif ch in char_to_idx:
                alignment[r, c] = char_to_idx[ch]
            else:
                raise ValueError(
                    f"Unknown character '{ch}' in sequence '{seq_order[r]}'"
                )

    result = {
        "alignment": alignment,
        "leaf_names": seq_order,
        "alphabet": alphabet,
    }

    if nh_parts:
        nh_str = " ".join(nh_parts)
        tree_result = parse_newick(nh_str)
        combined = combine_tree_alignment(tree_result, result)
        return combined

    return result


# ---------------------------------------------------------------------------
# MAF parser
# ---------------------------------------------------------------------------

def parse_maf(text, alphabet=None):
    """Parse MAF (Multiple Alignment Format) text.

    Species name = text before first '.' in source field.
    Concatenates columns across 'a' blocks. Missing species in some
    blocks are filled with gaps.

    Args:
        text: MAF string
        alphabet: list of characters, or None for auto-detect

    Returns:
        dict with alignment (N, C) int32, leaf_names, alphabet
    """
    blocks = []
    current_block = None

    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("a"):
            current_block = []
            blocks.append(current_block)
        elif line.startswith("s") and current_block is not None:
            parts = line.split()
            # s src start size strand srcSize sequence
            src = parts[1]
            seq = parts[-1]
            species = src.split(".")[0]
            current_block.append((species, seq))

    if not blocks:
        raise ValueError("No alignment blocks found in MAF input")

    # Collect all species across blocks
    all_species = []
    seen = set()
    for block in blocks:
        for species, _ in block:
            if species not in seen:
                all_species.append(species)
                seen.add(species)

    # Collect all characters for alphabet detection
    all_chars = set()
    for block in blocks:
        for _, seq in block:
            all_chars.update(seq.upper())
    all_chars -= _GAP_CHARS

    if alphabet is None:
        alphabet = detect_alphabet(all_chars)

    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    for ch, i in list(char_to_idx.items()):
        char_to_idx[ch.lower()] = i
    gap_idx = len(alphabet) + 1

    # Build alignment by concatenating blocks
    species_to_idx = {sp: i for i, sp in enumerate(all_species)}
    N = len(all_species)
    columns = []  # list of per-block column arrays

    for block in blocks:
        # Determine block width
        block_width = 0
        block_dict = {}
        for species, seq in block:
            block_width = len(seq)
            block_dict[species] = seq

        block_cols = np.full((N, block_width), gap_idx, dtype=np.int32)
        for species, seq in block:
            r = species_to_idx[species]
            for c, ch in enumerate(seq):
                if ch in _GAP_CHARS:
                    block_cols[r, c] = gap_idx
                elif ch in char_to_idx:
                    block_cols[r, c] = char_to_idx[ch]
                else:
                    raise ValueError(
                        f"Unknown character '{ch}' in species '{species}'"
                    )
        columns.append(block_cols)

    alignment = np.concatenate(columns, axis=1)

    return {
        "alignment": alignment,
        "leaf_names": all_species,
        "alphabet": alphabet,
    }


# ---------------------------------------------------------------------------
# Plain string parser
# ---------------------------------------------------------------------------

def parse_strings(sequences, alphabet=None):
    """Parse a list of equal-length strings into an alignment tensor.

    Args:
        sequences: list of strings (one per sequence)
        alphabet: list of characters, or None for auto-detect

    Returns:
        dict with alignment (N, C) int32, alphabet
    """
    if not sequences:
        raise ValueError("Empty sequence list")

    lengths = {len(s) for s in sequences}
    if len(lengths) > 1:
        raise ValueError(f"Unequal sequence lengths: {sorted(lengths)}")

    C = lengths.pop()
    if C == 0:
        raise ValueError("Empty sequences")

    all_chars = set()
    for s in sequences:
        all_chars.update(s.upper())
    all_chars -= _GAP_CHARS

    if alphabet is None:
        alphabet = detect_alphabet(all_chars)

    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    for ch, i in list(char_to_idx.items()):
        char_to_idx[ch.lower()] = i
    gap_idx = len(alphabet) + 1

    N = len(sequences)
    alignment = np.zeros((N, C), dtype=np.int32)
    for r, seq in enumerate(sequences):
        for c, ch in enumerate(seq):
            if ch in _GAP_CHARS:
                alignment[r, c] = gap_idx
            elif ch in char_to_idx:
                alignment[r, c] = char_to_idx[ch]
            else:
                raise ValueError(
                    f"Unknown character '{ch}' at row {r}, col {c}"
                )

    return {
        "alignment": alignment,
        "alphabet": alphabet,
    }


# ---------------------------------------------------------------------------
# Dictionary parser
# ---------------------------------------------------------------------------

def parse_dict(sequences, alphabet=None):
    """Parse a name→sequence dictionary into an alignment tensor.

    Args:
        sequences: dict mapping sequence name to string (e.g. {"human": "ACGT", ...})
        alphabet: list of characters, or None for auto-detect

    Returns:
        dict with alignment (N, C) int32, leaf_names, alphabet
    """
    if not sequences:
        raise ValueError("Empty sequence dictionary")

    names = list(sequences.keys())
    seqs = [sequences[name] for name in names]

    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise ValueError(f"Unequal sequence lengths: {sorted(lengths)}")

    C = lengths.pop()
    if C == 0:
        raise ValueError("Empty sequences")

    all_chars = set()
    for s in seqs:
        all_chars.update(s.upper())
    all_chars -= _GAP_CHARS

    if alphabet is None:
        alphabet = detect_alphabet(all_chars)

    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    for ch, i in list(char_to_idx.items()):
        char_to_idx[ch.lower()] = i
    gap_idx = len(alphabet) + 1

    N = len(seqs)
    alignment = np.zeros((N, C), dtype=np.int32)
    for r, seq in enumerate(seqs):
        for c, ch in enumerate(seq):
            if ch in _GAP_CHARS:
                alignment[r, c] = gap_idx
            elif ch in char_to_idx:
                alignment[r, c] = char_to_idx[ch]
            else:
                raise ValueError(
                    f"Unknown character '{ch}' in sequence '{names[r]}' at position {c}"
                )

    return {
        "alignment": alignment,
        "leaf_names": names,
        "alphabet": alphabet,
    }


# ---------------------------------------------------------------------------
# K-mer tokenization
# ---------------------------------------------------------------------------

class KmerIndex:
    """Maps between column tuples and output alignment indices.

    Provides O(1) lookup from column tuple to output index and vice versa.
    """

    def __init__(self, column_tuples):
        """
        Args:
            column_tuples: (T, k) array-like of column indices.
        """
        self.tuples = np.asarray(column_tuples, dtype=np.int64)
        self._lookup = {}
        for i in range(len(self.tuples)):
            self._lookup[tuple(self.tuples[i])] = i

    def tuple_to_idx(self, t):
        """Column tuple → index in the output alignment. Returns -1 if absent."""
        return self._lookup.get(tuple(t), -1)

    def idx_to_tuple(self, idx):
        """Index in the output alignment → column tuple."""
        return tuple(self.tuples[idx])

    def __len__(self):
        return len(self.tuples)

    def __repr__(self):
        return f"KmerIndex({len(self)} tuples, k={self.tuples.shape[1] if len(self) > 0 else 0})"


def sliding_windows(C, k, stride=None, offset=0, edge='truncate'):
    """Generate column index tuples for sliding-window k-mer tokenization.

    Args:
        C: number of columns in the alignment
        k: window size (k-mer length)
        stride: step between window starts. Default None → k (non-overlapping).
        offset: starting column index. Default 0.
        edge: handling of incomplete trailing window:
              'truncate' — drop incomplete trailing window (default)
              'pad' — include partial window, using -1 for out-of-bounds columns

    Returns:
        (M, k) int64 array of column indices. Entries of -1 indicate
        out-of-bounds positions (only with edge='pad').
    """
    if stride is None:
        stride = k

    if edge == 'truncate':
        starts = np.arange(offset, C - k + 1, stride)
    elif edge == 'pad':
        starts = np.arange(offset, C, stride)
    else:
        raise ValueError(f"Unknown edge mode: {edge!r}")

    if len(starts) == 0:
        return np.zeros((0, k), dtype=np.int64)

    # (M, k) array
    col_indices = starts[:, None] + np.arange(k)
    if edge == 'pad':
        col_indices = np.where(col_indices < C, col_indices, -1)

    return col_indices.astype(np.int64)


def all_column_ktuples(C, k, ordered=True):
    """Generate all k-tuples of column indices.

    WARNING: produces O(C^k) tuples. Use with caution for large C or k > 2.

    Args:
        C: number of columns
        k: tuple size
        ordered: if True, permutations (C * (C-1) for k=2);
                 if False, combinations (C choose k)

    Returns:
        (T, k) int64 array of column index tuples
    """
    if ordered:
        from itertools import permutations
        tuples = list(permutations(range(C), k))
    else:
        from itertools import combinations
        tuples = list(combinations(range(C), k))

    if len(tuples) == 0:
        return np.zeros((0, k), dtype=np.int64)

    return np.array(tuples, dtype=np.int64)


def kmer_tokenize(alignment, A, k_or_tuples, gap_mode='any', alphabet=None):
    """Convert single-character token alignment to k-mer tokens.

    Accepts either:
      - An integer k: generates non-overlapping contiguous windows of size k.
        C must be divisible by k (backward compatible with original API).
      - A (T, k) array-like of column index tuples: tokenizes arbitrary column
        groupings. Use sliding_windows() or all_column_ktuples() to generate
        tuples, or pass custom tuples directly. Entries of -1 in tuples are
        treated as unobserved positions.

    Token encoding for the output alignment:
    0..A^k-1 for observed k-mers, A^k for ungapped-unobserved, A^k+1 for gap.
    When gap_mode='all', partial gaps produce an illegal token (A^k+2).

    Args:
        alignment: (R, C) int32 tokens (0..A-1 observed, A ungapped-unobserved,
                   A+1 or -1 gap)
        A: single-character alphabet size
        k_or_tuples: either an int k (non-overlapping windows, C must be
                     divisible by k) or a (T, k) array-like of column tuples
        gap_mode: 'any' — gap in any position gaps the entire k-mer
                  'all' — only all-gap k-mers become gaps; partial gaps become
                          an illegal token (A^k+2)
        alphabet: optional list of A single-character labels for building
                  k-mer labels (e.g. ['A','C','G','T'])

    Returns:
        dict with:
            alignment: (R, T) int32 k-mer tokens
            A_kmer: int, k-mer alphabet size (A^k)
            index: KmerIndex mapping tuples ↔ output positions
            alphabet: list of A^k k-mer label strings (only if alphabet given)
    """
    alignment = np.asarray(alignment, dtype=np.int32)
    R, C = alignment.shape

    # Determine column_tuples
    if isinstance(k_or_tuples, (int, np.integer)):
        k = int(k_or_tuples)
        if C % k != 0:
            raise ValueError(
                f"Number of columns ({C}) not divisible by k ({k})")
        column_tuples = sliding_windows(C, k)
    else:
        column_tuples = np.asarray(k_or_tuples, dtype=np.int64)
        if column_tuples.ndim != 2:
            raise ValueError(
                f"column_tuples must be 2D (T, k), got shape {column_tuples.shape}")
        k = column_tuples.shape[1]

    T = len(column_tuples)
    A_k = A ** k
    index = KmerIndex(column_tuples)

    if T == 0:
        kmer_alphabet = _build_kmer_alphabet(alphabet, A, k)
        out = {
            'alignment': np.zeros((R, 0), dtype=np.int32),
            'A_kmer': A_k,
            'index': index,
        }
        if kmer_alphabet is not None:
            out['alphabet'] = kmer_alphabet
        return out

    # Build blocks: (R, T, k)
    # Handle -1 sentinel (out-of-bounds / padding → unobserved)
    has_sentinel = np.any(column_tuples < 0)
    if has_sentinel:
        # Replace -1 with 0 for indexing, then overwrite those positions
        safe_indices = np.where(column_tuples >= 0, column_tuples, 0)
        blocks = alignment[:, safe_indices]  # (R, T, k)
        # Set sentinel positions to unobserved token
        sentinel_mask = np.broadcast_to(
            column_tuples[None, :, :] < 0, blocks.shape)
        blocks = np.where(sentinel_mask, A, blocks)
    else:
        blocks = alignment[:, column_tuples]  # (R, T, k)

    # Classify each position
    is_observed = (blocks >= 0) & (blocks < A)
    is_gap = (blocks < 0) | (blocks > A)  # -1, A+1, etc.

    all_observed = is_observed.all(axis=2)
    has_gap = is_gap.any(axis=2)
    all_gap = is_gap.all(axis=2)

    # K-mer indices (only valid where all_observed)
    powers = A ** np.arange(k - 1, -1, -1)
    kmer_idx = np.sum(np.where(is_observed, blocks, 0) * powers, axis=2)

    # Build result
    gap_tok = A_k + 1
    unobs_tok = A_k
    illegal_tok = A_k + 2

    result = np.full((R, T), unobs_tok, dtype=np.int32)
    result[all_observed] = kmer_idx[all_observed]

    if gap_mode == 'any':
        result[has_gap] = gap_tok
    elif gap_mode == 'all':
        result[all_gap] = gap_tok
        partial_gap = has_gap & ~all_gap
        result[partial_gap] = illegal_tok
    else:
        raise ValueError(f"Unknown gap_mode: {gap_mode!r}")

    # K-mer alphabet labels
    kmer_alphabet = _build_kmer_alphabet(alphabet, A, k)

    out = {
        'alignment': result,
        'A_kmer': A_k,
        'index': index,
    }
    if kmer_alphabet is not None:
        out['alphabet'] = kmer_alphabet
    return out


def _build_kmer_alphabet(alphabet, A, k):
    """Build k-mer label list from single-character alphabet, or return None."""
    if alphabet is None:
        return None
    from itertools import product as itertools_product
    return [''.join(combo) for combo in itertools_product(alphabet, repeat=k)]


# ---------------------------------------------------------------------------
# Genetic code helpers
# ---------------------------------------------------------------------------

_STANDARD_GENETIC_CODE = {
    'AAA': 'K', 'AAC': 'N', 'AAG': 'K', 'AAT': 'N',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AGA': 'R', 'AGC': 'S', 'AGG': 'R', 'AGT': 'S',
    'ATA': 'I', 'ATC': 'I', 'ATG': 'M', 'ATT': 'I',
    'CAA': 'Q', 'CAC': 'H', 'CAG': 'Q', 'CAT': 'H',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'GAA': 'E', 'GAC': 'D', 'GAG': 'E', 'GAT': 'D',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'TAA': '*', 'TAC': 'Y', 'TAG': '*', 'TAT': 'Y',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TGA': '*', 'TGC': 'C', 'TGG': 'W', 'TGT': 'C',
    'TTA': 'L', 'TTC': 'F', 'TTG': 'L', 'TTT': 'F',
}


def genetic_code():
    """Return the standard genetic code as a structured dict.

    Codons are in ACGT lexicographic order (AAA, AAC, AAG, ..., TTT).
    Stop codons (TAA=48, TAG=50, TGA=56) are marked with '*'.

    Returns:
        dict with:
            codons: list of 64 codon strings
            amino_acids: list of 64 amino acid letters (stop = '*')
            sense_mask: (64,) bool — True for sense codons
            sense_indices: (61,) int — indices of sense codons in 0..63
            codon_to_sense: (64,) int — maps codon index to sense index (stop → -1)
            sense_codons: list of 61 sense codon strings
            sense_amino_acids: list of 61 amino acid letters
    """
    from itertools import product as itertools_product
    bases = list("ACGT")
    codons = [''.join(combo) for combo in itertools_product(bases, repeat=3)]
    amino_acids = [_STANDARD_GENETIC_CODE[c] for c in codons]

    sense_mask = np.array([aa != '*' for aa in amino_acids], dtype=bool)
    sense_indices = np.where(sense_mask)[0].astype(np.int64)

    codon_to_sense_map = np.full(64, -1, dtype=np.int64)
    sense_idx = 0
    for i in range(64):
        if sense_mask[i]:
            codon_to_sense_map[i] = sense_idx
            sense_idx += 1

    sense_codons = [codons[i] for i in sense_indices]
    sense_amino_acids = [amino_acids[i] for i in sense_indices]

    return {
        'codons': codons,
        'amino_acids': amino_acids,
        'sense_mask': sense_mask,
        'sense_indices': sense_indices,
        'codon_to_sense': codon_to_sense_map,
        'sense_codons': sense_codons,
        'sense_amino_acids': sense_amino_acids,
    }


def codon_to_sense(alignment, A=64):
    """Remap a 64-codon tokenized alignment to 61-sense-codon tokens.

    Stop codons become the gap token. Unobserved and gap tokens are
    remapped to the new alphabet size.

    Args:
        alignment: (N, C) int32 with tokens 0..63 for codons,
                   64 for ungapped-unobserved, 65 (or A+1) for gap
        A: codon alphabet size (default 64)

    Returns:
        dict with:
            alignment: (N, C) int32 with A_sense=61-state tokens
            A_sense: 61
            alphabet: list of 61 sense codon strings
    """
    alignment = np.asarray(alignment, dtype=np.int32)
    gc = genetic_code()
    codon_map = gc['codon_to_sense']  # (64,) int, stop → -1

    unobs_in = A       # 64
    gap_in = A + 1     # 65

    A_sense = 61
    unobs_out = A_sense       # 61
    gap_out = A_sense + 1     # 62

    result = np.full_like(alignment, unobs_out)
    for i in range(64):
        mask = alignment == i
        if codon_map[i] >= 0:
            result[mask] = codon_map[i]
        else:
            result[mask] = gap_out  # stop → gap

    result[alignment == unobs_in] = unobs_out
    result[(alignment == gap_in) | (alignment < 0)] = gap_out

    return {
        'alignment': result,
        'A_sense': A_sense,
        'alphabet': gc['sense_codons'],
    }


# ---------------------------------------------------------------------------
# Paired-column helpers (for site-pair coevolution models)
# ---------------------------------------------------------------------------

def split_paired_columns(alignment, paired_columns, A=20):
    """Split an alignment into paired and single-column alignments.

    For coevolution models that operate on pairs of columns (e.g. CherryML
    SiteRM with A=400 = 20x20 amino acid pairs).

    Args:
        alignment: (N, C) int32 with A-state tokens (0..A-1 observed,
                   A ungapped-unobserved, A+1 gap)
        paired_columns: list of (col_i, col_j) tuples
        A: single-column alphabet size (default 20 for amino acids)

    Returns:
        dict with:
            paired_alignment: (N, P) int32 with A_paired = A*A states
            singles_alignment: (N, S) int32 with A_singles = A states
            paired_columns: list of (int, int) — echoed back
            single_columns: list of int — columns not in any pair
            A_paired: A*A
            A_singles: A
            paired_index: KmerIndex for paired columns
            singles_index: KmerIndex for single columns
    """
    alignment = np.asarray(alignment, dtype=np.int32)
    N, C = alignment.shape

    A_paired = A * A

    # Determine which columns are in pairs
    paired_set = set()
    for ci, cj in paired_columns:
        paired_set.add(ci)
        paired_set.add(cj)
    single_columns = [c for c in range(C) if c not in paired_set]

    P = len(paired_columns)
    S = len(single_columns)

    # Build paired alignment via kmer_tokenize with k=2 tuples
    if P > 0:
        paired_tuples = np.array(paired_columns, dtype=np.int64)
        pt = kmer_tokenize(alignment, A, paired_tuples, gap_mode='any')
        paired_aln = pt['alignment']
        paired_index = pt['index']
    else:
        paired_aln = np.zeros((N, 0), dtype=np.int32)
        paired_index = KmerIndex(np.zeros((0, 2), dtype=np.int64))

    # Build singles alignment via kmer_tokenize with k=1 tuples
    if S > 0:
        singles_tuples = np.array(single_columns, dtype=np.int64).reshape(-1, 1)
        st = kmer_tokenize(alignment, A, singles_tuples, gap_mode='any')
        singles_aln = st['alignment']
        singles_index = st['index']
    else:
        singles_aln = np.zeros((N, 0), dtype=np.int32)
        singles_index = KmerIndex(np.zeros((0, 1), dtype=np.int64))

    return {
        'paired_alignment': paired_aln,
        'singles_alignment': singles_aln,
        'paired_columns': list(paired_columns),
        'single_columns': single_columns,
        'A_paired': A_paired,
        'A_singles': A,
        'paired_index': paired_index,
        'singles_index': singles_index,
    }


def merge_paired_columns(paired_posterior, singles_posterior, split_info):
    """Reassemble per-column posteriors from paired and single results.

    Marginalizes the A_paired=A*A dimensional paired posteriors into
    two A-dimensional single-column posteriors, then reassembles into
    the original column order.

    Args:
        paired_posterior: (A_paired, P) array — posterior for paired columns
        singles_posterior: (A_singles, S) array — posterior for single columns
        split_info: dict from split_paired_columns

    Returns:
        (A, C) array — posterior for all columns in original order
    """
    paired_posterior = np.asarray(paired_posterior, dtype=np.float64)
    singles_posterior = np.asarray(singles_posterior, dtype=np.float64)

    paired_columns = split_info['paired_columns']
    single_columns = split_info['single_columns']
    A = split_info['A_singles']
    A_paired = split_info['A_paired']

    P = len(paired_columns)
    S = len(single_columns)
    C = 2 * P + S  # total columns (each pair contributes 2)

    # Determine total number of original columns
    all_cols = set()
    for ci, cj in paired_columns:
        all_cols.add(ci)
        all_cols.add(cj)
    for c in single_columns:
        all_cols.add(c)
    C_total = max(all_cols) + 1 if all_cols else 0

    result = np.zeros((A, C_total), dtype=np.float64)

    # Marginalize paired posteriors
    for p, (ci, cj) in enumerate(paired_columns):
        # paired_posterior[:, p] is (A*A,) with state (i,j) = i*A+j
        pair_post = paired_posterior[:, p]  # (A*A,)
        pair_2d = pair_post[:A_paired].reshape(A, A)
        # Marginalize over j → posterior for column i
        result[:, ci] = pair_2d.sum(axis=1)
        # Marginalize over i → posterior for column j
        result[:, cj] = pair_2d.sum(axis=0)

    # Copy singles
    for s_idx, c in enumerate(single_columns):
        result[:, c] = singles_posterior[:, s_idx]

    return result


# ---------------------------------------------------------------------------
# Combine tree + alignment
# ---------------------------------------------------------------------------

def combine_tree_alignment(tree_result, alignment_result):
    """Map leaf sequences to tree positions by name matching.

    Creates full (R, C) alignment with internal node rows filled with
    ungapped-unobserved token (A = len(alphabet)).

    Args:
        tree_result: dict from parse_newick (parentIndex, distanceToParent,
                     leaf_names, node_names)
        alignment_result: dict from parse_fasta/stockholm/maf/strings
                          (alignment (N, C) int32, leaf_names, alphabet)

    Returns:
        dict with alignment (R, C) int32, parentIndex, distanceToParent,
        alphabet, leaf_names
    """
    tree_leaves = tree_result["leaf_names"]
    aln_names = alignment_result["leaf_names"]
    aln_data = alignment_result["alignment"]
    alphabet = alignment_result["alphabet"]

    A = len(alphabet)
    ungapped_unobserved = A  # token for internal nodes

    # Build name -> row index in alignment
    aln_name_to_row = {}
    for i, name in enumerate(aln_names):
        aln_name_to_row[name] = i

    # Verify all tree leaves appear in alignment
    missing = [name for name in tree_leaves if name not in aln_name_to_row]
    if missing:
        raise ValueError(
            f"Tree leaves not found in alignment: {missing}"
        )

    R = len(tree_result["parentIndex"])
    C = aln_data.shape[1]

    # Build full (R, C) alignment
    full_aln = np.full((R, C), ungapped_unobserved, dtype=np.int32)

    # Map tree node indices to leaf names
    node_names = tree_result["node_names"]
    parent_index = tree_result["parentIndex"]

    # Identify leaf nodes (no children)
    child_count = np.zeros(R, dtype=np.int32)
    for n in range(1, R):
        child_count[parent_index[n]] += 1
    is_leaf = child_count == 0

    leaf_idx = 0
    for n in range(R):
        if is_leaf[n]:
            leaf_name = tree_leaves[leaf_idx]
            aln_row = aln_name_to_row[leaf_name]
            full_aln[n] = aln_data[aln_row]
            leaf_idx += 1

    return CombinedResult(
        alignment=full_aln,
        tree=Tree(
            parentIndex=tree_result["parentIndex"],
            distanceToParent=tree_result["distanceToParent"],
        ),
        alphabet=alphabet,
        leaf_names=tree_leaves,
    )
