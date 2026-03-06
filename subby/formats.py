"""Parsers for standard phylogenetic file formats.

Converts Newick trees, FASTA/Stockholm/MAF alignments, and plain strings
into subby's internal representation: (R, C) int32 alignment tensor +
parentIndex/distanceToParent tree arrays.

Pure Python + NumPy, no external dependencies.
"""

import re
import numpy as np


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

    return {
        "alignment": full_aln,
        "parentIndex": tree_result["parentIndex"],
        "distanceToParent": tree_result["distanceToParent"],
        "alphabet": alphabet,
        "leaf_names": tree_leaves,
    }
