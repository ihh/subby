"""JSON I/O for subby: parse inputs, format outputs, validate."""

import json
import numpy as np

from .oracle import oracle
from .formats import (
    Tree,
    detect_alphabet,
    parse_newick,
    parse_fasta,
    parse_stockholm,
    parse_maf,
    parse_strings,
    parse_dict,
    combine_tree_alignment,
)


def _parse_model(spec):
    """Parse a model specification from JSON.

    Supports:
    - Named models: {"name": "jukes-cantor", "alphabetSize": 4}
                     {"name": "hky85", "kappa": 2.0, "pi": [...]}
                     {"name": "f81", "pi": [...]}
    - Rate matrix:   {"alphabet": [...], "rootProb": [...], "subRate": [[...]], "reversible": true/false/null}

    Returns:
        (model_dict, alphabet) where alphabet is a list of state labels or None
    """
    if "name" in spec:
        name = spec["name"].lower().replace("-", "").replace("_", "")
        if name == "jukescantor":
            A = spec["alphabetSize"]
            alphabet = spec.get("alphabet")
            return oracle.jukes_cantor_model(A), alphabet
        elif name == "hky85":
            kappa = float(spec["kappa"])
            pi = np.array(spec["pi"], dtype=np.float64)
            alphabet = spec.get("alphabet")
            return oracle.hky85_diag(kappa, pi), alphabet
        elif name == "f81":
            pi = np.array(spec["pi"], dtype=np.float64)
            alphabet = spec.get("alphabet")
            return oracle.f81_model(pi), alphabet
        else:
            raise ValueError(f"Unknown model name: {spec['name']}")
    elif "subRate" in spec:
        subRate = np.array(spec["subRate"], dtype=np.float64)
        rootProb = np.array(spec["rootProb"], dtype=np.float64)
        reversible = spec.get("reversible")  # None → auto-detect
        alphabet = spec.get("alphabet")
        model = oracle.model_from_rate_matrix(subRate, rootProb, reversible=reversible)
        return model, alphabet
    else:
        raise ValueError("Model spec must have 'name' (named model) or 'subRate' (rate matrix)")


def _parse_alignment(raw, alphabet):
    """Parse alignment from JSON.

    Accepts:
    - Integer matrix: [[0, 1], [2, 3], ...]
    - Character strings: ["AC", "GT", ...] (requires alphabet)
    - Character matrix: [["A","C"], ["G","T"], ...] (requires alphabet)

    Returns:
        (R, C) int32 numpy array
    """
    if not raw:
        raise ValueError("Alignment is empty")

    first = raw[0]
    if isinstance(first, str):
        # Character strings
        if alphabet is None:
            raise ValueError("Alphabet required when alignment uses character strings")
        char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
        gap_idx = len(alphabet) + 1  # gapped token
        R = len(raw)
        C = len(raw[0])
        alignment = np.zeros((R, C), dtype=np.int32)
        for r, seq in enumerate(raw):
            if len(seq) != C:
                raise ValueError(f"Row {r} has length {len(seq)}, expected {C}")
            for c, ch in enumerate(seq):
                if ch == "-":
                    alignment[r, c] = gap_idx
                elif ch in char_to_idx:
                    alignment[r, c] = char_to_idx[ch]
                else:
                    raise ValueError(f"Unknown character '{ch}' at row {r}, col {c}")
        return alignment
    elif isinstance(first, list):
        first_elem = first[0] if first else None
        if isinstance(first_elem, str):
            # Character matrix
            if alphabet is None:
                raise ValueError("Alphabet required when alignment uses characters")
            char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
            gap_idx = len(alphabet) + 1
            R = len(raw)
            C = len(raw[0])
            alignment = np.zeros((R, C), dtype=np.int32)
            for r, row in enumerate(raw):
                for c, ch in enumerate(row):
                    if ch == "-":
                        alignment[r, c] = gap_idx
                    elif ch in char_to_idx:
                        alignment[r, c] = char_to_idx[ch]
                    else:
                        raise ValueError(f"Unknown character '{ch}' at row {r}, col {c}")
            return alignment
        else:
            # Integer matrix
            return np.array(raw, dtype=np.int32)
    else:
        raise ValueError("Alignment must be a list of strings or a 2D integer array")


def load_input(data):
    """Parse a complete input specification from a dict (parsed JSON).

    Args:
        data: dict with keys 'model', 'tree', 'alignment', and optional 'compute'

    Returns:
        (alignment, tree, model, compute_list) where:
        - alignment: (R, C) int32 numpy array
        - tree: dict with 'parentIndex', 'distanceToParent'
        - model: oracle model dict
        - compute_list: list of output names to compute
    """
    # Parse model
    model_spec = data["model"]
    model, alphabet = _parse_model(model_spec)

    # Parse tree
    tree_spec = data["tree"]
    tree = Tree(
        parentIndex=np.array(tree_spec["parentIndex"], dtype=np.intp),
        distanceToParent=np.array(tree_spec["distanceToParent"], dtype=np.float64),
    )

    # Parse alignment
    alignment = _parse_alignment(data["alignment"], alphabet)

    # Parse compute list
    compute = data.get("compute", ["logLike", "counts", "rootProb"])

    return alignment, tree, model, compute


def run(data):
    """Run subby computations from a parsed JSON input dict.

    Args:
        data: dict with keys 'model', 'tree', 'alignment', and optional 'compute'

    Returns:
        dict with requested outputs (logLike, counts, dwellTimes, rootProb)
    """
    alignment, tree, model, compute = load_input(data)
    result = {}

    if "logLike" in compute:
        ll = oracle.LogLike(alignment, tree, model)
        result["logLike"] = ll.tolist()

    if "counts" in compute:
        counts = oracle.Counts(alignment, tree, model)  # (A, A, C)
        result["counts"] = counts.tolist()
        # Extract dwell times from diagonal
        A = counts.shape[0]
        dwell = np.array([counts[a, a, :] for a in range(A)])  # (A, C)
        result["dwellTimes"] = dwell.tolist()

    if "rootProb" in compute:
        rp = oracle.RootProb(alignment, tree, model)  # (A, C)
        result["rootProb"] = rp.tolist()

    return result


def load_json(path_or_file):
    """Load JSON from a file path or file object.

    Args:
        path_or_file: str (file path) or file-like object

    Returns:
        parsed dict
    """
    if isinstance(path_or_file, str):
        with open(path_or_file) as f:
            return json.load(f)
    else:
        return json.load(path_or_file)


def format_output(result, indent=2):
    """Format output dict as JSON string.

    Args:
        result: dict from run()
        indent: JSON indentation level

    Returns:
        JSON string
    """
    return json.dumps(result, indent=indent)
