#!/usr/bin/env python3
"""Generate golden test files from the Python oracle.

These JSON files serve as the cross-language test oracle for WebGPU and WASM
implementations. Each file contains deterministic inputs and expected outputs
for a specific test case.

Test cases:
  1. 5-node JC A=4
  2. 7-node HKY85 A=4
  3. 7-node JC A=64
  4. 7-node JC A=4 with mixed gaps
  5. 7-node JC A=4 with all gaps (one column)

Usage:
    python scripts/generate_golden_tests.py
"""

import json
import os

import numpy as np

from subby.oracle import oracle


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'tests', 'golden'
)


def _to_list(arr):
    """Convert numpy array to nested Python list for JSON serialization."""
    return arr.tolist()


def _make_balanced_tree(R, seed=0):
    """Build a balanced binary tree with R nodes and deterministic branch lengths."""
    rng = np.random.RandomState(seed)
    parentIndex = np.zeros(R, dtype=np.int32)
    parentIndex[0] = -1
    for i in range(1, R):
        parentIndex[i] = (i - 1) // 2
    distances = rng.uniform(0.01, 0.5, size=R)
    distances[0] = 0.0
    return parentIndex, distances


def _run_full_pipeline(alignment, parentIndex, distances, model, f81_fast=False):
    """Run all oracle functions and return inputs + outputs as a dict."""
    tree = {'parentIndex': parentIndex, 'distanceToParent': distances}
    A = len(model['pi'])

    # Intermediates
    sub_matrices = oracle.compute_sub_matrices(model, distances)
    U, logNormU, logLike = oracle.upward_pass(alignment, tree, sub_matrices, model['pi'])
    D, logNormD = oracle.downward_pass(U, logNormU, tree, sub_matrices, model['pi'], alignment)

    if f81_fast:
        counts = oracle.f81_counts(
            U, D, logNormU, logNormD, logLike,
            distances, model['pi'], parentIndex,
        )
    else:
        J = oracle.compute_J(model['eigenvalues'], distances)
        U_tilde, D_tilde = oracle.eigenbasis_project(U, D, model)
        C_eigen = oracle.accumulate_C(D_tilde, U_tilde, J, logNormU, logNormD, logLike, parentIndex)
        counts = oracle.back_transform(C_eigen, model)

    root_prob = oracle.RootProb(alignment, tree, model)
    branch_mask = oracle.compute_branch_mask(alignment, parentIndex, A)

    result = {
        'inputs': {
            'alignment': _to_list(alignment),
            'parentIndex': _to_list(parentIndex),
            'distanceToParent': _to_list(distances),
            'model': {
                'eigenvalues': _to_list(model['eigenvalues']),
                'eigenvectors': _to_list(model['eigenvectors']),
                'pi': _to_list(model['pi']),
            },
            'A': A,
        },
        'intermediates': {
            'sub_matrices': _to_list(sub_matrices),
            'U': _to_list(U),
            'logNormU': _to_list(logNormU),
            'D': _to_list(D),
            'logNormD': _to_list(logNormD),
        },
        'outputs': {
            'logLike': _to_list(logLike),
            'counts': _to_list(counts),
            'root_prob': _to_list(root_prob),
            'branch_mask': _to_list(branch_mask),
        },
    }

    if not f81_fast:
        result['intermediates']['J'] = _to_list(J)
        result['intermediates']['U_tilde'] = _to_list(U_tilde)
        result['intermediates']['D_tilde'] = _to_list(D_tilde)
        result['intermediates']['C_eigen'] = _to_list(C_eigen)

    return result


def generate_case_1():
    """5-node JC A=4."""
    R = 5
    parentIndex, distances = _make_balanced_tree(R, seed=1001)
    rng = np.random.RandomState(2001)
    alignment = rng.randint(0, 4, size=(R, 8)).astype(np.int32)
    model = oracle.jukes_cantor_model(4)
    return _run_full_pipeline(alignment, parentIndex, distances, model)


def generate_case_2():
    """7-node HKY85 A=4."""
    R = 7
    parentIndex, distances = _make_balanced_tree(R, seed=1002)
    rng = np.random.RandomState(2002)
    alignment = rng.randint(0, 4, size=(R, 8)).astype(np.int32)
    model = oracle.hky85_diag(2.0, np.array([0.3, 0.2, 0.25, 0.25]))
    return _run_full_pipeline(alignment, parentIndex, distances, model)


def generate_case_3():
    """7-node JC A=64."""
    R = 7
    parentIndex, distances = _make_balanced_tree(R, seed=1003)
    rng = np.random.RandomState(2003)
    alignment = rng.randint(0, 64, size=(R, 6)).astype(np.int32)
    model = oracle.jukes_cantor_model(64)
    return _run_full_pipeline(alignment, parentIndex, distances, model)


def generate_case_4():
    """7-node JC A=4 with mixed gaps."""
    R = 7
    parentIndex, distances = _make_balanced_tree(R, seed=1004)
    rng = np.random.RandomState(2004)
    alignment = rng.randint(0, 4, size=(R, 8)).astype(np.int32)
    # Introduce gaps: set some entries to -1 (gap) and A (ungapped-unobserved)
    alignment[3, 0] = -1
    alignment[4, 0] = -1
    alignment[5, 1] = -1
    alignment[6, 1] = -1
    alignment[3, 2] = 4  # ungapped-unobserved
    alignment[5, 3] = -1
    alignment[6, 3] = -1
    alignment[3, 3] = -1
    alignment[4, 3] = -1
    model = oracle.jukes_cantor_model(4)
    return _run_full_pipeline(alignment, parentIndex, distances, model)


def generate_case_5():
    """7-node JC A=4 with all-gap column."""
    R = 7
    parentIndex, distances = _make_balanced_tree(R, seed=1005)
    rng = np.random.RandomState(2005)
    alignment = rng.randint(0, 4, size=(R, 4)).astype(np.int32)
    # Make column 0 all gaps
    alignment[:, 0] = -1
    model = oracle.jukes_cantor_model(4)
    return _run_full_pipeline(alignment, parentIndex, distances, model)


def generate_mixture_case():
    """7-node mixture of 3 rate-scaled JC4 models."""
    R = 7
    parentIndex, distances = _make_balanced_tree(R, seed=1006)
    rng = np.random.RandomState(2006)
    alignment = rng.randint(0, 4, size=(R, 8)).astype(np.int32)
    tree = {'parentIndex': parentIndex, 'distanceToParent': distances}

    base_model = oracle.jukes_cantor_model(4)
    rates = [0.5, 1.0, 2.0]
    models = [oracle.scale_model(base_model, r) for r in rates]
    log_weights = np.log(np.ones(3) / 3.0)

    # Compute per-component log-likelihoods
    log_likes = np.zeros((3, 8), dtype=np.float64)
    for k in range(3):
        log_likes[k, :] = oracle.LogLike(alignment, tree, models[k])

    posteriors = oracle.mixture_posterior(log_likes, log_weights)

    return {
        'inputs': {
            'alignment': _to_list(alignment),
            'parentIndex': _to_list(parentIndex),
            'distanceToParent': _to_list(distances),
            'base_model': {
                'eigenvalues': _to_list(base_model['eigenvalues']),
                'eigenvectors': _to_list(base_model['eigenvectors']),
                'pi': _to_list(base_model['pi']),
            },
            'rates': rates,
            'log_weights': _to_list(log_weights),
        },
        'outputs': {
            'log_likes': _to_list(log_likes),
            'posteriors': _to_list(posteriors),
        },
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cases = [
        ('5node_jc4', generate_case_1),
        ('7node_hky85_4', generate_case_2),
        ('7node_jc64', generate_case_3),
        ('7node_jc4_mixed_gaps', generate_case_4),
        ('7node_jc4_all_gaps', generate_case_5),
        ('7node_mixture_jc4', generate_mixture_case),
    ]

    for name, fn in cases:
        print(f'Generating {name}...')
        result = fn()
        path = os.path.join(OUTPUT_DIR, f'{name}.json')
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  -> {path}')

    print(f'\nAll {len(cases)} golden files generated.')


if __name__ == '__main__':
    main()
