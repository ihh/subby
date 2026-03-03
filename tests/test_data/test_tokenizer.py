"""Tests for the pure-JAX FSM tokenizer."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax.numpy as jnp
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.tokenizer import tokenize_msa


class TestSUBSTokens:

    def test_identity_mapping(self):
        """SUBS tokens should be identity: A=0,C=1,G=2,T=3,N=4,gap=5."""
        msa = jnp.array([[0, 1, 2, 3, 4, 5]], dtype=jnp.int32)
        tok = tokenize_msa(msa)
        np.testing.assert_array_equal(tok.subs, msa)


class TestPHASETokens:

    def test_ungapped_is_zero(self):
        """Non-gap positions should get phase token 0 (NUC)."""
        msa = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        tok = tokenize_msa(msa)
        np.testing.assert_array_equal(tok.phase, jnp.zeros((1, 4), dtype=jnp.int32))

    def test_gap_mod3(self):
        """Gap positions should get 1 + (gap_len_mod_3)."""
        # Row: A, gap, gap, gap, A, gap, gap
        msa = jnp.array([[0, 5, 5, 5, 0, 5, 5]], dtype=jnp.int32)
        tok = tokenize_msa(msa)
        # gap_len_mod3: first gap=0->tok=1, second=1->tok=2, third=2->tok=3
        # then reset, gap=0->tok=1, gap=1->tok=2
        expected = jnp.array([[0, 1, 2, 3, 0, 1, 2]], dtype=jnp.int32)
        np.testing.assert_array_equal(tok.phase, expected)


class TestTRIPLETTokens:

    def test_simple_triplet(self):
        """A C G -> triplet at C = A*16 + C*4 + G = 0*16 + 1*4 + 2 = 6."""
        msa = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        tok = tokenize_msa(msa)
        # Position 0: prev=BOS(-1) -> gapped=64
        # Position 1: prev=A(0), curr=C(1), next=G(2) -> 0*16+1*4+2 = 6
        # Position 2: next=EOS(-1) -> gapped=64
        expected = jnp.array([[64, 6, 64]], dtype=jnp.int32)
        np.testing.assert_array_equal(tok.triplets, expected)

    def test_gap_skipping(self):
        """Prev/next should skip gaps: A, gap, C, G."""
        msa = jnp.array([[0, 5, 1, 2]], dtype=jnp.int32)
        tok = tokenize_msa(msa)
        # Position 0: prev=BOS -> 64
        # Position 1: gap -> 64
        # Position 2: prev=A(0), curr=C(1), next=G(2) -> 0*16+1*4+2 = 6
        # Position 3: next=EOS -> 64
        expected = jnp.array([[64, 64, 6, 64]], dtype=jnp.int32)
        np.testing.assert_array_equal(tok.triplets, expected)

    def test_N_makes_gap(self):
        """If any of prev/curr/next is N, emit gap (64)."""
        msa = jnp.array([[0, 4, 2]], dtype=jnp.int32)  # A, N, G
        tok = tokenize_msa(msa)
        # Position 1: curr=N -> 64
        expected_1 = 64
        assert tok.triplets[0, 1] == expected_1

    def test_all_nucs(self):
        """A C G T A -> middle three should have valid triplets."""
        msa = jnp.array([[0, 1, 2, 3, 0]], dtype=jnp.int32)
        tok = tokenize_msa(msa)
        # pos 1: prev=A(0), curr=C(1), next=G(2) -> 0*16+1*4+2 = 6
        assert tok.triplets[0, 1] == 6
        # pos 2: prev=C(1), curr=G(2), next=T(3) -> 1*16+2*4+3 = 27
        assert tok.triplets[0, 2] == 27
        # pos 3: prev=G(2), curr=T(3), next=A(0) -> 2*16+3*4+0 = 44
        assert tok.triplets[0, 3] == 44


class TestANNOTTokens:

    def test_no_annotations(self):
        """Without annotations: non-root rows are gapped, root is ungapped-unobs."""
        msa = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        tok = tokenize_msa(msa, annotations=None, root_row=0, M=15)
        # Root row 0: ungapped -> 15 (M)
        assert tok.annot[0, 0] == 15
        assert tok.annot[0, 1] == 15
        # Non-root row 1: no annotations -> 16 (M+1, gapped)
        assert tok.annot[1, 0] == 16
        assert tok.annot[1, 1] == 16

    def test_with_annotations(self):
        """With annotations: label values preserved, -1 -> gapped."""
        msa = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        annot = jnp.array([[-1, -1], [5, 10]], dtype=jnp.int32)
        tok = tokenize_msa(msa, annotations=annot, root_row=0, M=15)
        # Root row 0: ungapped-unobs regardless of annotations
        assert tok.annot[0, 0] == 15
        assert tok.annot[0, 1] == 15
        # Row 1: annotation labels
        assert tok.annot[1, 0] == 5
        assert tok.annot[1, 1] == 10

    def test_root_gap_is_gapped(self):
        """Root row gaps should be gapped (M+1)."""
        msa = jnp.array([[5, 0]], dtype=jnp.int32)  # gap, A
        tok = tokenize_msa(msa, root_row=0, M=15)
        assert tok.annot[0, 0] == 16  # gap at root -> M+1
        assert tok.annot[0, 1] == 15  # ungapped at root -> M


class TestMultiRow:

    def test_vmap_over_rows(self):
        """Tokenizer should handle multiple rows correctly."""
        msa = jnp.array([
            [0, 1, 2, 3],
            [5, 5, 0, 1],
            [2, 3, 5, 0],
        ], dtype=jnp.int32)
        tok = tokenize_msa(msa)
        assert tok.subs.shape == (3, 4)
        assert tok.triplets.shape == (3, 4)
        assert tok.phase.shape == (3, 4)
        assert tok.annot.shape == (3, 4)
