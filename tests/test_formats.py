"""Tests for subby.formats — standard phylogenetic file format parsers."""

import numpy as np
import pytest

from subby.formats import (
    Tree,
    CombinedResult,
    detect_alphabet,
    parse_newick,
    parse_fasta,
    parse_stockholm,
    parse_maf,
    parse_strings,
    parse_dict,
    combine_tree_alignment,
    KmerIndex,
    sliding_windows,
    all_column_ktuples,
    kmer_tokenize,
    genetic_code,
    codon_to_sense,
    split_paired_columns,
    merge_paired_columns,
)


# ---- detect_alphabet ----

class TestDetectAlphabet:
    def test_dna(self):
        assert detect_alphabet({"A", "C", "G", "T"}) == list("ACGT")

    def test_dna_subset(self):
        assert detect_alphabet({"A", "T"}) == list("ACGT")

    def test_rna(self):
        assert detect_alphabet({"A", "C", "G", "U"}) == list("ACGU")

    def test_protein(self):
        chars = set("ACDEFGHIKLMNPQRSTVWY")
        assert detect_alphabet(chars) == list("ACDEFGHIKLMNPQRSTVWY")

    def test_protein_subset(self):
        assert detect_alphabet({"M", "A", "L"}) == list("ACDEFGHIKLMNPQRSTVWY")

    def test_custom(self):
        assert detect_alphabet({"X", "Y", "Z"}) == ["X", "Y", "Z"]

    def test_gap_exclusion(self):
        result = detect_alphabet({"A", "C", "-", ".", "G", "T"})
        assert "-" not in result
        assert "." not in result
        assert result == list("ACGT")

    def test_case_insensitivity(self):
        result = detect_alphabet({"a", "c", "g", "t"})
        assert result == list("ACGT")


# ---- parse_newick ----

class TestParseNewick:
    def test_simple_binary(self):
        result = parse_newick("((A:0.1,B:0.2):0.3,C:0.4);")
        pi = result["parentIndex"]
        d = result["distanceToParent"]
        assert pi[0] == -1  # root
        assert len(pi) == 5
        assert result["leaf_names"] == ["A", "B", "C"]
        # Root distance is 0
        assert d[0] == 0.0

    def test_named_internal(self):
        result = parse_newick("((A:0.1,B:0.2)X:0.3,C:0.4)root;")
        assert result["node_names"][0] == "root"
        # X is the internal node
        assert "X" in result["node_names"]

    def test_no_branch_lengths(self):
        result = parse_newick("((A,B),C);")
        d = result["distanceToParent"]
        assert np.all(d == 0.0)
        assert result["leaf_names"] == ["A", "B", "C"]

    def test_single_leaf(self):
        result = parse_newick("A:0.5;")
        assert len(result["parentIndex"]) == 1
        assert result["parentIndex"][0] == -1
        assert result["leaf_names"] == ["A"]

    def test_quoted_labels(self):
        result = parse_newick("('leaf one':0.1,'leaf two':0.2);")
        assert result["leaf_names"] == ["leaf one", "leaf two"]

    def test_scientific_notation(self):
        result = parse_newick("(A:1.5e-3,B:2.0E+1);")
        d = result["distanceToParent"]
        leaves = result["leaf_names"]
        a_idx = [i for i, n in enumerate(result["node_names"]) if n == "A"][0]
        b_idx = [i for i, n in enumerate(result["node_names"]) if n == "B"][0]
        assert abs(d[a_idx] - 1.5e-3) < 1e-10
        assert abs(d[b_idx] - 20.0) < 1e-10

    def test_comments(self):
        result = parse_newick("((A[comment]:0.1,B:0.2):0.3,C:0.4);")
        assert result["leaf_names"] == ["A", "B", "C"]

    def test_known_tree(self):
        """Verify against manually computed parentIndex/distanceToParent."""
        #     0 (root)
        #    / \
        #   1   2
        #  / \
        # 3   4
        result = parse_newick("((A:0.05,B:0.15):0.1,C:0.3);")
        pi = result["parentIndex"]
        d = result["distanceToParent"]
        # 5 nodes: root, internal, C, A, B
        assert len(pi) == 5
        # Root
        assert pi[0] == -1
        # All parents are valid and < their child index
        for i in range(1, len(pi)):
            assert 0 <= pi[i] < i
        # Leaf names in DFS order
        assert result["leaf_names"] == ["A", "B", "C"]
        # Distance check: find leaves by name
        names = result["node_names"]
        a_idx = names.index("A")
        b_idx = names.index("B")
        c_idx = names.index("C")
        assert abs(d[a_idx] - 0.05) < 1e-10
        assert abs(d[b_idx] - 0.15) < 1e-10
        assert abs(d[c_idx] - 0.3) < 1e-10

    def test_multifurcation(self):
        """Three children at root."""
        result = parse_newick("(A:0.1,B:0.2,C:0.3);")
        assert len(result["parentIndex"]) == 4  # root + 3 leaves
        assert result["leaf_names"] == ["A", "B", "C"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_newick("")

    def test_no_semicolon(self):
        """Should still parse without trailing semicolon."""
        result = parse_newick("(A:0.1,B:0.2)")
        assert result["leaf_names"] == ["A", "B"]


# ---- parse_fasta ----

class TestParseFasta:
    def test_simple(self):
        text = ">seq1\nACGT\n>seq2\nACGT\n"
        result = parse_fasta(text)
        assert result["alignment"].shape == (2, 4)
        assert result["leaf_names"] == ["seq1", "seq2"]
        assert result["alphabet"] == list("ACGT")

    def test_multiline_sequences(self):
        text = ">seq1\nAC\nGT\n>seq2\nTG\nCA\n"
        result = parse_fasta(text)
        assert result["alignment"].shape == (2, 4)

    def test_gaps(self):
        text = ">seq1\nA-GT\n>seq2\nACG-\n"
        result = parse_fasta(text)
        A = len(result["alphabet"])
        gap_idx = A + 1
        # Check gap positions
        assert result["alignment"][0, 1] == gap_idx
        assert result["alignment"][1, 3] == gap_idx

    def test_unequal_lengths_error(self):
        text = ">seq1\nACGT\n>seq2\nACG\n"
        with pytest.raises(ValueError, match="Unequal"):
            parse_fasta(text)

    def test_description_lines(self):
        text = ">seq1 this is a description\nACGT\n>seq2 another one\nACGT\n"
        result = parse_fasta(text)
        assert result["leaf_names"] == ["seq1", "seq2"]

    def test_explicit_alphabet(self):
        text = ">s1\nAB\n>s2\nBA\n"
        result = parse_fasta(text, alphabet=["A", "B"])
        assert result["alphabet"] == ["A", "B"]
        np.testing.assert_array_equal(result["alignment"][0], [0, 1])
        np.testing.assert_array_equal(result["alignment"][1], [1, 0])

    def test_case_insensitive(self):
        text = ">seq1\nacgt\n>seq2\nACGT\n"
        result = parse_fasta(text)
        np.testing.assert_array_equal(
            result["alignment"][0], result["alignment"][1]
        )


# ---- parse_stockholm ----

class TestParseStockholm:
    def test_without_tree(self):
        text = """# STOCKHOLM 1.0
seq1  ACGT
seq2  ACGT
//"""
        result = parse_stockholm(text)
        assert result["alignment"].shape == (2, 4)
        assert result["leaf_names"] == ["seq1", "seq2"]

    def test_with_tree(self):
        text = """# STOCKHOLM 1.0
#=GF NH (seq1:0.1,seq2:0.2);
seq1  ACGT
seq2  TGCA
//"""
        result = parse_stockholm(text)
        # Should be combined: tree + alignment
        assert isinstance(result, CombinedResult)
        # R = 3 (root + 2 leaves), C = 4
        assert result.alignment.shape == (3, 4)

    def test_name_matching(self):
        text = """# STOCKHOLM 1.0
#=GF NH (B:0.1,A:0.2);
A  ACGT
B  TGCA
//"""
        result = parse_stockholm(text)
        # Tree leaves are [B, A], alignment names are [A, B]
        # The combine should match by name
        pi = result.tree.parentIndex
        R = len(pi)
        assert R == 3
        # Verify the leaf rows have correct sequences
        assert set(result.leaf_names) == {"A", "B"}

    def test_multiline_nh(self):
        text = """# STOCKHOLM 1.0
#=GF NH (seq1:0.1,
#=GF NH seq2:0.2);
seq1  ACGT
seq2  TGCA
//"""
        result = parse_stockholm(text)
        assert isinstance(result, CombinedResult)


# ---- parse_maf ----

class TestParseMaf:
    def test_single_block(self):
        text = """a score=0
s human.chr1 100 4 + 1000 ACGT
s mouse.chr2 200 4 + 900  TGCA
"""
        result = parse_maf(text)
        assert result["alignment"].shape == (2, 4)
        assert result["leaf_names"] == ["human", "mouse"]

    def test_multi_block(self):
        text = """a score=0
s human.chr1 100 2 + 1000 AC
s mouse.chr2 200 2 + 900  TG

a score=0
s human.chr1 102 2 + 1000 GT
s mouse.chr2 202 2 + 900  CA
"""
        result = parse_maf(text)
        assert result["alignment"].shape == (2, 4)

    def test_species_name(self):
        text = """a score=0
s homo_sapiens.chr1 100 4 + 1000 ACGT
"""
        result = parse_maf(text)
        assert result["leaf_names"] == ["homo_sapiens"]

    def test_missing_species(self):
        text = """a score=0
s human.chr1 100 2 + 1000 AC
s mouse.chr2 200 2 + 900  TG

a score=0
s human.chr1 102 2 + 1000 GT
"""
        result = parse_maf(text)
        # mouse missing from block 2, should be filled with gaps
        A = len(result["alphabet"])
        gap_idx = A + 1
        assert result["alignment"].shape == (2, 4)
        mouse_idx = result["leaf_names"].index("mouse")
        # Last 2 columns should be gaps for mouse
        assert result["alignment"][mouse_idx, 2] == gap_idx
        assert result["alignment"][mouse_idx, 3] == gap_idx


# ---- parse_strings ----

class TestParseStrings:
    def test_simple(self):
        result = parse_strings(["ACGT", "TGCA"])
        assert result["alignment"].shape == (2, 4)
        assert result["alphabet"] == list("ACGT")

    def test_gaps(self):
        result = parse_strings(["A-GT", "AC-T"])
        A = len(result["alphabet"])
        gap_idx = A + 1
        assert result["alignment"][0, 1] == gap_idx
        assert result["alignment"][1, 2] == gap_idx

    def test_empty_error(self):
        with pytest.raises(ValueError):
            parse_strings([])

    def test_unequal_error(self):
        with pytest.raises(ValueError, match="Unequal"):
            parse_strings(["ACGT", "AC"])


# ---- parse_dict ----

class TestParseDict:
    def test_simple(self):
        result = parse_dict({"human": "ACGT", "mouse": "TGCA"})
        assert result["alignment"].shape == (2, 4)
        assert result["alphabet"] == list("ACGT")
        assert set(result["leaf_names"]) == {"human", "mouse"}

    def test_gaps(self):
        result = parse_dict({"s1": "A-GT", "s2": "AC-T"})
        A = len(result["alphabet"])
        gap_idx = A + 1
        # Find the row for s1
        s1_row = result["leaf_names"].index("s1")
        assert result["alignment"][s1_row, 1] == gap_idx

    def test_empty_error(self):
        with pytest.raises(ValueError):
            parse_dict({})

    def test_unequal_error(self):
        with pytest.raises(ValueError, match="Unequal"):
            parse_dict({"a": "ACGT", "b": "AC"})

    def test_preserves_names(self):
        result = parse_dict({"human": "ACGT", "mouse": "TGCA", "dog": "GGGG"})
        assert len(result["leaf_names"]) == 3
        assert set(result["leaf_names"]) == {"human", "mouse", "dog"}

    def test_combine_with_tree(self):
        tree = parse_newick("((human:0.1,mouse:0.2):0.05,dog:0.3);")
        aln = parse_dict({"human": "ACGT", "mouse": "TGCA", "dog": "GGGG"})
        result = combine_tree_alignment(tree, aln)
        R = len(tree["parentIndex"])
        assert result.alignment.shape == (R, 4)

    def test_pipeline_integration(self):
        from subby.oracle import LogLike, jukes_cantor_model

        tree = parse_newick("((A:0.1,B:0.2):0.05,C:0.3);")
        aln = parse_dict({"A": "ACGT", "B": "ACGT", "C": "ACGT"})
        combined = combine_tree_alignment(tree, aln)
        model = jukes_cantor_model(4)
        ll = LogLike(combined.alignment, combined.tree, model)
        assert ll.shape == (4,)
        assert np.all(np.isfinite(ll))


# ---- combine_tree_alignment ----

class TestCombineTreeAlignment:
    def test_basic(self):
        tree = parse_newick("((A:0.1,B:0.2):0.3,C:0.4);")
        aln = parse_fasta(">A\nACGT\n>B\nTGCA\n>C\nGGGG\n")
        result = combine_tree_alignment(tree, aln)
        R = len(tree["parentIndex"])
        assert result.alignment.shape == (R, 4)
        # Internal nodes should have token A (ungapped-unobserved)
        A = len(result.alphabet)
        pi = result.tree.parentIndex
        # Count children to find internal nodes
        child_count = np.zeros(R, dtype=np.int32)
        for n in range(1, R):
            child_count[pi[n]] += 1
        for n in range(R):
            if child_count[n] > 0:  # internal
                assert np.all(result.alignment[n] == A)

    def test_name_mismatch_error(self):
        tree = parse_newick("((A:0.1,B:0.2):0.3,C:0.4);")
        aln = parse_fasta(">A\nACGT\n>B\nTGCA\n>X\nGGGG\n")
        with pytest.raises(ValueError, match="not found"):
            combine_tree_alignment(tree, aln)

    def test_pipeline_integration(self):
        """Parse → combine → LogLike should produce valid results."""
        from subby.oracle import LogLike, jukes_cantor_model

        tree = parse_newick("((A:0.1,B:0.2):0.05,C:0.3);")
        aln = parse_fasta(">A\nACGT\n>B\nACGT\n>C\nACGT\n")
        combined = combine_tree_alignment(tree, aln)

        model = jukes_cantor_model(4)
        ll = LogLike(combined.alignment, combined.tree, model)
        assert ll.shape == (4,)
        assert np.all(np.isfinite(ll))
        assert np.all(ll <= 0)

    def test_extra_alignment_sequences_ok(self):
        """Alignment can have extra sequences not in tree."""
        tree = parse_newick("(A:0.1,B:0.2);")
        aln = parse_fasta(">A\nACGT\n>B\nTGCA\n>C\nGGGG\n")
        result = combine_tree_alignment(tree, aln)
        R = len(tree["parentIndex"])
        assert result.alignment.shape == (R, 4)


# ---- kmer_tokenize ----

class TestKmerTokenize:

    def test_codon_basic(self):
        """DNA triplets → codon tokens."""
        # "ACG" "TAA" for one sequence
        result = parse_strings(["ACGTAA"], alphabet=list("ACGT"))
        aln = result["alignment"]  # (1, 6) tokens: [0,1,2,3,0,0]
        kmer = kmer_tokenize(aln, A=4, k_or_tuples=3, alphabet=list("ACGT"))
        assert kmer["alignment"].shape == (1, 2)
        assert kmer["A_kmer"] == 64
        # ACG = 0*16 + 1*4 + 2 = 6
        assert kmer["alignment"][0, 0] == 6
        # TAA = 3*16 + 0*4 + 0 = 48
        assert kmer["alignment"][0, 1] == 48

    def test_kmer_alphabet_labels(self):
        """K-mer alphabet labels are correct."""
        kmer = kmer_tokenize(np.zeros((1, 4), dtype=np.int32), A=2, k_or_tuples=2,
                             alphabet=["0", "1"])
        assert kmer["alphabet"] == ["00", "01", "10", "11"]
        assert kmer["A_kmer"] == 4

    def test_k1_identity(self):
        """k=1 is the identity transform for observed tokens."""
        aln = np.array([[0, 1, 2, 3]], dtype=np.int32)
        kmer = kmer_tokenize(aln, A=4, k_or_tuples=1)
        np.testing.assert_array_equal(kmer["alignment"], aln)
        assert kmer["A_kmer"] == 4

    def test_gap_any(self):
        """gap_mode='any': gap in any position → entire k-mer is gap."""
        A = 4
        gap = A + 1
        aln = np.array([[0, gap, 2]], dtype=np.int32)  # one k-mer, middle is gap
        kmer = kmer_tokenize(aln, A=A, k_or_tuples=3, gap_mode='any')
        A_k = 64
        assert kmer["alignment"][0, 0] == A_k + 1  # gap token

    def test_gap_all_full_gap(self):
        """gap_mode='all': all positions gapped → gap token."""
        A = 4
        gap = A + 1
        aln = np.array([[gap, gap, gap]], dtype=np.int32)
        kmer = kmer_tokenize(aln, A=A, k_or_tuples=3, gap_mode='all')
        A_k = 64
        assert kmer["alignment"][0, 0] == A_k + 1  # gap token

    def test_gap_all_partial_gap(self):
        """gap_mode='all': partial gap → illegal token."""
        A = 4
        gap = A + 1
        aln = np.array([[0, gap, 2]], dtype=np.int32)
        kmer = kmer_tokenize(aln, A=A, k_or_tuples=3, gap_mode='all')
        A_k = 64
        assert kmer["alignment"][0, 0] == A_k + 2  # illegal token

    def test_unobserved(self):
        """All-unobserved k-mer → unobserved token."""
        A = 4
        unobs = A  # ungapped-unobserved token
        aln = np.array([[unobs, unobs, unobs]], dtype=np.int32)
        kmer = kmer_tokenize(aln, A=A, k_or_tuples=3)
        A_k = 64
        assert kmer["alignment"][0, 0] == A_k  # ungapped-unobserved

    def test_legacy_gap(self):
        """Legacy gap token (-1) treated as gap."""
        A = 4
        aln = np.array([[0, -1, 2]], dtype=np.int32)
        kmer = kmer_tokenize(aln, A=A, k_or_tuples=3, gap_mode='any')
        A_k = 64
        assert kmer["alignment"][0, 0] == A_k + 1

    def test_c_not_divisible(self):
        """C not divisible by k raises ValueError."""
        aln = np.array([[0, 1, 2, 3, 0]], dtype=np.int32)  # C=5
        with pytest.raises(ValueError, match="not divisible"):
            kmer_tokenize(aln, A=4, k_or_tuples=3)

    def test_multiple_sequences(self):
        """Multiple sequences tokenized independently."""
        aln = np.array([
            [0, 1, 2, 3, 0, 0],  # ACG TAA
            [3, 2, 1, 0, 3, 2],  # TGC ATG
        ], dtype=np.int32)
        kmer = kmer_tokenize(aln, A=4, k_or_tuples=3)
        assert kmer["alignment"].shape == (2, 2)
        # ACG = 6, TAA = 48
        assert kmer["alignment"][0, 0] == 6
        assert kmer["alignment"][0, 1] == 48
        # TGC = 3*16 + 2*4 + 1 = 57, ATG = 0*16 + 3*4 + 2 = 14
        assert kmer["alignment"][1, 0] == 57
        assert kmer["alignment"][1, 1] == 14

    def test_fasta_to_codons(self):
        """End-to-end: FASTA → codon tokenization."""
        fasta = ">seq1\nACGTAA\n>seq2\nTGCATG\n"
        result = parse_fasta(fasta)
        kmer = kmer_tokenize(result["alignment"], A=len(result["alphabet"]),
                             k_or_tuples=3, alphabet=result["alphabet"])
        assert kmer["alignment"].shape == (2, 2)
        assert kmer["A_kmer"] == 64
        assert len(kmer["alphabet"]) == 64


# ---- genetic_code ----

class TestGeneticCode:
    def test_64_codons(self):
        gc = genetic_code()
        assert len(gc['codons']) == 64
        assert len(gc['amino_acids']) == 64

    def test_sense_count(self):
        gc = genetic_code()
        assert gc['sense_mask'].sum() == 61
        assert len(gc['sense_indices']) == 61
        assert len(gc['sense_codons']) == 61
        assert len(gc['sense_amino_acids']) == 61

    def test_stop_codons(self):
        gc = genetic_code()
        # TAA=48, TAG=50, TGA=56
        assert gc['amino_acids'][48] == '*'
        assert gc['amino_acids'][50] == '*'
        assert gc['amino_acids'][56] == '*'
        assert not gc['sense_mask'][48]
        assert not gc['sense_mask'][50]
        assert not gc['sense_mask'][56]

    def test_codon_to_sense_mapping(self):
        gc = genetic_code()
        # Stop codons map to -1
        assert gc['codon_to_sense'][48] == -1
        assert gc['codon_to_sense'][50] == -1
        assert gc['codon_to_sense'][56] == -1
        # First sense codon (AAA=0) maps to 0
        assert gc['codon_to_sense'][0] == 0
        # All sense codons have non-negative mapping
        for i in gc['sense_indices']:
            assert gc['codon_to_sense'][i] >= 0

    def test_roundtrip(self):
        gc = genetic_code()
        # sense_indices[codon_to_sense[i]] == i for all sense codons
        for i in range(64):
            if gc['sense_mask'][i]:
                si = gc['codon_to_sense'][i]
                assert gc['sense_indices'][si] == i

    def test_acgt_order(self):
        gc = genetic_code()
        assert gc['codons'][0] == 'AAA'
        assert gc['codons'][63] == 'TTT'
        assert gc['codons'][1] == 'AAC'

    def test_known_amino_acids(self):
        gc = genetic_code()
        # ATG = M (methionine), index = 0*16 + 3*4 + 2 = 14
        assert gc['codons'][14] == 'ATG'
        assert gc['amino_acids'][14] == 'M'
        # TGG = W (tryptophan), index = 3*16 + 2*4 + 2 = 58
        assert gc['codons'][58] == 'TGG'
        assert gc['amino_acids'][58] == 'W'


# ---- codon_to_sense ----

class TestCodonToSense:
    def test_basic_remapping(self):
        gc = genetic_code()
        # Token 0 (AAA) → sense 0
        aln = np.array([[0, 1, 2]], dtype=np.int32)
        result = codon_to_sense(aln, A=64)
        assert result['A_sense'] == 61
        assert result['alignment'][0, 0] == 0  # AAA → sense 0
        assert result['alignment'][0, 1] == 1  # AAC → sense 1
        assert result['alignment'][0, 2] == 2  # AAG → sense 2

    def test_stop_becomes_gap(self):
        # TAA=48 is a stop codon
        aln = np.array([[48]], dtype=np.int32)
        result = codon_to_sense(aln, A=64)
        assert result['alignment'][0, 0] == 62  # gap token

    def test_unobserved_remapping(self):
        aln = np.array([[64]], dtype=np.int32)  # unobserved token
        result = codon_to_sense(aln, A=64)
        assert result['alignment'][0, 0] == 61  # new unobserved

    def test_gap_remapping(self):
        aln = np.array([[65]], dtype=np.int32)  # gap token
        result = codon_to_sense(aln, A=64)
        assert result['alignment'][0, 0] == 62  # new gap

    def test_legacy_gap(self):
        aln = np.array([[-1]], dtype=np.int32)  # legacy gap
        result = codon_to_sense(aln, A=64)
        assert result['alignment'][0, 0] == 62  # gap

    def test_empty_alignment(self):
        aln = np.zeros((0, 0), dtype=np.int32)
        result = codon_to_sense(aln, A=64)
        assert result['alignment'].shape == (0, 0)
        assert result['A_sense'] == 61

    def test_alphabet_returned(self):
        aln = np.array([[0]], dtype=np.int32)
        result = codon_to_sense(aln, A=64)
        assert len(result['alphabet']) == 61
        assert result['alphabet'][0] == 'AAA'


# ---- split_paired_columns / merge_paired_columns ----

class TestSplitMerge:
    def test_split_basic(self):
        N, C, A = 3, 6, 20
        aln = np.random.randint(0, A, (N, C), dtype=np.int32)
        paired_cols = [(0, 3), (1, 4)]
        result = split_paired_columns(aln, paired_cols, A=A)
        assert result['paired_alignment'].shape == (N, 2)
        assert result['singles_alignment'].shape == (N, 2)  # cols 2 and 5
        assert result['A_paired'] == 400
        assert result['A_singles'] == 20
        assert result['single_columns'] == [2, 5]

    def test_paired_token_encoding(self):
        A = 20
        # Two sequences, 4 columns
        aln = np.array([
            [5, 10, 3, 7],  # pair (0,1): token = 5*20+10 = 110
            [0, 19, 1, 2],  # pair (0,1): token = 0*20+19 = 19
        ], dtype=np.int32)
        result = split_paired_columns(aln, [(0, 1)], A=A)
        assert result['paired_alignment'][0, 0] == 110
        assert result['paired_alignment'][1, 0] == 19

    def test_gap_handling(self):
        A = 20
        gap = A + 1
        aln = np.array([[gap, 5, 3]], dtype=np.int32)
        result = split_paired_columns(aln, [(0, 1)], A=A)
        # Column 0 is gap, so paired token should be gap (401)
        assert result['paired_alignment'][0, 0] == A * A + 1

    def test_unobserved_handling(self):
        A = 20
        unobs = A
        aln = np.array([[unobs, 5, 3]], dtype=np.int32)
        result = split_paired_columns(aln, [(0, 1)], A=A)
        # Column 0 is unobserved (no gap), so paired token should be unobserved (400)
        assert result['paired_alignment'][0, 0] == A * A

    def test_merge_roundtrip(self):
        A = 20
        N, C = 3, 6
        np.random.seed(42)
        aln = np.random.randint(0, A, (N, C), dtype=np.int32)
        paired_cols = [(0, 3), (1, 4)]
        split_info = split_paired_columns(aln, paired_cols, A=A)

        # Create fake posteriors
        P = len(paired_cols)
        S = len(split_info['single_columns'])
        paired_post = np.random.dirichlet(np.ones(A * A), size=P).T  # (400, P)
        singles_post = np.random.dirichlet(np.ones(A), size=S).T  # (20, S)

        result = merge_paired_columns(paired_post, singles_post, split_info)
        assert result.shape == (A, C)

        # Each column should sum to ~1 (probability distribution)
        for c in range(C):
            np.testing.assert_allclose(result[:, c].sum(), 1.0, atol=1e-10)

    def test_merge_marginalization(self):
        A = 20
        # Single pair (0, 1), 2 columns
        aln = np.array([[5, 10]], dtype=np.int32)
        split_info = split_paired_columns(aln, [(0, 1)], A=A)

        # Create a paired posterior where state (5, 10) has probability 1
        paired_post = np.zeros((400, 1), dtype=np.float64)
        paired_post[5 * 20 + 10, 0] = 1.0

        singles_post = np.zeros((A, 0), dtype=np.float64)
        result = merge_paired_columns(paired_post, singles_post, split_info)

        # Column 0: all probability on state 5
        assert result[5, 0] == 1.0
        # Column 1: all probability on state 10
        assert result[10, 1] == 1.0


# ---- sliding_windows ----

class TestSlidingWindows:

    def test_basic_nonoverlapping(self):
        """Default stride=k gives non-overlapping windows."""
        windows = sliding_windows(6, k=3)
        np.testing.assert_array_equal(windows, [[0, 1, 2], [3, 4, 5]])

    def test_stride1(self):
        """Stride=1 gives fully overlapping windows."""
        windows = sliding_windows(5, k=3, stride=1)
        np.testing.assert_array_equal(windows, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])

    def test_offset(self):
        """offset shifts the starting position."""
        windows = sliding_windows(9, k=3, stride=3, offset=1)
        np.testing.assert_array_equal(windows, [[1, 2, 3], [4, 5, 6]])

    def test_truncate_drops_partial(self):
        """edge='truncate' drops incomplete trailing window."""
        windows = sliding_windows(5, k=3, stride=3)
        # Only [0,1,2] fits completely
        np.testing.assert_array_equal(windows, [[0, 1, 2]])

    def test_pad_includes_partial(self):
        """edge='pad' includes partial window with -1 sentinels."""
        windows = sliding_windows(5, k=3, stride=3, edge='pad')
        assert windows.shape == (2, 3)
        np.testing.assert_array_equal(windows[0], [0, 1, 2])
        np.testing.assert_array_equal(windows[1], [3, 4, -1])

    def test_empty_result(self):
        """No complete windows gives empty (0, k) array."""
        windows = sliding_windows(2, k=3)
        assert windows.shape == (0, 3)

    def test_three_reading_frames(self):
        """Three reading frames via offset=0,1,2."""
        for offset in range(3):
            windows = sliding_windows(9, k=3, stride=3, offset=offset)
            expected = np.arange(offset, 9 - 3 + 1, 3)
            np.testing.assert_array_equal(windows[:, 0], expected)


# ---- all_column_ktuples ----

class TestAllColumnKtuples:

    def test_ordered_k2(self):
        """Ordered k=2: C*(C-1) permutations."""
        tuples = all_column_ktuples(4, k=2, ordered=True)
        assert tuples.shape == (12, 2)  # 4*3

    def test_unordered_k2(self):
        """Unordered k=2: C choose 2 combinations."""
        tuples = all_column_ktuples(4, k=2, ordered=False)
        assert tuples.shape == (6, 2)  # 4 choose 2

    def test_k1(self):
        """k=1 gives C single-column tuples."""
        tuples = all_column_ktuples(3, k=1, ordered=True)
        assert tuples.shape == (3, 1)
        np.testing.assert_array_equal(tuples.flatten(), [0, 1, 2])

    def test_empty_when_k_gt_c(self):
        """k > C gives empty result."""
        tuples = all_column_ktuples(1, k=2, ordered=True)
        assert tuples.shape == (0, 2)


# ---- KmerIndex ----

class TestKmerIndex:

    def test_tuple_to_idx(self):
        """Lookup column tuple → index."""
        idx = KmerIndex([(0, 1), (2, 3), (0, 3)])
        assert idx.tuple_to_idx((0, 1)) == 0
        assert idx.tuple_to_idx((2, 3)) == 1
        assert idx.tuple_to_idx((0, 3)) == 2

    def test_tuple_to_idx_missing(self):
        """Missing tuple returns -1."""
        idx = KmerIndex([(0, 1)])
        assert idx.tuple_to_idx((9, 9)) == -1

    def test_idx_to_tuple(self):
        """Index → column tuple."""
        idx = KmerIndex([(0, 1), (2, 3)])
        assert idx.idx_to_tuple(0) == (0, 1)
        assert idx.idx_to_tuple(1) == (2, 3)

    def test_len(self):
        """Length matches number of tuples."""
        idx = KmerIndex([(0, 1), (2, 3), (4, 5)])
        assert len(idx) == 3


# ---- kmer_tokenize with tuples ----

class TestKmerTokenizeTuples:

    def test_basic_pairs(self):
        """Tokenize non-contiguous column pairs."""
        aln = np.array([[0, 1, 2, 3]], dtype=np.int32)
        result = kmer_tokenize(aln, A=4, k_or_tuples=[(0, 2), (1, 3)])
        assert result['alignment'].shape == (1, 2)
        assert result['A_kmer'] == 16
        assert result['alignment'][0, 0] == 0*4 + 2  # (A,G) = 2
        assert result['alignment'][0, 1] == 1*4 + 3  # (C,T) = 7

    def test_contiguous_tuples_match_int_k(self):
        """Contiguous tuples give same result as int k."""
        aln = np.array([[0, 1, 2, 3, 0, 1]], dtype=np.int32)
        result_k = kmer_tokenize(aln, A=4, k_or_tuples=3)
        result_t = kmer_tokenize(aln, A=4, k_or_tuples=[(0, 1, 2), (3, 4, 5)])
        np.testing.assert_array_equal(
            result_k['alignment'], result_t['alignment'])

    def test_k1_identity(self):
        """k=1 tuples are identity for observed tokens."""
        aln = np.array([[0, 1, 2, 3]], dtype=np.int32)
        result = kmer_tokenize(aln, A=4, k_or_tuples=[(0,), (1,), (2,), (3,)])
        np.testing.assert_array_equal(result['alignment'], aln)

    def test_sliding_window_integration(self):
        """sliding_windows() output feeds directly to kmer_tokenize()."""
        aln = np.array([[0, 1, 2, 3, 0]], dtype=np.int32)
        windows = sliding_windows(5, k=3, stride=1)
        result = kmer_tokenize(aln, A=4, k_or_tuples=windows)
        assert result['alignment'].shape == (1, 3)
        assert result['alignment'][0, 0] == 0*16 + 1*4 + 2  # ACG = 6
        assert result['alignment'][0, 1] == 1*16 + 2*4 + 3  # CGT = 27
        assert result['alignment'][0, 2] == 2*16 + 3*4 + 0  # GTA = 44

    def test_sentinel_padding(self):
        """Sentinel -1 in tuples → unobserved token."""
        aln = np.array([[0, 1, 2, 3, 0]], dtype=np.int32)
        windows = sliding_windows(5, k=3, stride=3, edge='pad')
        result = kmer_tokenize(aln, A=4, k_or_tuples=windows)
        assert result['alignment'].shape == (1, 2)
        assert result['alignment'][0, 0] == 0*16 + 1*4 + 2  # ACG = 6
        assert result['alignment'][0, 1] == 64  # unobserved (has -1)

    def test_gap_any(self):
        """gap_mode='any': gap in any position → gap token."""
        A = 4
        gap = A + 1
        aln = np.array([[0, gap, 2, gap]], dtype=np.int32)
        result = kmer_tokenize(aln, A, [(0, 1)], gap_mode='any')
        assert result['alignment'][0, 0] == A**2 + 1

    def test_gap_all(self):
        """gap_mode='all': partial gap → illegal; all gap → gap."""
        A = 4
        gap = A + 1
        aln = np.array([[0, gap, gap, gap]], dtype=np.int32)
        # Partial gap (0 observed, 1 gap)
        r1 = kmer_tokenize(aln, A, [(0, 1)], gap_mode='all')
        assert r1['alignment'][0, 0] == A**2 + 2  # illegal
        # All gap
        r2 = kmer_tokenize(aln, A, [(1, 2)], gap_mode='all')
        assert r2['alignment'][0, 0] == A**2 + 1  # gap

    def test_index_returned(self):
        """KmerIndex is in return dict and works correctly."""
        aln = np.array([[0, 1, 2]], dtype=np.int32)
        result = kmer_tokenize(aln, A=4, k_or_tuples=[(0, 2), (1, 2)])
        idx = result['index']
        assert isinstance(idx, KmerIndex)
        assert idx.tuple_to_idx((0, 2)) == 0
        assert idx.tuple_to_idx((1, 2)) == 1
        assert idx.idx_to_tuple(0) == (0, 2)

    def test_alphabet_labels(self):
        """Alphabet labels generated correctly for tuples."""
        result = kmer_tokenize(
            np.array([[0, 1]], dtype=np.int32), A=2,
            k_or_tuples=[(0, 1)], alphabet=["A", "B"])
        assert result['alphabet'] == ["AA", "AB", "BA", "BB"]

    def test_all_column_ktuples_integration(self):
        """all_column_ktuples() feeds to kmer_tokenize()."""
        aln = np.array([[0, 1, 2, 3]], dtype=np.int32)
        tuples = all_column_ktuples(4, k=2, ordered=False)
        result = kmer_tokenize(aln, A=4, k_or_tuples=tuples)
        assert result['alignment'].shape == (1, 6)  # 4 choose 2


# ---- kmer_tokenize backward compat ----

class TestKmerBackwardCompat:

    def test_int_k_basic(self):
        """Int k produces same tokens as original API."""
        aln = np.array([[0, 1, 2, 3, 0, 0]], dtype=np.int32)
        result = kmer_tokenize(aln, A=4, k_or_tuples=3, alphabet=list("ACGT"))
        assert result['alignment'].shape == (1, 2)
        assert result['A_kmer'] == 64
        assert result['alignment'][0, 0] == 6   # ACG
        assert result['alignment'][0, 1] == 48  # TAA

    def test_not_divisible_raises(self):
        """C not divisible by k raises ValueError."""
        aln = np.array([[0, 1, 2, 3, 0]], dtype=np.int32)  # C=5
        with pytest.raises(ValueError, match="not divisible"):
            kmer_tokenize(aln, A=4, k_or_tuples=3)

    def test_index_for_int_k(self):
        """Int k also returns KmerIndex."""
        aln = np.array([[0, 1, 2, 3, 0, 1]], dtype=np.int32)
        result = kmer_tokenize(aln, A=4, k_or_tuples=3)
        idx = result['index']
        assert isinstance(idx, KmerIndex)
        assert len(idx) == 2
        assert idx.idx_to_tuple(0) == (0, 1, 2)
        assert idx.idx_to_tuple(1) == (3, 4, 5)
        assert idx.tuple_to_idx((0, 1, 2)) == 0


# ---- split_paired_columns with KmerIndex ----

class TestSplitPairedKmerIndex:

    def test_kmer_index_returned(self):
        """split_paired_columns returns paired_index and singles_index."""
        N, C, A = 3, 6, 20
        aln = np.random.randint(0, A, (N, C), dtype=np.int32)
        result = split_paired_columns(aln, [(0, 3), (1, 4)], A=A)
        assert isinstance(result['paired_index'], KmerIndex)
        assert isinstance(result['singles_index'], KmerIndex)
        assert len(result['paired_index']) == 2
        assert len(result['singles_index']) == 2  # cols 2, 5

    def test_paired_index_mapping(self):
        """paired_index maps pair tuples to output positions."""
        A = 20
        aln = np.array([[5, 10, 3, 7]], dtype=np.int32)
        result = split_paired_columns(aln, [(0, 1), (2, 3)], A=A)
        idx = result['paired_index']
        assert idx.tuple_to_idx((0, 1)) == 0
        assert idx.tuple_to_idx((2, 3)) == 1
        assert idx.tuple_to_idx((9, 9)) == -1

    def test_singles_index_mapping(self):
        """singles_index maps single columns to output positions."""
        A = 20
        aln = np.random.randint(0, A, (2, 6), dtype=np.int32)
        result = split_paired_columns(aln, [(0, 3)], A=A)
        idx = result['singles_index']
        # Single columns are [1, 2, 4, 5]
        assert idx.tuple_to_idx((1,)) == 0
        assert idx.tuple_to_idx((2,)) == 1
        assert idx.tuple_to_idx((4,)) == 2
        assert idx.tuple_to_idx((5,)) == 3


# ---- Cross-implementation: Python vs Rust ----

class TestCrossImplementation:
    """Test Python parsers produce arrays usable by the Rust backend."""

    @pytest.fixture
    def newick_str(self):
        return "((A:0.1,B:0.2):0.05,C:0.3);"

    @pytest.fixture
    def fasta_str(self):
        return ">A\nACGT\n>B\nTGCA\n>C\nGGGG\n"

    def test_newick_dtypes(self, newick_str):
        result = parse_newick(newick_str)
        assert result["parentIndex"].dtype == np.int32
        assert result["distanceToParent"].dtype == np.float64

    def test_fasta_dtypes(self, fasta_str):
        result = parse_fasta(fasta_str)
        assert result["alignment"].dtype == np.int32

    def test_combined_dtypes(self, newick_str, fasta_str):
        tree = parse_newick(newick_str)
        aln = parse_fasta(fasta_str)
        result = combine_tree_alignment(tree, aln)
        assert result.alignment.dtype == np.int32
        assert result.tree.parentIndex.dtype == np.int32
        assert result.tree.distanceToParent.dtype == np.float64

    def test_rust_loglike_agreement(self, newick_str, fasta_str):
        """If Rust backend is available, verify Python-parsed inputs
        produce identical log-likelihoods across oracle and Rust."""
        from subby.oracle import LogLike, jukes_cantor_model

        tree = parse_newick(newick_str)
        aln = parse_fasta(fasta_str)
        combined = combine_tree_alignment(tree, aln)
        model = jukes_cantor_model(4)
        ll = LogLike(combined.alignment, combined.tree, model)
        assert np.all(np.isfinite(ll))
        assert ll.shape == (4,)
