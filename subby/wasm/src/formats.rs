//! Parsers for standard phylogenetic file formats.
//!
//! Converts Newick trees, FASTA/Stockholm/MAF alignments, and plain strings
//! into subby's internal representation: flat i32 alignment + parentIndex /
//! distanceToParent arrays.

/// Parsed tree in DFS preorder.
#[derive(Debug, Clone)]
pub struct ParsedTree {
    pub parent_index: Vec<i32>,
    pub distance_to_parent: Vec<f64>,
    pub leaf_names: Vec<String>,
    pub node_names: Vec<Option<String>>,
}

/// Parsed alignment (leaf sequences only).
#[derive(Debug, Clone)]
pub struct ParsedAlignment {
    /// Flat (N * C) i32, row-major.
    pub alignment: Vec<i32>,
    pub leaf_names: Vec<String>,
    pub alphabet: Vec<char>,
    pub n: usize,
    pub c: usize,
}

/// Combined tree + alignment for subby pipeline.
#[derive(Debug, Clone)]
pub struct CombinedInput {
    /// Flat (R * C) i32, row-major.
    pub alignment: Vec<i32>,
    pub parent_index: Vec<i32>,
    pub distance_to_parent: Vec<f64>,
    pub alphabet: Vec<char>,
    pub leaf_names: Vec<String>,
    pub r: usize,
    pub c: usize,
}

// ---- Alphabet detection ----

const DNA: &[char] = &['A', 'C', 'G', 'T'];
const RNA: &[char] = &['A', 'C', 'G', 'U'];
const PROTEIN: &[char] = &[
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
];

fn is_gap(ch: char) -> bool {
    ch == '-' || ch == '.'
}

/// Auto-detect alphabet from a set of characters.
pub fn detect_alphabet(chars: &std::collections::HashSet<char>) -> Vec<char> {
    let upper: std::collections::HashSet<char> = chars
        .iter()
        .map(|c| c.to_ascii_uppercase())
        .filter(|c| !is_gap(*c))
        .collect();

    let is_subset = |set: &std::collections::HashSet<char>, arr: &[char]| -> bool {
        set.iter().all(|c| arr.contains(c))
    };

    if is_subset(&upper, DNA) {
        return DNA.to_vec();
    }
    if is_subset(&upper, RNA) {
        return RNA.to_vec();
    }
    if is_subset(&upper, PROTEIN) {
        return PROTEIN.to_vec();
    }
    let mut sorted: Vec<char> = upper.into_iter().collect();
    sorted.sort();
    sorted
}

// ---- Newick parser ----

fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth: i32 = 0;
    let mut in_quote = false;
    for ch in s.chars() {
        if ch == '\'' && depth == 0 {
            in_quote = !in_quote;
            out.push(ch);
        } else if in_quote {
            out.push(ch);
        } else if ch == '[' {
            depth += 1;
        } else if ch == ']' {
            depth -= 1;
        } else if depth == 0 {
            out.push(ch);
        }
    }
    out
}

struct NewickTokenizer {
    chars: Vec<char>,
    pos: usize,
}

impl NewickTokenizer {
    fn new(s: &str) -> Self {
        NewickTokenizer {
            chars: s.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn consume(&mut self, expected: Option<char>) -> char {
        let ch = self.chars[self.pos];
        if let Some(e) = expected {
            assert_eq!(ch, e, "Expected '{}' at position {}", e, self.pos);
        }
        self.pos += 1;
        ch
    }

    fn read_label(&mut self) -> Option<String> {
        if self.pos >= self.chars.len() {
            return None;
        }
        if self.chars[self.pos] == '\'' {
            Some(self.read_quoted())
        } else {
            self.read_unquoted()
        }
    }

    fn read_quoted(&mut self) -> String {
        self.consume(Some('\''));
        let mut parts = String::new();
        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            self.pos += 1;
            if ch == '\'' {
                if self.pos < self.chars.len() && self.chars[self.pos] == '\'' {
                    parts.push('\'');
                    self.pos += 1;
                } else {
                    return parts;
                }
            } else {
                parts.push(ch);
            }
        }
        panic!("Unterminated quoted label");
    }

    fn read_unquoted(&mut self) -> Option<String> {
        let start = self.pos;
        while self.pos < self.chars.len() && !"(),:;".contains(self.chars[self.pos]) {
            self.pos += 1;
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        let trimmed = text.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }

    fn read_branch_length(&mut self) -> Option<f64> {
        if self.pos < self.chars.len() && self.chars[self.pos] == ':' {
            self.consume(Some(':'));
            let start = self.pos;
            while self.pos < self.chars.len() && !"(),;".contains(self.chars[self.pos]) {
                self.pos += 1;
            }
            let text: String = self.chars[start..self.pos].iter().collect();
            let trimmed = text.trim();
            if trimmed.is_empty() {
                Some(0.0)
            } else {
                Some(trimmed.parse::<f64>().expect("Invalid branch length"))
            }
        } else {
            None
        }
    }
}

struct TempNode {
    name: Option<String>,
    dist: f64,
    children: Vec<usize>,
}

/// Parse a Newick tree string.
pub fn parse_newick(newick_str: &str) -> ParsedTree {
    let s = strip_comments(newick_str.trim());
    let s = s.trim_end_matches(';').trim();
    assert!(!s.is_empty(), "Empty Newick string");

    let mut tok = NewickTokenizer::new(s);
    let mut nodes: Vec<TempNode> = Vec::new();

    fn parse_node(tok: &mut NewickTokenizer, nodes: &mut Vec<TempNode>) -> usize {
        let mut children = Vec::new();
        if tok.peek() == Some('(') {
            tok.consume(Some('('));
            children.push(parse_node(tok, nodes));
            while tok.peek() == Some(',') {
                tok.consume(Some(','));
                children.push(parse_node(tok, nodes));
            }
            tok.consume(Some(')'));
        }
        let name = tok.read_label();
        let dist = tok.read_branch_length().unwrap_or(0.0);
        let idx = nodes.len();
        nodes.push(TempNode { name, dist, children });
        idx
    }

    let root_idx = parse_node(&mut tok, &mut nodes);

    let mut parent_index = Vec::new();
    let mut distance_to_parent = Vec::new();
    let mut node_names_out = Vec::new();
    let mut leaf_names = Vec::new();

    fn dfs(
        old_idx: usize,
        parent_new: i32,
        nodes: &[TempNode],
        parent_index: &mut Vec<i32>,
        distance_to_parent: &mut Vec<f64>,
        node_names: &mut Vec<Option<String>>,
        leaf_names: &mut Vec<String>,
    ) {
        let _new_idx = parent_index.len();
        let node = &nodes[old_idx];
        parent_index.push(parent_new);
        distance_to_parent.push(if parent_new >= 0 { node.dist } else { 0.0 });
        node_names.push(node.name.clone());
        if node.children.is_empty() {
            leaf_names.push(node.name.clone().unwrap_or_default());
        }
        let new_idx_i32 = (_new_idx) as i32;
        for &child_old in &node.children {
            dfs(
                child_old, new_idx_i32, nodes, parent_index,
                distance_to_parent, node_names, leaf_names,
            );
        }
    }

    dfs(
        root_idx, -1, &nodes,
        &mut parent_index, &mut distance_to_parent,
        &mut node_names_out, &mut leaf_names,
    );

    ParsedTree {
        parent_index,
        distance_to_parent,
        leaf_names,
        node_names: node_names_out,
    }
}

// ---- Helper: build char map ----

fn build_char_map(alphabet: &[char]) -> std::collections::HashMap<char, i32> {
    let mut map = std::collections::HashMap::new();
    for (i, &ch) in alphabet.iter().enumerate() {
        map.insert(ch, i as i32);
        map.insert(ch.to_ascii_lowercase(), i as i32);
    }
    map
}

// ---- FASTA parser ----

/// Parse FASTA-formatted alignment text.
pub fn parse_fasta(text: &str, alphabet: Option<&[char]>) -> ParsedAlignment {
    let mut sequences: Vec<String> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut current_seq: Vec<String> = Vec::new();
    let mut has_name = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('>') {
            if has_name {
                sequences.push(current_seq.join(""));
            }
            let header = &line[1..].trim();
            let name = header.split_whitespace().next().unwrap_or("").to_string();
            names.push(name);
            current_seq = Vec::new();
            has_name = true;
        } else {
            current_seq.push(line.to_string());
        }
    }
    if has_name {
        sequences.push(current_seq.join(""));
    }

    assert!(!sequences.is_empty(), "No sequences found in FASTA input");

    let c = sequences[0].len();
    for (i, s) in sequences.iter().enumerate() {
        assert_eq!(s.len(), c, "Unequal sequence length at row {}", i);
    }
    assert!(c > 0, "Empty sequences in FASTA input");

    let mut all_chars = std::collections::HashSet::new();
    for s in &sequences {
        for ch in s.chars() {
            if !is_gap(ch) {
                all_chars.insert(ch.to_ascii_uppercase());
            }
        }
    }

    let alpha: Vec<char> = match alphabet {
        Some(a) => a.to_vec(),
        None => detect_alphabet(&all_chars),
    };

    let char_map = build_char_map(&alpha);
    let gap_idx = alpha.len() as i32 + 1;
    let n = sequences.len();
    let mut alignment = vec![0i32; n * c];

    for (r, seq) in sequences.iter().enumerate() {
        for (col, ch) in seq.chars().enumerate() {
            if is_gap(ch) {
                alignment[r * c + col] = gap_idx;
            } else if let Some(&idx) = char_map.get(&ch) {
                alignment[r * c + col] = idx;
            } else if let Some(&idx) = char_map.get(&ch.to_ascii_lowercase()) {
                alignment[r * c + col] = idx;
            } else {
                panic!("Unknown character '{}' in sequence '{}' at position {}", ch, names[r], col);
            }
        }
    }

    ParsedAlignment { alignment, leaf_names: names, alphabet: alpha, n, c }
}

// ---- Plain string parser ----

/// Parse a list of equal-length strings into an alignment tensor.
pub fn parse_strings(sequences: &[&str], alphabet: Option<&[char]>) -> ParsedAlignment {
    assert!(!sequences.is_empty(), "Empty sequence list");

    let c = sequences[0].len();
    for (i, s) in sequences.iter().enumerate() {
        assert_eq!(s.len(), c, "Unequal sequence length at row {}", i);
    }
    assert!(c > 0, "Empty sequences");

    let mut all_chars = std::collections::HashSet::new();
    for s in sequences {
        for ch in s.chars() {
            if !is_gap(ch) {
                all_chars.insert(ch.to_ascii_uppercase());
            }
        }
    }

    let alpha: Vec<char> = match alphabet {
        Some(a) => a.to_vec(),
        None => detect_alphabet(&all_chars),
    };

    let char_map = build_char_map(&alpha);
    let gap_idx = alpha.len() as i32 + 1;
    let n = sequences.len();
    let mut alignment = vec![0i32; n * c];

    for (r, seq) in sequences.iter().enumerate() {
        for (col, ch) in seq.chars().enumerate() {
            if is_gap(ch) {
                alignment[r * c + col] = gap_idx;
            } else if let Some(&idx) = char_map.get(&ch) {
                alignment[r * c + col] = idx;
            } else {
                panic!("Unknown character '{}' at row {}, col {}", ch, r, col);
            }
        }
    }

    ParsedAlignment {
        alignment,
        leaf_names: Vec::new(),
        alphabet: alpha,
        n,
        c,
    }
}

// ---- Dictionary parser ----

/// Parse a name→sequence map into an alignment tensor.
pub fn parse_dict(
    sequences: &std::collections::HashMap<String, String>,
    alphabet: Option<&[char]>,
) -> ParsedAlignment {
    assert!(!sequences.is_empty(), "Empty sequence dictionary");

    let names: Vec<String> = sequences.keys().cloned().collect();
    let seqs: Vec<&str> = names.iter().map(|n| sequences[n].as_str()).collect();

    let c = seqs[0].len();
    for (i, s) in seqs.iter().enumerate() {
        assert_eq!(s.len(), c, "Unequal sequence length at '{}'", names[i]);
    }
    assert!(c > 0, "Empty sequences");

    let mut all_chars = std::collections::HashSet::new();
    for s in &seqs {
        for ch in s.chars() {
            if !is_gap(ch) {
                all_chars.insert(ch.to_ascii_uppercase());
            }
        }
    }

    let alpha: Vec<char> = match alphabet {
        Some(a) => a.to_vec(),
        None => detect_alphabet(&all_chars),
    };

    let char_map = build_char_map(&alpha);
    let gap_idx = alpha.len() as i32 + 1;
    let n = seqs.len();
    let mut alignment = vec![0i32; n * c];

    for (r, seq) in seqs.iter().enumerate() {
        for (col, ch) in seq.chars().enumerate() {
            if is_gap(ch) {
                alignment[r * c + col] = gap_idx;
            } else if let Some(&idx) = char_map.get(&ch) {
                alignment[r * c + col] = idx;
            } else if let Some(&idx) = char_map.get(&ch.to_ascii_lowercase()) {
                alignment[r * c + col] = idx;
            } else {
                panic!("Unknown character '{}' in sequence '{}' at position {}", ch, names[r], col);
            }
        }
    }

    ParsedAlignment {
        alignment,
        leaf_names: names,
        alphabet: alpha,
        n,
        c,
    }
}

// ---- Combine tree + alignment ----

/// Map leaf sequences to tree positions by name matching.
pub fn combine_tree_alignment(tree: &ParsedTree, aln: &ParsedAlignment) -> CombinedInput {
    let a = aln.alphabet.len();
    let ungapped_unobserved = a as i32;

    let mut aln_name_to_row = std::collections::HashMap::new();
    for (i, name) in aln.leaf_names.iter().enumerate() {
        aln_name_to_row.insert(name.as_str(), i);
    }

    for name in &tree.leaf_names {
        assert!(
            aln_name_to_row.contains_key(name.as_str()),
            "Tree leaf '{}' not found in alignment",
            name
        );
    }

    let r = tree.parent_index.len();
    let c = aln.c;

    let mut full_aln = vec![ungapped_unobserved; r * c];

    let mut child_count = vec![0u32; r];
    for n in 1..r {
        child_count[tree.parent_index[n] as usize] += 1;
    }

    let mut leaf_idx = 0usize;
    for n in 0..r {
        if child_count[n] == 0 {
            let leaf_name = &tree.leaf_names[leaf_idx];
            let aln_row = aln_name_to_row[leaf_name.as_str()];
            for col in 0..c {
                full_aln[n * c + col] = aln.alignment[aln_row * c + col];
            }
            leaf_idx += 1;
        }
    }

    CombinedInput {
        alignment: full_aln,
        parent_index: tree.parent_index.clone(),
        distance_to_parent: tree.distance_to_parent.clone(),
        alphabet: aln.alphabet.clone(),
        leaf_names: tree.leaf_names.clone(),
        r,
        c,
    }
}

// ---- Genetic code ----

/// Standard genetic code.
#[derive(Debug, Clone)]
pub struct GeneticCode {
    pub codons: Vec<String>,           // 64 codon strings
    pub amino_acids: Vec<char>,        // 64 amino acid letters (stop = '*')
    pub sense_mask: Vec<bool>,         // (64,) true for sense codons
    pub sense_indices: Vec<usize>,     // (61,) indices of sense codons in 0..63
    pub codon_to_sense: Vec<i32>,      // (64,) maps codon idx to sense idx (stop -> -1)
    pub sense_codons: Vec<String>,     // 61 sense codon strings
    pub sense_amino_acids: Vec<char>,  // 61 amino acid letters
}

pub fn genetic_code() -> GeneticCode {
    let bases = ['A', 'C', 'G', 'T'];
    let mut codons = Vec::with_capacity(64);
    for &b1 in &bases {
        for &b2 in &bases {
            for &b3 in &bases {
                codons.push(format!("{}{}{}", b1, b2, b3));
            }
        }
    }

    let code_table: std::collections::HashMap<&str, char> = [
        ("AAA",'K'),("AAC",'N'),("AAG",'K'),("AAT",'N'),
        ("ACA",'T'),("ACC",'T'),("ACG",'T'),("ACT",'T'),
        ("AGA",'R'),("AGC",'S'),("AGG",'R'),("AGT",'S'),
        ("ATA",'I'),("ATC",'I'),("ATG",'M'),("ATT",'I'),
        ("CAA",'Q'),("CAC",'H'),("CAG",'Q'),("CAT",'H'),
        ("CCA",'P'),("CCC",'P'),("CCG",'P'),("CCT",'P'),
        ("CGA",'R'),("CGC",'R'),("CGG",'R'),("CGT",'R'),
        ("CTA",'L'),("CTC",'L'),("CTG",'L'),("CTT",'L'),
        ("GAA",'E'),("GAC",'D'),("GAG",'E'),("GAT",'D'),
        ("GCA",'A'),("GCC",'A'),("GCG",'A'),("GCT",'A'),
        ("GGA",'G'),("GGC",'G'),("GGG",'G'),("GGT",'G'),
        ("GTA",'V'),("GTC",'V'),("GTG",'V'),("GTT",'V'),
        ("TAA",'*'),("TAC",'Y'),("TAG",'*'),("TAT",'Y'),
        ("TCA",'S'),("TCC",'S'),("TCG",'S'),("TCT",'S'),
        ("TGA",'*'),("TGC",'C'),("TGG",'W'),("TGT",'C'),
        ("TTA",'L'),("TTC",'F'),("TTG",'L'),("TTT",'F'),
    ].iter().cloned().collect();

    let amino_acids: Vec<char> = codons.iter().map(|c| *code_table.get(c.as_str()).unwrap()).collect();
    let sense_mask: Vec<bool> = amino_acids.iter().map(|&aa| aa != '*').collect();
    let mut sense_indices = Vec::with_capacity(61);
    let mut codon_to_sense = vec![-1i32; 64];
    let mut sense_idx = 0i32;
    for i in 0..64 {
        if sense_mask[i] {
            sense_indices.push(i);
            codon_to_sense[i] = sense_idx;
            sense_idx += 1;
        }
    }
    let sense_codons: Vec<String> = sense_indices.iter().map(|&i| codons[i].clone()).collect();
    let sense_amino_acids: Vec<char> = sense_indices.iter().map(|&i| amino_acids[i]).collect();

    GeneticCode {
        codons, amino_acids, sense_mask, sense_indices,
        codon_to_sense, sense_codons, sense_amino_acids,
    }
}

/// Remap a 64-codon tokenized alignment to 61-sense-codon tokens.
/// Stop codons become gap tokens.
pub fn codon_to_sense(alignment: &[i32], n: usize, c: usize, a: usize) -> Vec<i32> {
    let gc = genetic_code();
    let codon_map = &gc.codon_to_sense;
    let a_sense = 61usize;
    let unobs_in = a as i32;       // 64
    let _gap_in = a as i32 + 1;    // 65
    let unobs_out = a_sense as i32; // 61
    let gap_out = a_sense as i32 + 1; // 62

    let mut result = vec![unobs_out; n * c];
    for i in 0..n * c {
        let tok = alignment[i];
        if tok >= 0 && tok < 64 {
            result[i] = if codon_map[tok as usize] >= 0 { codon_map[tok as usize] } else { gap_out };
        } else if tok == unobs_in {
            result[i] = unobs_out;
        } else {
            result[i] = gap_out;
        }
    }
    result
}

// ---- K-mer tokenization ----

/// Result of k-mer tokenization.
#[derive(Debug, Clone)]
pub struct KmerResult {
    pub alignment: Vec<i32>,
    pub a_kmer: usize,
    pub n: usize,
    pub c_k: usize,
}

pub fn kmer_tokenize(alignment: &[i32], n: usize, c: usize, a: usize, k: usize, gap_mode: &str) -> KmerResult {
    if c % k != 0 {
        panic!("Number of columns ({}) not divisible by k ({})", c, k);
    }
    let c_k = c / k;
    let a_k = a.pow(k as u32);
    let gap_tok = a_k as i32 + 1;
    let unobs_tok = a_k as i32;
    let illegal_tok = a_k as i32 + 2;

    let mut result = vec![unobs_tok; n * c_k];

    for row in 0..n {
        for ck in 0..c_k {
            let base = row * c + ck * k;
            let mut all_obs = true;
            let mut has_gap = false;
            let mut all_gap = true;
            let mut kmer_idx: i32 = 0;

            for p in 0..k {
                let tok = alignment[base + p];
                let is_obs = tok >= 0 && tok < a as i32;
                let is_gap = tok < 0 || tok > a as i32;
                if !is_obs { all_obs = false; }
                if is_gap { has_gap = true; }
                if !is_gap { all_gap = false; }
                if is_obs {
                    kmer_idx = kmer_idx * a as i32 + tok;
                } else {
                    kmer_idx *= a as i32;
                }
            }

            let out_idx = row * c_k + ck;
            if all_obs {
                result[out_idx] = kmer_idx;
            } else if gap_mode == "any" && has_gap {
                result[out_idx] = gap_tok;
            } else if gap_mode == "all" && all_gap {
                result[out_idx] = gap_tok;
            } else if gap_mode == "all" && has_gap {
                result[out_idx] = illegal_tok;
            }
            // else: stays unobs_tok (default)
        }
    }

    KmerResult { alignment: result, a_kmer: a_k, n, c_k }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_alphabet_dna() {
        let chars: std::collections::HashSet<char> = "ACGT".chars().collect();
        assert_eq!(detect_alphabet(&chars), vec!['A', 'C', 'G', 'T']);
    }

    #[test]
    fn test_detect_alphabet_rna() {
        let chars: std::collections::HashSet<char> = "ACGU".chars().collect();
        assert_eq!(detect_alphabet(&chars), vec!['A', 'C', 'G', 'U']);
    }

    #[test]
    fn test_detect_alphabet_gap_exclusion() {
        let chars: std::collections::HashSet<char> = "ACGT-.".chars().collect();
        let result = detect_alphabet(&chars);
        assert!(!result.contains(&'-'));
        assert!(!result.contains(&'.'));
        assert_eq!(result, vec!['A', 'C', 'G', 'T']);
    }

    #[test]
    fn test_parse_newick_simple() {
        let result = parse_newick("((A:0.1,B:0.2):0.3,C:0.4);");
        assert_eq!(result.parent_index[0], -1);
        assert_eq!(result.parent_index.len(), 5);
        assert_eq!(result.leaf_names, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_parse_newick_no_lengths() {
        let result = parse_newick("((A,B),C);");
        assert!(result.distance_to_parent.iter().all(|&d| d == 0.0));
        assert_eq!(result.leaf_names, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_parse_newick_single_leaf() {
        let result = parse_newick("A:0.5;");
        assert_eq!(result.parent_index.len(), 1);
        assert_eq!(result.parent_index[0], -1);
        assert_eq!(result.leaf_names, vec!["A"]);
    }

    #[test]
    fn test_parse_newick_scientific_notation() {
        let result = parse_newick("(A:1.5e-3,B:2.0E+1);");
        let a_idx = result.node_names.iter().position(|n| n.as_deref() == Some("A")).unwrap();
        let b_idx = result.node_names.iter().position(|n| n.as_deref() == Some("B")).unwrap();
        assert!((result.distance_to_parent[a_idx] - 1.5e-3).abs() < 1e-10);
        assert!((result.distance_to_parent[b_idx] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_fasta_simple() {
        let result = parse_fasta(">seq1\nACGT\n>seq2\nACGT\n", None);
        assert_eq!(result.n, 2);
        assert_eq!(result.c, 4);
        assert_eq!(result.leaf_names, vec!["seq1", "seq2"]);
        assert_eq!(result.alphabet, vec!['A', 'C', 'G', 'T']);
    }

    #[test]
    fn test_parse_fasta_gaps() {
        let result = parse_fasta(">s1\nA-GT\n>s2\nACG-\n", None);
        let gap_idx = result.alphabet.len() as i32 + 1;
        assert_eq!(result.alignment[0 * 4 + 1], gap_idx);
        assert_eq!(result.alignment[1 * 4 + 3], gap_idx);
    }

    #[test]
    fn test_parse_strings_simple() {
        let result = parse_strings(&["ACGT", "TGCA"], None);
        assert_eq!(result.n, 2);
        assert_eq!(result.c, 4);
    }

    #[test]
    fn test_combine_tree_alignment() {
        let tree = parse_newick("((A:0.1,B:0.2):0.3,C:0.4);");
        let aln = parse_fasta(">A\nACGT\n>B\nTGCA\n>C\nGGGG\n", None);
        let result = combine_tree_alignment(&tree, &aln);
        assert_eq!(result.r, 5);
        assert_eq!(result.c, 4);
        // Internal nodes should have token A (= 4 for DNA)
        let a_token = result.alphabet.len() as i32;
        let child_count: Vec<u32> = {
            let mut cc = vec![0u32; result.r];
            for n in 1..result.r {
                cc[result.parent_index[n] as usize] += 1;
            }
            cc
        };
        for n in 0..result.r {
            if child_count[n] > 0 {
                for col in 0..result.c {
                    assert_eq!(result.alignment[n * result.c + col], a_token);
                }
            }
        }
    }

    #[test]
    fn test_parse_dict() {
        let mut seqs = std::collections::HashMap::new();
        seqs.insert("human".to_string(), "ACGT".to_string());
        seqs.insert("mouse".to_string(), "TGCA".to_string());
        let result = parse_dict(&seqs, None);
        assert_eq!(result.n, 2);
        assert_eq!(result.c, 4);
        assert_eq!(result.alphabet, vec!['A', 'C', 'G', 'T']);
        assert!(result.leaf_names.contains(&"human".to_string()));
        assert!(result.leaf_names.contains(&"mouse".to_string()));
    }

    #[test]
    fn test_newick_parent_ordering() {
        let result = parse_newick("((A:0.1,B:0.2):0.3,C:0.4);");
        for i in 1..result.parent_index.len() {
            assert!(result.parent_index[i] >= 0);
            assert!((result.parent_index[i] as usize) < i);
        }
    }
}

#[cfg(test)]
mod genetic_tests {
    use super::*;

    #[test]
    fn test_genetic_code_sense_count() {
        let gc = genetic_code();
        assert_eq!(gc.sense_indices.len(), 61);
        assert_eq!(gc.sense_mask.iter().filter(|&&m| m).count(), 61);
    }

    #[test]
    fn test_genetic_code_stop_codons() {
        let gc = genetic_code();
        // TAA=48, TAG=50, TGA=56
        assert_eq!(gc.amino_acids[48], '*');
        assert_eq!(gc.amino_acids[50], '*');
        assert_eq!(gc.amino_acids[56], '*');
        assert!(!gc.sense_mask[48]);
        assert!(!gc.sense_mask[50]);
        assert!(!gc.sense_mask[56]);
    }

    #[test]
    fn test_codon_to_sense_basic() {
        // Token 0 (AAA) is sense codon index 0
        let aln = vec![0i32, 1, 48]; // AAA, AAC, TAA(stop)
        let result = codon_to_sense(&aln, 1, 3, 64);
        assert_eq!(result[0], 0); // AAA -> sense 0
        assert_eq!(result[1], 1); // AAC -> sense 1
        assert_eq!(result[2], 62); // TAA (stop) -> gap (62)
    }

    #[test]
    fn test_kmer_tokenize_basic() {
        // ACG = 0*16 + 1*4 + 2 = 6
        let aln = vec![0, 1, 2, 3, 0, 0]; // ACG, TAA
        let result = kmer_tokenize(&aln, 1, 6, 4, 3, "any");
        assert_eq!(result.c_k, 2);
        assert_eq!(result.a_kmer, 64);
        assert_eq!(result.alignment[0], 6); // ACG
        assert_eq!(result.alignment[1], 48); // TAA = 3*16 + 0*4 + 0
    }
}
