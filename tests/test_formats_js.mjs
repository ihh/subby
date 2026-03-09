/**
 * Tests for formats.js — standard phylogenetic file format parsers and utilities.
 *
 * Mirrors tests/test_formats.py (Python) and subby/wasm/src/formats.rs (Rust).
 *
 * Run: node tests/test_formats_js.mjs
 */

import {
  detectAlphabet,
  parseNewick,
  parseFasta,
  parseStockholm,
  parseMaf,
  parseStrings,
  parseDict,
  combineTreeAlignment,
  geneticCode,
  codonToSense,
  KmerIndex,
  slidingWindows,
  allColumnKtuples,
  kmerTokenize,
} from '../subby/webgpu/formats.js';

// ---- Helpers ----
let passed = 0, failed = 0, total = 0;

function assert(cond, msg) {
  if (!cond) throw new Error(`Assertion failed: ${msg}`);
}

function assertEqual(a, b, msg) {
  if (a !== b) throw new Error(`${msg}: ${JSON.stringify(a)} !== ${JSON.stringify(b)}`);
}

function assertClose(a, b, tol, msg) {
  if (Math.abs(a - b) > tol) throw new Error(`${msg}: ${a} vs ${b} (tol=${tol})`);
}

function assertArrayEqual(a, b, msg) {
  if (a.length !== b.length) throw new Error(`${msg}: length ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) throw new Error(`${msg}[${i}]: ${a[i]} vs ${b[i]}`);
  }
}

function assertThrows(fn, pattern, msg) {
  let threw = false;
  try { fn(); } catch (e) {
    threw = true;
    if (pattern && !e.message.includes(pattern)) {
      throw new Error(`${msg}: expected error containing "${pattern}", got "${e.message}"`);
    }
  }
  if (!threw) throw new Error(`${msg}: expected error but none thrown`);
}

function runTest(name, fn) {
  total++;
  try {
    fn();
    passed++;
    process.stdout.write(`  ok  ${name}\n`);
  } catch (e) {
    failed++;
    process.stdout.write(`  FAIL ${name}: ${e.message}\n`);
  }
}

/** Get alignment value at row r, col c from flat array */
function alnAt(alignment, C, r, c) {
  return alignment[r * C + c];
}

// ==================== detectAlphabet ====================

console.log('--- detectAlphabet ---');

runTest('dna', () => {
  const result = detectAlphabet(new Set(['A', 'C', 'G', 'T']));
  assertArrayEqual(result, ['A', 'C', 'G', 'T'], 'DNA');
});

runTest('dna subset', () => {
  const result = detectAlphabet(new Set(['A', 'T']));
  assertArrayEqual(result, ['A', 'C', 'G', 'T'], 'DNA subset');
});

runTest('rna', () => {
  const result = detectAlphabet(new Set(['A', 'C', 'G', 'U']));
  assertArrayEqual(result, ['A', 'C', 'G', 'U'], 'RNA');
});

runTest('protein', () => {
  const chars = new Set('ACDEFGHIKLMNPQRSTVWY'.split(''));
  const result = detectAlphabet(chars);
  assertArrayEqual(result, 'ACDEFGHIKLMNPQRSTVWY'.split(''), 'protein');
});

runTest('protein subset', () => {
  const result = detectAlphabet(new Set(['M', 'A', 'L']));
  assertArrayEqual(result, 'ACDEFGHIKLMNPQRSTVWY'.split(''), 'protein subset');
});

runTest('custom', () => {
  const result = detectAlphabet(new Set(['X', 'Y', 'Z']));
  assertArrayEqual(result, ['X', 'Y', 'Z'], 'custom');
});

runTest('gap exclusion', () => {
  const result = detectAlphabet(new Set(['A', 'C', '-', '.', 'G', 'T']));
  assert(!result.includes('-'), 'no dash');
  assert(!result.includes('.'), 'no dot');
  assertArrayEqual(result, ['A', 'C', 'G', 'T'], 'gap exclusion');
});

runTest('case insensitivity', () => {
  const result = detectAlphabet(new Set(['a', 'c', 'g', 't']));
  assertArrayEqual(result, ['A', 'C', 'G', 'T'], 'case');
});

// ==================== parseNewick ====================

console.log('\n--- parseNewick ---');

runTest('simple binary', () => {
  const result = parseNewick('((A:0.1,B:0.2):0.3,C:0.4);');
  assert(result.parentIndex[0] === -1, 'root');
  assertEqual(result.parentIndex.length, 5, 'node count');
  assertArrayEqual(result.leafNames, ['A', 'B', 'C'], 'leaves');
  assertClose(result.distanceToParent[0], 0.0, 1e-10, 'root dist');
});

runTest('named internal', () => {
  const result = parseNewick('((A:0.1,B:0.2)X:0.3,C:0.4)root;');
  assertEqual(result.nodeNames[0], 'root', 'root name');
  assert(result.nodeNames.includes('X'), 'internal name');
});

runTest('no branch lengths', () => {
  const result = parseNewick('((A,B),C);');
  for (let i = 0; i < result.distanceToParent.length; i++) {
    assertClose(result.distanceToParent[i], 0.0, 1e-10, `dist[${i}]`);
  }
  assertArrayEqual(result.leafNames, ['A', 'B', 'C'], 'leaves');
});

runTest('single leaf', () => {
  const result = parseNewick('A:0.5;');
  assertEqual(result.parentIndex.length, 1, 'single node');
  assertEqual(result.parentIndex[0], -1, 'root');
  assertArrayEqual(result.leafNames, ['A'], 'leaves');
});

runTest('quoted labels', () => {
  const result = parseNewick("('leaf one':0.1,'leaf two':0.2);");
  assertArrayEqual(result.leafNames, ['leaf one', 'leaf two'], 'quoted leaves');
});

runTest('scientific notation', () => {
  const result = parseNewick('(A:1.5e-3,B:2.0E+1);');
  const aIdx = result.nodeNames.indexOf('A');
  const bIdx = result.nodeNames.indexOf('B');
  assertClose(result.distanceToParent[aIdx], 1.5e-3, 1e-10, 'A dist');
  assertClose(result.distanceToParent[bIdx], 20.0, 1e-10, 'B dist');
});

runTest('comments', () => {
  const result = parseNewick('((A[comment]:0.1,B:0.2):0.3,C:0.4);');
  assertArrayEqual(result.leafNames, ['A', 'B', 'C'], 'leaves');
});

runTest('known tree', () => {
  const result = parseNewick('((A:0.05,B:0.15):0.1,C:0.3);');
  assertEqual(result.parentIndex.length, 5, 'node count');
  assertEqual(result.parentIndex[0], -1, 'root');
  // All parents valid and < child index
  for (let i = 1; i < result.parentIndex.length; i++) {
    assert(result.parentIndex[i] >= 0 && result.parentIndex[i] < i, `parent[${i}]`);
  }
  assertArrayEqual(result.leafNames, ['A', 'B', 'C'], 'leaves');
  const aIdx = result.nodeNames.indexOf('A');
  const bIdx = result.nodeNames.indexOf('B');
  const cIdx = result.nodeNames.indexOf('C');
  assertClose(result.distanceToParent[aIdx], 0.05, 1e-10, 'A dist');
  assertClose(result.distanceToParent[bIdx], 0.15, 1e-10, 'B dist');
  assertClose(result.distanceToParent[cIdx], 0.3, 1e-10, 'C dist');
});

runTest('multifurcation', () => {
  const result = parseNewick('(A:0.1,B:0.2,C:0.3);');
  assertEqual(result.parentIndex.length, 4, 'root + 3 leaves');
  assertArrayEqual(result.leafNames, ['A', 'B', 'C'], 'leaves');
});

runTest('empty raises', () => {
  assertThrows(() => parseNewick(''), 'Empty', 'empty newick');
});

runTest('no semicolon', () => {
  const result = parseNewick('(A:0.1,B:0.2)');
  assertArrayEqual(result.leafNames, ['A', 'B'], 'leaves');
});

// ==================== parseFasta ====================

console.log('\n--- parseFasta ---');

runTest('simple', () => {
  const result = parseFasta('>seq1\nACGT\n>seq2\nACGT\n');
  assertEqual(result.N, 2, 'N');
  assertEqual(result.C, 4, 'C');
  assertArrayEqual(result.leafNames, ['seq1', 'seq2'], 'names');
  assertArrayEqual(result.alphabet, ['A', 'C', 'G', 'T'], 'alphabet');
});

runTest('multiline sequences', () => {
  const result = parseFasta('>seq1\nAC\nGT\n>seq2\nTG\nCA\n');
  assertEqual(result.N, 2, 'N');
  assertEqual(result.C, 4, 'C');
});

runTest('gaps', () => {
  const result = parseFasta('>seq1\nA-GT\n>seq2\nACG-\n');
  const A = result.alphabet.length;
  const gapIdx = A + 1;
  assertEqual(alnAt(result.alignment, result.C, 0, 1), gapIdx, 'seq1 col1 gap');
  assertEqual(alnAt(result.alignment, result.C, 1, 3), gapIdx, 'seq2 col3 gap');
});

runTest('unequal lengths error', () => {
  assertThrows(() => parseFasta('>seq1\nACGT\n>seq2\nACG\n'), 'Unequal', 'unequal');
});

runTest('description lines', () => {
  const result = parseFasta('>seq1 this is a description\nACGT\n>seq2 another one\nACGT\n');
  assertArrayEqual(result.leafNames, ['seq1', 'seq2'], 'names');
});

runTest('explicit alphabet', () => {
  const result = parseFasta('>s1\nAB\n>s2\nBA\n', ['A', 'B']);
  assertArrayEqual(result.alphabet, ['A', 'B'], 'alphabet');
  assertEqual(alnAt(result.alignment, result.C, 0, 0), 0, 's1[0]');
  assertEqual(alnAt(result.alignment, result.C, 0, 1), 1, 's1[1]');
  assertEqual(alnAt(result.alignment, result.C, 1, 0), 1, 's2[0]');
  assertEqual(alnAt(result.alignment, result.C, 1, 1), 0, 's2[1]');
});

runTest('case insensitive', () => {
  const result = parseFasta('>seq1\nacgt\n>seq2\nACGT\n');
  for (let c = 0; c < result.C; c++) {
    assertEqual(
      alnAt(result.alignment, result.C, 0, c),
      alnAt(result.alignment, result.C, 1, c),
      `col ${c}`,
    );
  }
});

// ==================== parseStockholm ====================

console.log('\n--- parseStockholm ---');

runTest('without tree', () => {
  const text = `# STOCKHOLM 1.0
seq1  ACGT
seq2  ACGT
//`;
  const result = parseStockholm(text);
  assertEqual(result.N, 2, 'N');
  assertEqual(result.C, 4, 'C');
  assertArrayEqual(result.leafNames, ['seq1', 'seq2'], 'names');
});

runTest('with tree', () => {
  const text = `# STOCKHOLM 1.0
#=GF NH (seq1:0.1,seq2:0.2);
seq1  ACGT
seq2  TGCA
//`;
  const result = parseStockholm(text);
  // Should be combined: tree + alignment
  assert(result.parentIndex !== undefined, 'has parentIndex');
  assertEqual(result.R, 3, 'R=3 (root + 2 leaves)');
  assertEqual(result.C, 4, 'C=4');
});

runTest('name matching', () => {
  const text = `# STOCKHOLM 1.0
#=GF NH (B:0.1,A:0.2);
A  ACGT
B  TGCA
//`;
  const result = parseStockholm(text);
  assertEqual(result.R, 3, 'R=3');
  const leafSet = new Set(result.leafNames);
  assert(leafSet.has('A'), 'has A');
  assert(leafSet.has('B'), 'has B');
});

runTest('multiline NH', () => {
  const text = `# STOCKHOLM 1.0
#=GF NH (seq1:0.1,
#=GF NH seq2:0.2);
seq1  ACGT
seq2  TGCA
//`;
  const result = parseStockholm(text);
  assert(result.parentIndex !== undefined, 'combined');
});

// ==================== parseMaf ====================

console.log('\n--- parseMaf ---');

runTest('single block', () => {
  const text = `a score=0
s human.chr1 100 4 + 1000 ACGT
s mouse.chr2 200 4 + 900  TGCA
`;
  const result = parseMaf(text);
  assertEqual(result.N, 2, 'N');
  assertEqual(result.C, 4, 'C');
  assertArrayEqual(result.leafNames, ['human', 'mouse'], 'names');
});

runTest('multi block', () => {
  const text = `a score=0
s human.chr1 100 2 + 1000 AC
s mouse.chr2 200 2 + 900  TG

a score=0
s human.chr1 102 2 + 1000 GT
s mouse.chr2 202 2 + 900  CA
`;
  const result = parseMaf(text);
  assertEqual(result.N, 2, 'N');
  assertEqual(result.C, 4, 'C');
});

runTest('species name', () => {
  const text = `a score=0
s homo_sapiens.chr1 100 4 + 1000 ACGT
`;
  const result = parseMaf(text);
  assertArrayEqual(result.leafNames, ['homo_sapiens'], 'names');
});

runTest('missing species', () => {
  const text = `a score=0
s human.chr1 100 2 + 1000 AC
s mouse.chr2 200 2 + 900  TG

a score=0
s human.chr1 102 2 + 1000 GT
`;
  const result = parseMaf(text);
  assertEqual(result.N, 2, 'N');
  assertEqual(result.C, 4, 'C');
  const A = result.alphabet.length;
  const gapIdx = A + 1;
  const mouseIdx = result.leafNames.indexOf('mouse');
  // Last 2 columns should be gaps for mouse
  assertEqual(alnAt(result.alignment, result.C, mouseIdx, 2), gapIdx, 'mouse col2 gap');
  assertEqual(alnAt(result.alignment, result.C, mouseIdx, 3), gapIdx, 'mouse col3 gap');
});

// ==================== parseStrings ====================

console.log('\n--- parseStrings ---');

runTest('simple', () => {
  const result = parseStrings(['ACGT', 'TGCA']);
  assertEqual(result.N, 2, 'N');
  assertEqual(result.C, 4, 'C');
  assertArrayEqual(result.alphabet, ['A', 'C', 'G', 'T'], 'alphabet');
});

runTest('gaps', () => {
  const result = parseStrings(['A-GT', 'AC-T']);
  const A = result.alphabet.length;
  const gapIdx = A + 1;
  assertEqual(alnAt(result.alignment, result.C, 0, 1), gapIdx, 'row0 col1');
  assertEqual(alnAt(result.alignment, result.C, 1, 2), gapIdx, 'row1 col2');
});

runTest('empty error', () => {
  assertThrows(() => parseStrings([]), 'Empty', 'empty');
});

runTest('unequal error', () => {
  assertThrows(() => parseStrings(['ACGT', 'AC']), 'Unequal', 'unequal');
});

// ==================== parseDict ====================

console.log('\n--- parseDict ---');

runTest('simple', () => {
  const result = parseDict({ human: 'ACGT', mouse: 'TGCA' });
  assertEqual(result.N, 2, 'N');
  assertEqual(result.C, 4, 'C');
  assertArrayEqual(result.alphabet, ['A', 'C', 'G', 'T'], 'alphabet');
  const names = new Set(result.leafNames);
  assert(names.has('human'), 'has human');
  assert(names.has('mouse'), 'has mouse');
});

runTest('gaps', () => {
  const result = parseDict({ s1: 'A-GT', s2: 'AC-T' });
  const A = result.alphabet.length;
  const gapIdx = A + 1;
  const s1Row = result.leafNames.indexOf('s1');
  assertEqual(alnAt(result.alignment, result.C, s1Row, 1), gapIdx, 's1 col1');
});

runTest('empty error', () => {
  assertThrows(() => parseDict({}), 'Empty', 'empty');
});

runTest('unequal error', () => {
  assertThrows(() => parseDict({ a: 'ACGT', b: 'AC' }), 'Unequal', 'unequal');
});

runTest('preserves names', () => {
  const result = parseDict({ human: 'ACGT', mouse: 'TGCA', dog: 'GGGG' });
  assertEqual(result.leafNames.length, 3, 'count');
  const names = new Set(result.leafNames);
  assert(names.has('human'), 'human');
  assert(names.has('mouse'), 'mouse');
  assert(names.has('dog'), 'dog');
});

// ==================== combineTreeAlignment ====================

console.log('\n--- combineTreeAlignment ---');

runTest('basic', () => {
  const tree = parseNewick('((A:0.1,B:0.2):0.3,C:0.4);');
  const aln = parseFasta('>A\nACGT\n>B\nTGCA\n>C\nGGGG\n');
  const result = combineTreeAlignment(tree, aln);
  const R = tree.parentIndex.length;
  assertEqual(result.R, R, 'R');
  assertEqual(result.C, 4, 'C');
  // Internal nodes should have token A (ungapped-unobserved)
  const A = result.alphabet.length;
  const childCount = new Int32Array(R);
  for (let n = 1; n < R; n++) childCount[result.parentIndex[n]]++;
  for (let n = 0; n < R; n++) {
    if (childCount[n] > 0) {  // internal
      for (let c = 0; c < result.C; c++) {
        assertEqual(alnAt(result.alignment, result.C, n, c), A, `internal[${n}][${c}]`);
      }
    }
  }
});

runTest('name mismatch error', () => {
  const tree = parseNewick('((A:0.1,B:0.2):0.3,C:0.4);');
  const aln = parseFasta('>A\nACGT\n>B\nTGCA\n>X\nGGGG\n');
  assertThrows(() => combineTreeAlignment(tree, aln), 'not found', 'name mismatch');
});

runTest('extra alignment sequences ok', () => {
  const tree = parseNewick('(A:0.1,B:0.2);');
  const aln = parseFasta('>A\nACGT\n>B\nTGCA\n>C\nGGGG\n');
  const result = combineTreeAlignment(tree, aln);
  const R = tree.parentIndex.length;
  assertEqual(result.R, R, 'R');
  assertEqual(result.C, 4, 'C');
});

// ==================== geneticCode ====================

console.log('\n--- geneticCode ---');

runTest('64 codons', () => {
  const gc = geneticCode();
  assertEqual(gc.codons.length, 64, 'codons');
  assertEqual(gc.aminoAcids.length, 64, 'aminoAcids');
});

runTest('sense count', () => {
  const gc = geneticCode();
  let senseCount = 0;
  for (const m of gc.senseMask) if (m) senseCount++;
  assertEqual(senseCount, 61, 'sense count');
  assertEqual(gc.senseIndices.length, 61, 'senseIndices');
  assertEqual(gc.senseCodons.length, 61, 'senseCodons');
  assertEqual(gc.senseAminoAcids.length, 61, 'senseAminoAcids');
});

runTest('stop codons', () => {
  const gc = geneticCode();
  // TAA=48, TAG=50, TGA=56
  assertEqual(gc.aminoAcids[48], '*', 'TAA');
  assertEqual(gc.aminoAcids[50], '*', 'TAG');
  assertEqual(gc.aminoAcids[56], '*', 'TGA');
  assert(!gc.senseMask[48], 'TAA not sense');
  assert(!gc.senseMask[50], 'TAG not sense');
  assert(!gc.senseMask[56], 'TGA not sense');
});

runTest('codon to sense mapping', () => {
  const gc = geneticCode();
  // Stop codons map to -1
  assertEqual(gc.codonToSense[48], -1, 'TAA');
  assertEqual(gc.codonToSense[50], -1, 'TAG');
  assertEqual(gc.codonToSense[56], -1, 'TGA');
  // First sense codon (AAA=0) maps to 0
  assertEqual(gc.codonToSense[0], 0, 'AAA');
  // All sense codons have non-negative mapping
  for (let i = 0; i < gc.senseIndices.length; i++) {
    assert(gc.codonToSense[gc.senseIndices[i]] >= 0, `sense[${i}]`);
  }
});

runTest('roundtrip', () => {
  const gc = geneticCode();
  for (let i = 0; i < 64; i++) {
    if (gc.senseMask[i]) {
      const si = gc.codonToSense[i];
      assertEqual(gc.senseIndices[si], i, `roundtrip[${i}]`);
    }
  }
});

runTest('ACGT order', () => {
  const gc = geneticCode();
  assertEqual(gc.codons[0], 'AAA', 'first');
  assertEqual(gc.codons[63], 'TTT', 'last');
  assertEqual(gc.codons[1], 'AAC', 'second');
});

runTest('known amino acids', () => {
  const gc = geneticCode();
  // ATG = M (methionine), index = 0*16 + 3*4 + 2 = 14
  assertEqual(gc.codons[14], 'ATG', 'ATG codon');
  assertEqual(gc.aminoAcids[14], 'M', 'ATG->M');
  // TGG = W (tryptophan), index = 3*16 + 2*4 + 2 = 58
  assertEqual(gc.codons[58], 'TGG', 'TGG codon');
  assertEqual(gc.aminoAcids[58], 'W', 'TGG->W');
});

// ==================== codonToSense ====================

console.log('\n--- codonToSense ---');

runTest('basic remapping', () => {
  // Token 0 (AAA) → sense 0, 1 (AAC) → sense 1, 2 (AAG) → sense 2
  const aln = new Int32Array([0, 1, 2]);
  const result = codonToSense(aln, 64);
  assertEqual(result.A_sense, 61, 'A_sense');
  assertEqual(result.alignment[0], 0, 'AAA -> 0');
  assertEqual(result.alignment[1], 1, 'AAC -> 1');
  assertEqual(result.alignment[2], 2, 'AAG -> 2');
});

runTest('stop becomes gap', () => {
  // TAA=48 is a stop codon
  const aln = new Int32Array([48]);
  const result = codonToSense(aln, 64);
  assertEqual(result.alignment[0], 62, 'stop -> gap');
});

runTest('unobserved remapping', () => {
  const aln = new Int32Array([64]);  // unobserved token
  const result = codonToSense(aln, 64);
  assertEqual(result.alignment[0], 61, 'unobs -> 61');
});

runTest('gap remapping', () => {
  const aln = new Int32Array([65]);  // gap token
  const result = codonToSense(aln, 64);
  assertEqual(result.alignment[0], 62, 'gap -> 62');
});

runTest('legacy gap', () => {
  const aln = new Int32Array([-1]);  // legacy gap
  const result = codonToSense(aln);
  assertEqual(result.alignment[0], 62, 'legacy gap -> 62');
});

runTest('alphabet returned', () => {
  const aln = new Int32Array([0]);
  const result = codonToSense(aln, 64);
  assertEqual(result.alphabet.length, 61, 'alphabet length');
  assertEqual(result.alphabet[0], 'AAA', 'first codon');
});

// ==================== KmerIndex ====================

console.log('\n--- KmerIndex ---');

runTest('tuple to idx', () => {
  const idx = new KmerIndex([[0, 1], [2, 3], [0, 3]]);
  assertEqual(idx.tupleToIdx([0, 1]), 0, 'first');
  assertEqual(idx.tupleToIdx([2, 3]), 1, 'second');
  assertEqual(idx.tupleToIdx([0, 3]), 2, 'third');
});

runTest('tuple to idx missing', () => {
  const idx = new KmerIndex([[0, 1]]);
  assertEqual(idx.tupleToIdx([9, 9]), -1, 'missing');
});

runTest('idx to tuple', () => {
  const idx = new KmerIndex([[0, 1], [2, 3]]);
  assertArrayEqual(idx.idxToTuple(0), [0, 1], 'first');
  assertArrayEqual(idx.idxToTuple(1), [2, 3], 'second');
});

runTest('length', () => {
  const idx = new KmerIndex([[0, 1], [2, 3], [4, 5]]);
  assertEqual(idx.length, 3, 'length');
});

runTest('k property', () => {
  const idx = new KmerIndex([[0, 1, 2], [3, 4, 5]]);
  assertEqual(idx.k, 3, 'k');
});

runTest('flat Int32Array constructor', () => {
  const flat = new Int32Array([0, 1, 2, 3, 4, 5]);
  const idx = new KmerIndex(flat, 2);
  assertEqual(idx.length, 3, 'length');
  assertEqual(idx.k, 2, 'k');
  assertArrayEqual(idx.idxToTuple(0), [0, 1], 'first');
  assertArrayEqual(idx.idxToTuple(1), [2, 3], 'second');
  assertArrayEqual(idx.idxToTuple(2), [4, 5], 'third');
});

runTest('empty', () => {
  const idx = new KmerIndex([]);
  assertEqual(idx.length, 0, 'length');
});

// ==================== slidingWindows ====================

console.log('\n--- slidingWindows ---');

runTest('basic non-overlapping', () => {
  const sw = slidingWindows(6, 3);
  assertEqual(sw.T, 2, 'T');
  assertEqual(sw.k, 3, 'k');
  // First window: [0,1,2]
  assertArrayEqual([sw.tuples[0], sw.tuples[1], sw.tuples[2]], [0, 1, 2], 'win0');
  // Second window: [3,4,5]
  assertArrayEqual([sw.tuples[3], sw.tuples[4], sw.tuples[5]], [3, 4, 5], 'win1');
});

runTest('stride 1', () => {
  const sw = slidingWindows(5, 3, 1);
  assertEqual(sw.T, 3, 'T');
  // Windows: [0,1,2], [1,2,3], [2,3,4]
  assertArrayEqual([sw.tuples[0], sw.tuples[1], sw.tuples[2]], [0, 1, 2], 'win0');
  assertArrayEqual([sw.tuples[3], sw.tuples[4], sw.tuples[5]], [1, 2, 3], 'win1');
  assertArrayEqual([sw.tuples[6], sw.tuples[7], sw.tuples[8]], [2, 3, 4], 'win2');
});

runTest('offset', () => {
  const sw = slidingWindows(9, 3, 3, 1);
  assertEqual(sw.T, 2, 'T');
  assertArrayEqual([sw.tuples[0], sw.tuples[1], sw.tuples[2]], [1, 2, 3], 'win0');
  assertArrayEqual([sw.tuples[3], sw.tuples[4], sw.tuples[5]], [4, 5, 6], 'win1');
});

runTest('truncate drops partial', () => {
  const sw = slidingWindows(5, 3, 3);
  assertEqual(sw.T, 1, 'T');
  assertArrayEqual([sw.tuples[0], sw.tuples[1], sw.tuples[2]], [0, 1, 2], 'win0');
});

runTest('pad includes partial', () => {
  const sw = slidingWindows(5, 3, 3, 0, 'pad');
  assertEqual(sw.T, 2, 'T');
  assertArrayEqual([sw.tuples[0], sw.tuples[1], sw.tuples[2]], [0, 1, 2], 'win0');
  assertArrayEqual([sw.tuples[3], sw.tuples[4], sw.tuples[5]], [3, 4, -1], 'win1 padded');
});

runTest('empty result', () => {
  const sw = slidingWindows(2, 3);
  assertEqual(sw.T, 0, 'T');
  assertEqual(sw.tuples.length, 0, 'empty tuples');
});

runTest('three reading frames', () => {
  for (let offset = 0; offset < 3; offset++) {
    const sw = slidingWindows(9, 3, 3, offset);
    const starts = [];
    for (let i = 0; i < sw.T; i++) starts.push(sw.tuples[i * 3]);
    const expected = [];
    for (let s = offset; s <= 9 - 3; s += 3) expected.push(s);
    assertArrayEqual(starts, expected, `frame ${offset}`);
  }
});

// ==================== allColumnKtuples ====================

console.log('\n--- allColumnKtuples ---');

runTest('ordered k=2', () => {
  const result = allColumnKtuples(4, 2, true);
  assertEqual(result.T, 12, 'T = 4*3');
  assertEqual(result.k, 2, 'k');
});

runTest('unordered k=2', () => {
  const result = allColumnKtuples(4, 2, false);
  assertEqual(result.T, 6, 'T = C(4,2)');
  assertEqual(result.k, 2, 'k');
});

runTest('k=1', () => {
  const result = allColumnKtuples(3, 1, true);
  assertEqual(result.T, 3, 'T');
  assertArrayEqual(Array.from(result.tuples), [0, 1, 2], 'values');
});

runTest('empty when k > C', () => {
  const result = allColumnKtuples(1, 2, true);
  assertEqual(result.T, 0, 'T');
});

runTest('ordered k=2 no self pairs', () => {
  const result = allColumnKtuples(3, 2, true);
  // Should have 3*2 = 6 permutations, none with same column twice
  assertEqual(result.T, 6, 'T');
  for (let i = 0; i < result.T; i++) {
    const a = result.tuples[i * 2];
    const b = result.tuples[i * 2 + 1];
    assert(a !== b, `tuple ${i}: ${a} !== ${b}`);
  }
});

runTest('unordered k=2 ascending', () => {
  const result = allColumnKtuples(4, 2, false);
  // All tuples should have first < second (combinations)
  for (let i = 0; i < result.T; i++) {
    const a = result.tuples[i * 2];
    const b = result.tuples[i * 2 + 1];
    assert(a < b, `combo ${i}: ${a} < ${b}`);
  }
});

// ==================== kmerTokenize ====================

console.log('\n--- kmerTokenize ---');

runTest('codon basic', () => {
  // "ACGTAA" → tokens [0,1,2,3,0,0]
  const result = parseStrings(['ACGTAA'], ['A', 'C', 'G', 'T']);
  const kmer = kmerTokenize(result.alignment, result.N, result.C, 4, 3, 'any', ['A', 'C', 'G', 'T']);
  assertEqual(kmer.C_k, 2, 'C_k');
  assertEqual(kmer.A_kmer, 64, 'A_kmer');
  // ACG = 0*16 + 1*4 + 2 = 6
  assertEqual(kmer.alignment[0], 6, 'ACG');
  // TAA = 3*16 + 0*4 + 0 = 48
  assertEqual(kmer.alignment[1], 48, 'TAA');
});

runTest('kmer alphabet labels', () => {
  const aln = new Int32Array(4);  // 1 seq, 4 cols, all zeros
  const kmer = kmerTokenize(aln, 1, 4, 2, 2, 'any', ['0', '1']);
  assertArrayEqual(kmer.alphabet, ['00', '01', '10', '11'], 'alphabet');
  assertEqual(kmer.A_kmer, 4, 'A_kmer');
});

runTest('k1 identity', () => {
  const aln = new Int32Array([0, 1, 2, 3]);
  const kmer = kmerTokenize(aln, 1, 4, 4, 1);
  assertArrayEqual(Array.from(kmer.alignment), [0, 1, 2, 3], 'identity');
  assertEqual(kmer.A_kmer, 4, 'A_kmer');
});

runTest('gap any', () => {
  const A = 4;
  const gap = A + 1;
  const aln = new Int32Array([0, gap, 2]);  // 1 seq, 3 cols
  const kmer = kmerTokenize(aln, 1, 3, A, 3, 'any');
  assertEqual(kmer.alignment[0], 64 + 1, 'gap token');
});

runTest('gap all full gap', () => {
  const A = 4;
  const gap = A + 1;
  const aln = new Int32Array([gap, gap, gap]);
  const kmer = kmerTokenize(aln, 1, 3, A, 3, 'all');
  assertEqual(kmer.alignment[0], 64 + 1, 'gap token');
});

runTest('gap all partial gap', () => {
  const A = 4;
  const gap = A + 1;
  const aln = new Int32Array([0, gap, 2]);
  const kmer = kmerTokenize(aln, 1, 3, A, 3, 'all');
  assertEqual(kmer.alignment[0], 64 + 2, 'illegal token');
});

runTest('unobserved', () => {
  const A = 4;
  const unobs = A;
  const aln = new Int32Array([unobs, unobs, unobs]);
  const kmer = kmerTokenize(aln, 1, 3, A, 3);
  assertEqual(kmer.alignment[0], 64, 'unobserved token');
});

runTest('legacy gap', () => {
  const A = 4;
  const aln = new Int32Array([0, -1, 2]);
  const kmer = kmerTokenize(aln, 1, 3, A, 3, 'any');
  assertEqual(kmer.alignment[0], 64 + 1, 'gap');
});

runTest('C not divisible', () => {
  const aln = new Int32Array([0, 1, 2, 3, 0]);  // C=5
  assertThrows(() => kmerTokenize(aln, 1, 5, 4, 3), 'not divisible', 'error');
});

runTest('multiple sequences', () => {
  // Row 0: [0,1,2,3,0,0] → ACG=6, TAA=48
  // Row 1: [3,2,1,0,3,2] → TGC=57, ATG=14
  const aln = new Int32Array([0,1,2,3,0,0, 3,2,1,0,3,2]);
  const kmer = kmerTokenize(aln, 2, 6, 4, 3);
  assertEqual(kmer.C_k, 2, 'C_k');
  assertEqual(kmer.alignment[0], 6, 'seq0 ACG');
  assertEqual(kmer.alignment[1], 48, 'seq0 TAA');
  assertEqual(kmer.alignment[2], 57, 'seq1 TGC');
  assertEqual(kmer.alignment[3], 14, 'seq1 ATG');
});

runTest('fasta to codons', () => {
  const fasta = '>seq1\nACGTAA\n>seq2\nTGCATG\n';
  const result = parseFasta(fasta);
  const kmer = kmerTokenize(result.alignment, result.N, result.C, result.alphabet.length, 3, 'any', result.alphabet);
  assertEqual(kmer.N, 2, 'N');
  assertEqual(kmer.C_k, 2, 'C_k');
  assertEqual(kmer.A_kmer, 64, 'A_kmer');
  assertEqual(kmer.alphabet.length, 64, 'alphabet length');
});

// ==================== kmerTokenize with tuples ====================

console.log('\n--- kmerTokenize with tuples ---');

runTest('basic pairs', () => {
  const aln = new Int32Array([0, 1, 2, 3]);  // 1 seq, 4 cols
  const result = kmerTokenize(aln, 1, 4, 4, [[0, 2], [1, 3]]);
  assertEqual(result.C_k, 2, 'C_k');
  assertEqual(result.A_kmer, 16, 'A_kmer');
  assertEqual(result.alignment[0], 0*4 + 2, '(A,G) = 2');
  assertEqual(result.alignment[1], 1*4 + 3, '(C,T) = 7');
});

runTest('contiguous tuples match int k', () => {
  const aln = new Int32Array([0, 1, 2, 3, 0, 1]);  // 1 seq, 6 cols
  const resultK = kmerTokenize(aln, 1, 6, 4, 3);
  const resultT = kmerTokenize(aln, 1, 6, 4, [[0, 1, 2], [3, 4, 5]]);
  assertArrayEqual(Array.from(resultK.alignment), Array.from(resultT.alignment), 'match');
});

runTest('k1 tuples identity', () => {
  const aln = new Int32Array([0, 1, 2, 3]);
  const result = kmerTokenize(aln, 1, 4, 4, [[0], [1], [2], [3]]);
  assertArrayEqual(Array.from(result.alignment), [0, 1, 2, 3], 'identity');
});

runTest('sliding window integration', () => {
  const aln = new Int32Array([0, 1, 2, 3, 0]);  // 1 seq, 5 cols
  const sw = slidingWindows(5, 3, 1);
  const result = kmerTokenize(aln, 1, 5, 4, sw);
  assertEqual(result.C_k, 3, 'C_k');
  assertEqual(result.alignment[0], 0*16 + 1*4 + 2, 'ACG = 6');
  assertEqual(result.alignment[1], 1*16 + 2*4 + 3, 'CGT = 27');
  assertEqual(result.alignment[2], 2*16 + 3*4 + 0, 'GTA = 44');
});

runTest('sentinel padding', () => {
  const aln = new Int32Array([0, 1, 2, 3, 0]);  // 1 seq, 5 cols
  const sw = slidingWindows(5, 3, 3, 0, 'pad');
  const result = kmerTokenize(aln, 1, 5, 4, sw);
  assertEqual(result.C_k, 2, 'C_k');
  assertEqual(result.alignment[0], 0*16 + 1*4 + 2, 'ACG = 6');
  assertEqual(result.alignment[1], 64, 'unobserved (has -1)');
});

runTest('gap any tuples', () => {
  const A = 4;
  const gap = A + 1;
  const aln = new Int32Array([0, gap, 2, gap]);  // 1 seq, 4 cols
  const result = kmerTokenize(aln, 1, 4, A, [[0, 1]], 'any');
  assertEqual(result.alignment[0], A**2 + 1, 'gap token');
});

runTest('gap all tuples', () => {
  const A = 4;
  const gap = A + 1;
  const aln = new Int32Array([0, gap, gap, gap]);  // 1 seq, 4 cols
  // Partial gap (0 observed, 1 gap)
  const r1 = kmerTokenize(aln, 1, 4, A, [[0, 1]], 'all');
  assertEqual(r1.alignment[0], A**2 + 2, 'illegal');
  // All gap
  const r2 = kmerTokenize(aln, 1, 4, A, [[1, 2]], 'all');
  assertEqual(r2.alignment[0], A**2 + 1, 'gap');
});

runTest('index returned', () => {
  const aln = new Int32Array([0, 1, 2]);  // 1 seq, 3 cols
  const result = kmerTokenize(aln, 1, 3, 4, [[0, 2], [1, 2]]);
  const idx = result.index;
  assert(idx instanceof KmerIndex, 'is KmerIndex');
  assertEqual(idx.tupleToIdx([0, 2]), 0, 'first');
  assertEqual(idx.tupleToIdx([1, 2]), 1, 'second');
  assertArrayEqual(idx.idxToTuple(0), [0, 2], 'reverse');
});

runTest('alphabet labels tuples', () => {
  const aln = new Int32Array([0, 1]);  // 1 seq, 2 cols
  const result = kmerTokenize(aln, 1, 2, 2, [[0, 1]], 'any', ['A', 'B']);
  assertArrayEqual(result.alphabet, ['AA', 'AB', 'BA', 'BB'], 'alphabet');
});

runTest('all column ktuples integration', () => {
  const aln = new Int32Array([0, 1, 2, 3]);  // 1 seq, 4 cols
  const tuples = allColumnKtuples(4, 2, false);
  const result = kmerTokenize(aln, 1, 4, 4, tuples);
  assertEqual(result.C_k, 6, 'C(4,2) = 6');
});

// ==================== kmerTokenize backward compat ====================

console.log('\n--- kmerTokenize backward compat ---');

runTest('int k basic', () => {
  const aln = new Int32Array([0, 1, 2, 3, 0, 0]);  // 1 seq, 6 cols
  const result = kmerTokenize(aln, 1, 6, 4, 3, 'any', ['A', 'C', 'G', 'T']);
  assertEqual(result.C_k, 2, 'C_k');
  assertEqual(result.A_kmer, 64, 'A_kmer');
  assertEqual(result.alignment[0], 6, 'ACG');
  assertEqual(result.alignment[1], 48, 'TAA');
});

runTest('not divisible raises', () => {
  const aln = new Int32Array([0, 1, 2, 3, 0]);
  assertThrows(() => kmerTokenize(aln, 1, 5, 4, 3), 'not divisible', 'error');
});

runTest('index for int k', () => {
  const aln = new Int32Array([0, 1, 2, 3, 0, 1]);  // 1 seq, 6 cols
  const result = kmerTokenize(aln, 1, 6, 4, 3);
  const idx = result.index;
  assert(idx instanceof KmerIndex, 'is KmerIndex');
  assertEqual(idx.length, 2, 'length');
  assertArrayEqual(idx.idxToTuple(0), [0, 1, 2], 'first');
  assertArrayEqual(idx.idxToTuple(1), [3, 4, 5], 'second');
  assertEqual(idx.tupleToIdx([0, 1, 2]), 0, 'lookup');
});

// ==================== Cross-validation: type checks ====================

console.log('\n--- Type checks ---');

runTest('newick types', () => {
  const result = parseNewick('((A:0.1,B:0.2):0.05,C:0.3);');
  assert(result.parentIndex instanceof Int32Array, 'parentIndex Int32Array');
  assert(result.distanceToParent instanceof Float64Array, 'distanceToParent Float64Array');
});

runTest('fasta types', () => {
  const result = parseFasta('>A\nACGT\n>B\nTGCA\n');
  assert(result.alignment instanceof Int32Array, 'alignment Int32Array');
});

runTest('combined types', () => {
  const tree = parseNewick('((A:0.1,B:0.2):0.05,C:0.3);');
  const aln = parseFasta('>A\nACGT\n>B\nTGCA\n>C\nGGGG\n');
  const result = combineTreeAlignment(tree, aln);
  assert(result.alignment instanceof Int32Array, 'alignment Int32Array');
  assert(result.parentIndex instanceof Int32Array, 'parentIndex Int32Array');
  assert(result.distanceToParent instanceof Float64Array, 'distanceToParent Float64Array');
});

runTest('slidingWindows types', () => {
  const sw = slidingWindows(6, 3);
  assert(sw.tuples instanceof Int32Array, 'tuples Int32Array');
});

runTest('allColumnKtuples types', () => {
  const result = allColumnKtuples(4, 2);
  assert(result.tuples instanceof Int32Array, 'tuples Int32Array');
});

runTest('kmerTokenize types', () => {
  const aln = new Int32Array([0, 1, 2, 3, 0, 1]);
  const result = kmerTokenize(aln, 1, 6, 4, 3);
  assert(result.alignment instanceof Int32Array, 'alignment Int32Array');
  assert(result.index instanceof KmerIndex, 'index KmerIndex');
});

// ---- Summary ----
console.log(`\n${passed}/${total} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
