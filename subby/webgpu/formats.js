/**
 * Parsers for standard phylogenetic file formats.
 *
 * Converts Newick trees, FASTA/Stockholm/MAF alignments, and plain strings
 * into subby's internal representation: Int32Array alignment + parentIndex /
 * distanceToParent arrays.
 *
 * Pure JavaScript, no external dependencies.
 */

const DNA = ['A', 'C', 'G', 'T'];
const RNA = ['A', 'C', 'G', 'U'];
const PROTEIN = 'ACDEFGHIKLMNPQRSTVWY'.split('');
const GAP_CHARS = new Set(['-', '.']);

/**
 * Auto-detect alphabet from a set of characters.
 * Recognizes DNA (ACGT), RNA (ACGU), protein (20 AAs), else sorted unique.
 * @param {Set<string>} chars
 * @returns {string[]}
 */
export function detectAlphabet(chars) {
  const upper = new Set();
  for (const ch of chars) {
    const u = ch.toUpperCase();
    if (!GAP_CHARS.has(u)) upper.add(u);
  }
  const isSubset = (set, arr) => {
    for (const ch of set) if (!arr.includes(ch)) return false;
    return true;
  };
  if (isSubset(upper, DNA)) return [...DNA];
  if (isSubset(upper, RNA)) return [...RNA];
  if (isSubset(upper, PROTEIN)) return [...PROTEIN];
  return [...upper].sort();
}

// ---- Newick parser ----

function stripComments(s) {
  const out = [];
  let depth = 0;
  let inQuote = false;
  for (const ch of s) {
    if (ch === "'" && depth === 0) {
      inQuote = !inQuote;
      out.push(ch);
    } else if (inQuote) {
      out.push(ch);
    } else if (ch === '[') {
      depth++;
    } else if (ch === ']') {
      depth--;
    } else if (depth === 0) {
      out.push(ch);
    }
  }
  return out.join('');
}

class NewickTokenizer {
  constructor(s) {
    this.s = s;
    this.pos = 0;
  }

  peek() {
    return this.pos < this.s.length ? this.s[this.pos] : null;
  }

  consume(expected) {
    const ch = this.s[this.pos];
    if (expected !== undefined && ch !== expected) {
      throw new Error(`Expected '${expected}' at position ${this.pos}, got '${ch}'`);
    }
    this.pos++;
    return ch;
  }

  readLabel() {
    if (this.pos >= this.s.length) return null;
    if (this.s[this.pos] === "'") return this._readQuoted();
    return this._readUnquoted();
  }

  _readQuoted() {
    this.consume("'");
    const parts = [];
    while (this.pos < this.s.length) {
      const ch = this.s[this.pos++];
      if (ch === "'") {
        if (this.pos < this.s.length && this.s[this.pos] === "'") {
          parts.push("'");
          this.pos++;
        } else {
          return parts.join('');
        }
      } else {
        parts.push(ch);
      }
    }
    throw new Error('Unterminated quoted label');
  }

  _readUnquoted() {
    const start = this.pos;
    while (this.pos < this.s.length && !'(),:;'.includes(this.s[this.pos])) {
      this.pos++;
    }
    const text = this.s.slice(start, this.pos).trim();
    return text || null;
  }

  readBranchLength() {
    if (this.pos < this.s.length && this.s[this.pos] === ':') {
      this.consume(':');
      const start = this.pos;
      while (this.pos < this.s.length && !'(),;'.includes(this.s[this.pos])) {
        this.pos++;
      }
      const text = this.s.slice(start, this.pos).trim();
      return text ? parseFloat(text) : 0.0;
    }
    return null;
  }
}

/**
 * Parse a Newick tree string.
 * @param {string} newickStr
 * @returns {{parentIndex: Int32Array, distanceToParent: Float64Array,
 *            leafNames: string[], nodeNames: (string|null)[], R: number}}
 */
export function parseNewick(newickStr) {
  let s = stripComments(newickStr.trim());
  while (s.endsWith(';')) s = s.slice(0, -1);
  s = s.trim();
  if (!s) throw new Error('Empty Newick string');

  const tok = new NewickTokenizer(s);
  const nodes = []; // [{name, dist, children}]

  function parseNode() {
    const children = [];
    if (tok.peek() === '(') {
      tok.consume('(');
      children.push(parseNode());
      while (tok.peek() === ',') {
        tok.consume(',');
        children.push(parseNode());
      }
      tok.consume(')');
    }
    const name = tok.readLabel();
    let dist = tok.readBranchLength();
    if (dist === null) dist = 0.0;
    const idx = nodes.length;
    nodes.push({ name, dist, children });
    return idx;
  }

  const rootIdx = parseNode();

  const parentIndex = [];
  const distanceToParent = [];
  const nodeNames = [];
  const leafNames = [];

  function dfs(oldIdx, parentNew) {
    const newIdx = parentIndex.length;
    const { name, dist, children } = nodes[oldIdx];
    parentIndex.push(parentNew);
    distanceToParent.push(parentNew >= 0 ? dist : 0.0);
    nodeNames.push(name);
    if (children.length === 0) leafNames.push(name);
    for (const childOld of children) dfs(childOld, newIdx);
  }

  dfs(rootIdx, -1);

  return {
    parentIndex: new Int32Array(parentIndex),
    distanceToParent: new Float64Array(distanceToParent),
    leafNames,
    nodeNames,
    R: parentIndex.length,
  };
}

// ---- Helper: build char→index map ----

function buildCharMap(alphabet) {
  const map = new Map();
  for (let i = 0; i < alphabet.length; i++) {
    map.set(alphabet[i], i);
    map.set(alphabet[i].toLowerCase(), i);
  }
  return map;
}

// ---- FASTA parser ----

/**
 * Parse FASTA-formatted alignment text.
 * @param {string} text
 * @param {string[]|null} [alphabet=null]
 * @returns {{alignment: Int32Array, leafNames: string[], alphabet: string[],
 *            N: number, C: number}}
 */
export function parseFasta(text, alphabet = null) {
  const sequences = [];
  const names = [];
  let currentName = null;
  let currentSeq = [];

  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith('>')) {
      if (currentName !== null) sequences.push(currentSeq.join(''));
      const header = trimmed.slice(1).trim();
      currentName = header.split(/\s+/)[0] || '';
      names.push(currentName);
      currentSeq = [];
    } else {
      currentSeq.push(trimmed);
    }
  }
  if (currentName !== null) sequences.push(currentSeq.join(''));

  if (sequences.length === 0) throw new Error('No sequences found in FASTA input');

  const lengths = new Set(sequences.map(s => s.length));
  if (lengths.size > 1) throw new Error(`Unequal sequence lengths in FASTA: ${[...lengths].sort()}`);

  const C = [...lengths][0];
  if (C === 0) throw new Error('Empty sequences in FASTA input');

  const allChars = new Set();
  for (const s of sequences) for (const ch of s) allChars.add(ch.toUpperCase());
  for (const g of GAP_CHARS) allChars.delete(g);

  if (!alphabet) alphabet = detectAlphabet(allChars);

  const charMap = buildCharMap(alphabet);
  const gapIdx = alphabet.length + 1;
  const N = sequences.length;
  const alignment = new Int32Array(N * C);

  for (let r = 0; r < N; r++) {
    const seq = sequences[r];
    for (let c = 0; c < C; c++) {
      const ch = seq[c];
      if (GAP_CHARS.has(ch)) {
        alignment[r * C + c] = gapIdx;
      } else if (charMap.has(ch)) {
        alignment[r * C + c] = charMap.get(ch);
      } else {
        throw new Error(`Unknown character '${ch}' in sequence '${names[r]}' at position ${c}`);
      }
    }
  }

  return { alignment, leafNames: names, alphabet, N, C };
}

// ---- Stockholm parser ----

/**
 * Parse Stockholm-format alignment text.
 * @param {string} text
 * @param {string[]|null} [alphabet=null]
 * @returns {{alignment: Int32Array, leafNames: string[], alphabet: string[],
 *            N: number, C: number, parentIndex?: Int32Array,
 *            distanceToParent?: Float64Array, R?: number}}
 */
export function parseStockholm(text, alphabet = null) {
  const seqData = new Map();
  const seqOrder = [];
  const nhParts = [];

  for (const line of text.split('\n')) {
    const trimmed = line.trimEnd();
    if (!trimmed || trimmed.startsWith('# STOCKHOLM')) continue;
    if (trimmed === '//') break;
    if (trimmed.startsWith('#=GF NH')) {
      nhParts.push(trimmed.slice(7).trim());
      continue;
    }
    if (trimmed.startsWith('#')) continue;

    const parts = trimmed.split(/\s+/);
    if (parts.length >= 2) {
      const name = parts[0];
      const seq = parts[1];
      if (!seqData.has(name)) {
        seqData.set(name, []);
        seqOrder.push(name);
      }
      seqData.get(name).push(seq);
    }
  }

  if (seqData.size === 0) throw new Error('No sequences found in Stockholm input');

  const sequences = seqOrder.map(name => seqData.get(name).join(''));
  const lengths = new Set(sequences.map(s => s.length));
  if (lengths.size > 1) throw new Error(`Unequal sequence lengths in Stockholm`);
  const C = [...lengths][0];

  const allChars = new Set();
  for (const s of sequences) for (const ch of s) allChars.add(ch.toUpperCase());
  for (const g of GAP_CHARS) allChars.delete(g);

  if (!alphabet) alphabet = detectAlphabet(allChars);

  const charMap = buildCharMap(alphabet);
  const gapIdx = alphabet.length + 1;
  const N = sequences.length;
  const alignment = new Int32Array(N * C);

  for (let r = 0; r < N; r++) {
    const seq = sequences[r];
    for (let c = 0; c < C; c++) {
      const ch = seq[c];
      if (GAP_CHARS.has(ch)) {
        alignment[r * C + c] = gapIdx;
      } else if (charMap.has(ch)) {
        alignment[r * C + c] = charMap.get(ch);
      } else {
        throw new Error(`Unknown character '${ch}' in sequence '${seqOrder[r]}'`);
      }
    }
  }

  const result = { alignment, leafNames: seqOrder, alphabet, N, C };

  if (nhParts.length > 0) {
    const nhStr = nhParts.join(' ');
    const treeResult = parseNewick(nhStr);
    return combineTreeAlignment(treeResult, result);
  }

  return result;
}

// ---- MAF parser ----

/**
 * Parse MAF (Multiple Alignment Format) text.
 * @param {string} text
 * @param {string[]|null} [alphabet=null]
 * @returns {{alignment: Int32Array, leafNames: string[], alphabet: string[],
 *            N: number, C: number}}
 */
export function parseMaf(text, alphabet = null) {
  const blocks = [];
  let currentBlock = null;

  for (const line of text.split('\n')) {
    const trimmed = line.trimEnd();
    if (trimmed.startsWith('a')) {
      currentBlock = [];
      blocks.push(currentBlock);
    } else if (trimmed.startsWith('s') && currentBlock !== null) {
      const parts = trimmed.split(/\s+/);
      const src = parts[1];
      const seq = parts[parts.length - 1];
      const species = src.split('.')[0];
      currentBlock.push({ species, seq });
    }
  }

  if (blocks.length === 0) throw new Error('No alignment blocks found in MAF input');

  const allSpecies = [];
  const seen = new Set();
  for (const block of blocks) {
    for (const { species } of block) {
      if (!seen.has(species)) {
        allSpecies.push(species);
        seen.add(species);
      }
    }
  }

  const allChars = new Set();
  for (const block of blocks) {
    for (const { seq } of block) {
      for (const ch of seq) allChars.add(ch.toUpperCase());
    }
  }
  for (const g of GAP_CHARS) allChars.delete(g);

  if (!alphabet) alphabet = detectAlphabet(allChars);

  const charMap = buildCharMap(alphabet);
  const gapIdx = alphabet.length + 1;
  const speciesIdx = new Map();
  allSpecies.forEach((sp, i) => speciesIdx.set(sp, i));
  const N = allSpecies.length;

  // Compute total columns
  let totalC = 0;
  for (const block of blocks) {
    if (block.length > 0) totalC += block[0].seq.length;
  }

  const alignment = new Int32Array(N * totalC);
  alignment.fill(gapIdx);

  let colOffset = 0;
  for (const block of blocks) {
    const blockWidth = block.length > 0 ? block[0].seq.length : 0;
    for (const { species, seq } of block) {
      const r = speciesIdx.get(species);
      for (let c = 0; c < seq.length; c++) {
        const ch = seq[c];
        if (GAP_CHARS.has(ch)) {
          alignment[r * totalC + colOffset + c] = gapIdx;
        } else if (charMap.has(ch)) {
          alignment[r * totalC + colOffset + c] = charMap.get(ch);
        } else {
          throw new Error(`Unknown character '${ch}' in species '${species}'`);
        }
      }
    }
    colOffset += blockWidth;
  }

  return { alignment, leafNames: allSpecies, alphabet, N, C: totalC };
}

// ---- Plain string parser ----

/**
 * Parse a list of equal-length strings into an alignment.
 * @param {string[]} sequences
 * @param {string[]|null} [alphabet=null]
 * @returns {{alignment: Int32Array, alphabet: string[], N: number, C: number}}
 */
export function parseStrings(sequences, alphabet = null) {
  if (sequences.length === 0) throw new Error('Empty sequence list');

  const lengths = new Set(sequences.map(s => s.length));
  if (lengths.size > 1) throw new Error(`Unequal sequence lengths: ${[...lengths].sort()}`);

  const C = [...lengths][0];
  if (C === 0) throw new Error('Empty sequences');

  const allChars = new Set();
  for (const s of sequences) for (const ch of s) allChars.add(ch.toUpperCase());
  for (const g of GAP_CHARS) allChars.delete(g);

  if (!alphabet) alphabet = detectAlphabet(allChars);

  const charMap = buildCharMap(alphabet);
  const gapIdx = alphabet.length + 1;
  const N = sequences.length;
  const alignment = new Int32Array(N * C);

  for (let r = 0; r < N; r++) {
    const seq = sequences[r];
    for (let c = 0; c < C; c++) {
      const ch = seq[c];
      if (GAP_CHARS.has(ch)) {
        alignment[r * C + c] = gapIdx;
      } else if (charMap.has(ch)) {
        alignment[r * C + c] = charMap.get(ch);
      } else {
        throw new Error(`Unknown character '${ch}' at row ${r}, col ${c}`);
      }
    }
  }

  return { alignment, alphabet, N, C };
}

// ---- Dictionary parser ----

/**
 * Parse a name→sequence object into an alignment.
 * @param {Object} sequences - e.g. { "human": "ACGT", "mouse": "TGCA" }
 * @param {string[]|null} [alphabet=null]
 * @returns {{alignment: Int32Array, leafNames: string[], alphabet: string[],
 *            N: number, C: number}}
 */
export function parseDict(sequences, alphabet = null) {
  const names = Object.keys(sequences);
  if (names.length === 0) throw new Error('Empty sequence dictionary');

  const seqs = names.map(name => sequences[name]);
  const lengths = new Set(seqs.map(s => s.length));
  if (lengths.size > 1) throw new Error(`Unequal sequence lengths: ${[...lengths].sort()}`);

  const C = [...lengths][0];
  if (C === 0) throw new Error('Empty sequences');

  const allChars = new Set();
  for (const s of seqs) for (const ch of s) allChars.add(ch.toUpperCase());
  for (const g of GAP_CHARS) allChars.delete(g);

  if (!alphabet) alphabet = detectAlphabet(allChars);

  const charMap = buildCharMap(alphabet);
  const gapIdx = alphabet.length + 1;
  const N = seqs.length;
  const alignment = new Int32Array(N * C);

  for (let r = 0; r < N; r++) {
    const seq = seqs[r];
    for (let c = 0; c < C; c++) {
      const ch = seq[c];
      if (GAP_CHARS.has(ch)) {
        alignment[r * C + c] = gapIdx;
      } else if (charMap.has(ch)) {
        alignment[r * C + c] = charMap.get(ch);
      } else {
        throw new Error(`Unknown character '${ch}' in sequence '${names[r]}' at position ${c}`);
      }
    }
  }

  return { alignment, leafNames: names, alphabet, N, C };
}

// ---- Combine tree + alignment ----

/**
 * Map leaf sequences to tree positions by name matching.
 * Creates full (R, C) alignment with internal node rows filled with token A.
 * @param {object} treeResult - from parseNewick
 * @param {object} alignmentResult - from parseFasta/parseStockholm/etc
 * @returns {{alignment: Int32Array, parentIndex: Int32Array,
 *            distanceToParent: Float64Array, alphabet: string[],
 *            leafNames: string[], R: number, C: number}}
 */
export function combineTreeAlignment(treeResult, alignmentResult) {
  const treeLeaves = treeResult.leafNames;
  const alnNames = alignmentResult.leafNames;
  const alnData = alignmentResult.alignment;
  const alphabet = alignmentResult.alphabet;
  const alnC = alignmentResult.C || (alnData.length / alnNames.length);

  const A = alphabet.length;
  const ungappedUnobserved = A;

  const alnNameToRow = new Map();
  alnNames.forEach((name, i) => alnNameToRow.set(name, i));

  const missing = treeLeaves.filter(name => !alnNameToRow.has(name));
  if (missing.length > 0) {
    throw new Error(`Tree leaves not found in alignment: ${missing}`);
  }

  const R = treeResult.parentIndex.length;
  const C = Math.round(alnC);
  const fullAln = new Int32Array(R * C);
  fullAln.fill(ungappedUnobserved);

  const parentIndex = treeResult.parentIndex;
  const childCount = new Int32Array(R);
  for (let n = 1; n < R; n++) childCount[parentIndex[n]]++;

  let leafIdx = 0;
  for (let n = 0; n < R; n++) {
    if (childCount[n] === 0) {
      const leafName = treeLeaves[leafIdx];
      const alnRow = alnNameToRow.get(leafName);
      for (let c = 0; c < C; c++) {
        fullAln[n * C + c] = alnData[alnRow * C + c];
      }
      leafIdx++;
    }
  }

  return {
    alignment: fullAln,
    parentIndex: treeResult.parentIndex,
    distanceToParent: treeResult.distanceToParent,
    alphabet,
    leafNames: treeLeaves,
    R,
    C,
  };
}

// ---- Genetic code helpers ----

/**
 * Return the standard genetic code: 64 codons, sense/stop classification,
 * and mapping from 64-codon indices to 61-sense-codon indices.
 * @returns {{codons: string[], aminoAcids: string[], senseMask: boolean[],
 *            senseIndices: Int32Array, codonToSense: Int32Array,
 *            senseCodons: string[], senseAminoAcids: string[]}}
 */
export function geneticCode() {
  const bases = ['A', 'C', 'G', 'T'];
  const codons = [];
  for (const b1 of bases)
    for (const b2 of bases)
      for (const b3 of bases)
        codons.push(b1 + b2 + b3);

  const codeTable = {
    'AAA':'K','AAC':'N','AAG':'K','AAT':'N',
    'ACA':'T','ACC':'T','ACG':'T','ACT':'T',
    'AGA':'R','AGC':'S','AGG':'R','AGT':'S',
    'ATA':'I','ATC':'I','ATG':'M','ATT':'I',
    'CAA':'Q','CAC':'H','CAG':'Q','CAT':'H',
    'CCA':'P','CCC':'P','CCG':'P','CCT':'P',
    'CGA':'R','CGC':'R','CGG':'R','CGT':'R',
    'CTA':'L','CTC':'L','CTG':'L','CTT':'L',
    'GAA':'E','GAC':'D','GAG':'E','GAT':'D',
    'GCA':'A','GCC':'A','GCG':'A','GCT':'A',
    'GGA':'G','GGC':'G','GGG':'G','GGT':'G',
    'GTA':'V','GTC':'V','GTG':'V','GTT':'V',
    'TAA':'*','TAC':'Y','TAG':'*','TAT':'Y',
    'TCA':'S','TCC':'S','TCG':'S','TCT':'S',
    'TGA':'*','TGC':'C','TGG':'W','TGT':'C',
    'TTA':'L','TTC':'F','TTG':'L','TTT':'F',
  };

  const aminoAcids = codons.map(c => codeTable[c]);
  const senseMask = aminoAcids.map(aa => aa !== '*');
  const senseIndices = [];
  const codonToSenseMap = new Int32Array(64).fill(-1);
  let senseIdx = 0;
  for (let i = 0; i < 64; i++) {
    if (senseMask[i]) {
      senseIndices.push(i);
      codonToSenseMap[i] = senseIdx++;
    }
  }
  const senseCodons = senseIndices.map(i => codons[i]);
  const senseAminoAcids = senseIndices.map(i => aminoAcids[i]);

  return {
    codons, aminoAcids, senseMask,
    senseIndices: new Int32Array(senseIndices),
    codonToSense: codonToSenseMap,
    senseCodons, senseAminoAcids,
  };
}

// ---- Codon-to-sense remapping ----

/**
 * Remap a 64-codon tokenized alignment to 61-sense-codon tokens.
 * Stop codons are mapped to gap.
 * @param {Int32Array} alignment - flat token array (64-codon encoding)
 * @param {number} [A=64] - input alphabet size (64 for full codons)
 * @returns {{alignment: Int32Array, A_sense: number, alphabet: string[]}}
 */
export function codonToSense(alignment, A = 64) {
  const gc = geneticCode();
  const codonMap = gc.codonToSense;
  const N = alignment.length;
  const ASense = 61;
  const unobsIn = A;        // 64
  const gapIn = A + 1;      // 65
  const unobsOut = ASense;   // 61
  const gapOut = ASense + 1; // 62

  const result = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    const tok = alignment[i];
    if (tok >= 0 && tok < 64) {
      result[i] = codonMap[tok] >= 0 ? codonMap[tok] : gapOut;
    } else if (tok === unobsIn) {
      result[i] = unobsOut;
    } else {
      result[i] = gapOut;
    }
  }
  return { alignment: result, A_sense: ASense, alphabet: gc.senseCodons };
}

// ---- K-mer index ----

/**
 * Maps between column tuples and output alignment indices.
 * Provides O(1) lookup from column tuple to output index and vice versa.
 */
export class KmerIndex {
  /**
   * @param {Int32Array|number[][]} tuples - (T, k) row-major column index tuples.
   *   If Int32Array, interpreted as flat T*k row-major. Must also pass k.
   *   If array of arrays, k is inferred from inner length.
   * @param {number} [k] - required when tuples is a flat Int32Array
   */
  constructor(tuples, k) {
    if (tuples instanceof Int32Array ||
        (typeof BigInt64Array !== 'undefined' && tuples instanceof BigInt64Array) ||
        (ArrayBuffer.isView(tuples) && !Array.isArray(tuples))) {
      // Flat typed array: need k
      if (k == null) throw new Error('k is required when tuples is a flat typed array');
      this._k = k;
      this._flat = new Int32Array(tuples);
      this._T = this._flat.length / k;
      if (this._flat.length % k !== 0) {
        throw new Error(`Flat tuples length (${this._flat.length}) not divisible by k (${k})`);
      }
    } else if (Array.isArray(tuples)) {
      // Array of arrays
      this._T = tuples.length;
      if (this._T === 0) {
        this._k = k || 0;
        this._flat = new Int32Array(0);
      } else {
        this._k = tuples[0].length;
        this._flat = new Int32Array(this._T * this._k);
        for (let i = 0; i < this._T; i++) {
          for (let j = 0; j < this._k; j++) {
            this._flat[i * this._k + j] = tuples[i][j];
          }
        }
      }
    } else {
      throw new Error('tuples must be Int32Array or array of arrays');
    }

    // Build lookup: tuple string → index
    this._lookup = new Map();
    for (let i = 0; i < this._T; i++) {
      const key = this._tupleKey(i);
      this._lookup.set(key, i);
    }
  }

  /** @private */
  _tupleKey(idx) {
    const offset = idx * this._k;
    let key = '';
    for (let j = 0; j < this._k; j++) {
      if (j > 0) key += ',';
      key += this._flat[offset + j];
    }
    return key;
  }

  /** @private */
  _arrayToKey(arr) {
    let key = '';
    for (let j = 0; j < arr.length; j++) {
      if (j > 0) key += ',';
      key += arr[j];
    }
    return key;
  }

  /**
   * Column tuple to index in the output alignment.
   * @param {number[]} t - column tuple
   * @returns {number} index, or -1 if absent
   */
  tupleToIdx(t) {
    const key = this._arrayToKey(t);
    const idx = this._lookup.get(key);
    return idx !== undefined ? idx : -1;
  }

  /**
   * Index in the output alignment to column tuple.
   * @param {number} idx
   * @returns {number[]} column indices
   */
  idxToTuple(idx) {
    const offset = idx * this._k;
    const result = new Array(this._k);
    for (let j = 0; j < this._k; j++) {
      result[j] = this._flat[offset + j];
    }
    return result;
  }

  /** Number of tuples */
  get length() { return this._T; }

  /** k-mer size */
  get k() { return this._k; }
}

// ---- Sliding windows ----

/**
 * Generate column index tuples for sliding-window k-mer tokenization.
 *
 * @param {number} C - number of columns in the alignment
 * @param {number} k - window size (k-mer length)
 * @param {number|null} [stride=null] - step between window starts. Default null -> k (non-overlapping).
 * @param {number} [offset=0] - starting column index
 * @param {string} [edge='truncate'] - handling of incomplete trailing window:
 *   'truncate' - drop incomplete trailing window (default)
 *   'pad' - include partial window, using -1 for out-of-bounds columns
 * @returns {{tuples: Int32Array, T: number, k: number}}
 *   tuples is flat T*k row-major Int32Array of column indices.
 *   Entries of -1 indicate out-of-bounds positions (only with edge='pad').
 */
export function slidingWindows(C, k, stride = null, offset = 0, edge = 'truncate') {
  if (stride === null) stride = k;

  let starts;
  if (edge === 'truncate') {
    // starts from offset up to C - k (inclusive), stepping by stride
    starts = [];
    for (let s = offset; s <= C - k; s += stride) {
      starts.push(s);
    }
  } else if (edge === 'pad') {
    starts = [];
    for (let s = offset; s < C; s += stride) {
      starts.push(s);
    }
  } else {
    throw new Error(`Unknown edge mode: '${edge}'`);
  }

  const T = starts.length;
  const tuples = new Int32Array(T * k);

  for (let i = 0; i < T; i++) {
    const base = starts[i];
    for (let j = 0; j < k; j++) {
      const col = base + j;
      tuples[i * k + j] = (col < C) ? col : -1;
    }
  }

  return { tuples, T, k };
}

// ---- All column k-tuples ----

/**
 * Generate all k-tuples of column indices.
 *
 * WARNING: produces O(C^k) tuples. Use with caution for large C or k > 2.
 *
 * @param {number} C - number of columns
 * @param {number} k - tuple size
 * @param {boolean} [ordered=true] - if true, permutations (C*(C-1) for k=2);
 *   if false, combinations (C choose k)
 * @returns {{tuples: Int32Array, T: number, k: number}}
 *   tuples is flat T*k row-major Int32Array of column index tuples.
 */
export function allColumnKtuples(C, k, ordered = true) {
  const result = [];

  if (ordered) {
    // Generate permutations of k elements from 0..C-1
    _permutations(C, k, [], result);
  } else {
    // Generate combinations of k elements from 0..C-1
    _combinations(C, k, 0, [], result);
  }

  const T = result.length;
  const tuples = new Int32Array(T * k);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < k; j++) {
      tuples[i * k + j] = result[i][j];
    }
  }

  return { tuples, T, k };
}

/** @private Generate permutations of k elements from 0..n-1 (no repeats) */
function _permutations(n, k, current, result) {
  if (current.length === k) {
    result.push([...current]);
    return;
  }
  for (let i = 0; i < n; i++) {
    if (current.includes(i)) continue;
    current.push(i);
    _permutations(n, k, current, result);
    current.pop();
  }
}

/** @private Generate combinations of k elements from start..n-1 */
function _combinations(n, k, start, current, result) {
  if (current.length === k) {
    result.push([...current]);
    return;
  }
  for (let i = start; i < n; i++) {
    current.push(i);
    _combinations(n, k, i + 1, current, result);
    current.pop();
  }
}

// ---- Build k-mer alphabet helper ----

/**
 * Build k-mer label list from single-character alphabet, or return null.
 * @private
 */
function _buildKmerAlphabet(alphabet, A, k) {
  if (alphabet === null) return null;
  const Ak = Math.pow(A, k);
  const labels = [];
  const indices = new Array(k).fill(0);
  for (let i = 0; i < Ak; i++) {
    let label = '';
    let val = i;
    for (let p = k - 1; p >= 0; p--) {
      indices[p] = val % A;
      val = Math.floor(val / A);
    }
    for (let p = 0; p < k; p++) label += alphabet[indices[p]];
    labels.push(label);
  }
  return labels;
}

// ---- K-mer tokenization ----

/**
 * Convert single-character token alignment to k-mer tokens.
 *
 * Accepts either:
 *   - A number k: generates non-overlapping contiguous windows of size k.
 *     C must be divisible by k (backward compatible).
 *   - An Int32Array (flat T*k row-major) or array of arrays: column tuples,
 *     each of length k. Use slidingWindows() or allColumnKtuples() to generate
 *     tuples, or pass custom tuples directly. Entries of -1 in tuples are
 *     treated as unobserved positions.
 *
 * Token encoding: 0..A^k-1 observed, A^k ungapped-unobserved, A^k+1 gap.
 * When gapMode='all', partial gaps produce illegal token (A^k+2).
 *
 * @param {Int32Array} alignment - flat (N*C) token array
 * @param {number} N - number of sequences
 * @param {number} C - number of columns
 * @param {number} A - single-character alphabet size
 * @param {number|Int32Array|number[][]} kOrTuples - either an int k or column tuples
 * @param {string} [gapMode='any'] - 'any' or 'all'
 * @param {string[]|null} [alphabet=null] - single-char labels for k-mer labels
 * @returns {{alignment: Int32Array, A_kmer: number, N: number, C_k: number,
 *            index: KmerIndex, alphabet: string[]|null}}
 */
export function kmerTokenize(alignment, N, C, A, kOrTuples, gapMode = 'any', alphabet = null) {
  let k, T, columnTuples;

  if (typeof kOrTuples === 'number') {
    // Backward-compatible: integer k → non-overlapping contiguous windows
    k = kOrTuples;
    if (C % k !== 0) throw new Error(`Number of columns (${C}) not divisible by k (${k})`);
    const sw = slidingWindows(C, k);
    columnTuples = sw.tuples;
    T = sw.T;
  } else if (kOrTuples instanceof Int32Array || (ArrayBuffer.isView(kOrTuples) && !(kOrTuples instanceof Array))) {
    // Flat typed array: need to infer k from context
    // User must ensure it's T*k row-major. We need k to interpret it.
    // Convention: when passing flat Int32Array, the caller should also set
    // .k on the array or we infer from the shape. Since JS typed arrays
    // don't carry shape, we require array of arrays or the slidingWindows
    // result. For flat Int32Array, we look for a .k property or require
    // it to have been generated by slidingWindows/allColumnKtuples.
    // Fallback: we cannot infer k from a flat array alone. Require array of arrays.
    throw new Error(
      'Flat typed array passed to kOrTuples without shape info. ' +
      'Use slidingWindows() or allColumnKtuples() which return {tuples, T, k}, ' +
      'then pass the result object or use array-of-arrays format.'
    );
  } else if (Array.isArray(kOrTuples)) {
    // Array of arrays
    if (kOrTuples.length === 0) {
      throw new Error('Empty column tuples array');
    }
    k = kOrTuples[0].length;
    T = kOrTuples.length;
    columnTuples = new Int32Array(T * k);
    for (let i = 0; i < T; i++) {
      if (kOrTuples[i].length !== k) {
        throw new Error(`Inconsistent tuple length at index ${i}: expected ${k}, got ${kOrTuples[i].length}`);
      }
      for (let j = 0; j < k; j++) {
        columnTuples[i * k + j] = kOrTuples[i][j];
      }
    }
  } else if (typeof kOrTuples === 'object' && kOrTuples !== null &&
             'tuples' in kOrTuples && 'T' in kOrTuples && 'k' in kOrTuples) {
    // Result object from slidingWindows or allColumnKtuples
    columnTuples = kOrTuples.tuples;
    T = kOrTuples.T;
    k = kOrTuples.k;
  } else {
    throw new Error('kOrTuples must be a number, array of arrays, or {tuples, T, k} object');
  }

  const Ak = Math.pow(A, k);
  const gapTok = Ak + 1;
  const unobsTok = Ak;
  const illegalTok = Ak + 2;

  const index = new KmerIndex(columnTuples, k);

  if (T === 0) {
    const kmerAlphabet = _buildKmerAlphabet(alphabet, A, k);
    return {
      alignment: new Int32Array(0),
      A_kmer: Ak,
      N,
      C_k: 0,
      index,
      alphabet: kmerAlphabet,
    };
  }

  // Check for -1 sentinels in column tuples
  let hasSentinel = false;
  for (let i = 0; i < columnTuples.length; i++) {
    if (columnTuples[i] < 0) { hasSentinel = true; break; }
  }

  const result = new Int32Array(N * T);

  for (let n = 0; n < N; n++) {
    for (let t = 0; t < T; t++) {
      const tupleOffset = t * k;
      let allObs = true, hasGap = false, allGap = true;
      let kmerIdx = 0;

      for (let p = 0; p < k; p++) {
        const col = columnTuples[tupleOffset + p];
        let tok;
        if (col < 0) {
          // Sentinel: treat as unobserved
          tok = A;
        } else {
          tok = alignment[n * C + col];
        }

        const isObs = tok >= 0 && tok < A;
        const isGap = tok < 0 || tok > A;

        if (!isObs) allObs = false;
        if (isGap) hasGap = true;
        if (!isGap) allGap = false;

        if (isObs) kmerIdx = kmerIdx * A + tok;
        else kmerIdx = kmerIdx * A;
      }

      const outIdx = n * T + t;
      if (allObs) {
        result[outIdx] = kmerIdx;
      } else if (gapMode === 'any' && hasGap) {
        result[outIdx] = gapTok;
      } else if (gapMode === 'all' && allGap) {
        result[outIdx] = gapTok;
      } else if (gapMode === 'all' && hasGap) {
        result[outIdx] = illegalTok;
      } else {
        result[outIdx] = unobsTok;
      }
    }
  }

  const kmerAlphabet = _buildKmerAlphabet(alphabet, A, k);

  return { alignment: result, A_kmer: Ak, N, C_k: T, index, alphabet: kmerAlphabet };
}
