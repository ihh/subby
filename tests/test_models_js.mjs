/**
 * Tests for models.js diagonalization helpers + preset models.
 *
 * Run: node tests/test_models_js.mjs
 *
 * Uses deterministic seeded PRNG for reproducible "random" test data.
 */

import {
  jukesCantor, f81, hky85,
  diagonalize, diagonalizeIrreversible, diagonalizeAuto,
} from '../subby/webgpu/models.js';

// ---- Deterministic PRNG (LCG) for reproducible data ----
class Rng {
  constructor(seed) { this.state = BigInt(seed) & 0xFFFFFFFFFFFFFFFFn; }
  next() {
    this.state = (this.state * 6364136223846793005n + 1442695040888963407n) & 0xFFFFFFFFFFFFFFFFn;
    return Number((this.state >> 33n) & 0x7FFFFFFFn) / 0x80000000;
  }
  uniform(lo, hi) { return lo + (hi - lo) * this.next(); }
}

// ---- Helpers ----
let passed = 0, failed = 0, total = 0;

function assert(cond, msg) {
  if (!cond) throw new Error(`Assertion failed: ${msg}`);
}

function assertClose(a, b, tol, msg) {
  if (Math.abs(a - b) > tol) throw new Error(`${msg}: ${a} vs ${b} (tol=${tol})`);
}

function assertArrayClose(a, b, tol, msg) {
  assert(a.length === b.length, `${msg}: length mismatch ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tol) {
      throw new Error(`${msg}[${i}]: ${a[i]} vs ${b[i]} (tol=${tol})`);
    }
  }
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

/** Matrix multiply: C = A @ B (row-major, N×N) */
function matmul(A, B, N) {
  const C = new Float64Array(N * N);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
      C[i * N + j] = s;
    }
  return C;
}

/** Matrix exponential via Taylor series (real). */
function matExp(Q, t, A) {
  const M = new Float64Array(A * A);
  for (let i = 0; i < A * A; i++) M[i] = Q[i] * t;
  const result = new Float64Array(A * A);
  for (let i = 0; i < A; i++) result[i * A + i] = 1.0;
  let term = new Float64Array(result);
  for (let n = 1; n <= 60; n++) {
    const next = matmul(term, M, A);
    for (let i = 0; i < A * A; i++) next[i] /= n;
    term = next;
    for (let i = 0; i < A * A; i++) result[i] += term[i];
  }
  return result;
}

/**
 * Reconstruct P(t) = exp(Qt) from reversible eigendecomposition.
 * S = D^{1/2} Q D^{-1/2} = V diag(λ) V^T, so
 * P(t) = D^{-1/2} V diag(exp(λ*t)) V^T D^{1/2}  where D = diag(pi)
 */
function revExpQt(eigenvalues, eigenvectors, t, A, pi) {
  const expD = new Float64Array(A);
  for (let i = 0; i < A; i++) expD[i] = Math.exp(eigenvalues[i] * t);
  const sqrtPi = new Float64Array(A), invSqrtPi = new Float64Array(A);
  for (let i = 0; i < A; i++) {
    sqrtPi[i] = Math.sqrt(pi[i]);
    invSqrtPi[i] = 1.0 / sqrtPi[i];
  }
  const result = new Float64Array(A * A);
  for (let i = 0; i < A; i++) {
    for (let j = 0; j < A; j++) {
      let s = 0;
      for (let k = 0; k < A; k++) {
        s += invSqrtPi[i] * eigenvectors[i * A + k] * expD[k] * eigenvectors[j * A + k] * sqrtPi[j];
      }
      result[i * A + j] = s;
    }
  }
  return result;
}

/** Reconstruct exp(Qt) from irreversible complex eigendecomposition */
function irrevExpQt(eigenvalues_complex, eigenvectors_complex, eigenvectors_inv_complex, t, A) {
  const expDRe = new Float64Array(A), expDIm = new Float64Array(A);
  for (let k = 0; k < A; k++) {
    const lr = eigenvalues_complex[2 * k], li = eigenvalues_complex[2 * k + 1];
    const er = Math.exp(lr * t);
    expDRe[k] = er * Math.cos(li * t);
    expDIm[k] = er * Math.sin(li * t);
  }
  const resultRe = new Float64Array(A * A);
  for (let i = 0; i < A; i++) {
    for (let j = 0; j < A; j++) {
      let sr = 0;
      for (let k = 0; k < A; k++) {
        const vr = eigenvectors_complex[2 * (i * A + k)];
        const vi = eigenvectors_complex[2 * (i * A + k) + 1];
        const ter = vr * expDRe[k] - vi * expDIm[k];
        const tei = vr * expDIm[k] + vi * expDRe[k];
        const ir = eigenvectors_inv_complex[2 * (k * A + j)];
        const ii = eigenvectors_inv_complex[2 * (k * A + j) + 1];
        sr += ter * ir - tei * ii;
      }
      resultRe[i * A + j] = sr;
    }
  }
  return resultRe;
}

/** Build a random rate matrix Q with given pi as stationary distribution. */
function makeRateMatrix(A, pi, seed, reversible) {
  const rng = new Rng(seed);
  const Q = new Float64Array(A * A);
  if (reversible) {
    for (let i = 0; i < A; i++) {
      for (let j = i + 1; j < A; j++) {
        const s = rng.uniform(0.1, 2.0);
        Q[i * A + j] = s * pi[j];
        Q[j * A + i] = s * pi[i];
      }
    }
  } else {
    for (let i = 0; i < A; i++) {
      for (let j = 0; j < A; j++) {
        if (i !== j) Q[i * A + j] = rng.uniform(0.1, 2.0);
      }
    }
  }
  for (let i = 0; i < A; i++) {
    let rowSum = 0;
    for (let j = 0; j < A; j++) if (i !== j) rowSum += Q[i * A + j];
    Q[i * A + i] = -rowSum;
  }
  return Q;
}

/** Generate random stationary distribution */
function makePi(A, seed) {
  const rng = new Rng(seed);
  const raw = new Float64Array(A);
  let sum = 0;
  for (let i = 0; i < A; i++) { raw[i] = rng.uniform(0.1, 1.0); sum += raw[i]; }
  for (let i = 0; i < A; i++) raw[i] /= sum;
  return raw;
}

/** Build JC4 rate matrix (rows sum to 0). */
function jc4RateMatrix() {
  const Q = new Float64Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      Q[i * 4 + j] = i === j ? -1 : 1/3;
    }
  }
  return Q;
}

// ==================== Tests ====================

console.log('--- Preset models ---');

runTest('jukesCantor(4) eigenvalues', () => {
  const m = jukesCantor(4);
  assert(m.eigenvalues.length === 4, 'length');
  assertClose(m.eigenvalues[0], 0, 1e-12, 'lambda0');
  for (let i = 1; i < 4; i++) assertClose(m.eigenvalues[i], -4/3, 1e-12, `lambda${i}`);
});

runTest('jukesCantor(4) pi', () => {
  const m = jukesCantor(4);
  for (let i = 0; i < 4; i++) assertClose(m.pi[i], 0.25, 1e-12, `pi${i}`);
});

runTest('jukesCantor(4) eigenvectors orthogonal', () => {
  const { eigenvectors: V } = jukesCantor(4);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let dot = 0;
      for (let k = 0; k < 4; k++) dot += V[k * 4 + i] * V[k * 4 + j];
      assertClose(dot, i === j ? 1 : 0, 1e-12, `V^T V [${i},${j}]`);
    }
  }
});

runTest('jukesCantor(4) exp(Qt) rows sum to 1', () => {
  const m = jukesCantor(4);
  const P = revExpQt(m.eigenvalues, m.eigenvectors, 0.5, 4, m.pi);
  for (let i = 0; i < 4; i++) {
    let rowSum = 0;
    for (let j = 0; j < 4; j++) rowSum += P[i * 4 + j];
    assertClose(rowSum, 1.0, 1e-10, `row${i} sum`);
  }
});

runTest('jukesCantor(20) eigenvalues', () => {
  const m = jukesCantor(20);
  assertClose(m.eigenvalues[0], 0, 1e-12, 'lambda0');
  for (let i = 1; i < 20; i++) assertClose(m.eigenvalues[i], -20/19, 1e-12, `lambda${i}`);
});

runTest('f81 eigenvalues', () => {
  const pi = [0.3, 0.2, 0.15, 0.35];
  const m = f81(pi);
  assertClose(m.eigenvalues[0], 0, 1e-12, 'lambda0');
  let piSqSum = 0;
  for (let i = 0; i < 4; i++) piSqSum += pi[i] * pi[i];
  const mu = 1 / (1 - piSqSum);
  for (let i = 1; i < 4; i++) assertClose(m.eigenvalues[i], -mu, 1e-12, `lambda${i}`);
});

runTest('hky85 exp(Qt) rows sum to 1', () => {
  const pi = [0.3, 0.2, 0.2, 0.3];
  const m = hky85(2.0, pi);
  const P = revExpQt(m.eigenvalues, m.eigenvectors, 0.5, 4, m.pi);
  for (let i = 0; i < 4; i++) {
    let rowSum = 0;
    for (let j = 0; j < 4; j++) rowSum += P[i * 4 + j];
    assertClose(rowSum, 1.0, 1e-10, `row${i} sum`);
  }
});

runTest('hky85 exp(Qt) entries non-negative', () => {
  const pi = [0.3, 0.2, 0.2, 0.3];
  const m = hky85(5.0, pi);
  const P = revExpQt(m.eigenvalues, m.eigenvectors, 0.1, 4, m.pi);
  for (let i = 0; i < 16; i++) assert(P[i] >= -1e-12, `P[${i}]=${P[i]} negative`);
});

runTest('hky85 extreme kappa', () => {
  const pi = [0.25, 0.25, 0.25, 0.25];
  const m = hky85(100, pi);
  assert(m.eigenvalues.length === 4, 'length');
  assert(m.eigenvalues[0] === 0, 'lambda0');
  for (let i = 1; i < 4; i++) assert(m.eigenvalues[i] < 0, `lambda${i} negative`);
});

console.log('\n--- diagonalize (reversible) ---');

runTest('diagonalize JC4 matches preset', () => {
  const pi = new Float64Array([0.25, 0.25, 0.25, 0.25]);
  const Q = jc4RateMatrix();
  const m = diagonalize(Q, pi);
  const sorted = Array.from(m.eigenvalues).sort();
  const expected = [-4/3, -4/3, -4/3, 0];
  for (let i = 0; i < 4; i++) assertClose(sorted[i], expected[i], 1e-10, `lambda${i}`);
});

runTest('diagonalize JC4 exp(Qt) matches matExp', () => {
  const pi = new Float64Array([0.25, 0.25, 0.25, 0.25]);
  const Q = jc4RateMatrix();
  const m = diagonalize(Q, pi);
  const Peigen = revExpQt(m.eigenvalues, m.eigenvectors, 0.3, 4, pi);
  const Ptaylor = matExp(Q, 0.3, 4);
  assertArrayClose(Peigen, Ptaylor, 1e-10, 'P(t)');
});

runTest('diagonalize HKY85 exp(Qt) matches matExp', () => {
  const pi = new Float64Array([0.3, 0.2, 0.2, 0.3]);
  const kappa = 3.0;
  const Q = new Float64Array(16);
  const piR = pi[0] + pi[2], piY = pi[1] + pi[3];
  const beta = 1 / (2 * piR * piY + 2 * kappa * (pi[0] * pi[2] + pi[1] * pi[3]));
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      if (i === j) continue;
      const isTransition = (i + j) % 2 === 0;
      Q[i * 4 + j] = beta * pi[j] * (isTransition ? kappa : 1);
    }
    let rowSum = 0;
    for (let j = 0; j < 4; j++) if (j !== i) rowSum += Q[i * 4 + j];
    Q[i * 4 + i] = -rowSum;
  }
  const m = diagonalize(Q, pi);
  const t = 0.3;
  const Peigen = revExpQt(m.eigenvalues, m.eigenvectors, t, 4, pi);
  const Ptaylor = matExp(Q, t, 4);
  assertArrayClose(Peigen, Ptaylor, 1e-8, 'P(t)');
});

runTest('diagonalize random A=4 reversible', () => {
  const pi = makePi(4, 100);
  const Q = makeRateMatrix(4, pi, 200, true);
  const m = diagonalize(Q, pi);
  const t = 0.2;
  const Peigen = revExpQt(m.eigenvalues, m.eigenvectors, t, 4, pi);
  const Ptaylor = matExp(Q, t, 4);
  assertArrayClose(Peigen, Ptaylor, 1e-8, 'P(t)');
});

runTest('diagonalize random A=8 reversible', () => {
  const pi = makePi(8, 110);
  const Q = makeRateMatrix(8, pi, 210, true);
  const m = diagonalize(Q, pi);
  const t = 0.15;
  const Peigen = revExpQt(m.eigenvalues, m.eigenvectors, t, 8, pi);
  const Ptaylor = matExp(Q, t, 8);
  assertArrayClose(Peigen, Ptaylor, 1e-6, 'P(t)');
});

runTest('diagonalize random A=20 reversible', () => {
  const pi = makePi(20, 120);
  const Q = makeRateMatrix(20, pi, 220, true);
  const m = diagonalize(Q, pi);
  const t = 0.1;
  const Peigen = revExpQt(m.eigenvalues, m.eigenvectors, t, 20, pi);
  const Ptaylor = matExp(Q, t, 20);
  assertArrayClose(Peigen, Ptaylor, 1e-5, 'P(t)');
});

runTest('diagonalize eigenvalues <= 0 for reversible Q', () => {
  const pi = makePi(4, 130);
  const Q = makeRateMatrix(4, pi, 230, true);
  const m = diagonalize(Q, pi);
  for (let i = 0; i < 4; i++) assert(m.eigenvalues[i] <= 1e-12, `lambda${i}=${m.eigenvalues[i]}`);
});

runTest('diagonalize eigenvectors orthogonal', () => {
  const pi = makePi(4, 140);
  const Q = makeRateMatrix(4, pi, 240, true);
  const { eigenvectors: V } = diagonalize(Q, pi);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let dot = 0;
      for (let k = 0; k < 4; k++) dot += V[k * 4 + i] * V[k * 4 + j];
      assertClose(dot, i === j ? 1 : 0, 1e-10, `V^TV[${i},${j}]`);
    }
  }
});

runTest('diagonalize exp(Qt) rows sum to 1 for several t', () => {
  const pi = makePi(4, 150);
  const Q = makeRateMatrix(4, pi, 250, true);
  const m = diagonalize(Q, pi);
  for (const t of [0.01, 0.1, 0.5, 1.0, 5.0]) {
    const P = revExpQt(m.eigenvalues, m.eigenvectors, t, 4, pi);
    for (let i = 0; i < 4; i++) {
      let rowSum = 0;
      for (let j = 0; j < 4; j++) rowSum += P[i * 4 + j];
      assertClose(rowSum, 1.0, 1e-10, `t=${t} row${i}`);
    }
  }
});

runTest('diagonalize multiple seeds A=4', () => {
  for (let seed = 160; seed < 170; seed++) {
    const pi = makePi(4, seed);
    const Q = makeRateMatrix(4, pi, seed + 1000, true);
    const m = diagonalize(Q, pi);
    const Peigen = revExpQt(m.eigenvalues, m.eigenvectors, 0.2, 4, pi);
    const Ptaylor = matExp(Q, 0.2, 4);
    assertArrayClose(Peigen, Ptaylor, 1e-8, `seed=${seed}`);
  }
});

console.log('\n--- diagonalizeIrreversible ---');

runTest('diagonalizeIrreversible A=4 exp(Qt) matches matExp', () => {
  const pi = makePi(4, 300);
  const Q = makeRateMatrix(4, pi, 400, false);
  const m = diagonalizeIrreversible(Q, pi);
  const t = 0.2;
  const Peigen = irrevExpQt(m.eigenvalues_complex, m.eigenvectors_complex, m.eigenvectors_inv_complex, t, 4);
  const Ptaylor = matExp(Q, t, 4);
  assertArrayClose(Peigen, Ptaylor, 1e-6, 'P(t)');
});

runTest('diagonalizeIrreversible A=4 rows sum to 1', () => {
  const pi = makePi(4, 310);
  const Q = makeRateMatrix(4, pi, 410, false);
  const m = diagonalizeIrreversible(Q, pi);
  for (const t of [0.05, 0.2, 0.5, 1.0]) {
    const P = irrevExpQt(m.eigenvalues_complex, m.eigenvectors_complex, m.eigenvectors_inv_complex, t, 4);
    for (let i = 0; i < 4; i++) {
      let rowSum = 0;
      for (let j = 0; j < 4; j++) rowSum += P[i * 4 + j];
      assertClose(rowSum, 1.0, 1e-8, `t=${t} row${i}`);
    }
  }
});

runTest('diagonalizeIrreversible A=4 entries non-negative', () => {
  const pi = makePi(4, 320);
  const Q = makeRateMatrix(4, pi, 420, false);
  const m = diagonalizeIrreversible(Q, pi);
  const P = irrevExpQt(m.eigenvalues_complex, m.eigenvectors_complex, m.eigenvectors_inv_complex, 0.3, 4);
  for (let i = 0; i < 16; i++) assert(P[i] >= -1e-8, `P[${i}]=${P[i]} negative`);
});

runTest('diagonalizeIrreversible A=8 exp(Qt) matches matExp', () => {
  const pi = makePi(8, 330);
  const Q = makeRateMatrix(8, pi, 430, false);
  const m = diagonalizeIrreversible(Q, pi);
  const t = 0.15;
  const Peigen = irrevExpQt(m.eigenvalues_complex, m.eigenvectors_complex, m.eigenvectors_inv_complex, t, 8);
  const Ptaylor = matExp(Q, t, 8);
  assertArrayClose(Peigen, Ptaylor, 1e-4, 'P(t)');
});

runTest('diagonalizeIrreversible multiple seeds A=4', () => {
  for (let seed = 500; seed < 510; seed++) {
    const pi = makePi(4, seed);
    const Q = makeRateMatrix(4, pi, seed + 1000, false);
    const m = diagonalizeIrreversible(Q, pi);
    const t = 0.25;
    const Peigen = irrevExpQt(m.eigenvalues_complex, m.eigenvectors_complex, m.eigenvectors_inv_complex, t, 4);
    const Ptaylor = matExp(Q, t, 4);
    assertArrayClose(Peigen, Ptaylor, 1e-5, `seed=${seed} P(t)`);
  }
});

runTest('diagonalizeIrreversible V * V_inv = I', () => {
  const pi = makePi(4, 340);
  const Q = makeRateMatrix(4, pi, 440, false);
  const m = diagonalizeIrreversible(Q, pi);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let sr = 0, si = 0;
      for (let k = 0; k < 4; k++) {
        const vr = m.eigenvectors_complex[2 * (i * 4 + k)];
        const vi = m.eigenvectors_complex[2 * (i * 4 + k) + 1];
        const ir = m.eigenvectors_inv_complex[2 * (k * 4 + j)];
        const ii = m.eigenvectors_inv_complex[2 * (k * 4 + j) + 1];
        sr += vr * ir - vi * ii;
        si += vr * ii + vi * ir;
      }
      assertClose(sr, i === j ? 1 : 0, 1e-8, `VV_inv[${i},${j}] re`);
      assertClose(si, 0, 1e-8, `VV_inv[${i},${j}] im`);
    }
  }
});

runTest('diagonalizeIrreversible has zero eigenvalue', () => {
  const pi = makePi(4, 350);
  const Q = makeRateMatrix(4, pi, 450, false);
  const m = diagonalizeIrreversible(Q, pi);
  let hasZero = false;
  for (let i = 0; i < 4; i++) {
    const re = m.eigenvalues_complex[2 * i], im = m.eigenvalues_complex[2 * i + 1];
    if (Math.abs(re) < 1e-8 && Math.abs(im) < 1e-8) hasZero = true;
  }
  assert(hasZero, 'should have eigenvalue 0');
});

console.log('\n--- diagonalizeAuto ---');

runTest('diagonalizeAuto reversible Q returns eigenvalues (no _complex)', () => {
  const pi = makePi(4, 600);
  const Q = makeRateMatrix(4, pi, 700, true);
  const m = diagonalizeAuto(Q, pi);
  assert(m.eigenvalues !== undefined, 'has eigenvalues');
  assert(m.eigenvectors !== undefined, 'has eigenvectors');
  assert(m.eigenvalues_complex === undefined, 'no eigenvalues_complex');
});

runTest('diagonalizeAuto irreversible Q returns _complex fields', () => {
  const pi = makePi(4, 610);
  const Q = makeRateMatrix(4, pi, 710, false);
  const m = diagonalizeAuto(Q, pi);
  assert(m.eigenvalues_complex !== undefined, 'has eigenvalues_complex');
  assert(m.eigenvectors_complex !== undefined, 'has eigenvectors_complex');
  assert(m.eigenvectors_inv_complex !== undefined, 'has eigenvectors_inv_complex');
});

runTest('diagonalizeAuto reversible exp(Qt) correct', () => {
  const pi = makePi(4, 620);
  const Q = makeRateMatrix(4, pi, 720, true);
  const m = diagonalizeAuto(Q, pi);
  const Peigen = revExpQt(m.eigenvalues, m.eigenvectors, 0.3, 4, pi);
  const Ptaylor = matExp(Q, 0.3, 4);
  assertArrayClose(Peigen, Ptaylor, 1e-8, 'P(t)');
});

runTest('diagonalizeAuto irreversible exp(Qt) correct', () => {
  const pi = makePi(4, 630);
  const Q = makeRateMatrix(4, pi, 730, false);
  const m = diagonalizeAuto(Q, pi);
  const Peigen = irrevExpQt(m.eigenvalues_complex, m.eigenvectors_complex, m.eigenvectors_inv_complex, 0.3, 4);
  const Ptaylor = matExp(Q, 0.3, 4);
  assertArrayClose(Peigen, Ptaylor, 1e-6, 'P(t)');
});

console.log('\n--- Cross-validation: rev diag vs irrev diag on reversible Q ---');

runTest('rev and irrev diag give same P(t) for reversible A=4', () => {
  const pi = makePi(4, 800);
  const Q = makeRateMatrix(4, pi, 900, true);
  const t = 0.25;
  const mRev = diagonalize(Q, pi);
  const mIrrev = diagonalizeIrreversible(Q, pi);
  const Prev = revExpQt(mRev.eigenvalues, mRev.eigenvectors, t, 4, pi);
  const Pirrev = irrevExpQt(mIrrev.eigenvalues_complex, mIrrev.eigenvectors_complex, mIrrev.eigenvectors_inv_complex, t, 4);
  assertArrayClose(Prev, Pirrev, 1e-6, 'P(t) rev vs irrev');
});

runTest('rev and irrev diag give same P(t) for reversible A=8', () => {
  const pi = makePi(8, 810);
  const Q = makeRateMatrix(8, pi, 910, true);
  const t = 0.15;
  const mRev = diagonalize(Q, pi);
  const mIrrev = diagonalizeIrreversible(Q, pi);
  const Prev = revExpQt(mRev.eigenvalues, mRev.eigenvectors, t, 8, pi);
  const Pirrev = irrevExpQt(mIrrev.eigenvalues_complex, mIrrev.eigenvectors_complex, mIrrev.eigenvectors_inv_complex, t, 8);
  assertArrayClose(Prev, Pirrev, 1e-4, 'P(t) rev vs irrev A=8');
});

console.log('\n--- Edge cases ---');

runTest('diagonalize identity-like Q (all zeros)', () => {
  const Q = new Float64Array(16);
  const pi = new Float64Array([0.25, 0.25, 0.25, 0.25]);
  const m = diagonalize(Q, pi);
  for (let i = 0; i < 4; i++) assertClose(m.eigenvalues[i], 0, 1e-12, `lambda${i}`);
});

runTest('diagonalizeIrreversible near-reversible Q', () => {
  const pi = makePi(4, 1000);
  const Q = makeRateMatrix(4, pi, 1100, true);
  Q[0 * 4 + 1] += 0.001;
  Q[0 * 4 + 0] -= 0.001;
  const m = diagonalizeIrreversible(Q, pi);
  const Peigen = irrevExpQt(m.eigenvalues_complex, m.eigenvectors_complex, m.eigenvectors_inv_complex, 0.3, 4);
  const Ptaylor = matExp(Q, 0.3, 4);
  assertArrayClose(Peigen, Ptaylor, 1e-6, 'P(t) near-reversible');
});

runTest('diagonalize A=2 (binary alphabet)', () => {
  const pi = new Float64Array([0.6, 0.4]);
  const Q = new Float64Array([
    -0.4, 0.4,
    0.6, -0.6,
  ]);
  const m = diagonalize(Q, pi);
  const P = revExpQt(m.eigenvalues, m.eigenvectors, 0.5, 2, pi);
  const Pt = matExp(Q, 0.5, 2);
  assertArrayClose(P, Pt, 1e-10, 'P(t)');
});

runTest('diagonalizeAuto tolerance parameter works', () => {
  const pi = makePi(4, 1200);
  const Q = makeRateMatrix(4, pi, 1300, true);
  Q[0 * 4 + 1] += 1e-12;
  Q[0 * 4 + 0] -= 1e-12;
  const m = diagonalizeAuto(Q, pi);
  assert(m.eigenvalues !== undefined, 'should use reversible path');
  assert(m.eigenvalues_complex === undefined, 'should not have complex fields');
});

runTest('diagonalizeAuto with large asymmetry', () => {
  const pi = makePi(4, 1210);
  const Q = makeRateMatrix(4, pi, 1310, true);
  Q[0 * 4 + 1] += 1.0;
  Q[0 * 4 + 0] -= 1.0;
  const m = diagonalizeAuto(Q, pi);
  assert(m.eigenvalues_complex !== undefined, 'should use irreversible path');
});

console.log('\n--- Deterministic regression ---');

runTest('diagonalize deterministic (A=4, seed=42)', () => {
  const pi = makePi(4, 42);
  const Q = makeRateMatrix(4, pi, 84, true);
  const m = diagonalize(Q, pi);
  const sorted = Array.from(m.eigenvalues).sort();
  let sum = 0;
  for (let i = 0; i < 4; i++) {
    assert(sorted[i] <= 1e-12, `lambda${i} should be <= 0`);
    sum += sorted[i];
  }
  assert(Math.abs(sum) > 0.1, `sum=${sum} should be non-trivial`);
  const P = revExpQt(m.eigenvalues, m.eigenvectors, 0.1, 4, pi);
  const Pt = matExp(Q, 0.1, 4);
  assertArrayClose(P, Pt, 1e-8, 'regression P(t)');
});

runTest('diagonalizeIrreversible deterministic (A=4, seed=42)', () => {
  const pi = makePi(4, 42);
  const Q = makeRateMatrix(4, pi, 84, false);
  const m = diagonalizeIrreversible(Q, pi);
  const P = irrevExpQt(m.eigenvalues_complex, m.eigenvectors_complex, m.eigenvectors_inv_complex, 0.1, 4);
  const Pt = matExp(Q, 0.1, 4);
  assertArrayClose(P, Pt, 1e-6, 'regression P(t)');
});

// ---- Summary ----
console.log(`\n${passed}/${total} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
