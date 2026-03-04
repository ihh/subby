/**
 * Preset substitution models for the phylogenetic engine.
 *
 * Each model returns { eigenvalues, eigenvectors, pi } ready for
 * engine.LogLike / engine.Counts / engine.RootProb.
 */

/**
 * Build an orthonormal basis with v0 as the first column via modified Gram-Schmidt.
 * @param {number[]} v0 - first basis vector (length A)
 * @param {number} A - alphabet size
 * @returns {Float64Array} (A*A) row-major eigenvector matrix
 */
function qrOrthonormalBasis(v0, A) {
  // Candidate columns: [v0, e0, e1, ..., e_{A-1}]
  const cols = [v0.slice()];
  for (let i = 0; i < A; i++) {
    const e = new Array(A).fill(0);
    e[i] = 1;
    cols.push(e);
  }

  const basis = [];
  for (const col of cols) {
    if (basis.length >= A) break;
    const v = col.slice();
    for (const b of basis) {
      let dot = 0;
      for (let i = 0; i < A; i++) dot += v[i] * b[i];
      for (let i = 0; i < A; i++) v[i] -= dot * b[i];
    }
    let norm = 0;
    for (let i = 0; i < A; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (norm > 1e-12) {
      for (let i = 0; i < A; i++) v[i] /= norm;
      basis.push(v);
    }
  }

  const result = new Float64Array(A * A);
  for (let row = 0; row < A; row++) {
    for (let col = 0; col < A; col++) {
      result[row * A + col] = basis[col][row];
    }
  }
  return result;
}

/**
 * Jukes-Cantor model for an A-state alphabet.
 * @param {number} A - alphabet size (e.g. 4 for DNA)
 * @returns {{ eigenvalues: Float64Array, eigenvectors: Float64Array, pi: Float64Array }}
 */
export function jukesCantor(A) {
  const pi = new Float64Array(A).fill(1 / A);
  const mu = A / (A - 1);
  const eigenvalues = new Float64Array(A);
  for (let i = 1; i < A; i++) eigenvalues[i] = -mu;

  const v0 = new Array(A).fill(1 / Math.sqrt(A));
  const eigenvectors = qrOrthonormalBasis(v0, A);

  return { eigenvalues, eigenvectors, pi };
}

/**
 * F81 model (Felsenstein 1981) with non-uniform equilibrium frequencies.
 * @param {number[]|Float64Array} pi - equilibrium frequencies (length A)
 * @returns {{ eigenvalues: Float64Array, eigenvectors: Float64Array, pi: Float64Array }}
 */
export function f81(pi) {
  const A = pi.length;
  const piArr = new Float64Array(pi);
  let piSqSum = 0;
  for (let i = 0; i < A; i++) piSqSum += piArr[i] * piArr[i];
  const mu = 1 / (1 - piSqSum);

  const eigenvalues = new Float64Array(A);
  for (let i = 1; i < A; i++) eigenvalues[i] = -mu;

  const sqrtPi = new Array(A);
  for (let i = 0; i < A; i++) sqrtPi[i] = Math.sqrt(piArr[i]);
  const eigenvectors = qrOrthonormalBasis(sqrtPi, A);

  return { eigenvalues, eigenvectors, pi: piArr };
}

/**
 * HKY85 model (Hasegawa, Kishino & Yano 1985) for DNA.
 * @param {number} kappa - transition/transversion ratio
 * @param {number[]|Float64Array} pi - equilibrium frequencies [A, C, G, T]
 * @returns {{ eigenvalues: Float64Array, eigenvectors: Float64Array, pi: Float64Array }}
 */
export function hky85(kappa, pi) {
  const [piA, piC, piG, piT] = pi;
  const piR = piA + piG;
  const piY = piC + piT;
  const piArr = new Float64Array(pi);

  const beta = 1 / (2 * piR * piY + 2 * kappa * (piA * piG + piC * piT));

  const eigenvalues = new Float64Array([
    0,
    -beta,
    -beta * (piR + kappa * piY),
    -beta * (piY + kappa * piR),
  ]);

  const sqrtPi = [Math.sqrt(piA), Math.sqrt(piC), Math.sqrt(piG), Math.sqrt(piT)];

  // w0 = sqrt(pi)
  const w0 = sqrtPi.slice();

  // w1: purine-pyrimidine split
  const norm1 = Math.sqrt(piR * piY);
  const w1 = [
    sqrtPi[0] * piY / norm1,
    -sqrtPi[1] * piR / norm1,
    sqrtPi[2] * piY / norm1,
    -sqrtPi[3] * piR / norm1,
  ];

  // w2: within-pyrimidine
  const norm2 = Math.sqrt(piC * piT * piY);
  const w2 = [0, sqrtPi[1] * piT / norm2, 0, -sqrtPi[3] * piC / norm2];

  // w3: within-purine
  const norm3 = Math.sqrt(piA * piG * piR);
  const w3 = [sqrtPi[0] * piG / norm3, 0, -sqrtPi[2] * piA / norm3, 0];

  const eigenvectors = new Float64Array(16);
  for (let a = 0; a < 4; a++) {
    eigenvectors[a * 4 + 0] = w0[a];
    eigenvectors[a * 4 + 1] = w1[a];
    eigenvectors[a * 4 + 2] = w2[a];
    eigenvectors[a * 4 + 3] = w3[a];
  }

  return { eigenvalues, eigenvectors, pi: piArr };
}
