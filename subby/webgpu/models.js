/**
 * Preset substitution models and diagonalization helpers for the phylogenetic engine.
 *
 * Preset models return { eigenvalues, eigenvectors, pi } ready for
 * engine.LogLike / engine.Counts / engine.RootProb.
 *
 * Diagonalization helpers convert raw rate matrices into model objects:
 * - diagonalize(Q, pi) — reversible (symmetric eigen)
 * - diagonalizeIrreversible(Q, pi) — irreversible (general eigen)
 * - diagonalizeAuto(Q, pi) — auto-detect
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

// ---- Internal linear algebra helpers ----

/**
 * Symmetric eigendecomposition via cyclic Jacobi iteration.
 * @param {Float64Array} S - (A*A) row-major symmetric matrix (mutated)
 * @param {number} A
 * @returns {{ eigenvalues: Float64Array, eigenvectors: Float64Array }}
 */
function eigSymmetric(S, A) {
  const V = new Float64Array(A * A);
  for (let i = 0; i < A; i++) V[i * A + i] = 1.0;

  for (let iter = 0; iter < 100; iter++) {
    let maxOff = 0;
    for (let i = 0; i < A; i++) {
      for (let j = i + 1; j < A; j++) {
        const v = Math.abs(S[i * A + j]);
        if (v > maxOff) maxOff = v;
      }
    }
    if (maxOff < 1e-14) break;

    for (let p = 0; p < A; p++) {
      for (let q = p + 1; q < A; q++) {
        const Spq = S[p * A + q];
        if (Math.abs(Spq) < 1e-15) continue;

        const tau = (S[q * A + q] - S[p * A + p]) / (2 * Spq);
        const t = (tau === 0) ? 1 : Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1.0 / Math.sqrt(1 + t * t);
        const s = t * c;

        const Spp = S[p * A + p];
        S[p * A + p] = Spp - t * Spq;
        S[q * A + q] = S[q * A + q] + t * Spq;
        S[p * A + q] = 0;
        S[q * A + p] = 0;

        for (let r = 0; r < A; r++) {
          if (r === p || r === q) continue;
          const Srp = S[r * A + p];
          const Srq = S[r * A + q];
          S[r * A + p] = c * Srp - s * Srq;
          S[p * A + r] = S[r * A + p];
          S[r * A + q] = s * Srp + c * Srq;
          S[q * A + r] = S[r * A + q];
        }

        for (let r = 0; r < A; r++) {
          const Vrp = V[r * A + p];
          const Vrq = V[r * A + q];
          V[r * A + p] = c * Vrp - s * Vrq;
          V[r * A + q] = s * Vrp + c * Vrq;
        }
      }
    }
  }

  const eigenvalues = new Float64Array(A);
  for (let i = 0; i < A; i++) eigenvalues[i] = S[i * A + i];
  return { eigenvalues, eigenvectors: V };
}

/**
 * General (non-symmetric) eigendecomposition via QR algorithm with shifts.
 * Returns complex eigenvalues and eigenvectors in interleaved (re,im) layout.
 * @param {Float64Array} M - (A*A) row-major matrix
 * @param {number} A
 * @returns {{ eigenvalues: Float64Array, eigenvectors: Float64Array, eigenvectorsInv: Float64Array }}
 */
function eigGeneral(M, A) {
  // Step 1: Reduce to upper Hessenberg via Householder
  const H = new Float64Array(M);
  for (let k = 0; k < A - 2; k++) {
    const x = new Float64Array(A - k - 1);
    for (let i = 0; i < x.length; i++) x[i] = H[(k + 1 + i) * A + k];
    let xNorm = 0;
    for (let i = 0; i < x.length; i++) xNorm += x[i] * x[i];
    xNorm = Math.sqrt(xNorm);
    if (xNorm < 1e-15) continue;
    x[0] += (x[0] >= 0 ? 1 : -1) * xNorm;
    let vNorm = 0;
    for (let i = 0; i < x.length; i++) vNorm += x[i] * x[i];
    vNorm = Math.sqrt(vNorm);
    for (let i = 0; i < x.length; i++) x[i] /= vNorm;

    for (let j = 0; j < A; j++) {
      let dot = 0;
      for (let i = 0; i < x.length; i++) dot += x[i] * H[(k + 1 + i) * A + j];
      for (let i = 0; i < x.length; i++) H[(k + 1 + i) * A + j] -= 2 * x[i] * dot;
    }
    for (let i = 0; i < A; i++) {
      let dot = 0;
      for (let j = 0; j < x.length; j++) dot += H[i * A + (k + 1 + j)] * x[j];
      for (let j = 0; j < x.length; j++) H[i * A + (k + 1 + j)] -= 2 * dot * x[j];
    }
  }

  // Step 2: QR iteration on Hessenberg H
  const eigenRe = new Float64Array(A);
  const eigenIm = new Float64Array(A);
  let n = A;

  for (let qrIter = 0; qrIter < 1000 && n > 0; qrIter++) {
    if (n === 1) {
      eigenRe[0] = H[0];
      eigenIm[0] = 0;
      n = 0;
      break;
    }

    const subDiag = Math.abs(H[(n - 1) * A + (n - 2)]);
    const diagSum = Math.abs(H[(n - 2) * A + (n - 2)]) + Math.abs(H[(n - 1) * A + (n - 1)]);
    if (subDiag < 1e-14 * diagSum + 1e-300) {
      eigenRe[n - 1] = H[(n - 1) * A + (n - 1)];
      eigenIm[n - 1] = 0;
      n--;
      continue;
    }

    const is2x2 = n === 2 ||
      Math.abs(H[(n - 2) * A + (n - 3)]) < 1e-14 * (Math.abs(H[(n - 3) * A + (n - 3)]) + Math.abs(H[(n - 2) * A + (n - 2)])) + 1e-300;
    if (is2x2) {
      const a11 = H[(n - 2) * A + (n - 2)], a12 = H[(n - 2) * A + (n - 1)];
      const a21 = H[(n - 1) * A + (n - 2)], a22 = H[(n - 1) * A + (n - 1)];
      const tr = a11 + a22, det = a11 * a22 - a12 * a21;
      const disc = tr * tr - 4 * det;
      if (disc >= 0) {
        const sq = Math.sqrt(disc);
        eigenRe[n - 2] = (tr + sq) / 2; eigenIm[n - 2] = 0;
        eigenRe[n - 1] = (tr - sq) / 2; eigenIm[n - 1] = 0;
      } else {
        const sq = Math.sqrt(-disc);
        eigenRe[n - 2] = tr / 2; eigenIm[n - 2] = sq / 2;
        eigenRe[n - 1] = tr / 2; eigenIm[n - 1] = -sq / 2;
      }
      n -= 2;
      continue;
    }

    // Wilkinson shift
    const wa = H[(n - 2) * A + (n - 2)], wb = H[(n - 2) * A + (n - 1)];
    const wc = H[(n - 1) * A + (n - 2)], wd = H[(n - 1) * A + (n - 1)];
    const wtr = wa + wd, wdet = wa * wd - wb * wc, wdisc = wtr * wtr - 4 * wdet;
    let shift;
    if (wdisc >= 0) {
      const sq = Math.sqrt(wdisc);
      const e1 = (wtr + sq) / 2, e2 = (wtr - sq) / 2;
      shift = Math.abs(e1 - wd) < Math.abs(e2 - wd) ? e1 : e2;
    } else {
      shift = wd;
    }

    for (let i = 0; i < n; i++) H[i * A + i] -= shift;
    const cs = new Float64Array(n - 1), sn = new Float64Array(n - 1);
    for (let i = 0; i < n - 1; i++) {
      const hi = H[i * A + i], hi1 = H[(i + 1) * A + i];
      const r = Math.sqrt(hi * hi + hi1 * hi1);
      if (r < 1e-300) { cs[i] = 1; sn[i] = 0; continue; }
      cs[i] = hi / r; sn[i] = hi1 / r;
      for (let j = 0; j < n; j++) {
        const t1 = H[i * A + j], t2 = H[(i + 1) * A + j];
        H[i * A + j] = cs[i] * t1 + sn[i] * t2;
        H[(i + 1) * A + j] = -sn[i] * t1 + cs[i] * t2;
      }
    }
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n; j++) {
        const t1 = H[j * A + i], t2 = H[j * A + (i + 1)];
        H[j * A + i] = cs[i] * t1 + sn[i] * t2;
        H[j * A + (i + 1)] = -sn[i] * t1 + cs[i] * t2;
      }
    }
    for (let i = 0; i < n; i++) H[i * A + i] += shift;
  }

  // Step 3: Eigenvectors via inverse iteration
  const eigenvalues = new Float64Array(2 * A);
  const eigenvectors = new Float64Array(2 * A * A);
  for (let idx = 0; idx < A; idx++) {
    eigenvalues[2 * idx] = eigenRe[idx];
    eigenvalues[2 * idx + 1] = eigenIm[idx];
  }

  for (let idx = 0; idx < A; idx++) {
    const lr = eigenRe[idx], li = eigenIm[idx];
    if (Math.abs(li) < 1e-14) {
      const shifted = new Float64Array(A * A);
      for (let i = 0; i < A * A; i++) shifted[i] = M[i];
      for (let i = 0; i < A; i++) shifted[i * A + i] -= lr + 1e-14;
      let v = new Float64Array(A);
      for (let i = 0; i < A; i++) v[i] = 1.0;
      for (let it = 0; it < 30; it++) {
        const w = _solveLU(shifted, v, A);
        let nm = 0;
        for (let i = 0; i < A; i++) nm += w[i] * w[i];
        nm = Math.sqrt(nm);
        if (nm < 1e-300) break;
        for (let i = 0; i < A; i++) v[i] = w[i] / nm;
      }
      for (let i = 0; i < A; i++) {
        eigenvectors[2 * (i * A + idx)] = v[i];
        eigenvectors[2 * (i * A + idx) + 1] = 0;
      }
    } else if (li > 0) {
      const vr = new Float64Array(A), vi = new Float64Array(A);
      for (let i = 0; i < A; i++) { vr[i] = 1.0; vi[i] = 0.5; }
      for (let it = 0; it < 30; it++) {
        const [wr, wi] = _solveComplexLU(M, lr + 1e-14, li, vr, vi, A);
        let nm = 0;
        for (let i = 0; i < A; i++) nm += wr[i] * wr[i] + wi[i] * wi[i];
        nm = Math.sqrt(nm);
        if (nm < 1e-300) break;
        for (let i = 0; i < A; i++) { vr[i] = wr[i] / nm; vi[i] = wi[i] / nm; }
      }
      for (let i = 0; i < A; i++) {
        eigenvectors[2 * (i * A + idx)] = vr[i];
        eigenvectors[2 * (i * A + idx) + 1] = vi[i];
      }
      for (let j = idx + 1; j < A; j++) {
        if (Math.abs(eigenRe[j] - lr) < 1e-12 && Math.abs(eigenIm[j] + li) < 1e-12) {
          for (let i = 0; i < A; i++) {
            eigenvectors[2 * (i * A + j)] = vr[i];
            eigenvectors[2 * (i * A + j) + 1] = -vi[i];
          }
          break;
        }
      }
    }
  }

  const eigenvectorsInv = _matInvComplex(eigenvectors, A);
  return { eigenvalues, eigenvectors, eigenvectorsInv };
}

/** Solve Ax=b via LU with partial pivoting. */
function _solveLU(A_mat, b, N) {
  const LU = new Float64Array(A_mat);
  const piv = new Int32Array(N);
  for (let i = 0; i < N; i++) piv[i] = i;
  for (let k = 0; k < N; k++) {
    let mx = 0, mr = k;
    for (let i = k; i < N; i++) { const v = Math.abs(LU[i * N + k]); if (v > mx) { mx = v; mr = i; } }
    if (mr !== k) {
      for (let j = 0; j < N; j++) { const t = LU[k * N + j]; LU[k * N + j] = LU[mr * N + j]; LU[mr * N + j] = t; }
      const t = piv[k]; piv[k] = piv[mr]; piv[mr] = t;
    }
    if (Math.abs(LU[k * N + k]) < 1e-300) continue;
    for (let i = k + 1; i < N; i++) {
      LU[i * N + k] /= LU[k * N + k];
      for (let j = k + 1; j < N; j++) LU[i * N + j] -= LU[i * N + k] * LU[k * N + j];
    }
  }
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) x[i] = b[piv[i]];
  for (let i = 0; i < N; i++) for (let j = 0; j < i; j++) x[i] -= LU[i * N + j] * x[j];
  for (let i = N - 1; i >= 0; i--) {
    for (let j = i + 1; j < N; j++) x[i] -= LU[i * N + j] * x[j];
    x[i] /= LU[i * N + i];
  }
  return x;
}

/** Solve (A - (lr+i*li)*I)(xr+i*xi) = (br+i*bi) via real 2N×2N system. */
function _solveComplexLU(A_mat, lr, li, br, bi, N) {
  const M2 = new Float64Array(4 * N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const v = A_mat[i * N + j] - (i === j ? lr : 0);
      M2[i * 2 * N + j] = v;
      M2[(N + i) * 2 * N + (N + j)] = v;
    }
    M2[i * 2 * N + (N + i)] = li;
    M2[(N + i) * 2 * N + i] = -li;
  }
  const rhs = new Float64Array(2 * N);
  for (let i = 0; i < N; i++) { rhs[i] = br[i]; rhs[N + i] = bi[i]; }
  const sol = _solveLU(M2, rhs, 2 * N);
  return [sol.slice(0, N), sol.slice(N)];
}

/** Invert a complex matrix in interleaved (re,im) layout via Gauss-Jordan. */
function _matInvComplex(V, A) {
  const re = new Float64Array(A * A), im = new Float64Array(A * A);
  const ire = new Float64Array(A * A), iim = new Float64Array(A * A);
  for (let i = 0; i < A * A; i++) { re[i] = V[2 * i]; im[i] = V[2 * i + 1]; }
  for (let i = 0; i < A; i++) ire[i * A + i] = 1.0;

  for (let k = 0; k < A; k++) {
    let maxN = 0, mr = k;
    for (let i = k; i < A; i++) {
      const nr = re[i * A + k] * re[i * A + k] + im[i * A + k] * im[i * A + k];
      if (nr > maxN) { maxN = nr; mr = i; }
    }
    if (mr !== k) {
      for (let j = 0; j < A; j++) {
        let t;
        t = re[k * A + j]; re[k * A + j] = re[mr * A + j]; re[mr * A + j] = t;
        t = im[k * A + j]; im[k * A + j] = im[mr * A + j]; im[mr * A + j] = t;
        t = ire[k * A + j]; ire[k * A + j] = ire[mr * A + j]; ire[mr * A + j] = t;
        t = iim[k * A + j]; iim[k * A + j] = iim[mr * A + j]; iim[mr * A + j] = t;
      }
    }
    const pr = re[k * A + k], pim = im[k * A + k];
    const den = pr * pr + pim * pim;
    if (den < 1e-300) continue;
    const ivR = pr / den, ivI = -pim / den;
    for (let j = 0; j < A; j++) {
      const ar = re[k * A + j], ai = im[k * A + j];
      re[k * A + j] = ar * ivR - ai * ivI;
      im[k * A + j] = ar * ivI + ai * ivR;
      const cr = ire[k * A + j], ci = iim[k * A + j];
      ire[k * A + j] = cr * ivR - ci * ivI;
      iim[k * A + j] = cr * ivI + ci * ivR;
    }
    for (let i = 0; i < A; i++) {
      if (i === k) continue;
      const fr = re[i * A + k], fi = im[i * A + k];
      for (let j = 0; j < A; j++) {
        const rKj = re[k * A + j], iKj = im[k * A + j];
        re[i * A + j] -= fr * rKj - fi * iKj;
        im[i * A + j] -= fr * iKj + fi * rKj;
        const irKj = ire[k * A + j], iiKj = iim[k * A + j];
        ire[i * A + j] -= fr * irKj - fi * iiKj;
        iim[i * A + j] -= fr * iiKj + fi * irKj;
      }
    }
  }
  const result = new Float64Array(2 * A * A);
  for (let i = 0; i < A * A; i++) { result[2 * i] = ire[i]; result[2 * i + 1] = iim[i]; }
  return result;
}

// ---- Exported diagonalization functions ----

/**
 * Diagonalize a reversible rate matrix.
 * Symmetrizes S_ij = Q_ij * sqrt(pi_i / pi_j), eigendecomposes S.
 * @param {Float64Array|number[]} Q - (A*A) row-major rate matrix
 * @param {Float64Array|number[]} pi - (A,) equilibrium frequencies
 * @returns {{ eigenvalues: Float64Array, eigenvectors: Float64Array, pi: Float64Array }}
 */
export function diagonalize(Q, pi) {
  const A = pi.length;
  const piArr = new Float64Array(pi);
  const sqrtPi = new Float64Array(A);
  const invSqrtPi = new Float64Array(A);
  for (let i = 0; i < A; i++) {
    sqrtPi[i] = Math.sqrt(piArr[i]);
    invSqrtPi[i] = 1.0 / sqrtPi[i];
  }
  const S = new Float64Array(A * A);
  for (let i = 0; i < A; i++) {
    for (let j = 0; j < A; j++) {
      S[i * A + j] = Q[i * A + j] * sqrtPi[i] * invSqrtPi[j];
    }
  }
  for (let i = 0; i < A; i++) {
    for (let j = i + 1; j < A; j++) {
      const avg = 0.5 * (S[i * A + j] + S[j * A + i]);
      S[i * A + j] = avg;
      S[j * A + i] = avg;
    }
  }
  const { eigenvalues, eigenvectors } = eigSymmetric(S, A);
  return { eigenvalues, eigenvectors, pi: piArr };
}

/**
 * Diagonalize an irreversible rate matrix.
 * Eigendecomposes Q directly. Returns complex interleaved (re,im) layout.
 * @param {Float64Array|number[]} Q - (A*A) row-major rate matrix
 * @param {Float64Array|number[]} pi - (A,) stationary distribution
 * @returns {{ eigenvalues_complex: Float64Array, eigenvectors_complex: Float64Array, eigenvectors_inv_complex: Float64Array, pi: Float64Array }}
 */
export function diagonalizeIrreversible(Q, pi) {
  const A = pi.length;
  const piArr = new Float64Array(pi);
  const { eigenvalues, eigenvectors, eigenvectorsInv } = eigGeneral(new Float64Array(Q), A);
  return {
    eigenvalues_complex: eigenvalues,
    eigenvectors_complex: eigenvectors,
    eigenvectors_inv_complex: eigenvectorsInv,
    pi: piArr,
  };
}

/**
 * Auto-detect reversibility and diagonalize accordingly.
 * Checks detailed balance (pi_i Q_ij ≈ pi_j Q_ji).
 * @param {Float64Array|number[]} Q - (A*A) row-major rate matrix
 * @param {Float64Array|number[]} pi - (A,) equilibrium frequencies
 * @param {number} [tol=1e-10] - tolerance for detailed balance check
 * @returns {Object} DiagModel or IrrevDiagModel
 */
export function diagonalizeAuto(Q, pi, tol = 1e-10) {
  const A = pi.length;
  let reversible = true;
  for (let i = 0; i < A && reversible; i++) {
    for (let j = i + 1; j < A && reversible; j++) {
      if (Math.abs(pi[i] * Q[i * A + j] - pi[j] * Q[j * A + i]) > tol) {
        reversible = false;
      }
    }
  }
  return reversible ? diagonalize(Q, pi) : diagonalizeIrreversible(Q, pi);
}

/**
 * Goldman-Yang 1994 (GY94) codon model.
 * Builds a 61x61 rate matrix over sense codons with transition/transversion
 * ratio kappa and nonsynonymous/synonymous ratio omega.
 * @param {number} omega - dN/dS ratio
 * @param {number} kappa - transition/transversion ratio
 * @param {number[]|Float64Array|null} [pi=null] - equilibrium frequencies (length 61); uniform if null
 * @returns {{ eigenvalues: Float64Array, eigenvectors: Float64Array, pi: Float64Array }}
 */
export function gy94(omega, kappa, pi = null) {
  const A = 61;
  if (pi === null) {
    pi = new Float64Array(A).fill(1 / A);
  }
  const piArr = new Float64Array(pi);

  // Build codon table
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
  const senseIndices = [];
  for (let i = 0; i < 64; i++) if (aminoAcids[i] !== '*') senseIndices.push(i);

  const transitions = new Set(['AG', 'GA', 'CT', 'TC']);

  // Build Q matrix (61x61)
  const Q = new Float64Array(A * A);
  for (let si = 0; si < A; si++) {
    const idxI = senseIndices[si];
    const codonI = codons[idxI];
    const aaI = aminoAcids[idxI];
    for (let sj = 0; sj < A; sj++) {
      if (si === sj) continue;
      const idxJ = senseIndices[sj];
      const codonJ = codons[idxJ];
      // Count nucleotide differences
      let ndiff = 0, diffNucI = '', diffNucJ = '';
      for (let p = 0; p < 3; p++) {
        if (codonI[p] !== codonJ[p]) { ndiff++; diffNucI = codonI[p]; diffNucJ = codonJ[p]; }
      }
      if (ndiff !== 1) continue;
      const isTs = transitions.has(diffNucI + diffNucJ);
      const aaJ = aminoAcids[idxJ];
      const isNonsyn = aaI !== aaJ;
      let rate = piArr[sj];
      if (isTs) rate *= kappa;
      if (isNonsyn) rate *= omega;
      Q[si * A + sj] = rate;
    }
  }

  // Set diagonal so rows sum to zero
  for (let i = 0; i < A; i++) {
    let rowSum = 0;
    for (let j = 0; j < A; j++) rowSum += Q[i * A + j];
    Q[i * A + i] = -rowSum;
  }

  // Normalize so expected rate = 1
  let expectedRate = 0;
  for (let i = 0; i < A; i++) expectedRate -= piArr[i] * Q[i * A + i];
  for (let i = 0; i < A * A; i++) Q[i] /= expectedRate;

  return diagonalize(Q, piArr);
}
