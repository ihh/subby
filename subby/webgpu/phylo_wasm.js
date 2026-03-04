/**
 * PhyloWASM — WASM fallback implementation of phylogenetic sufficient statistics.
 *
 * Mirrors the PhyloGPU API, delegating to Rust/WASM functions compiled via
 * wasm-bindgen. All computation in f64 precision.
 */

export class PhyloWASM {
  constructor(wasmModule) {
    this.wasm = wasmModule;
  }

  /**
   * Initialize the PhyloWASM engine.
   * @param {string} wasmUrl - URL to the .wasm file
   * @returns {PhyloWASM}
   */
  static async create(wasmUrl) {
    // wasm-bindgen generates an init function
    const wasm = await import(wasmUrl);
    await wasm.default();
    return new PhyloWASM(wasm);
  }

  /**
   * Create from an already-initialized wasm module.
   */
  static fromModule(wasmModule) {
    return new PhyloWASM(wasmModule);
  }

  /**
   * Compute per-column log-likelihoods.
   *
   * @param {Int32Array} alignment - (R*C) flat
   * @param {Int32Array} parentIndex - (R,)
   * @param {Float64Array} distances - (R,)
   * @param {Float64Array} eigenvalues - (A,)
   * @param {Float64Array} eigenvectors - (A*A,)
   * @param {Float64Array} pi - (A,)
   * @returns {Float64Array} (C,) log-likelihoods
   */
  async LogLike(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi) {
    return this.wasm.wasm_log_like(
      new Int32Array(alignment),
      new Int32Array(parentIndex),
      new Float64Array(distances),
      new Float64Array(eigenvalues),
      new Float64Array(eigenvectors),
      new Float64Array(pi),
    );
  }

  /**
   * Compute expected substitution counts and dwell times.
   *
   * @returns {Float64Array} (A*A*C,) counts tensor
   */
  async Counts(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi, f81Fast = false) {
    return this.wasm.wasm_counts(
      new Int32Array(alignment),
      new Int32Array(parentIndex),
      new Float64Array(distances),
      new Float64Array(eigenvalues),
      new Float64Array(eigenvectors),
      new Float64Array(pi),
      f81Fast,
    );
  }

  /**
   * Compute posterior root state distribution.
   *
   * @returns {Float64Array} (A*C,) root probabilities
   */
  async RootProb(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi) {
    return this.wasm.wasm_root_prob(
      new Int32Array(alignment),
      new Int32Array(parentIndex),
      new Float64Array(distances),
      new Float64Array(eigenvalues),
      new Float64Array(eigenvectors),
      new Float64Array(pi),
    );
  }

  /**
   * Compute mixture posteriors.
   *
   * @param {Array<{eigenvalues, eigenvectors, pi}>} models
   * @param {Float64Array} logWeights - (K,)
   * @returns {Float64Array} (K*C,) posteriors
   */
  async MixturePosterior(alignment, parentIndex, distances, models, logWeights) {
    const K = models.length;
    const R = parentIndex.length;
    const C = alignment.length / R;

    const logLikes = new Float64Array(K * C);
    for (let k = 0; k < K; k++) {
      const ll = await this.LogLike(
        alignment, parentIndex, distances,
        models[k].eigenvalues, models[k].eigenvectors, models[k].pi,
      );
      logLikes.set(ll, k * C);
    }

    // Softmax on CPU (simple, small K)
    const posteriors = new Float64Array(K * C);
    for (let c = 0; c < C; c++) {
      let maxVal = -Infinity;
      for (let k = 0; k < K; k++) {
        const lj = logLikes[k * C + c] + logWeights[k];
        if (lj > maxVal) maxVal = lj;
      }
      let denom = 0;
      for (let k = 0; k < K; k++) {
        posteriors[k * C + c] = Math.exp(logLikes[k * C + c] + logWeights[k] - maxVal);
        denom += posteriors[k * C + c];
      }
      for (let k = 0; k < K; k++) {
        posteriors[k * C + c] /= denom;
      }
    }

    return posteriors;
  }

  /**
   * Compute branch mask.
   */
  computeBranchMask(alignment, parentIndex, A) {
    return this.wasm.wasm_branch_mask(
      new Int32Array(alignment),
      new Int32Array(parentIndex),
      A,
    );
  }

  destroy() {
    // No-op for WASM
  }
}
