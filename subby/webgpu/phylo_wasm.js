/**
 * PhyloWASM — WASM fallback implementation of phylogenetic sufficient statistics.
 *
 * Mirrors the PhyloGPU API, delegating to Rust/WASM functions compiled via
 * wasm-bindgen. All computation in f64 precision.
 *
 * Supports both reversible and irreversible models. The 4th argument can be:
 * - Positional args: (eigenvalues, eigenvectors, pi) — reversible
 * - Model object with `pi` property — auto-detect:
 *   - { eigenvalues, eigenvectors, pi } → reversible
 *   - { eigenvalues_complex, eigenvectors_complex, eigenvectors_inv_complex, pi } → irreversible
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
   * Detect whether the 4th argument is a model object or positional eigenvalues.
   * @returns {{ irrev: boolean, model: Object }|null} null if positional args
   */
  static _parseModelArg(arg4, arg5) {
    if (arg4 && typeof arg4 === 'object' && arg4.pi !== undefined && arg5 === undefined) {
      const irrev = arg4.eigenvalues_complex !== undefined;
      return { irrev, model: arg4 };
    }
    return null;
  }

  /**
   * Compute per-column log-likelihoods.
   *
   * Accepts either:
   *   LogLike(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi)
   *   LogLike(alignment, parentIndex, distances, model)
   *
   * @returns {Float64Array} (C,) log-likelihoods
   */
  async LogLike(alignment, parentIndex, distances, arg4, arg5, arg6) {
    const parsed = PhyloWASM._parseModelArg(arg4, arg5);
    if (parsed && parsed.irrev) {
      const m = parsed.model;
      return this.wasm.wasm_log_like_irrev(
        new Int32Array(alignment),
        new Int32Array(parentIndex),
        new Float64Array(distances),
        new Float64Array(m.eigenvalues_complex),
        new Float64Array(m.eigenvectors_complex),
        new Float64Array(m.eigenvectors_inv_complex),
        new Float64Array(m.pi),
      );
    }
    // Reversible path (positional or model object)
    const eigenvalues = parsed ? parsed.model.eigenvalues : arg4;
    const eigenvectors = parsed ? parsed.model.eigenvectors : arg5;
    const pi = parsed ? parsed.model.pi : arg6;
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
   * Accepts either:
   *   Counts(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi, f81Fast?)
   *   Counts(alignment, parentIndex, distances, model)
   *
   * @returns {Float64Array} (A*A*C,) counts tensor
   */
  async Counts(alignment, parentIndex, distances, arg4, arg5, arg6, arg7) {
    const parsed = PhyloWASM._parseModelArg(arg4, arg5);
    if (parsed && parsed.irrev) {
      const m = parsed.model;
      return this.wasm.wasm_counts_irrev(
        new Int32Array(alignment),
        new Int32Array(parentIndex),
        new Float64Array(distances),
        new Float64Array(m.eigenvalues_complex),
        new Float64Array(m.eigenvectors_complex),
        new Float64Array(m.eigenvectors_inv_complex),
        new Float64Array(m.pi),
      );
    }
    const eigenvalues = parsed ? parsed.model.eigenvalues : arg4;
    const eigenvectors = parsed ? parsed.model.eigenvectors : arg5;
    const pi = parsed ? parsed.model.pi : arg6;
    const f81Fast = parsed ? false : (arg7 || false);
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
   * Compute per-branch expected substitution counts and dwell times.
   *
   * Accepts either:
   *   BranchCounts(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi, f81Fast?)
   *   BranchCounts(alignment, parentIndex, distances, model)
   *
   * @returns {Float64Array} (R*A*A*C,) per-branch counts
   */
  async BranchCounts(alignment, parentIndex, distances, arg4, arg5, arg6, arg7) {
    const parsed = PhyloWASM._parseModelArg(arg4, arg5);
    if (parsed && parsed.irrev) {
      const m = parsed.model;
      return this.wasm.wasm_branch_counts_irrev(
        new Int32Array(alignment),
        new Int32Array(parentIndex),
        new Float64Array(distances),
        new Float64Array(m.eigenvalues_complex),
        new Float64Array(m.eigenvectors_complex),
        new Float64Array(m.eigenvectors_inv_complex),
        new Float64Array(m.pi),
      );
    }
    const eigenvalues = parsed ? parsed.model.eigenvalues : arg4;
    const eigenvectors = parsed ? parsed.model.eigenvectors : arg5;
    const pi = parsed ? parsed.model.pi : arg6;
    const f81Fast = parsed ? false : (arg7 || false);
    return this.wasm.wasm_branch_counts(
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
   * Accepts either:
   *   RootProb(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi)
   *   RootProb(alignment, parentIndex, distances, model)
   *
   * @returns {Float64Array} (A*C,) root probabilities
   */
  async RootProb(alignment, parentIndex, distances, arg4, arg5, arg6) {
    const parsed = PhyloWASM._parseModelArg(arg4, arg5);
    if (parsed && parsed.irrev) {
      const m = parsed.model;
      return this.wasm.wasm_root_prob_irrev(
        new Int32Array(alignment),
        new Int32Array(parentIndex),
        new Float64Array(distances),
        new Float64Array(m.eigenvalues_complex),
        new Float64Array(m.eigenvectors_complex),
        new Float64Array(m.eigenvectors_inv_complex),
        new Float64Array(m.pi),
      );
    }
    const eigenvalues = parsed ? parsed.model.eigenvalues : arg4;
    const eigenvectors = parsed ? parsed.model.eigenvectors : arg5;
    const pi = parsed ? parsed.model.pi : arg6;
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
        alignment, parentIndex, distances, models[k],
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
   * Expected substitution counts and dwell times for a single CTMC branch.
   *
   * Accepts either:
   *   ExpectedCounts(eigenvalues, eigenvectors, pi, t)   — reversible
   *   ExpectedCounts(model, t)                           — auto-detect
   *
   * @returns {Float64Array} (A^4,) result[a*A^3 + b*A^2 + i*A + j]
   */
  async ExpectedCounts(arg1, arg2, arg3, arg4) {
    if (typeof arg1 === 'object' && arg1.pi !== undefined && typeof arg2 === 'number') {
      // Model object + t
      const model = arg1;
      const t = arg2;
      if (model.eigenvalues_complex !== undefined) {
        return this.wasm.wasm_expected_counts_irrev(
          new Float64Array(model.eigenvalues_complex),
          new Float64Array(model.eigenvectors_complex),
          new Float64Array(model.eigenvectors_inv_complex),
          new Float64Array(model.pi),
          t,
        );
      }
      return this.wasm.wasm_expected_counts(
        new Float64Array(model.eigenvalues),
        new Float64Array(model.eigenvectors),
        new Float64Array(model.pi),
        t,
      );
    }
    // Positional: eigenvalues, eigenvectors, pi, t
    return this.wasm.wasm_expected_counts(
      new Float64Array(arg1),
      new Float64Array(arg2),
      new Float64Array(arg3),
      arg4,
    );
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

  /**
   * Create an InsideOutside table for querying posteriors.
   *
   * Accepts either:
   *   InsideOutside(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi)
   *   InsideOutside(alignment, parentIndex, distances, model)
   *
   * @returns {PhyloWASMInsideOutside}
   */
  async InsideOutside(alignment, parentIndex, distances, arg4, arg5, arg6) {
    const parsed = PhyloWASM._parseModelArg(arg4, arg5);
    let wasmIO;
    if (parsed && parsed.irrev) {
      const m = parsed.model;
      wasmIO = this.wasm.WasmInsideOutside.create_irrev(
        new Int32Array(alignment),
        new Int32Array(parentIndex),
        new Float64Array(distances),
        new Float64Array(m.eigenvalues_complex),
        new Float64Array(m.eigenvectors_complex),
        new Float64Array(m.eigenvectors_inv_complex),
        new Float64Array(m.pi),
      );
    } else {
      const eigenvalues = parsed ? parsed.model.eigenvalues : arg4;
      const eigenvectors = parsed ? parsed.model.eigenvectors : arg5;
      const pi = parsed ? parsed.model.pi : arg6;
      wasmIO = this.wasm.WasmInsideOutside.create(
        new Int32Array(alignment),
        new Int32Array(parentIndex),
        new Float64Array(distances),
        new Float64Array(eigenvalues),
        new Float64Array(eigenvectors),
        new Float64Array(pi),
      );
    }
    const R = parentIndex.length;
    const A = (parsed ? parsed.model.pi : arg6).length;
    const C = alignment.length / R;
    return new PhyloWASMInsideOutside(wasmIO, R, C, A);
  }

  destroy() {
    // No-op for WASM
  }
}

/**
 * InsideOutside table wrapper for WASM backend.
 * Provides log_likelihood, counts, node_posterior, branch_posterior methods.
 */
class PhyloWASMInsideOutside {
  constructor(wasmIO, R, C, A) {
    this._io = wasmIO;
    this.R = R;
    this.C = C;
    this.A = A;
  }

  /** Per-column log-likelihoods. @returns {Float64Array} (C,) */
  get log_likelihood() {
    return this._io.log_likelihood();
  }

  /** Expected substitution counts. @returns {Float64Array} (A*A*C,) */
  counts(f81Fast = false) {
    return this._io.counts(f81Fast);
  }

  /** Per-branch expected counts. @returns {Float64Array} (R*A*A*C,) */
  branch_counts(f81Fast = false) {
    return this._io.branch_counts(f81Fast);
  }

  /**
   * Posterior state distribution at node(s).
   * @param {number|null} node - Node index, or null for all nodes.
   * @returns {Float64Array} (A*C,) for single node or (R*A*C,) for all.
   */
  node_posterior(node = null) {
    return this._io.node_posterior(node === null ? -1 : node);
  }

  /**
   * Joint branch posterior of parent-child states.
   * @param {number|null} node - Child node index (> 0), or null for all.
   * @returns {Float64Array} (A*A*C,) for single or (R*A*A*C,) for all.
   */
  branch_posterior(node = null) {
    return this._io.branch_posterior(node === null ? -1 : node);
  }

  destroy() {
    if (this._io && this._io.free) {
      this._io.free();
    }
    this._io = null;
  }
}
