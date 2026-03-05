/**
 * PhyloGPU — WebGPU implementation of phylogenetic sufficient statistics.
 *
 * Provides LogLike, Counts, RootProb, and MixturePosterior using WGSL
 * compute shaders. All arrays are flattened to 1D storage buffers in
 * row-major (C-order) layout.
 *
 * f32 precision with rescaling to prevent underflow.
 */

// Shader source loading (inline for portability)
async function loadShader(url) {
  const resp = await fetch(url);
  return resp.text();
}

export class PhyloGPU {
  constructor(device, shaders) {
    this.device = device;
    this.shaders = shaders;
    this._pipelines = {};
  }

  /**
   * Initialize the PhyloGPU engine.
   * @param {string} shaderBasePath - URL prefix for .wgsl files
   * @returns {PhyloGPU}
   */
  static async create(shaderBasePath = './shaders/') {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice();

    const shaderNames = [
      'token_to_likelihood',
      'compute_sub_matrices',
      'upward_step',
      'downward_step',
      'compute_J',
      'eigenbasis_project',
      'accumulate_C',
      'back_transform',
      'f81_fast',
      'mixture_posterior',
      'compute_sub_matrices_complex',
      'compute_J_complex',
      'eigenbasis_project_complex',
      'accumulate_C_complex',
      'back_transform_complex',
    ];

    const shaders = {};
    await Promise.all(shaderNames.map(async (name) => {
      shaders[name] = await loadShader(`${shaderBasePath}${name}.wgsl`);
    }));

    return new PhyloGPU(device, shaders);
  }

  /**
   * Create from pre-loaded shader source strings.
   */
  static async createFromSources(shaderSources) {
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice();
    return new PhyloGPU(device, shaderSources);
  }

  _getOrCreatePipeline(name) {
    if (!this._pipelines[name]) {
      const module = this.device.createShaderModule({ code: this.shaders[name] });
      this._pipelines[name] = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    }
    return this._pipelines[name];
  }

  _createBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) {
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new data.constructor(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }

  _createUniformBuffer(data) {
    return this._createBuffer(data, GPUBufferUsage.UNIFORM);
  }

  _createStorageBuffer(size) {
    return this.device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  async _readBuffer(buf, size) {
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buf, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return result;
  }

  /**
   * Build children_of and sibling arrays (CPU-side, small).
   */
  _childrenOf(parentIndex) {
    const R = parentIndex.length;
    const leftChild = new Int32Array(R).fill(-1);
    const rightChild = new Int32Array(R).fill(-1);
    const sibling = new Int32Array(R).fill(-1);

    for (let n = 1; n < R; n++) {
      const p = parentIndex[n];
      if (leftChild[p] === -1) leftChild[p] = n;
      else rightChild[p] = n;
    }

    for (let n = 1; n < R; n++) {
      const p = parentIndex[n];
      sibling[n] = (leftChild[p] === n) ? rightChild[p] : leftChild[p];
    }

    return { leftChild, rightChild, sibling };
  }

  /**
   * Compute branch mask on CPU (simple boolean logic).
   */
  computeBranchMask(alignment, parentIndex, A) {
    const R = alignment.length / (alignment.length / parentIndex.length);
    // alignment is (R, C) stored flat
    const C = alignment.length / parentIndex.length;
    const R2 = parentIndex.length;

    const childCount = new Int32Array(R2);
    for (let n = 1; n < R2; n++) childCount[parentIndex[n]]++;
    const isLeaf = childCount.map(c => c === 0);

    const isUngappedLeaf = new Uint8Array(R2 * C);
    for (let r = 0; r < R2; r++) {
      for (let c = 0; c < C; c++) {
        const tok = alignment[r * C + c];
        isUngappedLeaf[r * C + c] = (isLeaf[r] && tok >= 0 && tok <= A) ? 1 : 0;
      }
    }

    const hasUngapped = new Uint8Array(isUngappedLeaf);
    for (let n = R2 - 1; n >= 1; n--) {
      const p = parentIndex[n];
      for (let c = 0; c < C; c++) {
        if (hasUngapped[n * C + c]) hasUngapped[p * C + c] = 1;
      }
    }

    const ungappedChildCount = new Int32Array(R2 * C);
    for (let n = 1; n < R2; n++) {
      const p = parentIndex[n];
      for (let c = 0; c < C; c++) {
        if (hasUngapped[n * C + c]) ungappedChildCount[p * C + c]++;
      }
    }

    const isSteiner = new Uint8Array(R2 * C);
    for (let r = 0; r < R2; r++) {
      for (let c = 0; c < C; c++) {
        if (isUngappedLeaf[r * C + c] || ungappedChildCount[r * C + c] >= 2) {
          isSteiner[r * C + c] = 1;
        }
      }
    }

    for (let n = 1; n < R2; n++) {
      const p = parentIndex[n];
      for (let c = 0; c < C; c++) {
        if (isSteiner[p * C + c] && hasUngapped[n * C + c]) {
          isSteiner[n * C + c] = 1;
        }
      }
    }

    const branchMask = new Uint8Array(R2 * C);
    for (let n = 1; n < R2; n++) {
      const p = parentIndex[n];
      for (let c = 0; c < C; c++) {
        branchMask[n * C + c] = (isSteiner[n * C + c] && isSteiner[p * C + c]) ? 1 : 0;
      }
    }

    return branchMask;
  }

  /**
   * Run token_to_likelihood shader.
   * @returns {GPUBuffer} (R*C*A) f32
   */
  _tokenToLikelihood(encoder, alignmentBuf, R, C, A) {
    const pipeline = this._getOrCreatePipeline('token_to_likelihood');
    const params = this._createUniformBuffer(new Uint32Array([R, C, A]));
    const outBuf = this._createStorageBuffer(R * C * A * 4);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: alignmentBuf } },
        { binding: 2, resource: { buffer: outBuf } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((R * C) / 64));
    pass.end();

    return outBuf;
  }

  /**
   * Run compute_sub_matrices shader.
   * @returns {GPUBuffer} (R*A*A) f32
   */
  _computeSubMatrices(encoder, eigenvalsBuf, eigenvecsBuf, piBuf, distancesBuf, R, A) {
    const pipeline = this._getOrCreatePipeline('compute_sub_matrices');
    const params = this._createUniformBuffer(new Uint32Array([R, A, 0, 0]));

    const outBuf = this._createStorageBuffer(R * A * A * 4);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: eigenvalsBuf } },
        { binding: 2, resource: { buffer: eigenvecsBuf } },
        { binding: 3, resource: { buffer: piBuf } },
        { binding: 4, resource: { buffer: distancesBuf } },
        { binding: 5, resource: { buffer: outBuf } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(R / 64));
    pass.end();

    return outBuf;
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
   * Run compute_sub_matrices_complex shader (irreversible model).
   * @returns {GPUBuffer} (R*A*A) f32 real output
   */
  _computeSubMatricesComplex(encoder, eigenvalsBuf, eigenvecsBuf, eigenvecsInvBuf, distancesBuf, R, A) {
    const pipeline = this._getOrCreatePipeline('compute_sub_matrices_complex');
    const params = this._createUniformBuffer(new Uint32Array([R, A, 0, 0]));
    const outBuf = this._createStorageBuffer(R * A * A * 4);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: eigenvalsBuf } },
        { binding: 2, resource: { buffer: eigenvecsBuf } },
        { binding: 3, resource: { buffer: eigenvecsInvBuf } },
        { binding: 4, resource: { buffer: distancesBuf } },
        { binding: 5, resource: { buffer: outBuf } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(R / 64));
    pass.end();

    return outBuf;
  }

  /**
   * Eigensub counts pipeline for irreversible (complex) model.
   */
  async _eigensubCountsComplex(UData, DData, logNormUData, logNormDData,
                                logLike, model, distances, R, C, A) {
    // compute_J_complex
    const eigenvalsBuf = this._createBuffer(new Float32Array(model.eigenvalues_complex));
    const distBuf = this._createBuffer(new Float32Array(distances));
    const JBuf = this._createStorageBuffer(2 * R * A * A * 4);

    const encoder = this.device.createCommandEncoder();
    {
      const pipeline = this._getOrCreatePipeline('compute_J_complex');
      const params = this._createUniformBuffer(new Uint32Array([R, A, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: eigenvalsBuf } },
          { binding: 2, resource: { buffer: distBuf } },
          { binding: 3, resource: { buffer: JBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(R / 64));
      pass.end();
    }

    // eigenbasis_project_complex
    const UBuf = this._createBuffer(new Float32Array(UData));
    const DBuf = this._createBuffer(new Float32Array(DData));
    const eigenvecsBuf = this._createBuffer(new Float32Array(model.eigenvectors_complex));
    const eigenvecsInvBuf = this._createBuffer(new Float32Array(model.eigenvectors_inv_complex));
    const U_tildeBuf = this._createStorageBuffer(2 * R * C * A * 4);
    const D_tildeBuf = this._createStorageBuffer(2 * R * C * A * 4);

    {
      const pipeline = this._getOrCreatePipeline('eigenbasis_project_complex');
      const params = this._createUniformBuffer(new Uint32Array([R, C, A, 0]));
      const bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: eigenvecsBuf } },
          { binding: 2, resource: { buffer: eigenvecsInvBuf } },
          { binding: 3, resource: { buffer: UBuf } },
          { binding: 4, resource: { buffer: DBuf } },
          { binding: 5, resource: { buffer: U_tildeBuf } },
          { binding: 6, resource: { buffer: D_tildeBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((R * C) / 64));
      pass.end();
    }

    // accumulate_C_complex
    const logNormUBuf = this._createBuffer(new Float32Array(logNormUData));
    const logNormDBuf = this._createBuffer(new Float32Array(logNormDData));
    const logLikeBuf = this._createBuffer(new Float32Array(logLike));
    const C_eigenBuf = this._createStorageBuffer(2 * A * A * C * 4);

    {
      const pipeline = this._getOrCreatePipeline('accumulate_C_complex');
      const params = this._createUniformBuffer(new Uint32Array([R, C, A, 0]));
      const bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: D_tildeBuf } },
          { binding: 2, resource: { buffer: U_tildeBuf } },
          { binding: 3, resource: { buffer: JBuf } },
          { binding: 4, resource: { buffer: logNormUBuf } },
          { binding: 5, resource: { buffer: logNormDBuf } },
          { binding: 6, resource: { buffer: logLikeBuf } },
          { binding: 7, resource: { buffer: C_eigenBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((A * A * C) / 64));
      pass.end();
    }

    // back_transform_complex
    const countsBuf = this._createStorageBuffer(A * A * C * 4);
    {
      const pipeline = this._getOrCreatePipeline('back_transform_complex');
      const params = this._createUniformBuffer(new Uint32Array([C, A, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: eigenvecsBuf } },
          { binding: 2, resource: { buffer: eigenvecsInvBuf } },
          { binding: 3, resource: { buffer: eigenvalsBuf } },
          { binding: 4, resource: { buffer: C_eigenBuf } },
          { binding: 5, resource: { buffer: countsBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((A * A * C) / 64));
      pass.end();
    }

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const result = await this._readBuffer(countsBuf, A * A * C * 4);

    for (const b of [eigenvalsBuf, distBuf, JBuf, UBuf, DBuf, eigenvecsBuf, eigenvecsInvBuf,
                      U_tildeBuf, D_tildeBuf, logNormUBuf, logNormDBuf, logLikeBuf,
                      C_eigenBuf, countsBuf]) {
      b.destroy();
    }

    return result;
  }

  /**
   * Run the full upward pass (R-1 sequential dispatches).
   */
  _upwardPass(encoder, likelihoodBuf, logNormUBuf, subMatBuf, parentIndex, R, C, A) {
    const pipeline = this._getOrCreatePipeline('upward_step');

    for (let n = R - 1; n >= 1; n--) {
      const child = n;
      const parent = parentIndex[n];

      const params = this._createUniformBuffer(new Uint32Array([C, A, child, parent]));

      // Extract sub-matrix for this child's branch: offset child*A*A, size A*A
      // We pass the full buffer and compute offset in the shader... but our shader
      // takes a separate A*A buffer. So we create a view buffer.
      // Actually, for simplicity, pass full subMat and child index via params.
      // But our shader expects a separate subMatrix binding of size A*A.
      // We'll create a temp buffer with just this branch's sub-matrix.
      const smOffset = child * A * A * 4;
      const smSize = A * A * 4;

      const smBuf = this._createStorageBuffer(smSize);
      encoder.copyBufferToBuffer(subMatBuf, smOffset, smBuf, 0, smSize);

      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: smBuf } },
          { binding: 2, resource: { buffer: likelihoodBuf } },
          { binding: 3, resource: { buffer: logNormUBuf } },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(C / 64));
      pass.end();
    }
  }

  /**
   * Run the full downward pass (R-1 sequential dispatches).
   */
  _downwardPass(encoder, UBuf, logNormUBuf, DBuf, logNormDBuf,
                subMatBuf, rootProbBuf, obsLikeBuf, parentIndex, R, C, A) {
    const pipeline = this._getOrCreatePipeline('downward_step');
    const { sibling } = this._childrenOf(parentIndex);

    for (let n = 1; n < R; n++) {
      const parent = parentIndex[n];
      const sib = sibling[n];

      const params = this._createUniformBuffer(new Uint32Array([
        C, A, n, parent, sib, parent === 0 ? 1 : 0, 0, 0,
      ]));

      // Sibling's sub-matrix
      const sibSmSize = A * A * 4;
      const sibSmBuf = this._createStorageBuffer(sibSmSize);
      encoder.copyBufferToBuffer(subMatBuf, sib * A * A * 4, sibSmBuf, 0, sibSmSize);

      // Parent's sub-matrix
      const parSmBuf = this._createStorageBuffer(sibSmSize);
      encoder.copyBufferToBuffer(subMatBuf, parent * A * A * 4, parSmBuf, 0, sibSmSize);

      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: sibSmBuf } },
          { binding: 2, resource: { buffer: parSmBuf } },
          { binding: 3, resource: { buffer: UBuf } },
          { binding: 4, resource: { buffer: logNormUBuf } },
          { binding: 5, resource: { buffer: DBuf } },
          { binding: 6, resource: { buffer: logNormDBuf } },
          { binding: 7, resource: { buffer: rootProbBuf } },
          { binding: 8, resource: { buffer: obsLikeBuf } },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(C / 64));
      pass.end();
    }
  }

  /**
   * Compute per-column log-likelihoods.
   *
   * Accepts either:
   *   LogLike(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi)
   *   LogLike(alignment, parentIndex, distances, model)
   *
   * @returns {Float32Array} (C,) log-likelihoods
   */
  async LogLike(alignment, parentIndex, distances, arg4, arg5, arg6) {
    const parsed = PhyloGPU._parseModelArg(arg4, arg5);
    const pi = parsed ? parsed.model.pi : arg6;
    const R = parentIndex.length;
    const A = pi.length;
    const C = alignment.length / R;

    const alignBuf = this._createBuffer(new Int32Array(alignment));
    const distBuf = this._createBuffer(new Float32Array(distances));

    const encoder = this.device.createCommandEncoder();

    // 1. Token to likelihood
    const likeBuf = this._tokenToLikelihood(encoder, alignBuf, R, C, A);

    // 2. Compute sub-matrices (reversible or complex)
    let subMatBuf, extraBufs;
    if (parsed && parsed.irrev) {
      const m = parsed.model;
      const evBuf = this._createBuffer(new Float32Array(m.eigenvalues_complex));
      const vecBuf = this._createBuffer(new Float32Array(m.eigenvectors_complex));
      const invBuf = this._createBuffer(new Float32Array(m.eigenvectors_inv_complex));
      subMatBuf = this._computeSubMatricesComplex(encoder, evBuf, vecBuf, invBuf, distBuf, R, A);
      extraBufs = [evBuf, vecBuf, invBuf];
    } else {
      const eigenvalues = parsed ? parsed.model.eigenvalues : arg4;
      const eigenvectors = parsed ? parsed.model.eigenvectors : arg5;
      const eigenvalsBuf = this._createBuffer(new Float32Array(eigenvalues));
      const eigenvecsBuf = this._createBuffer(new Float32Array(eigenvectors));
      const piBuf = this._createBuffer(new Float32Array(pi));
      subMatBuf = this._computeSubMatrices(encoder, eigenvalsBuf, eigenvecsBuf, piBuf, distBuf, R, A);
      extraBufs = [eigenvalsBuf, eigenvecsBuf, piBuf];
    }

    // 3. LogNormU buffer
    const logNormUBuf = this._createStorageBuffer(R * C * 4);
    this.device.queue.writeBuffer(logNormUBuf, 0, new Float32Array(R * C));

    // 4. Upward pass
    this._upwardPass(encoder, likeBuf, logNormUBuf, subMatBuf, parentIndex, R, C, A);

    // Submit and wait
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    // 5. Compute logLike on CPU from root inside vectors
    const likeData = await this._readBuffer(likeBuf, R * C * A * 4);
    const logNormUData = await this._readBuffer(logNormUBuf, R * C * 4);

    const logLike = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let a = 0; a < A; a++) {
        s += pi[a] * likeData[0 * C * A + c * A + a];
      }
      logLike[c] = logNormUData[0 * C + c] + Math.log(s);
    }

    // Cleanup
    for (const b of [alignBuf, distBuf, likeBuf, subMatBuf, logNormUBuf, ...extraBufs]) {
      b.destroy();
    }

    return logLike;
  }

  /**
   * Compute expected substitution counts and dwell times.
   *
   * Accepts either:
   *   Counts(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi, f81Fast?)
   *   Counts(alignment, parentIndex, distances, model)
   *
   * @returns {Float32Array} (A*A*C,) row-major counts tensor
   */
  async Counts(alignment, parentIndex, distances, arg4, arg5, arg6, arg7) {
    const parsed = PhyloGPU._parseModelArg(arg4, arg5);
    const pi = parsed ? parsed.model.pi : arg6;
    const eigenvalues = parsed ? parsed.model.eigenvalues : arg4;
    const eigenvectors = parsed ? parsed.model.eigenvectors : arg5;
    const f81Fast = parsed ? false : (arg7 || false);
    const isIrrev = parsed && parsed.irrev;
    const R = parentIndex.length;
    const A = pi.length;
    const C = alignment.length / R;

    const alignBuf = this._createBuffer(new Int32Array(alignment));
    const distBuf = this._createBuffer(new Float32Array(distances));

    const encoder = this.device.createCommandEncoder();

    const likeBuf = this._tokenToLikelihood(encoder, alignBuf, R, C, A);

    // Compute sub-matrices
    let subMatBuf, smExtraBufs;
    if (isIrrev) {
      const m = parsed.model;
      const evBuf = this._createBuffer(new Float32Array(m.eigenvalues_complex));
      const vecBuf = this._createBuffer(new Float32Array(m.eigenvectors_complex));
      const invBuf = this._createBuffer(new Float32Array(m.eigenvectors_inv_complex));
      subMatBuf = this._computeSubMatricesComplex(encoder, evBuf, vecBuf, invBuf, distBuf, R, A);
      smExtraBufs = [evBuf, vecBuf, invBuf];
    } else {
      const eigenvalsBuf = this._createBuffer(new Float32Array(eigenvalues));
      const eigenvecsBuf = this._createBuffer(new Float32Array(eigenvectors));
      const piBuf = this._createBuffer(new Float32Array(pi));
      subMatBuf = this._computeSubMatrices(encoder, eigenvalsBuf, eigenvecsBuf, piBuf, distBuf, R, A);
      smExtraBufs = [eigenvalsBuf, eigenvecsBuf, piBuf];
    }

    const logNormUBuf = this._createStorageBuffer(R * C * 4);
    this.device.queue.writeBuffer(logNormUBuf, 0, new Float32Array(R * C));

    this._upwardPass(encoder, likeBuf, logNormUBuf, subMatBuf, parentIndex, R, C, A);

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const UData = await this._readBuffer(likeBuf, R * C * A * 4);
    const logNormUData = await this._readBuffer(logNormUBuf, R * C * 4);

    const logLike = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let a = 0; a < A; a++) s += pi[a] * UData[c * A + a];
      logLike[c] = logNormUData[c] + Math.log(s);
    }

    const UBuf = this._createBuffer(new Float32Array(UData));
    const logNormUBuf2 = this._createBuffer(new Float32Array(logNormUData));

    const obsLikeBuf = this._createStorageBuffer(R * C * A * 4);
    {
      const obsLike = new Float32Array(R * C * A);
      for (let r = 0; r < R; r++) {
        for (let c = 0; c < C; c++) {
          const tok = alignment[r * C + c];
          for (let a = 0; a < A; a++) {
            if (tok >= 0 && tok < A) {
              obsLike[(r * C + c) * A + a] = (a === tok) ? 1.0 : 0.0;
            } else {
              obsLike[(r * C + c) * A + a] = 1.0;
            }
          }
        }
      }
      this.device.queue.writeBuffer(obsLikeBuf, 0, obsLike);
    }

    const rootProbBuf = this._createBuffer(new Float32Array(pi));

    const DBuf = this._createStorageBuffer(R * C * A * 4);
    this.device.queue.writeBuffer(DBuf, 0, new Float32Array(R * C * A));
    const logNormDBuf = this._createStorageBuffer(R * C * 4);
    this.device.queue.writeBuffer(logNormDBuf, 0, new Float32Array(R * C));

    const subMatBuf2 = this._createBuffer(new Float32Array(
      await this._readBuffer(subMatBuf, R * A * A * 4)
    ));

    const encoder2 = this.device.createCommandEncoder();
    this._downwardPass(encoder2, UBuf, logNormUBuf2, DBuf, logNormDBuf,
                       subMatBuf2, rootProbBuf, obsLikeBuf, parentIndex, R, C, A);
    this.device.queue.submit([encoder2.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const DData = await this._readBuffer(DBuf, R * C * A * 4);
    const logNormDData = await this._readBuffer(logNormDBuf, R * C * 4);

    let countsData;
    if (isIrrev) {
      countsData = await this._eigensubCountsComplex(UData, DData, logNormUData, logNormDData,
                                                      logLike, parsed.model, distances, R, C, A);
    } else if (f81Fast) {
      countsData = await this._f81Fast(UData, DData, logNormUData, logNormDData,
                                        logLike, distances, pi, R, C, A);
    } else {
      countsData = await this._eigensubCounts(UData, DData, logNormUData, logNormDData,
                                               logLike, eigenvalues, eigenvectors, distances, pi, R, C, A);
    }

    for (const b of [alignBuf, distBuf, likeBuf, subMatBuf, logNormUBuf,
                      UBuf, logNormUBuf2, obsLikeBuf, rootProbBuf, DBuf, logNormDBuf,
                      subMatBuf2, ...smExtraBufs]) {
      b.destroy();
    }

    return countsData;
  }

  async _eigensubCounts(UData, DData, logNormUData, logNormDData,
                         logLike, eigenvalues, eigenvectors, distances, pi, R, C, A) {
    // compute_J
    const eigenvalsBuf = this._createBuffer(new Float32Array(eigenvalues));
    const distBuf = this._createBuffer(new Float32Array(distances));
    const JBuf = this._createStorageBuffer(R * A * A * 4);

    const encoder = this.device.createCommandEncoder();
    {
      const pipeline = this._getOrCreatePipeline('compute_J');
      const params = this._createUniformBuffer(new Uint32Array([R, A, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: eigenvalsBuf } },
          { binding: 2, resource: { buffer: distBuf } },
          { binding: 3, resource: { buffer: JBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(R / 64));
      pass.end();
    }

    // eigenbasis_project
    const UBuf = this._createBuffer(new Float32Array(UData));
    const DBuf = this._createBuffer(new Float32Array(DData));
    const eigenvecsBuf = this._createBuffer(new Float32Array(eigenvectors));
    const piBuf = this._createBuffer(new Float32Array(pi));
    const U_tildeBuf = this._createStorageBuffer(R * C * A * 4);
    const D_tildeBuf = this._createStorageBuffer(R * C * A * 4);

    {
      const pipeline = this._getOrCreatePipeline('eigenbasis_project');
      const params = this._createUniformBuffer(new Uint32Array([R, C, A, 0]));
      const bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: eigenvecsBuf } },
          { binding: 2, resource: { buffer: piBuf } },
          { binding: 3, resource: { buffer: UBuf } },
          { binding: 4, resource: { buffer: DBuf } },
          { binding: 5, resource: { buffer: U_tildeBuf } },
          { binding: 6, resource: { buffer: D_tildeBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((R * C) / 64));
      pass.end();
    }

    // accumulate_C
    const logNormUBuf = this._createBuffer(new Float32Array(logNormUData));
    const logNormDBuf = this._createBuffer(new Float32Array(logNormDData));
    const logLikeBuf = this._createBuffer(new Float32Array(logLike));
    const C_eigenBuf = this._createStorageBuffer(A * A * C * 4);

    {
      const pipeline = this._getOrCreatePipeline('accumulate_C');
      const params = this._createUniformBuffer(new Uint32Array([R, C, A, 0]));
      const bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: D_tildeBuf } },
          { binding: 2, resource: { buffer: U_tildeBuf } },
          { binding: 3, resource: { buffer: JBuf } },
          { binding: 4, resource: { buffer: logNormUBuf } },
          { binding: 5, resource: { buffer: logNormDBuf } },
          { binding: 6, resource: { buffer: logLikeBuf } },
          { binding: 7, resource: { buffer: C_eigenBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((A * A * C) / 64));
      pass.end();
    }

    // back_transform
    const countsBuf = this._createStorageBuffer(A * A * C * 4);
    {
      const pipeline = this._getOrCreatePipeline('back_transform');
      const params = this._createUniformBuffer(new Uint32Array([C, A, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: eigenvecsBuf } },
          { binding: 2, resource: { buffer: eigenvalsBuf } },
          { binding: 3, resource: { buffer: C_eigenBuf } },
          { binding: 4, resource: { buffer: countsBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((A * A * C) / 64));
      pass.end();
    }

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const result = await this._readBuffer(countsBuf, A * A * C * 4);

    for (const b of [eigenvalsBuf, distBuf, JBuf, UBuf, DBuf, eigenvecsBuf, piBuf,
                      U_tildeBuf, D_tildeBuf, logNormUBuf, logNormDBuf, logLikeBuf,
                      C_eigenBuf, countsBuf]) {
      b.destroy();
    }

    return result;
  }

  async _f81Fast(UData, DData, logNormUData, logNormDData,
                  logLike, distances, pi, R, C, A) {
    // F81 fast path: compute mu, then dispatch shader
    let piSqSum = 0;
    for (let a = 0; a < A; a++) piSqSum += pi[a] * pi[a];
    const mu = 1.0 / (1.0 - piSqSum);

    const UBuf = this._createBuffer(new Float32Array(UData));
    const DBuf = this._createBuffer(new Float32Array(DData));
    const logNormUBuf = this._createBuffer(new Float32Array(logNormUData));
    const logNormDBuf = this._createBuffer(new Float32Array(logNormDData));
    const logLikeBuf = this._createBuffer(new Float32Array(logLike));
    const distBuf = this._createBuffer(new Float32Array(distances));
    const piBuf = this._createBuffer(new Float32Array(pi));
    const muBuf = this._createUniformBuffer(new Float32Array([mu]));
    const countsBuf = this._createStorageBuffer(A * A * C * 4);

    const pipeline = this._getOrCreatePipeline('f81_fast');
    const params = this._createUniformBuffer(new Uint32Array([R, C, A, 0]));

    const bg = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: UBuf } },
        { binding: 2, resource: { buffer: DBuf } },
        { binding: 3, resource: { buffer: logNormUBuf } },
        { binding: 4, resource: { buffer: logNormDBuf } },
        { binding: 5, resource: { buffer: logLikeBuf } },
        { binding: 6, resource: { buffer: distBuf } },
        { binding: 7, resource: { buffer: piBuf } },
        { binding: 8, resource: { buffer: muBuf } },
        { binding: 9, resource: { buffer: countsBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil((A * A * C) / 64));
    pass.end();

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const result = await this._readBuffer(countsBuf, A * A * C * 4);

    for (const b of [UBuf, DBuf, logNormUBuf, logNormDBuf, logLikeBuf,
                      distBuf, piBuf, muBuf, countsBuf]) {
      b.destroy();
    }

    return result;
  }

  /**
   * Compute posterior root state distribution.
   *
   * Accepts either:
   *   RootProb(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi)
   *   RootProb(alignment, parentIndex, distances, model)
   *
   * @returns {Float32Array} (A*C,) row-major
   */
  async RootProb(alignment, parentIndex, distances, arg4, arg5, arg6) {
    const parsed = PhyloGPU._parseModelArg(arg4, arg5);
    const pi = parsed ? parsed.model.pi : arg6;
    const R = parentIndex.length;
    const A = pi.length;
    const C = alignment.length / R;

    const alignBuf = this._createBuffer(new Int32Array(alignment));
    const distBuf = this._createBuffer(new Float32Array(distances));

    const encoder = this.device.createCommandEncoder();
    const likeBuf = this._tokenToLikelihood(encoder, alignBuf, R, C, A);

    let subMatBuf, extraBufs;
    if (parsed && parsed.irrev) {
      const m = parsed.model;
      const evBuf = this._createBuffer(new Float32Array(m.eigenvalues_complex));
      const vecBuf = this._createBuffer(new Float32Array(m.eigenvectors_complex));
      const invBuf = this._createBuffer(new Float32Array(m.eigenvectors_inv_complex));
      subMatBuf = this._computeSubMatricesComplex(encoder, evBuf, vecBuf, invBuf, distBuf, R, A);
      extraBufs = [evBuf, vecBuf, invBuf];
    } else {
      const eigenvalues = parsed ? parsed.model.eigenvalues : arg4;
      const eigenvectors = parsed ? parsed.model.eigenvectors : arg5;
      const eigenvalsBuf = this._createBuffer(new Float32Array(eigenvalues));
      const eigenvecsBuf = this._createBuffer(new Float32Array(eigenvectors));
      const piBuf = this._createBuffer(new Float32Array(pi));
      subMatBuf = this._computeSubMatrices(encoder, eigenvalsBuf, eigenvecsBuf, piBuf, distBuf, R, A);
      extraBufs = [eigenvalsBuf, eigenvecsBuf, piBuf];
    }

    const logNormUBuf = this._createStorageBuffer(R * C * 4);
    this.device.queue.writeBuffer(logNormUBuf, 0, new Float32Array(R * C));
    this._upwardPass(encoder, likeBuf, logNormUBuf, subMatBuf, parentIndex, R, C, A);

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const UData = await this._readBuffer(likeBuf, R * C * A * 4);
    const logNormUData = await this._readBuffer(logNormUBuf, R * C * 4);

    const result = new Float32Array(A * C);
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let a = 0; a < A; a++) s += pi[a] * UData[c * A + a];
      const logLike = logNormUData[c] + Math.log(s);
      const logScale = logNormUData[c] - logLike;
      const scale = Math.exp(logScale);
      for (let a = 0; a < A; a++) {
        result[a * C + c] = pi[a] * UData[c * A + a] * scale;
      }
    }

    for (const b of [alignBuf, distBuf, likeBuf, subMatBuf, logNormUBuf, ...extraBufs]) {
      b.destroy();
    }

    return result;
  }

  /**
   * Compute mixture posteriors.
   *
   * @param {Array<{eigenvalues, eigenvectors, pi}>} models - K model specs
   * @param {Float32Array} logWeights - (K,)
   * @returns {Float32Array} (K*C,) posteriors
   */
  async MixturePosterior(alignment, parentIndex, distances, models, logWeights) {
    const K = models.length;
    const R = parentIndex.length;
    const C = alignment.length / R;

    // Compute per-component log-likelihoods
    const logLikes = new Float32Array(K * C);
    for (let k = 0; k < K; k++) {
      const ll = await this.LogLike(
        alignment, parentIndex, distances, models[k],
      );
      logLikes.set(ll, k * C);
    }

    // Run mixture_posterior shader
    const logLikesBuf = this._createBuffer(new Float32Array(logLikes));
    const logWeightsBuf = this._createBuffer(new Float32Array(logWeights));
    const posteriorsBuf = this._createStorageBuffer(K * C * 4);

    const pipeline = this._getOrCreatePipeline('mixture_posterior');
    const params = this._createUniformBuffer(new Uint32Array([K, C, 0, 0]));

    const bg = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: logLikesBuf } },
        { binding: 2, resource: { buffer: logWeightsBuf } },
        { binding: 3, resource: { buffer: posteriorsBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(C / 64));
    pass.end();

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const result = await this._readBuffer(posteriorsBuf, K * C * 4);

    for (const b of [logLikesBuf, logWeightsBuf, posteriorsBuf]) {
      b.destroy();
    }

    return result;
  }

  /**
   * Create an InsideOutside table for querying posteriors.
   *
   * Accepts either:
   *   InsideOutside(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi)
   *   InsideOutside(alignment, parentIndex, distances, model)
   *
   * @returns {PhyloGPUInsideOutside}
   */
  async InsideOutside(alignment, parentIndex, distances, arg4, arg5, arg6) {
    const parsed = PhyloGPU._parseModelArg(arg4, arg5);
    const pi = parsed ? parsed.model.pi : arg6;
    const isIrrev = parsed && parsed.irrev;
    const R = parentIndex.length;
    const A = pi.length;
    const C = alignment.length / R;

    const alignBuf = this._createBuffer(new Int32Array(alignment));
    const distBuf = this._createBuffer(new Float32Array(distances));

    const encoder = this.device.createCommandEncoder();
    const likeBuf = this._tokenToLikelihood(encoder, alignBuf, R, C, A);

    // Compute sub-matrices
    let subMatBuf, smExtraBufs;
    if (isIrrev) {
      const m = parsed.model;
      const evBuf = this._createBuffer(new Float32Array(m.eigenvalues_complex));
      const vecBuf = this._createBuffer(new Float32Array(m.eigenvectors_complex));
      const invBuf = this._createBuffer(new Float32Array(m.eigenvectors_inv_complex));
      subMatBuf = this._computeSubMatricesComplex(encoder, evBuf, vecBuf, invBuf, distBuf, R, A);
      smExtraBufs = [evBuf, vecBuf, invBuf];
    } else {
      const eigenvalues = parsed ? parsed.model.eigenvalues : arg4;
      const eigenvectors = parsed ? parsed.model.eigenvectors : arg5;
      const eigenvalsBuf = this._createBuffer(new Float32Array(eigenvalues));
      const eigenvecsBuf = this._createBuffer(new Float32Array(eigenvectors));
      const piBuf = this._createBuffer(new Float32Array(pi));
      subMatBuf = this._computeSubMatrices(encoder, eigenvalsBuf, eigenvecsBuf, piBuf, distBuf, R, A);
      smExtraBufs = [eigenvalsBuf, eigenvecsBuf, piBuf];
    }

    const logNormUBuf = this._createStorageBuffer(R * C * 4);
    this.device.queue.writeBuffer(logNormUBuf, 0, new Float32Array(R * C));
    this._upwardPass(encoder, likeBuf, logNormUBuf, subMatBuf, parentIndex, R, C, A);

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const UData = await this._readBuffer(likeBuf, R * C * A * 4);
    const logNormUData = await this._readBuffer(logNormUBuf, R * C * 4);

    // Compute logLike on CPU
    const logLike = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let a = 0; a < A; a++) s += pi[a] * UData[c * A + a];
      logLike[c] = logNormUData[c] + Math.log(s);
    }

    // Observation likelihoods for downward pass
    const obsLikeBuf = this._createStorageBuffer(R * C * A * 4);
    {
      const obsLike = new Float32Array(R * C * A);
      for (let r = 0; r < R; r++) {
        for (let c = 0; c < C; c++) {
          const tok = alignment[r * C + c];
          for (let a = 0; a < A; a++) {
            if (tok >= 0 && tok < A) {
              obsLike[(r * C + c) * A + a] = (a === tok) ? 1.0 : 0.0;
            } else {
              obsLike[(r * C + c) * A + a] = 1.0;
            }
          }
        }
      }
      this.device.queue.writeBuffer(obsLikeBuf, 0, obsLike);
    }

    const UBuf2 = this._createBuffer(new Float32Array(UData));
    const logNormUBuf2 = this._createBuffer(new Float32Array(logNormUData));
    const rootProbBuf = this._createBuffer(new Float32Array(pi));

    const DBuf = this._createStorageBuffer(R * C * A * 4);
    this.device.queue.writeBuffer(DBuf, 0, new Float32Array(R * C * A));
    const logNormDBuf = this._createStorageBuffer(R * C * 4);
    this.device.queue.writeBuffer(logNormDBuf, 0, new Float32Array(R * C));

    const subMatData = await this._readBuffer(subMatBuf, R * A * A * 4);
    const subMatBuf2 = this._createBuffer(new Float32Array(subMatData));

    const encoder2 = this.device.createCommandEncoder();
    this._downwardPass(encoder2, UBuf2, logNormUBuf2, DBuf, logNormDBuf,
                       subMatBuf2, rootProbBuf, obsLikeBuf, parentIndex, R, C, A);
    this.device.queue.submit([encoder2.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const DData = await this._readBuffer(DBuf, R * C * A * 4);
    const logNormDData = await this._readBuffer(logNormDBuf, R * C * 4);

    // Fix root D: set D[0] = pi (rescaled)
    let maxPi = 0;
    for (let a = 0; a < A; a++) { if (pi[a] > maxPi) maxPi = pi[a]; }
    if (maxPi < 1e-30) maxPi = 1e-30;
    for (let c = 0; c < C; c++) {
      for (let a = 0; a < A; a++) {
        DData[c * A + a] = pi[a] / maxPi;
      }
      logNormDData[c] = Math.log(maxPi);
    }

    // Cleanup GPU buffers
    for (const b of [alignBuf, distBuf, likeBuf, subMatBuf, logNormUBuf,
                      UBuf2, logNormUBuf2, obsLikeBuf, rootProbBuf,
                      DBuf, logNormDBuf, subMatBuf2, ...smExtraBufs]) {
      b.destroy();
    }

    return new PhyloGPUInsideOutside(
      UData, DData, logNormUData, logNormDData, logLike,
      subMatData, pi, parentIndex, R, C, A,
      isIrrev ? parsed.model : null,
      isIrrev ? null : { eigenvalues: parsed ? parsed.model.eigenvalues : arg4,
                          eigenvectors: parsed ? parsed.model.eigenvectors : arg5 },
      distances, this,
    );
  }

  destroy() {
    this.device.destroy();
  }
}

/**
 * InsideOutside table for PhyloGPU.
 * Stores CPU-side copies of U, D, logNorm arrays and computes posteriors on CPU.
 */
class PhyloGPUInsideOutside {
  constructor(U, D, logNormU, logNormD, logLike, subMats, pi,
              parentIndex, R, C, A, irrevModel, revModel, distances, gpu) {
    this._U = U;               // Float32Array (R*C*A)
    this._D = D;               // Float32Array (R*C*A)
    this._logNormU = logNormU;  // Float32Array (R*C)
    this._logNormD = logNormD;  // Float32Array (R*C)
    this._logLike = logLike;    // Float32Array (C)
    this._subMats = subMats;    // Float32Array (R*A*A)
    this._pi = pi;
    this._parentIndex = parentIndex;
    this.R = R;
    this.C = C;
    this.A = A;
    this._irrevModel = irrevModel;
    this._revModel = revModel;
    this._distances = distances;
    this._gpu = gpu;
  }

  /** Per-column log-likelihoods. @returns {Float32Array} (C,) */
  get log_likelihood() {
    return this._logLike;
  }

  /** Expected substitution counts via eigensub pipeline on GPU. */
  async counts(f81Fast = false) {
    const { R, C, A } = this;
    if (this._irrevModel) {
      return this._gpu._eigensubCountsComplex(
        this._U, this._D, this._logNormU, this._logNormD,
        this._logLike, this._irrevModel, this._distances, R, C, A,
      );
    } else if (f81Fast) {
      return this._gpu._f81Fast(
        this._U, this._D, this._logNormU, this._logNormD,
        this._logLike, this._distances, this._pi, R, C, A,
      );
    } else {
      return this._gpu._eigensubCounts(
        this._U, this._D, this._logNormU, this._logNormD,
        this._logLike, this._revModel.eigenvalues, this._revModel.eigenvectors,
        this._distances, this._pi, R, C, A,
      );
    }
  }

  /**
   * Posterior state distribution at node(s).
   * @param {number|null} node - Node index, or null for all nodes.
   * @returns {Float32Array} (A*C,) for single or (R*A*C,) for all.
   */
  node_posterior(node = null) {
    const { R, C, A } = this;
    if (node !== null) {
      const q = new Float32Array(A * C);
      for (let c = 0; c < C; c++) {
        const logScale = this._logNormU[node * C + c]
          + this._logNormD[node * C + c]
          - this._logLike[c];
        const scale = Math.exp(logScale);
        let sum = 0;
        if (node === 0) {
          for (let a = 0; a < A; a++) {
            const val = this._D[c * A + a] * this._U[c * A + a] * scale;
            q[a * C + c] = val;
            sum += val;
          }
        } else {
          const mBase = node * A * A;
          const uBase = (node * C + c) * A;
          const dBase = (node * C + c) * A;
          for (let j = 0; j < A; j++) {
            let dT = 0;
            for (let a = 0; a < A; a++) {
              dT += this._D[dBase + a] * this._subMats[mBase + a * A + j];
            }
            const val = dT * this._U[uBase + j] * scale;
            q[j * C + c] = val;
            sum += val;
          }
        }
        if (sum > 0) {
          for (let a = 0; a < A; a++) q[a * C + c] /= sum;
        }
      }
      return q;
    } else {
      const result = new Float32Array(R * A * C);
      for (let n = 0; n < R; n++) {
        const single = this.node_posterior(n);
        result.set(single, n * A * C);
      }
      return result;
    }
  }

  /**
   * Joint branch posterior of parent-child states.
   * @param {number|null} node - Child node (> 0), or null for all.
   * @returns {Float32Array} (A*A*C,) for single or (R*A*A*C,) for all.
   */
  branch_posterior(node = null) {
    const { R, C, A } = this;
    if (node !== null) {
      const joint = new Float32Array(A * A * C);
      for (let c = 0; c < C; c++) {
        const logScale = this._logNormD[node * C + c]
          + this._logNormU[node * C + c]
          - this._logLike[c];
        const scale = Math.exp(logScale);
        const dBase = (node * C + c) * A;
        const uBase = (node * C + c) * A;
        const mBase = node * A * A;
        let sum = 0;
        for (let i = 0; i < A; i++) {
          for (let j = 0; j < A; j++) {
            const val = this._D[dBase + i]
              * this._subMats[mBase + i * A + j]
              * this._U[uBase + j]
              * scale;
            joint[i * A * C + j * C + c] = val;
            sum += val;
          }
        }
        if (sum > 0) {
          for (let i = 0; i < A; i++) {
            for (let j = 0; j < A; j++) {
              joint[i * A * C + j * C + c] /= sum;
            }
          }
        }
      }
      return joint;
    } else {
      const result = new Float32Array(R * A * A * C);
      for (let n = 1; n < R; n++) {
        const single = this.branch_posterior(n);
        result.set(single, n * A * A * C);
      }
      return result;
    }
  }

  destroy() {
    // No GPU resources to free (all read back to CPU)
  }
}
