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
   * @param {Int32Array} alignment - (R*C) flat, row-major
   * @param {Int32Array} parentIndex - (R,)
   * @param {Float32Array} distances - (R,)
   * @param {Float32Array} eigenvalues - (A,)
   * @param {Float32Array} eigenvectors - (A*A,) row-major
   * @param {Float32Array} pi - (A,)
   * @returns {Float32Array} (C,) log-likelihoods
   */
  async LogLike(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi) {
    const R = parentIndex.length;
    const A = pi.length;
    const C = alignment.length / R;

    const alignBuf = this._createBuffer(new Int32Array(alignment));
    const eigenvalsBuf = this._createBuffer(new Float32Array(eigenvalues));
    const eigenvecsBuf = this._createBuffer(new Float32Array(eigenvectors));
    const piBuf = this._createBuffer(new Float32Array(pi));
    const distBuf = this._createBuffer(new Float32Array(distances));

    const encoder = this.device.createCommandEncoder();

    // 1. Token to likelihood
    const likeBuf = this._tokenToLikelihood(encoder, alignBuf, R, C, A);

    // 2. Compute sub-matrices
    const subMatBuf = this._computeSubMatrices(encoder, eigenvalsBuf, eigenvecsBuf, piBuf, distBuf, R, A);

    // 3. LogNormU buffer
    const logNormUBuf = this._createStorageBuffer(R * C * 4);
    // Zero it
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
    for (const b of [alignBuf, eigenvalsBuf, eigenvecsBuf, piBuf, distBuf,
                      likeBuf, subMatBuf, logNormUBuf]) {
      b.destroy();
    }

    return logLike;
  }

  /**
   * Compute expected substitution counts and dwell times.
   *
   * @returns {Float32Array} (A*A*C,) row-major counts tensor
   */
  async Counts(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi, f81Fast = false) {
    const R = parentIndex.length;
    const A = pi.length;
    const C = alignment.length / R;

    const alignBuf = this._createBuffer(new Int32Array(alignment));
    const eigenvalsBuf = this._createBuffer(new Float32Array(eigenvalues));
    const eigenvecsBuf = this._createBuffer(new Float32Array(eigenvectors));
    const piBuf = this._createBuffer(new Float32Array(pi));
    const distBuf = this._createBuffer(new Float32Array(distances));

    const encoder = this.device.createCommandEncoder();

    const likeBuf = this._tokenToLikelihood(encoder, alignBuf, R, C, A);
    const subMatBuf = this._computeSubMatrices(encoder, eigenvalsBuf, eigenvecsBuf, piBuf, distBuf, R, A);
    const logNormUBuf = this._createStorageBuffer(R * C * 4);
    this.device.queue.writeBuffer(logNormUBuf, 0, new Float32Array(R * C));

    this._upwardPass(encoder, likeBuf, logNormUBuf, subMatBuf, parentIndex, R, C, A);

    // We need to read logLike first, then do downward pass, so submit here
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    // Read U, logNormU for logLike computation
    const UData = await this._readBuffer(likeBuf, R * C * A * 4);
    const logNormUData = await this._readBuffer(logNormUBuf, R * C * 4);

    // Compute logLike on CPU
    const logLike = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let a = 0; a < A; a++) s += pi[a] * UData[c * A + a];
      logLike[c] = logNormUData[c] + Math.log(s);
    }

    // Re-upload U (it's been read-back from likeBuf which was read-write)
    const UBuf = this._createBuffer(new Float32Array(UData));
    const logNormUBuf2 = this._createBuffer(new Float32Array(logNormUData));

    // Observation likelihood for downward pass
    const obsLikeBuf = this._createStorageBuffer(R * C * A * 4);
    {
      // Re-compute obsLike from alignment
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

    // Downward pass
    const DBuf = this._createStorageBuffer(R * C * A * 4);
    this.device.queue.writeBuffer(DBuf, 0, new Float32Array(R * C * A));
    const logNormDBuf = this._createStorageBuffer(R * C * 4);
    this.device.queue.writeBuffer(logNormDBuf, 0, new Float32Array(R * C));

    // Re-compute sub-matrices (same as before, need a fresh buffer)
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

    // Now accumulate counts
    let countsData;
    if (f81Fast) {
      countsData = await this._f81Fast(UData, DData, logNormUData, logNormDData,
                                        logLike, distances, pi, R, C, A);
    } else {
      countsData = await this._eigensubCounts(UData, DData, logNormUData, logNormDData,
                                               logLike, eigenvalues, eigenvectors, distances, pi, R, C, A);
    }

    // Cleanup
    for (const b of [alignBuf, eigenvalsBuf, eigenvecsBuf, piBuf, distBuf,
                      likeBuf, subMatBuf, logNormUBuf, UBuf, logNormUBuf2,
                      obsLikeBuf, rootProbBuf, DBuf, logNormDBuf, subMatBuf2]) {
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
   * @returns {Float32Array} (A*C,) row-major
   */
  async RootProb(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi) {
    const R = parentIndex.length;
    const A = pi.length;
    const C = alignment.length / R;

    // Reuse LogLike path to get U and logNormU
    const alignBuf = this._createBuffer(new Int32Array(alignment));
    const eigenvalsBuf = this._createBuffer(new Float32Array(eigenvalues));
    const eigenvecsBuf = this._createBuffer(new Float32Array(eigenvectors));
    const piBuf = this._createBuffer(new Float32Array(pi));
    const distBuf = this._createBuffer(new Float32Array(distances));

    const encoder = this.device.createCommandEncoder();
    const likeBuf = this._tokenToLikelihood(encoder, alignBuf, R, C, A);
    const subMatBuf = this._computeSubMatrices(encoder, eigenvalsBuf, eigenvecsBuf, piBuf, distBuf, R, A);
    const logNormUBuf = this._createStorageBuffer(R * C * 4);
    this.device.queue.writeBuffer(logNormUBuf, 0, new Float32Array(R * C));
    this._upwardPass(encoder, likeBuf, logNormUBuf, subMatBuf, parentIndex, R, C, A);

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const UData = await this._readBuffer(likeBuf, R * C * A * 4);
    const logNormUData = await this._readBuffer(logNormUBuf, R * C * 4);

    // Compute logLike and root posterior on CPU
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

    for (const b of [alignBuf, eigenvalsBuf, eigenvecsBuf, piBuf, distBuf,
                      likeBuf, subMatBuf, logNormUBuf]) {
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
        alignment, parentIndex, distances,
        models[k].eigenvalues, models[k].eigenvectors, models[k].pi,
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

  destroy() {
    this.device.destroy();
  }
}
