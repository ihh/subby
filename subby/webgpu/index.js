/**
 * Unified phylogenetic engine entry point.
 *
 * Feature-detects WebGPU and falls back to WASM.
 *
 * Usage:
 *   const engine = await createPhyloEngine({ wasmUrl: './phylo_wasm_bg.wasm' });
 *   const logLike = await engine.LogLike(alignment, parentIndex, distances,
 *                                         eigenvalues, eigenvectors, pi);
 *   engine.destroy();
 */

import { PhyloGPU } from './phylo_gpu.js';
import { PhyloWASM } from './phylo_wasm.js';

/**
 * Create a phylogenetic computation engine.
 *
 * Tries WebGPU first; falls back to WASM if unavailable.
 *
 * @param {Object} options
 * @param {string} [options.shaderBasePath='./shaders/'] - URL prefix for WGSL shaders
 * @param {string} [options.wasmUrl] - URL to WASM module (required for fallback)
 * @param {string} [options.backend] - Force 'webgpu' or 'wasm' backend
 * @param {Object} [options.shaderSources] - Pre-loaded shader source strings (optional, bypasses fetch)
 * @param {Object} [options.wasmModule] - Pre-initialized WASM module (optional, bypasses fetch)
 * @returns {Promise<{engine: PhyloGPU|PhyloWASM, backend: string}>}
 */
export async function createPhyloEngine(options = {}) {
  const {
    shaderBasePath = './shaders/',
    wasmUrl,
    backend,
    shaderSources,
    wasmModule,
  } = options;

  // Try WebGPU
  if (backend !== 'wasm') {
    try {
      if (typeof navigator !== 'undefined' && navigator.gpu) {
        let engine;
        if (shaderSources) {
          engine = await PhyloGPU.createFromSources(shaderSources);
        } else {
          engine = await PhyloGPU.create(shaderBasePath);
        }
        return { engine, backend: 'webgpu' };
      }
    } catch (e) {
      if (backend === 'webgpu') {
        throw new Error(`WebGPU requested but unavailable: ${e.message}`);
      }
      console.warn('WebGPU initialization failed, falling back to WASM:', e.message);
    }
  }

  // Fall back to WASM
  if (backend === 'webgpu') {
    throw new Error('WebGPU requested but not available');
  }

  if (wasmModule) {
    return { engine: PhyloWASM.fromModule(wasmModule), backend: 'wasm' };
  }

  if (!wasmUrl) {
    throw new Error(
      'Neither WebGPU nor WASM available. Provide wasmUrl option for WASM fallback.'
    );
  }

  const engine = await PhyloWASM.create(wasmUrl);
  return { engine, backend: 'wasm' };
}

export { PhyloGPU } from './phylo_gpu.js';
export { PhyloWASM } from './phylo_wasm.js';
export { jukesCantor, f81, hky85, gy94, diagonalize, diagonalizeIrreversible, diagonalizeAuto } from './models.js';
export { detectAlphabet, parseNewick, parseFasta, parseStockholm, parseMaf, parseStrings, parseDict, combineTreeAlignment, geneticCode, codonToSense, KmerIndex, slidingWindows, allColumnKtuples, kmerTokenize } from './formats.js';
