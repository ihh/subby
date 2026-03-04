# src/phylo/webgpu/

WebGPU compute shader implementation of phylogenetic sufficient statistics, plus unified JS entry point.

## Contents

- `shaders/` — WGSL compute shaders (one per algorithmic kernel)
- `phylo_gpu.js` — `PhyloGPU` class: device init, shader compilation, buffer management, dispatch
- `phylo_wasm.js` — `PhyloWASM` class: same API, delegates to `src/phylo/wasm/` WASM module
- `index.js` — `createPhyloEngine()`: feature-detects WebGPU, falls back to WASM

All arrays are flattened to 1D storage buffers (row-major). f32 precision with log-rescaling to prevent underflow.

See `docs/api/webgpu.md` for API reference.
