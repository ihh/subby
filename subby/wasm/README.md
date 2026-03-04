# src/phylo/wasm/

Rust crate for phylogenetic sufficient statistics, compiled to WASM via wasm-bindgen.

Dual target: `cdylib` (WASM for browser) and `rlib` (native Rust for preprocessing). All internal computation in f64. Flat typed arrays across the wasm-bindgen boundary.

Build: `wasm-pack build --target web`

See `docs/api/wasm.md` for API reference. The JS wrapper is at `src/phylo/webgpu/phylo_wasm.js`.
