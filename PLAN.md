# subby: Implementation, Evaluation, and Publication Plan

## 1. Compute requirements

### 1.1 Development (local/cloud CPU/single-GPU)

- Prototyping featurization, data pipeline, tiling, and splitting logic
- Unit tests for phylo library (JAX, WebGPU, WASM, Rust)
- Small-scale model iteration on toy data
- Browser inference benchmarking
- **Estimated need**: 1x workstation with a single GPU (e.g. A6000 48GB or equivalent), or cloud equivalent (e.g. 1x A100 on-demand)

### 1.2 Data staging and preprocessing

- Download and preprocess MSA data (50-100 GB compressed for MultiZ 100-way; more for alternatives)
- Download and preprocess genome annotations (RefSeq, Ensembl, GENCODE)
- Download and preprocess RNA-Seq tracks (optional; species-dependent, ~10-50 GB per species)
- Tile, featurize, and serialize training examples
- **Estimated need**: ~2 TB local/cloud storage; CPU-heavy preprocessing (embarrassingly parallel across chromosomes/species); ~500 GPU-hours for precomputing phylogenetic features if done on GPU, or ~5,000 CPU-hours if done on CPU
- **Cost note**: Preprocessing is a one-time cost per dataset configuration. Intermediate results should be cached. Consider using a cloud bucket (GCS/S3) for shared access.

### 1.3 Training

- Tiberius (comparable model) trained for ~15 days on 4x A100 80GB = ~1,440 A100-hours
- subby uses Mamba (more efficient than LSTM for long sequences) but adds phylo featurization overhead
- **Estimated need for a full training run**: 4x A100 80GB for ~7-14 days (~700-1,400 A100-hours)
- **Estimated need for ablation studies** (8-12 ablations at reduced scale): ~2,000-4,000 A100-hours total
- **Estimated need for hyperparameter sweeps**: ~500-1,000 A100-hours
- **Total estimated GPU budget**: ~3,000-6,500 A100-hours (~$7,500-$16,000 at on-demand cloud rates; substantially less with reserved/spot instances or academic allocations)

## 2. Development of model and data pipeline tools

### 2.1 Phylogenetic sufficient statistics library (`src/phylo/`)

1. **JAX implementation** (`src/phylo/jax/`)
   - Extend `toy_felsenstein_pruning.py` to include root-to-leaf (outside) pass
   - Accumulate eigensubstitutions per Holmes & Rubin (2002)
   - Transform back to obtain per-column E[s_kl] and E[w_k]
   - Handle multiple ungapped components per column
   - Implement `LogLike(MSA, Tree, Model)`, `Counts(MSA, Tree, Model)`, `RootProb(MSA, Tree, Model)`
   - Support diagonalized rate matrix input; provide wrapper to diagonalize
   - Implement HKY85 closed-form eigendecomposition
   - Implement Jukes-Cantor / F81 fast path (O(CRA^4))
   - Implement mixture model posterior (SoftMax over components)
   - Comprehensive tests against known analytical results and brute-force enumeration

2. **Python oracle** (`src/phylo/oracle/`)
   - Simple imperative Python implementation (numpy-only, explicit for-loops)
   - Serves as the numerically trustworthy cross-language test oracle for WebGPU and WASM
   - No vectorization, no einsum, no JAX — obviously correct at the expense of speed
   - Golden test files generated from oracle outputs (`tests/test_phylo/golden/*.json`)

3. **WebGPU implementation** (`src/phylo/webgpu/`) — *higher priority*
   - WGSL compute shaders for the pruning recursion and eigensubstitution accumulation
   - Vanilla JS wrapper (`PhyloGPU` class) providing the same `LogLike`, `Counts`, `RootProb` API
   - Tree traversals dispatched one compute pass per branch step, all C columns parallel
   - f32 precision with rescaling; tested against golden files at atol=1e-3
   - Unified JS entry point (`createPhyloEngine()`) that feature-detects WebGPU

4. **WASM fallback** (`src/phylo/wasm/`)
   - Rust compiled to WASM via wasm-bindgen, exposing the same API
   - Single crate, dual target: `cdylib` (WASM) + `rlib` (native Rust tests)
   - f64 precision; tested against golden files at atol=1e-8
   - Fallback for browsers without WebGPU support
   - `PhyloWASM` JS wrapper class mirroring `PhyloGPU` API, integrated into `createPhyloEngine()` fallback

5. **Rust native** (`src/phylo/rust/`)
   - Native Rust implementation for command-line preprocessing
   - Shared core with the WASM target (same crate, different compilation targets)

### 2.2 Featurization pipeline (`src/data/`)

1. **Tokenizers** for each featurization scheme described in Model.md:
   - `[HKY85]`: nucleotide tokenization, N -> ungapped-unobserved, mixture of K rate-scaled HKY85 models
   - `[TRIPLETS]`: context-dependent triplet tokenization (previous/current/next non-gap nucleotide)
   - `[PHASE]`: gap-phase tokenization ({NUC, 0, 1, 2})
   - `[ANNOTATION]`: annotation label tokenization, root node as ungapped-unobserved

Implement all four efficiently as a finite state transducer that consumes each row character-by-character and emits four different tokens at each position.

2. **Feature extraction**: run `Counts`, `LogLike`, and/or `RootProb` for each tokenization on tiled MSA windows, concatenate feature vectors

3. **RNA-Seq preprocessing**: Borzoi-transform coverage/junction tracks into 16-bit float tensors

### 2.3 Model (`src/model/`)

1. **Bidirectional Mamba tower** (JAX)
   - Implement or adapt bidirectional Mamba with RmsNorm pre-normalization
   - Reverse complement data augmentation
   - Cross-attention layers for RNA-Seq integration

2. **Track encoder** (JAX)
   - Bidirectional Mamba on 6-channel RNA-Seq tensors (fwd + rev strand)
   - Shared weights across tracks

3. **HMM decoder head**
   - Interface with Machine Boss for Forward-Backward / Viterbi
   - 15 annotation states (Tiberius-compatible)
   - Differentiable Forward-Backward for training; Viterbi for inference

4. **WebGPU inference versions** (`src/model/webgpu/`, `src/inference/`)
   - Bidirectional Mamba inference in WebGPU compute shaders
   - Viterbi HMM decoder in JS/WASM
   - Model weight loading from quantized ONNX or custom format
   - Tiled inference with overlap-and-stitch

## 3. Dataset tilling

### 3.1 Data source alternatives

We present several alternatives for the MSA data, each with different trade-offs:

| Source | Species | Alignment size | Pros | Cons |
|--------|---------|----------------|------|------|
| **UCSC MultiZ 100-way** | 100 vertebrates (hg38 reference) | ~60 GB compressed MAF | Broad phylogenetic coverage; well-established; conservation scores available; Zarr version on HuggingFace | Human-centric (reference-biased); alignment is from 2015 |
| **Zoonomia 241-way** | 241 mammals (Cactus alignment) | ~200+ GB | Densest mammalian sampling; used by Tiberius (ClaMSA); recent (2023) | Mammals only; very large; requires Cactus HAL tools |
| **UCSC MultiZ 470-way** | 470 vertebrates (hg38) | ~150+ GB | Most species; extends 100-way | Very large; many low-quality assemblies |
| **Ensembl Compara EPO** | Variable (up to ~60 amniotes) | ~20-40 GB | High-quality curated alignments; handles rearrangements better | Fewer species; different format (EMF) |
| **Custom Cactus alignment** | User-selected species | Variable | Full control over species set and quality thresholds | Requires running Cactus (compute-intensive) |
| **Tiberius training set** | 37 mammals + NCBI RefSeq annotations | Genome FASTAs + annotations | Direct comparison with Tiberius; well-validated species set | Ab initio only (no MSA); would need separate alignment step |

**Recommendation**: Start with **UCSC MultiZ 100-way** (broad, well-supported, HuggingFace Zarr version available for fast loading). Use **Zoonomia 241-way** as a secondary/ablation dataset for mammalian-specific features. Use the **Tiberius 37-mammal** species set and annotations as our training annotation source for direct benchmarking.

### 3.2 Tiling strategy

- Tile genomes into fixed-length windows (e.g. 100 kb with 10 kb overlap, tunable)
- Each tile contains: MSA slice, annotation labels, optional RNA-Seq tracks
- Tiles are assigned to splits at the region level (see Section 4)
- At training time, sample random sub-windows (e.g. 10 kb) from tiles for efficiency; increase window size during training
- At inference time, process full tiles with overlap-and-stitch (average or max over overlapping predictions)
- Store tiles in a columnar format (e.g. TFRecord, WebDataset, or Zarr) for efficient random access and shuffling

## 4. Train / validation / test split partitioning

The splits must be disjoint along two axes to prevent homology-driven data leakage:

### 4.1 Species-level splitting

- **Training species**: ~30-35 mammals (following Tiberius: well-annotated species with BUSCO >90%)
- **Validation species**: 2 held-out species (e.g. leopard, rat — following Tiberius)
- **Test species**: 3 held-out species (e.g. human, cow, beluga whale — following Tiberius for comparability)
- No orthologous leakage: a species appears in exactly one split

### 4.2 Region-level splitting (within each species)

Within each species, genomic regions are partitioned to avoid homologous leakage between splits used for different purposes (e.g. early stopping on training species, or multi-species evaluation):

1. **Chromosome-level holdout** (baseline approach, following Enformer/Basenji):
   - Hold out specific chromosomes for validation (e.g. chr10) and test (e.g. chr8)
   - Simple, reproducible, standard in the field

2. **Paralog-aware refinement**:
   - Within chromosome-assigned splits, identify cross-split paralogs using sequence similarity clustering (MMseqs2 or CD-HIT at ~80% nucleotide identity)
   - Remove or reassign paralogous regions that span splits
   - Record removed regions for transparency

3. **Segmental duplication filtering**:
   - Mask or exclude known segmental duplications (available from UCSC segDup track) from test/validation to ensure clean evaluation

4. **Repeat masking**:
   - Use soft-masked genomes (RepeatMasker) throughout, following Tiberius
   - Repeat regions are included in training (as they contain real gene structure) but flagged

### 4.3 Split validation

- Compute all-vs-all BLAST or minimap2 between test and training tile sequences
- Report maximum sequence identity between any test tile and any training tile
- Target: no test tile shares >80% identity over >500 bp with any training tile
- Publish split definitions (chromosome assignments, excluded regions, species assignments) for reproducibility

## 5. Systematic training, evaluation, and ablation

### 5.1 Training protocol

1. **Phase 1: Featurization-only baseline** (~2 days, 4x A100)
   - Train a simple model (e.g. 1D CNN + linear head) on phylo features only
   - Purpose: validate featurization pipeline and establish a lower-bound baseline

2. **Phase 2: Full model pre-training** (~7 days, 4x A100)
   - Bidirectional Mamba tower on phylo features, without HMM head
   - Categorical cross-entropy loss on annotation labels
   - Adam optimizer, lr=1e-4, cosine schedule with warmup
   - Validate every 1,000 steps on held-out species

3. **Phase 3: HMM head fine-tuning** (~5 days, 4x A100)
   - Attach Machine Boss HMM decoder
   - Fine-tune end-to-end with reduced learning rate
   - Viterbi decoding for evaluation

4. **Phase 4: RNA-Seq integration** (optional, ~3 days, 4x A100)
   - Add track encoder and cross-attention layers
   - Fine-tune with RNA-Seq tracks for species where available

### 5.2 Evaluation metrics

- **Gene-level F1**: fraction of exactly predicted gene structures (Tiberius primary metric)
- **Exon-level F1**: sensitivity and specificity at exon boundaries
- **Nucleotide-level accuracy**: per-base annotation accuracy
- **Splice site F1**: precision/recall of donor/acceptor predictions
- **BUSCO completeness**: on predicted gene sets
- **Comparison targets**: Tiberius (ab initio and de novo), AUGUSTUS, Helixer, BRAKER3

### 5.3 Ablation studies

Ablations should be run at reduced scale (fewer species, shorter training) to be cost-effective. Each ablation modifies one axis:

| # | Ablation | Purpose |
|---|----------|---------|
| 1 | No phylo features (sequence only) | Quantify value of phylogenetic featurization |
| 2 | HKY85 only (no triplets, phase, annotation) | Quantify value of each feature group |
| 3 | Triplets only | Isolate triplet contribution |
| 4 | Phase only | Isolate phase contribution |
| 5 | No mixture model (single rate category) | Quantify value of rate heterogeneity |
| 6 | K=2,4,8,16 rate categories | Optimize K |
| 7 | F81 vs HKY85 for nucleotide features | Cost/quality tradeoff for substitution model |
| 8 | Vary Mamba layer count / width | Model capacity vs cost |
| 9 | No HMM head (softmax only) | Quantify value of structured decoder |
| 10 | No RNA-Seq cross-attention | Quantify RNA-Seq contribution |
| 11 | MultiZ 100-way vs Zoonomia 241-way | Alignment source comparison |
| 12 | Tile size: 50kb / 100kb / 200kb | Sequence context tradeoff |

### 5.4 Cost-effective ablation strategy

- Use a **reduced training set** (10 species, 5 chromosomes each) for ablations
- Train for **1/5 of full training steps** (sufficient to observe relative differences)
- Use **gene-level F1 on validation species** as the primary comparison metric
- Run ablations in parallel where GPU budget allows
- Estimated cost per ablation: ~150-250 A100-hours
- Total ablation budget: ~2,000-3,000 A100-hours

## 6. In-browser inference benchmarking

### 6.1 Model export pipeline

1. Export trained JAX model weights to ONNX (via jax2onnx or torch intermediary)
2. Quantize to float16 and int8 (ONNX quantization tools)
3. Benchmark accuracy degradation from quantization on validation set
4. Package WebGPU Mamba + phylo shaders and WASM fallbacks

### 6.2 Browser benchmarks

- **Metric**: wall-clock time to annotate a genomic region of fixed size (e.g. 100 kb, 1 Mb)
- **Environments**: Chrome (WebGPU), Firefox (WebGPU/WASM fallback), Safari (WASM)
- **Hardware**: representative consumer hardware (integrated GPU, discrete GPU, CPU-only)
- **Breakdown**: time spent in phylo featurization vs. Mamba inference vs. HMM decoding
- **Comparison**: in-browser vs. native Rust vs. JAX on same hardware
- **Memory profiling**: peak GPU/WASM memory usage per tile size

### 6.3 Targets

- Annotate 100 kb in <5 seconds on a mid-range laptop with WebGPU
- Annotate 100 kb in <30 seconds on CPU-only WASM fallback
- Model download size <50 MB (quantized)

## 7. Write-up

### 7.1 Manuscript structure (`paper/`)

1. **Introduction**: motivation for phylogenetically-informed gene annotation; limitations of current approaches; in-browser deployment as a novel contribution
2. **Methods**:
   - Phylogenetic sufficient statistics and eigensubstitution algorithm
   - Featurization schemes (HKY85 mixture, triplets, phase, annotation transfer)
   - Model architecture (bidirectional Mamba, cross-attention, HMM decoder)
   - Data pipeline, tiling, and split methodology
   - In-browser inference system (WebGPU/WASM)
3. **Results**:
   - Comparison with Tiberius, AUGUSTUS, Helixer, BRAKER3
   - Ablation studies
   - In-browser performance benchmarks
4. **Discussion**: interpretation of ablations; when phylo features help most; practical deployment scenarios; limitations
5. **Data and code availability**: all code, trained models, split definitions, and browser demo

### 7.2 Supplementary materials

- Full species lists and split assignments
- Per-species evaluation results
- Extended ablation tables
- Browser benchmark details across hardware configurations

### 7.3 Target venues

- Primary: *Bioinformatics* or *Genome Research* (peer-reviewed, where Tiberius was published)
- Alternative: *Nature Methods* (if results are sufficiently strong)
- Preprint: bioRxiv, concurrent with submission

## 8. Timeline (approximate)

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phylo library (JAX) | 3-4 weeks | — |
| Phylo library (WebGPU/WASM/Rust) | 3-4 weeks | JAX impl for test oracle |
| Data pipeline + tiling + splits | 2-3 weeks | Phylo library (JAX) |
| Model implementation (JAX) | 2-3 weeks | — |
| Data staging + preprocessing | 1-2 weeks | Data pipeline |
| Training (full + ablations) | 4-6 weeks | Model + data pipeline |
| Browser inference runtime | 2-3 weeks | Phylo (WebGPU/WASM) + trained model |
| Evaluation + benchmarking | 2-3 weeks | Trained model + browser runtime |
| Write-up | 3-4 weeks | All above |
| **Total** | **~22-32 weeks** | |

Many phases overlap (e.g. model implementation and data pipeline are independent; browser runtime can start once phylo WebGPU is done and model architecture is frozen).
