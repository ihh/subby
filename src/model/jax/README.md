# src/model/jax/

JAX/Flax implementation of the subby gene annotation model.

## Modules

- `tower.py` — `SubbyModel` (top-level), `MambaTower`, `TrackEncoder`, `TrackCrossAttention`
- `selectssm.py` — `BidirectionalMamba`, `SelectiveSSM` (selective state space layers)
- `ssmrecscan.py` — Memory-efficient recursive SSM scan with custom VJP

See `docs/api/model.md` for full API reference. Tests in `tests/test_model/test_tower.py`.
