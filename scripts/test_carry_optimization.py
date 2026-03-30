"""Test whether jax.lax.scan optimizes carry to in-place updates.

If the carry is copied each step, doubling the padding dimension should
roughly double the scan time. If XLA does in-place scatter updates,
the time should be insensitive to the padding size.
"""
import jax
import jax.numpy as jnp
import time

R = 63  # number of nodes (like a phylogenetic tree)
A = 4   # alphabet size


def make_scan(C):
    """Create a scan that does D.at[node].set(value) on an (R, C, A) carry."""
    parent_of = jnp.zeros(R - 1, dtype=jnp.int32)  # dummy parents (all root)
    for i in range(1, R):
        parent_of = parent_of.at[i - 1].set((i - 1) // 2)

    nodes = jnp.arange(1, R, dtype=jnp.int32)

    def step(carry, xs):
        D = carry
        node, parent = xs

        # Read parent's value from carry (like the downward pass)
        parent_D = D[parent, :, :]  # (C, A)

        # Some computation
        new_val = parent_D * 0.99 + 0.01

        # Write to node (scatter update)
        D = D.at[node, :, :].set(new_val)

        return D, None

    @jax.jit
    def run(D_init):
        D, _ = jax.lax.scan(step, D_init, (nodes, parent_of))
        return D

    return run


def benchmark(fn, D_init, warmup=3, trials=20):
    """Benchmark a JIT-compiled function."""
    # Warmup (includes compilation)
    for _ in range(warmup):
        result = fn(D_init)
        result.block_until_ready()

    # Timed trials
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        result = fn(D_init)
        result.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sorted(times)


print(f"Platform: {jax.default_backend()}")
print(f"R={R}, A={A}")
print(f"{'C':>6}  {'carry size':>12}  {'median ms':>10}  {'min ms':>10}")
print("-" * 50)

for C in [1, 4, 16, 64, 128, 256, 512, 1024]:
    fn = make_scan(C)
    D_init = jnp.ones((R, C, A))
    times = benchmark(fn, D_init)
    median = times[len(times) // 2] * 1000
    minimum = times[0] * 1000
    carry_floats = R * C * A
    print(f"{C:>6}  {carry_floats:>12}  {median:>10.3f}  {minimum:>10.3f}")
