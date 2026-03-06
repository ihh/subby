#!/usr/bin/env python3
"""Benchmark suite for eigensubstitution accumulation.

Times LogLike() and Counts() across backends, alphabet sizes,
alignment widths, and tree depths. Outputs JSON results keyed
by hardware ID.

Usage:
    python run_benchmarks.py [--backends jax_cpu,oracle] [--dry-run] [--reps N]
    python run_benchmarks.py --tables-only
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _cpu_model():
    """Best-effort CPU model string."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
            return out
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _collect_machine_stats():
    """Collect platform/hardware stats for the report."""
    stats = {}
    stats["uname"] = platform.uname()._asdict()
    stats["platform"] = platform.platform()
    stats["python_version"] = platform.python_version()

    if platform.system() == "Darwin":
        keys = [
            "hw.cpufrequency", "hw.cpufrequency_max", "hw.ncpu",
            "hw.physicalcpu", "hw.logicalcpu", "hw.memsize",
            "machdep.cpu.brand_string",
        ]
        for key in keys:
            try:
                val = subprocess.check_output(
                    ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL,
                ).strip()
                stats[key] = val
            except Exception:
                pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        stats["cpu_model"] = line.split(":", 1)[1].strip()
                        break
            stats["cpu_count"] = str(os.cpu_count())
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        stats["mem_total"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    try:
        uname_full = subprocess.check_output(["uname", "-a"], text=True).strip()
        stats["uname_string"] = uname_full
    except Exception:
        pass

    return stats


def _gpu_model():
    """Return GPU model if JAX sees one, else None."""
    try:
        import jax
        devs = jax.devices("gpu")
        if devs:
            return str(devs[0])
    except Exception:
        pass
    return None


def hardware_id():
    host = platform.node()
    cpu = _cpu_model()
    gpu = _gpu_model()
    parts = [host, cpu]
    if gpu:
        parts.append(gpu)
    return " / ".join(parts)


# ---------------------------------------------------------------------------
# Data generation (mirrors test patterns)
# ---------------------------------------------------------------------------

def make_tree(R, seed=0):
    """Balanced binary tree with R nodes (preorder)."""
    rng = np.random.RandomState(seed)
    parent_index = np.zeros(R, dtype=np.int32)
    parent_index[0] = -1
    for i in range(1, R):
        parent_index[i] = (i - 1) // 2
    distances = rng.uniform(0.01, 0.5, size=R).astype(np.float64)
    distances[0] = 0.0
    return parent_index, distances


def make_alignment(R, C, A, seed=1):
    """Random alignment with occasional gaps (-1)."""
    rng = np.random.RandomState(seed)
    alignment = rng.randint(0, A + 2, size=(R, C)).astype(np.int32)
    alignment[alignment >= A] = -1
    return alignment


# ---------------------------------------------------------------------------
# Backend wrappers
# ---------------------------------------------------------------------------

def _make_irrev_rate_matrix(A, seed=42):
    """Random valid irreversible rate matrix: off-diag >= 0, rows sum to 0."""
    rng = np.random.RandomState(seed)
    R = rng.uniform(0.01, 1.0, (A, A))
    np.fill_diagonal(R, 0.0)
    np.fill_diagonal(R, -R.sum(axis=1))
    pi = np.ones(A) / A
    rate = -np.sum(pi * np.diag(R))
    R /= rate
    return R, pi


def _get_jax_backend():
    """Return (fn_map, make_tree_fn, make_model_fns) for JAX."""
    import jax
    import jax.numpy as jnp
    from subby.jax import LogLike, Counts, BranchCounts, LogLikeCustomGrad
    from subby.jax.types import Tree
    from subby.jax.models import jukes_cantor_model, irrev_model_from_rate_matrix

    def make_tree_jax(parent_index, distances):
        return Tree(
            parentIndex=jnp.array(parent_index),
            distanceToParent=jnp.array(distances),
        )

    def make_model_jax(A, model_type="reversible"):
        if model_type == "irreversible":
            R_mat, pi = _make_irrev_rate_matrix(A)
            return irrev_model_from_rate_matrix(jnp.array(R_mat), jnp.array(pi))
        return jukes_cantor_model(A)

    def make_alignment_jax(alignment):
        return jnp.array(alignment)

    return LogLike, Counts, BranchCounts, make_tree_jax, make_model_jax, make_alignment_jax


def _get_oracle_backend():
    """Return (LogLike, Counts, make_tree_fn, make_model_fns) for oracle."""
    from subby.oracle import LogLike, Counts, BranchCounts
    from subby.oracle import jukes_cantor_model as oracle_jc
    from subby.oracle import irrev_model_from_rate_matrix as oracle_irrev

    def make_tree_oracle(parent_index, distances):
        return {
            "parentIndex": parent_index,
            "distanceToParent": distances,
        }

    def make_model_oracle(A, model_type="reversible"):
        if model_type == "irreversible":
            R_mat, pi = _make_irrev_rate_matrix(A)
            return oracle_irrev(R_mat, pi)
        return oracle_jc(A)

    def make_alignment_oracle(alignment):
        return alignment

    return LogLike, Counts, BranchCounts, make_tree_oracle, make_model_oracle, make_alignment_oracle


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_fn(fn, n_reps, timeout=None):
    """Time fn() over n_reps calls. Returns (mean_s, std_s).

    If timeout is set and any single call exceeds it, remaining reps
    are skipped but results from completed reps are still returned.
    """
    times = []
    for i in range(n_reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        if timeout and elapsed > timeout:
            break
    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "A": [4, 20, 64],
    "C": [100, 1000, 10000],
    "R": [7, 15, 31],
}

FUNCTIONS = ["LogLike", "Counts", "BranchCounts", "LogLikeGrad", "LogLikeCustomGrad", "ExpectedCounts"]
MODEL_TYPES = ["reversible", "irreversible"]


def run_benchmarks(backends, n_reps, dry_run=False, timeout=60.0, model_types=None):
    """Run the full benchmark grid. Returns list of result dicts.

    Args:
        timeout: Max seconds per single call. If a probe call exceeds
            this, the config is recorded with probe timing and larger
            configs for the same (backend, function, A) are skipped.
        model_types: List of model types to benchmark (default: MODEL_TYPES).
    """
    if model_types is None:
        model_types = MODEL_TYPES
    results = []
    hw = hardware_id()
    total = (
        len(PARAM_GRID["A"])
        * len(PARAM_GRID["C"])
        * len(PARAM_GRID["R"])
        * len(backends)
        * len(FUNCTIONS)
        * len(model_types)
    )
    done = 0

    for backend_name in backends:
        # Load backend
        if backend_name == "jax_cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
            try:
                ll_fn, counts_fn, bc_fn, mk_tree, mk_model, mk_align = _get_jax_backend()
            except ImportError as e:
                print(f"Skipping {backend_name}: {e}")
                done += len(PARAM_GRID["A"]) * len(PARAM_GRID["C"]) * len(PARAM_GRID["R"]) * len(FUNCTIONS) * len(model_types)
                continue
        elif backend_name == "jax_gpu":
            os.environ["JAX_PLATFORMS"] = "gpu"
            try:
                import jax
                if not jax.devices("gpu"):
                    raise RuntimeError("No GPU devices")
                ll_fn, counts_fn, bc_fn, mk_tree, mk_model, mk_align = _get_jax_backend()
            except Exception as e:
                print(f"Skipping {backend_name}: {e}")
                done += len(PARAM_GRID["A"]) * len(PARAM_GRID["C"]) * len(PARAM_GRID["R"]) * len(FUNCTIONS) * len(model_types)
                continue
        elif backend_name == "oracle":
            try:
                ll_fn, counts_fn, bc_fn, mk_tree, mk_model, mk_align = _get_oracle_backend()
            except ImportError as e:
                print(f"Skipping {backend_name}: {e}")
                done += len(PARAM_GRID["A"]) * len(PARAM_GRID["C"]) * len(PARAM_GRID["R"]) * len(FUNCTIONS) * len(model_types)
                continue
        else:
            print(f"Unknown backend: {backend_name}")
            continue

        # Track which (func, A, model_type) combos have hit the timeout wall,
        # so we can skip larger C/R values.
        timed_out = set()

        for model_type in model_types:
            for A in PARAM_GRID["A"]:
                model = mk_model(A, model_type)
                for R in PARAM_GRID["R"]:
                    parent_index, distances = make_tree(R)
                    tree = mk_tree(parent_index, distances)
                    for C in PARAM_GRID["C"]:
                        alignment_np = make_alignment(R, C, A)
                        alignment = mk_align(alignment_np)
                        for func_name in FUNCTIONS:
                            done += 1
                            mt_tag = "irrev" if model_type == "irreversible" else "rev"
                            label = f"[{done}/{total}] {backend_name} {func_name} {mt_tag} A={A} C={C} R={R}"

                            if dry_run:
                                print(f"  DRY RUN: {label}")
                                results.append({
                                    "backend": backend_name,
                                    "function": func_name,
                                    "model_type": model_type,
                                    "A": A,
                                    "C": C,
                                    "R": R,
                                    "mean_seconds": 0.0,
                                    "std_seconds": 0.0,
                                    "n_reps": n_reps,
                                    "hardware_id": hw,
                                })
                                continue

                            # Skip if a smaller config already timed out
                            skip_key = (func_name, A, model_type)
                            if skip_key in timed_out:
                                print(f"  {label} ... SKIPPED (smaller config exceeded {timeout}s)")
                                continue

                            fn_map = {"LogLike": ll_fn, "Counts": counts_fn, "BranchCounts": bc_fn}

                            # ExpectedCounts: standalone (no alignment/tree), only JAX/oracle
                            if func_name == "ExpectedCounts":
                                if backend_name not in ("jax_cpu", "jax_gpu", "oracle"):
                                    print(f"  {label} ... SKIPPED (ExpectedCounts requires JAX or oracle)")
                                    continue
                                if backend_name.startswith("jax"):
                                    from subby.jax import ExpectedCounts as _EC
                                else:
                                    from subby.oracle import ExpectedCounts as _EC
                                _model_for_ec = model
                                def _ec_fn(_model=_model_for_ec, _A=A):
                                    return _EC(_model, 0.3)
                                fn = None
                                _grad_fn = _ec_fn

                            # For gradient functions, only available with JAX
                            elif func_name in ("LogLikeGrad", "LogLikeCustomGrad") and backend_name not in ("jax_cpu", "jax_gpu"):
                                print(f"  {label} ... SKIPPED (gradient fn requires JAX)")
                                continue

                            if func_name == "ExpectedCounts":
                                pass  # already handled above
                            elif func_name == "LogLikeGrad":
                                import jax
                                import jax.numpy as jnp
                                from subby.jax.types import Tree as JTree
                                _parent_idx = tree.parentIndex if hasattr(tree, 'parentIndex') else tree['parentIndex']
                                def _grad_fn(alignment=alignment, tree=tree, model=model):
                                    def loss(d):
                                        t = JTree(parentIndex=_parent_idx, distanceToParent=d)
                                        return jnp.sum(ll_fn(alignment, t, model))
                                    _d = tree.distanceToParent if hasattr(tree, 'distanceToParent') else tree['distanceToParent']
                                    return jax.grad(loss)(_d)
                                fn = None  # use _grad_fn below
                            elif func_name == "LogLikeCustomGrad":
                                import jax
                                import jax.numpy as jnp
                                from subby.jax import LogLikeCustomGrad as LLCG
                                from subby.jax.types import Tree as JTree
                                _parent_idx = tree.parentIndex if hasattr(tree, 'parentIndex') else tree['parentIndex']
                                def _grad_fn(alignment=alignment, tree=tree, model=model):
                                    def loss(d):
                                        t = JTree(parentIndex=_parent_idx, distanceToParent=d)
                                        return jnp.sum(LLCG(alignment, t, model))
                                    _d = tree.distanceToParent if hasattr(tree, 'distanceToParent') else tree['distanceToParent']
                                    return jax.grad(loss)(_d)
                                fn = None  # use _grad_fn below
                            else:
                                fn = fn_map[func_name]
                                _grad_fn = None

                            call_fn = _grad_fn if _grad_fn is not None else lambda: fn(alignment, tree, model)

                            # Warmup / probe: single call to get rough timing
                            print(f"  {label} ...", end="", flush=True)
                            try:
                                t0 = time.perf_counter()
                                call_fn()
                                probe_time = time.perf_counter() - t0
                            except Exception as e:
                                print(f" ERROR: {e}")
                                continue

                            if probe_time > timeout:
                                print(f" {probe_time:.2f}s (probe > {timeout}s, skipping larger)")
                                timed_out.add(skip_key)
                                results.append({
                                    "backend": backend_name,
                                    "function": func_name,
                                    "model_type": model_type,
                                    "A": A,
                                    "C": C,
                                    "R": R,
                                    "mean_seconds": probe_time,
                                    "std_seconds": 0.0,
                                    "n_reps": 1,
                                    "hardware_id": hw,
                                })
                                continue

                            # Full timing (probe counts as warmup for JAX)
                            try:
                                mean_s, std_s = time_fn(
                                    call_fn,
                                    n_reps,
                                    timeout=timeout,
                                )
                            except Exception as e:
                                print(f" ERROR: {e}")
                                continue

                            print(f" {mean_s:.4f} ± {std_s:.4f} s")
                            results.append({
                                "backend": backend_name,
                                "function": func_name,
                                "model_type": model_type,
                                "A": A,
                                "C": C,
                                "R": R,
                                "mean_seconds": mean_s,
                                "std_seconds": std_s,
                                "n_reps": n_reps,
                                "hardware_id": hw,
                            })

    return results


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_results(results, out_dir="results"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    hostname = platform.node() or "unknown"
    filepath = out_path / f"{hostname}.json"
    data = {
        "hardware_id": results[0]["hardware_id"] if results else hardware_id(),
        "hostname": hostname,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine_stats": _collect_machine_stats(),
        "results": results,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filepath}")
    return filepath


def load_all_results(results_dir="results"):
    """Load all JSON result files. Returns list of record dicts."""
    records = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return records
    for fp in sorted(results_path.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        for r in data["results"]:
            r.setdefault("hardware_id", data.get("hardware_id", fp.stem))
            records.append(r)
    return records


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def _tex_escape(s):
    """Escape special LaTeX characters in a string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def _generate_machine_stats_table(results_dir, tables_dir):
    """Generate a LaTeX table fragment with machine stats per host."""
    results_path = Path(results_dir)
    tables_path = Path(tables_dir)
    tables_path.mkdir(parents=True, exist_ok=True)

    host_stats = {}
    for fp in sorted(results_path.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        hw_id = data.get("hardware_id", fp.stem)
        stats = data.get("machine_stats", {})
        host_stats[hw_id] = stats

    if not host_stats:
        return

    # Determine display rows
    display_keys = [
        ("uname_string", "System"),
        ("machdep.cpu.brand_string", "CPU"),
        ("cpu_model", "CPU"),
        ("hw.physicalcpu", "Physical cores"),
        ("hw.logicalcpu", "Logical cores"),
        ("cpu_count", "CPU count"),
        ("hw.cpufrequency_max", "CPU frequency (Hz)"),
        ("hw.memsize", "Memory (bytes)"),
        ("mem_total", "Memory"),
        ("python_version", "Python"),
        ("platform", "Platform"),
    ]

    filepath = tables_path / "machine_stats.tex"
    lines = []
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l p{0.7\textwidth}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Property} & \textbf{Value} \\")
    lines.append(r"\midrule")

    for hw_id, stats in host_stats.items():
        lines.append(r"\multicolumn{2}{l}{\textbf{" + _tex_escape(hw_id) + r"}} \\")
        lines.append(r"\midrule")
        seen_labels = set()
        for key, label in display_keys:
            if key in stats and label not in seen_labels:
                val = str(stats[key])
                # Format large numbers
                if key == "hw.memsize":
                    try:
                        gb = int(val) / (1024 ** 3)
                        val = f"{gb:.0f} GB"
                    except ValueError:
                        pass
                elif key == "hw.cpufrequency_max":
                    try:
                        ghz = int(val) / 1e9
                        val = f"{ghz:.2f} GHz"
                    except ValueError:
                        pass
                lines.append(f"{_tex_escape(label)} & {_tex_escape(val)} \\\\")
                seen_labels.add(label)
        lines.append(r"\midrule")

    # Remove trailing midrule
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Machine stats table written: {filepath}")


def generate_tables(results_dir="results", tables_dir="tables"):
    """Generate LaTeX table fragments from results JSON."""
    records = load_all_results(results_dir)
    if not records:
        print("No results found — skipping table generation.")
        return

    tables_path = Path(tables_dir)
    tables_path.mkdir(parents=True, exist_ok=True)

    _generate_machine_stats_table(results_dir, tables_dir)

    # Group by hardware
    hw_map = {}
    for r in records:
        hw = r["hardware_id"]
        hw_map.setdefault(hw, []).append(r)

    all_backends = sorted({r["backend"] for r in records})

    all_model_types = sorted({r.get("model_type", "reversible") for r in records})

    for hw, hw_records in hw_map.items():
        # Build lookup: (func, model_type, A, C, R, backend) -> (mean, std)
        lookup = {}
        for r in hw_records:
            mt = r.get("model_type", "reversible")
            key = (r["function"], mt, r["A"], r["C"], r["R"], r["backend"])
            lookup[key] = (r["mean_seconds"], r["std_seconds"])

        for func_name in FUNCTIONS:
            for model_type in all_model_types:
                mt_tag = "irrev" if model_type == "irreversible" else "rev"
                safe_hw = hw.replace("/", "-").replace(" ", "_")[:40]
                filename = f"{safe_hw}_{func_name}_{mt_tag}.tex"
                filepath = tables_path / filename

                lines = []
                col_spec = "rrr" + "r" * len(all_backends)
                lines.append(r"\begin{tabular}{" + col_spec + "}")
                lines.append(r"\toprule")
                header = r"$A$ & $C$ & $R$ & " + " & ".join(
                    r"\texttt{" + b.replace("_", r"\_") + "}" for b in all_backends
                )
                lines.append(header + r" \\")
                lines.append(r"\midrule")

                has_data = False
                for A in PARAM_GRID["A"]:
                    for C in PARAM_GRID["C"]:
                        for R in PARAM_GRID["R"]:
                            cells = [str(A), str(C), str(R)]
                            for b in all_backends:
                                key = (func_name, model_type, A, C, R, b)
                                if key in lookup:
                                    has_data = True
                                    mean, std = lookup[key]
                                    if mean < 0.01:
                                        cells.append(f"{mean:.4f} $\\pm$ {std:.4f}")
                                    elif mean < 1.0:
                                        cells.append(f"{mean:.3f} $\\pm$ {std:.3f}")
                                    else:
                                        cells.append(f"{mean:.2f} $\\pm$ {std:.2f}")
                                else:
                                    cells.append("---")
                            lines.append(" & ".join(cells) + r" \\")

                lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")

                if has_data:
                    with open(filepath, "w") as f:
                        f.write("\n".join(lines) + "\n")
                    print(f"Table written: {filepath}")

    # Generate include files for LaTeX
    for func_name in FUNCTIONS:
        for model_type in all_model_types:
            mt_tag = "irrev" if model_type == "irreversible" else "rev"
            mt_label = "irreversible" if model_type == "irreversible" else "reversible"
            include_lines = []
            for hw in hw_map:
                safe_hw = hw.replace("/", "-").replace(" ", "_")[:40]
                filename = f"{safe_hw}_{func_name}_{mt_tag}.tex"
                tex_path = tables_path / filename
                if tex_path.exists():
                    include_lines.append(
                        r"\begin{table}[H]"
                        "\n" r"\centering"
                        "\n" r"\caption{\texttt{" + func_name + r"()} " + mt_label + r" timings (seconds, mean $\pm$ std).}"
                        "\n" r"\small"
                        "\n" r"\input{tables/" + filename + "}"
                        "\n" r"\end{table}"
                        "\n"
                    )
            inc_file = tables_path / f"{func_name.lower()}_{mt_tag}_includes.tex"
            with open(inc_file, "w") as f:
                f.write("\n".join(include_lines) + "\n")
            print(f"Include file written: {inc_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eigensubstitution benchmark suite")
    parser.add_argument(
        "--backends",
        default="jax_cpu,oracle",
        help="Comma-separated backends to benchmark (default: jax_cpu,oracle)",
    )
    parser.add_argument(
        "--reps", type=int, default=5, help="Number of timing repetitions (default: 5)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print config without running"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Max seconds per single call before skipping larger configs (default: 60)",
    )
    parser.add_argument(
        "--model-types",
        default="reversible,irreversible",
        help="Comma-separated model types to benchmark (default: reversible,irreversible)",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Generate LaTeX tables from existing results (no benchmarking)",
    )
    args = parser.parse_args()

    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    tables_dir = script_dir / "tables"

    if args.tables_only:
        generate_tables(results_dir, tables_dir)
        return

    backends = [b.strip() for b in args.backends.split(",")]
    model_types = [m.strip() for m in args.model_types.split(",")]
    print(f"Hardware: {hardware_id()}")
    print(f"Backends: {backends}")
    print(f"Model types: {model_types}")
    print(f"Reps: {args.reps}")
    print(f"Grid: A={PARAM_GRID['A']} C={PARAM_GRID['C']} R={PARAM_GRID['R']}")
    print()

    results = run_benchmarks(backends, args.reps, dry_run=args.dry_run, timeout=args.timeout,
                             model_types=model_types)
    save_results(results, results_dir)
    generate_tables(results_dir, tables_dir)


if __name__ == "__main__":
    main()
