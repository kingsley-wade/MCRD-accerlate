"""Compare package solver output with legacy OpenCL implementation.

Since the OpenCL kernel uses in-place updates (potential race conditions), we compare
statistical properties and pattern morphology rather than pixel-by-pixel equality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from mcrd_accel.config import SolverConfig
from mcrd_accel.mcrd import run


def compute_statistics(field: np.ndarray) -> dict:
    """Compute summary statistics for a field."""
    return {
        "mean": float(np.mean(field)),
        "std": float(np.std(field)),
        "min": float(np.min(field)),
        "max": float(np.max(field)),
        "median": float(np.median(field)),
    }


def compute_spatial_correlation(field: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Compute radial spatial correlation up to max_lag pixels."""
    # Simplified: compute 1D autocorrelation along x and y, then average
    nx, ny = field.shape
    corr_x = np.correlate(field[ny // 2, :], field[ny // 2, :], mode="full")
    corr_y = np.correlate(field[:, nx // 2], field[:, nx // 2], mode="full")

    # Normalize
    corr_x = corr_x / corr_x[len(corr_x) // 2]
    corr_y = corr_y / corr_y[len(corr_y) // 2]

    # Take center portion up to max_lag
    center = len(corr_x) // 2
    corr = (corr_x[center : center + max_lag] + corr_y[center : center + max_lag]) / 2
    return corr


def try_load_opencl_results(opencl_dir: Path) -> dict | None:
    """Try to load OpenCL results from the paper repository.

    Returns None if OpenCL data is not available.
    """
    # Look for saved .mat files or .npy files
    mat_files = list(opencl_dir.glob("*.mat"))
    npy_files = list(opencl_dir.glob("*.npy"))

    if not mat_files and not npy_files:
        return None

    # Try to load .mat files
    try:
        from scipy import io

        data = {}
        for mat_file in mat_files:
            mat_data = io.loadmat(mat_file)
            # Look for u and v arrays
            for key in ["u", "v", "Us", "Vs"]:
                if key in mat_data:
                    data[key] = mat_data[key]
        return data if data else None
    except ImportError:
        return None


def compare_fields(our_u: np.ndarray, our_v: np.ndarray, ref_u: np.ndarray, ref_v: np.ndarray) -> dict:
    """Compare our solver output with reference (OpenCL) output."""
    if our_u.shape != ref_u.shape:
        return {"error": f"Shape mismatch: {our_u.shape} vs {ref_u.shape}"}

    # Pointwise difference (note: not expected to match exactly due to OpenCL races)
    diff_u = our_u - ref_u
    diff_v = our_v - ref_v

    return {
        "u": {
            "mean_abs_diff": float(np.mean(np.abs(diff_u))),
            "max_abs_diff": float(np.max(np.abs(diff_u))),
            "rms_diff": float(np.sqrt(np.mean(diff_u**2))),
            "our_stats": compute_statistics(our_u),
            "ref_stats": compute_statistics(ref_u),
        },
        "v": {
            "mean_abs_diff": float(np.mean(np.abs(diff_v))),
            "max_abs_diff": float(np.max(np.abs(diff_v))),
            "rms_diff": float(np.sqrt(np.mean(diff_v**2))),
            "our_stats": compute_statistics(our_v),
            "ref_stats": compute_statistics(ref_v),
        },
        "note": "OpenCL kernel has in-place updates (potential races). Exact pixel match is not expected.",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare MCRD solver with OpenCL")
    parser.add_argument("--Nx", type=int, default=128, help="Grid size")
    parser.add_argument("--n-steps", type=int, default=1000, help="Total steps")
    parser.add_argument("--opencl-dir", type=Path, help="Path to OpenCL results directory")
    parser.add_argument("--save", type=Path, help="Save comparison report to JSON")
    args = parser.parse_args()

    print("MCRD OpenCL Comparison")
    print("=" * 50)

    # Run our solver
    config = SolverConfig.fig2b(Nx=args.Nx, Ny=args.Nx, n_steps=args.n_steps)
    print(f"Running package solver: {config.Nx}x{config.Ny}, {config.n_steps} steps...")
    result = run(config, checkpoint_stride=0)

    our_u = result.final_state.u
    our_v = result.final_state.v

    print("\nOur solver final state:")
    print(f"  u: mean={our_u.mean():.4f}, std={our_u.std():.4f}")
    print(f"  v: mean={our_v.mean():.4f}, std={our_v.std():.4f}")

    # Try to load OpenCL results
    report = {
        "config": config.to_dict(),
        "our_solver": {
            "u_stats": compute_statistics(our_u),
            "v_stats": compute_statistics(our_v),
            "u_correlation": compute_spatial_correlation(our_u).tolist(),
            "v_correlation": compute_spatial_correlation(our_v).tolist(),
        },
    }

    if args.opencl_dir and args.opencl_dir.exists():
        print(f"\nLoading OpenCL results from {args.opencl_dir}...")
        opencl_data = try_load_opencl_results(args.opencl_dir)

        if opencl_data:
            # Extract u and v from OpenCL data
            # This depends on the exact format; adjust as needed
            ref_u = opencl_data.get("u", opencl_data.get("Us"))
            ref_v = opencl_data.get("v", opencl_data.get("Vs"))

            if ref_u is not None and ref_v is not None:
                # Flatten if needed
                if ref_u.ndim == 3:
                    ref_u = ref_u[:, :, -1]  # Take last time slice
                if ref_v.ndim == 3:
                    ref_v = ref_v[:, :, -1]

                print(f"OpenCL reference loaded: u shape {ref_u.shape}, v shape {ref_v.shape}")

                comparison = compare_fields(our_u, our_v, ref_u, ref_v)
                report["comparison"] = comparison

                print("\nComparison:")
                print(f"  u mean abs diff: {comparison['u']['mean_abs_diff']:.4f}")
                print(f"  u max abs diff:  {comparison['u']['max_abs_diff']:.4f}")
                print(f"  v mean abs diff: {comparison['v']['mean_abs_diff']:.4f}")
                print(f"  v max abs diff:  {comparison['v']['max_abs_diff']:.4f}")
                print(f"\n  Note: {comparison['note']}")
            else:
                print("  Could not extract u/v from OpenCL data")
                report["opencl_note"] = "Data format not recognized"
        else:
            print("  No OpenCL results found")
            report["opencl_note"] = "No .mat or .npy files found"
    else:
        print("\nNo OpenCL directory specified. Skipping comparison.")
        report["opencl_note"] = "No reference provided"

    # Save report
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.save}")

    print("\nDone!")


if __name__ == "__main__":
    main()
