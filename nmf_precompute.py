"""Precompute and cache dual-branch NMF features from NumPy matrix sequences."""

import argparse

import numpy as np

from rtqp_phase1.nmf import DualBranchNMFExtractor, NMFParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qos-type", choices=("rt", "tp", "synthetic"), required=True)
    parser.add_argument("--original", required=True, help="[time,user,service] .npy")
    parser.add_argument("--enhanced", required=True, help="[time,user,service] .npy")
    parser.add_argument("--mask", required=True, help="Boolean [time,user,service] .npy")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--tol", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-slices", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    extractor = DualBranchNMFExtractor(
        NMFParameters(rank=args.rank, max_iter=args.max_iter, tol=args.tol, seed=args.seed),
        cache_dir=args.cache_dir,
        qos_type=args.qos_type,
    )
    results = extractor.precompute(
        np.load(args.original, mmap_mode="r"),
        np.load(args.enhanced, mmap_mode="r"),
        np.load(args.mask, mmap_mode="r"),
        time_slices=args.time_slices,
        force=args.force,
    )
    for time, branches in results.items():
        print(
            f"t={time} original_cache={branches.original.loaded_from_cache} "
            f"enhanced_cache={branches.enhanced.loaded_from_cache}"
        )


if __name__ == "__main__":
    main()
