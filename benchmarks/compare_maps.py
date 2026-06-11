"""두 headless 지도(npz)의 구조 비교: 로버스트 extent, 상호 커버리지, 궤적.

Usage: python benchmarks/compare_maps.py A.npz B.npz [--radius 0.04]
"""

import argparse

import numpy as np
from scipy.spatial import cKDTree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--radius", type=float, default=0.04)
    args = ap.parse_args()

    da, db = np.load(args.a), np.load(args.b)
    for name, d in [(args.a, da), (args.b, db)]:
        pts = d["pts"]
        lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
        ext = hi - lo
        print(f"{name}: pts={len(pts):7d} "
              f"robust_extent=({ext[0]:.2f},{ext[1]:.2f},{ext[2]:.2f})")

    ta, tb = da["traj"], db["traj"]
    n = min(len(ta), len(tb))
    dt = np.linalg.norm(ta[:n] - tb[:n], axis=1)
    print(f"traj diff: median={np.median(dt):.4f} max={dt.max():.4f} (n={n})")

    tree_a, tree_b = cKDTree(da["pts"]), cKDTree(db["pts"])
    r = args.radius
    cov_a = (tree_b.query(da["pts"], k=1)[0] < r).mean()
    cov_b = (tree_a.query(db["pts"], k=1)[0] < r).mean()
    print(f"{args.a} covered by {args.b} (r={r}): {cov_a:.1%}")
    print(f"{args.b} covered by {args.a} (r={r}): {cov_b:.1%}")


if __name__ == "__main__":
    main()
