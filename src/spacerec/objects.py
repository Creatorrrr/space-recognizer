"""Object 3D localization (Phase 2) and the persistent world registry (Phase 4)."""

from __future__ import annotations

import numpy as np

from .detect import Detection


def localize_objects(detections: list[Detection], depth: np.ndarray,
                     K: np.ndarray, T_wc: np.ndarray
                     ) -> list[tuple[Detection, np.ndarray]]:
    """Mask-interior depth median -> camera-frame 3D -> world-frame position."""
    results = []
    for det in detections:
        if det.mask is not None and det.mask.any():
            ys, xs = np.nonzero(det.mask)
            z = float(np.median(depth[ys, xs]))
            u, v = float(np.median(xs)), float(np.median(ys))
        else:
            x0, y0, x1, y1 = det.box
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            z = float(depth[int(np.clip(v, 0, depth.shape[0] - 1)),
                            int(np.clip(u, 0, depth.shape[1] - 1))])
        if z <= 1e-6:
            continue
        cam = np.array([(u - K[0, 2]) / K[0, 0] * z,
                        (v - K[1, 2]) / K[1, 1] * z,
                        z])
        world = T_wc[:3, :3] @ cam + T_wc[:3, 3]
        results.append((det, world))
    return results
