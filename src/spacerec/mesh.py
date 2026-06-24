"""TSDF-backed mesh submaps.

Meshes are treated as rebuildable artifacts derived from RGB-D keyframe
evidence.  Geometry is stored in a submap-local frame and correction happens by
updating the submap anchor transform or by rebuilding the affected submap from
its retained views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import MeshCfg
from .geometry import SIM3_IDENTITY, Sim3, sim3_apply, sim3_on_pose


@dataclass
class TriMesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    colors: np.ndarray

    @classmethod
    def empty(cls) -> "TriMesh":
        return cls(
            vertices=np.empty((0, 3), np.float32),
            faces=np.empty((0, 3), np.int32),
            normals=np.empty((0, 3), np.float32),
            colors=np.empty((0, 3), np.uint8),
        )

    @property
    def n_vertices(self) -> int:
        return int(len(self.vertices))

    @property
    def n_faces(self) -> int:
        return int(len(self.faces))


@dataclass
class MeshView:
    depth: np.ndarray
    color: np.ndarray
    valid: np.ndarray
    K: np.ndarray
    T_wc: np.ndarray
    kf_id: int = -1


@dataclass
class MeshSubmap:
    submap_id: int
    anchor_pose: np.ndarray
    anchor_scale: float = 1.0
    window_ids: list[int] = field(default_factory=list)
    views: list[MeshView] = field(default_factory=list)
    mesh: TriMesh = field(default_factory=TriMesh.empty)
    version: int = 0
    pose_version: int = 0
    dirty: bool = True

    def local_pose_for(self, T_wc: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.anchor_pose) @ T_wc

    def global_vertices(self) -> np.ndarray:
        if self.mesh.n_vertices == 0:
            return np.empty((0, 3), np.float32)
        pts = self.mesh.vertices.astype(np.float64)
        return (self.anchor_scale * (pts @ self.anchor_pose[:3, :3].T)
                + self.anchor_pose[:3, 3]).astype(np.float32)

    def apply_sim3(self, T: Sim3) -> None:
        self.anchor_scale *= float(T[0])
        self.anchor_pose = sim3_on_pose(T, self.anchor_pose)
        self.pose_version += 1


class MeshMap:
    def __init__(self, cfg: MeshCfg):
        self.cfg = cfg
        self.submaps: dict[int, MeshSubmap] = {}
        self._next_id = 0
        self._changed: set[int] = set()
        self._removed: set[int] = set()
        self._T_gl_current: Sim3 = SIM3_IDENTITY
        self._T_gl_target: Sim3 = SIM3_IDENTITY

    def integrate_views(
        self,
        depths: np.ndarray,
        colors: np.ndarray,
        valids: np.ndarray,
        poses: np.ndarray,
        Ks: np.ndarray,
        window_ids: Iterable[int] | None = None,
        submap_id: int | None = None,
    ) -> MeshSubmap:
        views = _coerce_views(depths, colors, valids, poses, Ks, window_ids)
        if not views:
            submap = MeshSubmap(
                submap_id=self._alloc_id() if submap_id is None else submap_id,
                anchor_pose=np.eye(4, dtype=np.float64),
                window_ids=[],
            )
            submap.dirty = False
            self.submaps[submap.submap_id] = submap
            return submap

        sid = self._alloc_id() if submap_id is None else int(submap_id)
        submap = MeshSubmap(
            submap_id=sid,
            anchor_pose=views[0].T_wc.copy(),
            window_ids=[v.kf_id for v in views],
            views=views,
        )
        self._rebuild_submap(submap)
        self.submaps[sid] = submap
        self._changed.add(sid)
        self._enforce_cap()
        return submap

    def integrate_backend_result(self, result) -> MeshSubmap | None:
        if getattr(result, "view_depths", None) is None:
            return None
        return self.integrate_views(
            result.view_depths,
            result.view_colors,
            result.view_valid,
            result.view_poses,
            result.view_intrinsics,
            getattr(result, "window_ids", None),
        )

    def rebuild_submap(self, submap_id: int, views: list[MeshView] | None = None) -> MeshSubmap:
        submap = self.submaps[submap_id]
        if views is not None:
            submap.views = views
            if views:
                submap.anchor_pose = views[0].T_wc.copy()
        self._rebuild_submap(submap)
        self._changed.add(submap_id)
        return submap

    def apply_sim3_to_anchors(self, T: Sim3) -> None:
        for submap in self.submaps.values():
            submap.apply_sim3(T)
            self._changed.add(submap.submap_id)

    def set_correction_target(self, T: Sim3) -> None:
        self._T_gl_target = T

    def step_correction(self, alpha: float = 0.2) -> None:
        from .geometry import sim3_interp

        self._T_gl_current = sim3_interp(self._T_gl_current, self._T_gl_target, alpha)

    @property
    def T_global_live(self) -> Sim3:
        return self._T_gl_current

    def changed_submaps(self, clear: bool = True) -> list[MeshSubmap]:
        ids = sorted(self._changed)
        if clear:
            self._changed.clear()
        return [self.submaps[i] for i in ids if i in self.submaps]

    def removed_submaps(self, clear: bool = True) -> list[int]:
        ids = sorted(self._removed)
        if clear:
            self._removed.clear()
        return ids

    def raw_combined_mesh(self) -> TriMesh:
        selections = {
            sid: np.ones(submap.mesh.n_faces, dtype=bool)
            for sid, submap in self.submaps.items()
            if submap.mesh.n_faces > 0
        }
        return self._mesh_from_face_selections(selections)

    def canonical_mesh(self) -> TriMesh:
        selections = self._canonical_face_selections()
        return self._mesh_from_face_selections(selections)

    def combined_mesh(self, mode: str | None = None) -> TriMesh:
        mode = (mode or getattr(self.cfg, "render_mode", "canonical")).lower()
        if mode == "raw":
            return self.raw_combined_mesh()
        return self.canonical_mesh()

    def _mesh_from_face_selections(self, selections: dict[int, np.ndarray]) -> TriMesh:
        verts, faces, norms, colors = [], [], [], []
        offset = 0
        for submap in self.submaps.values():
            mesh = submap.mesh
            if mesh.n_vertices == 0 or mesh.n_faces == 0:
                continue
            keep = selections.get(submap.submap_id)
            if keep is None or not np.any(keep):
                continue
            kept_faces = mesh.faces[keep]
            used = np.unique(kept_faces.reshape(-1))
            remap = np.full(mesh.n_vertices, -1, np.int32)
            remap[used] = np.arange(len(used), dtype=np.int32)
            gv = submap.global_vertices()
            if len(mesh.normals) == mesh.n_vertices:
                gn = mesh.normals @ submap.anchor_pose[:3, :3].T
                kept_normals = gn[used].astype(np.float32)
            else:
                kept_normals = np.zeros((len(used), 3), np.float32)
            if len(mesh.colors) == mesh.n_vertices:
                kept_colors = mesh.colors[used]
            else:
                kept_colors = np.full((len(used), 3), 180, np.uint8)
            verts.append(gv[used])
            faces.append(remap[kept_faces] + offset)
            norms.append(kept_normals)
            colors.append(kept_colors)
            offset += len(used)
        if not verts:
            return TriMesh.empty()
        return TriMesh(
            vertices=np.concatenate(verts).astype(np.float32),
            faces=np.concatenate(faces).astype(np.int32),
            normals=np.concatenate(norms).astype(np.float32),
            colors=np.concatenate(colors).astype(np.uint8),
        )

    def _canonical_face_selections(self) -> dict[int, np.ndarray]:
        if not self.submaps:
            return {}
        cell = _canonical_cell_size(self.cfg)
        recency_rank = _recency_ranks(self.submaps.values())
        normal_cos = float(getattr(self.cfg, "canonical_normal_cos", 0.85))
        groups: dict[tuple[int, int, int], list[dict]] = {}
        face_refs: dict[int, list[tuple[tuple[int, int, int], int]]] = {}

        for sid, submap in self.submaps.items():
            mesh = submap.mesh
            if mesh.n_vertices == 0 or mesh.n_faces == 0:
                continue
            support = _submap_support(submap)
            if support < int(getattr(self.cfg, "canonical_min_support", 1)):
                continue
            score = _canonical_submap_score(submap, recency_rank.get(sid, 0), self.cfg)
            gv = submap.global_vertices()
            normals = _global_vertex_normals(submap)
            refs = []
            for face in mesh.faces:
                centroid = gv[face].mean(axis=0)
                normal = _face_normal(gv, normals, face)
                key = _canonical_cell_key(centroid, cell)
                group_idx = _find_normal_group(groups.setdefault(key, []), normal, normal_cos)
                if group_idx < 0:
                    group_idx = len(groups[key])
                    groups[key].append({"normal": normal, "score": score, "sid": sid})
                else:
                    current = groups[key][group_idx]
                    if (score > current["score"] + 1e-9
                            or (abs(score - current["score"]) <= 1e-9 and sid > current["sid"])):
                        current["normal"] = normal
                        current["score"] = score
                        current["sid"] = sid
                refs.append((key, group_idx))
            face_refs[sid] = refs

        selections: dict[int, np.ndarray] = {}
        for sid, submap in self.submaps.items():
            mesh = submap.mesh
            if mesh.n_faces == 0:
                continue
            keep = np.zeros(mesh.n_faces, dtype=bool)
            for i, (key, group_idx) in enumerate(face_refs.get(sid, [])):
                group = groups.get(key, [])
                keep[i] = group_idx < len(group) and group[group_idx]["sid"] == sid
            if np.any(keep):
                selections[sid] = keep
        return selections

    def export_ply(self, path: str | Path, mode: str | None = None) -> TriMesh:
        mesh = self.combined_mesh(mode)
        write_ply(path, mesh)
        return mesh

    def _alloc_id(self) -> int:
        sid = self._next_id
        self._next_id += 1
        return sid

    def _enforce_cap(self) -> None:
        cap = int(self.cfg.max_active_submaps)
        if cap <= 0:
            return
        while len(self.submaps) > cap:
            oldest = min(self.submaps)
            del self.submaps[oldest]
            self._changed.discard(oldest)
            self._removed.add(oldest)

    def _rebuild_submap(self, submap: MeshSubmap) -> None:
        submap.mesh = _integrate_tsdf(submap.views, submap.anchor_pose, self.cfg)
        submap.version += 1
        submap.dirty = False


def _coerce_views(
    depths: np.ndarray,
    colors: np.ndarray,
    valids: np.ndarray,
    poses: np.ndarray,
    Ks: np.ndarray,
    window_ids: Iterable[int] | None,
) -> list[MeshView]:
    depths = np.asarray(depths)
    colors = np.asarray(colors)
    valids = np.asarray(valids)
    poses = np.asarray(poses, dtype=np.float64)
    Ks = np.asarray(Ks, dtype=np.float64)
    if depths.ndim == 2:
        depths = depths[None]
    if colors.ndim == 3:
        colors = colors[None]
    if valids.ndim == 2:
        valids = valids[None]
    if Ks.ndim == 2:
        Ks = np.repeat(Ks[None], len(depths), axis=0)
    ids = list(range(len(depths))) if window_ids is None else list(window_ids)
    views: list[MeshView] = []
    for i in range(len(depths)):
        depth = np.asarray(depths[i], np.float32)
        valid = np.asarray(valids[i], bool)
        color = np.asarray(colors[i])
        if color.shape[:2] != depth.shape:
            color = cv2.resize(color, (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_AREA)
        if color.dtype != np.uint8:
            color = np.clip(color, 0, 255).astype(np.uint8)
        depth = np.where(valid & np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
        views.append(MeshView(
            depth=depth,
            color=color,
            valid=valid,
            K=Ks[i].copy(),
            T_wc=poses[i].copy(),
            kf_id=int(ids[i]) if i < len(ids) else i,
        ))
    return views


def _canonical_cell_size(cfg: MeshCfg) -> float:
    configured = float(getattr(cfg, "canonical_cell_size", 0.0))
    distance = float(getattr(cfg, "canonical_distance_m", 0.0))
    fallback = max(2.0 * float(cfg.voxel_size), float(cfg.trunc_margin))
    return max(configured, distance, fallback, 1e-6)


def _submap_support(submap: MeshSubmap) -> int:
    return max(len(submap.views), len(submap.window_ids), 1)


def _latest_window_id(submap: MeshSubmap) -> int:
    return max(submap.window_ids) if submap.window_ids else submap.submap_id


def _recency_ranks(submaps: Iterable[MeshSubmap]) -> dict[int, int]:
    ordered = sorted(submaps, key=lambda s: (_latest_window_id(s), s.submap_id))
    return {submap.submap_id: i for i, submap in enumerate(ordered)}


def _canonical_submap_score(submap: MeshSubmap, recency_rank: int, cfg: MeshCfg) -> float:
    support = float(_submap_support(submap))
    support_w = float(getattr(cfg, "canonical_support_weight", 1.0))
    residual_w = float(getattr(cfg, "canonical_residual_weight", 0.25))
    recency_w = float(getattr(cfg, "canonical_recency_weight", 0.10))
    residual = _submap_residual_proxy(submap, cfg)
    return support_w * support - residual_w * residual + recency_w * float(recency_rank)


def _submap_residual_proxy(submap: MeshSubmap, cfg: MeshCfg) -> float:
    if not submap.views or submap.mesh.n_vertices == 0:
        return 0.0
    verts = submap.global_vertices().astype(np.float64)
    if len(verts) > 512:
        sample = np.linspace(0, len(verts) - 1, 512).astype(np.int32)
        verts = verts[sample]
    verts_h = np.column_stack([verts, np.ones(len(verts))])
    residuals = []
    for view in submap.views:
        T_cw = np.linalg.inv(view.T_wc)
        cam = (T_cw @ verts_h.T).T[:, :3]
        z = cam[:, 2]
        in_front = z > 1e-6
        if not np.any(in_front):
            continue
        z_safe = np.where(in_front, z, 1.0)
        u = np.rint(view.K[0, 0] * cam[:, 0] / z_safe + view.K[0, 2]).astype(np.int32)
        v = np.rint(view.K[1, 1] * cam[:, 1] / z_safe + view.K[1, 2]).astype(np.int32)
        h, w = view.depth.shape
        inside = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(inside):
            continue
        idx = np.nonzero(inside)[0]
        measured = view.depth[v[idx], u[idx]]
        ok = view.valid[v[idx], u[idx]] & (measured > 0) & np.isfinite(measured)
        if np.any(ok):
            residuals.append(np.abs(measured[ok] - z[idx][ok]))
    if not residuals:
        return 1.0
    residual = float(np.median(np.concatenate(residuals)))
    tol = max(float(cfg.trunc_margin), 2.0 * float(cfg.voxel_size), 1e-6)
    return float(np.clip(residual / tol, 0.0, 4.0))


def _global_vertex_normals(submap: MeshSubmap) -> np.ndarray:
    mesh = submap.mesh
    if len(mesh.normals) == mesh.n_vertices:
        normals = mesh.normals.astype(np.float64) @ submap.anchor_pose[:3, :3].T
        n = np.linalg.norm(normals, axis=1, keepdims=True)
        return np.divide(normals, np.maximum(n, 1e-12)).astype(np.float32)
    return np.empty((0, 3), np.float32)


def _face_normal(vertices: np.ndarray, vertex_normals: np.ndarray, face: np.ndarray) -> np.ndarray:
    if len(vertex_normals) == len(vertices):
        normal = vertex_normals[face].mean(axis=0).astype(np.float64)
    else:
        a, b, c = vertices[face].astype(np.float64)
        normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (normal / norm).astype(np.float32)


def _canonical_cell_key(centroid: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    cell = np.floor(np.asarray(centroid, dtype=np.float64) / cell_size + 0.5).astype(np.int64)
    return int(cell[0]), int(cell[1]), int(cell[2])


def _find_normal_group(groups: list[dict], normal: np.ndarray, normal_cos: float) -> int:
    threshold = float(np.clip(normal_cos, 0.0, 1.0))
    for i, group in enumerate(groups):
        if abs(float(np.dot(normal, group["normal"]))) >= threshold:
            return i
    return -1


def _integrate_tsdf(views: list[MeshView], anchor_pose: np.ndarray, cfg: MeshCfg) -> TriMesh:
    if not views:
        return TriMesh.empty()
    import open3d as o3d

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(cfg.voxel_size),
        sdf_trunc=float(cfg.trunc_margin),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    anchor_inv = np.linalg.inv(anchor_pose)
    for view in views:
        h, w = view.depth.shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(view.color)),
            o3d.geometry.Image(np.ascontiguousarray(view.depth.astype(np.float32))),
            depth_scale=1.0,
            depth_trunc=float(cfg.depth_trunc_m),
            convert_rgb_to_intensity=False,
        )
        K = view.K
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(w), int(h), float(K[0, 0]), float(K[1, 1]),
            float(K[0, 2]), float(K[1, 2]),
        )
        T_lc = anchor_inv @ view.T_wc
        volume.integrate(rgbd, intrinsic, np.linalg.inv(T_lc))

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return TriMesh.empty()
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    vertex_colors = np.asarray(mesh.vertex_colors)
    if len(vertex_colors) == len(vertices):
        colors = np.clip(vertex_colors * 255.0, 0, 255).astype(np.uint8)
    else:
        colors = np.full((len(vertices), 3), 180, np.uint8)
    return _filter_by_surface_support(
        TriMesh(vertices=vertices, faces=faces, normals=normals, colors=colors),
        views,
        anchor_pose,
        cfg,
    )


def _filter_by_surface_support(
    mesh: TriMesh,
    views: list[MeshView],
    anchor_pose: np.ndarray,
    cfg: MeshCfg,
) -> TriMesh:
    min_obs = int(getattr(cfg, "min_surface_observations", 1))
    if min_obs <= 1 or mesh.n_vertices == 0 or mesh.n_faces == 0:
        return mesh
    support = np.zeros(mesh.n_vertices, np.int16)
    tol = max(float(cfg.trunc_margin), 2.0 * float(cfg.voxel_size))
    anchor_inv = np.linalg.inv(anchor_pose)
    verts_h = np.column_stack([mesh.vertices.astype(np.float64), np.ones(mesh.n_vertices)])
    for view in views:
        T_lc = anchor_inv @ view.T_wc
        T_cl = np.linalg.inv(T_lc)
        cam = (T_cl @ verts_h.T).T[:, :3]
        z = cam[:, 2]
        in_front = z > 1e-6
        if not np.any(in_front):
            continue
        z_safe = np.where(in_front, z, 1.0)
        u = np.rint(view.K[0, 0] * cam[:, 0] / z_safe + view.K[0, 2]).astype(np.int32)
        v = np.rint(view.K[1, 1] * cam[:, 1] / z_safe + view.K[1, 2]).astype(np.int32)
        h, w = view.depth.shape
        inside = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(inside):
            continue
        idx = np.nonzero(inside)[0]
        measured = view.depth[v[idx], u[idx]]
        ok = view.valid[v[idx], u[idx]] & (measured > 0) & (np.abs(measured - z[idx]) <= tol)
        support[idx[ok]] += 1
    supported = support >= min_obs
    face_keep = supported[mesh.faces].all(axis=1)
    if not np.any(face_keep):
        return TriMesh.empty()
    kept_faces = mesh.faces[face_keep]
    used = np.unique(kept_faces.reshape(-1))
    remap = np.full(mesh.n_vertices, -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return TriMesh(
        vertices=mesh.vertices[used],
        faces=remap[kept_faces],
        normals=mesh.normals[used] if len(mesh.normals) == mesh.n_vertices else np.empty((0, 3), np.float32),
        colors=mesh.colors[used] if len(mesh.colors) == mesh.n_vertices else np.full((len(used), 3), 180, np.uint8),
    )


def write_ply(path: str | Path, mesh: TriMesh) -> None:
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    if len(mesh.normals) == len(mesh.vertices):
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(mesh.normals.astype(np.float64))
    if len(mesh.colors) == len(mesh.vertices):
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.colors.astype(np.float64) / 255.0)
    o3d.io.write_triangle_mesh(str(path), o3d_mesh, write_ascii=False)


def save_meshmap(path: str | Path, meshmap: MeshMap) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "submap_ids": np.array(sorted(meshmap.submaps), dtype=np.int32),
    }
    for sid, submap in meshmap.submaps.items():
        prefix = f"submap_{sid}_"
        arrays[prefix + "anchor"] = submap.anchor_pose.astype(np.float64)
        arrays[prefix + "anchor_scale"] = np.array(submap.anchor_scale, dtype=np.float64)
        arrays[prefix + "window_ids"] = np.array(submap.window_ids, dtype=np.int32)
        arrays[prefix + "vertices"] = submap.mesh.vertices.astype(np.float32)
        arrays[prefix + "faces"] = submap.mesh.faces.astype(np.int32)
        arrays[prefix + "normals"] = submap.mesh.normals.astype(np.float32)
        arrays[prefix + "colors"] = submap.mesh.colors.astype(np.uint8)
    np.savez_compressed(path, **arrays)
    return len(meshmap.submaps)


def load_meshmap(path: str | Path, cfg: MeshCfg | None = None) -> MeshMap:
    data = np.load(path, allow_pickle=False)
    meshmap = MeshMap(cfg or MeshCfg())
    for sid in data["submap_ids"].astype(int).tolist():
        prefix = f"submap_{sid}_"
        submap = MeshSubmap(
            submap_id=sid,
            anchor_pose=data[prefix + "anchor"],
            anchor_scale=float(data[prefix + "anchor_scale"])
            if prefix + "anchor_scale" in data else 1.0,
            window_ids=data[prefix + "window_ids"].astype(int).tolist(),
            mesh=TriMesh(
                vertices=data[prefix + "vertices"],
                faces=data[prefix + "faces"],
                normals=data[prefix + "normals"],
                colors=data[prefix + "colors"],
            ),
            dirty=False,
        )
        meshmap.submaps[sid] = submap
        meshmap._next_id = max(meshmap._next_id, sid + 1)
    return meshmap
