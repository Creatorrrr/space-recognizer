import numpy as np
import pytest

from spacerec.config import MeshCfg
from spacerec.geometry import sim3_apply
from spacerec.mesh import MeshMap, MeshView, load_meshmap, save_meshmap


def _plane_views(z=1.0, n=2, w=64, h=48):
    K = np.array([[60.0, 0.0, w / 2], [0.0, 60.0, h / 2], [0.0, 0.0, 1.0]])
    depths, colors, valids, poses, Ks = [], [], [], [], []
    for i in range(n):
        depth = np.full((h, w), z, np.float32)
        color = np.full((h, w, 3), [80 + i * 20, 120, 180], np.uint8)
        pose = np.eye(4)
        pose[0, 3] = 0.02 * i
        depths.append(depth)
        colors.append(color)
        valids.append(depth > 0)
        poses.append(pose)
        Ks.append(K)
    return (
        np.stack(depths),
        np.stack(colors),
        np.stack(valids),
        np.stack(poses),
        np.stack(Ks),
    )


def _look_at(camera, target=np.zeros(3)):
    camera = np.asarray(camera, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z_axis = target - camera
    z_axis /= np.linalg.norm(z_axis)
    down = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(down, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = camera
    return T


def _ray_box_depth(T_wc, K, w=80, h=60, bounds=(-0.45, -0.3, -0.45, 0.45, 0.3, 0.45)):
    mins = np.array(bounds[:3], dtype=np.float64)
    maxs = np.array(bounds[3:], dtype=np.float64)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    x = (us - K[0, 2]) / K[0, 0]
    y = (vs - K[1, 2]) / K[1, 1]
    dirs_cam = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    origin = T_wc[:3, 3]
    dirs = dirs_cam @ T_wc[:3, :3].T
    inv = 1.0 / np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)
    t0 = (mins - origin) * inv
    t1 = (maxs - origin) * inv
    tmin = np.maximum.reduce(np.minimum(t0, t1), axis=1)
    tmax = np.minimum.reduce(np.maximum(t0, t1), axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0)
    depth = np.zeros(h * w, dtype=np.float32)
    depth[hit] = tmin[hit].astype(np.float32)
    return depth.reshape(h, w)


def _box_views(w=80, h=60):
    K = np.array([[80.0, 0.0, w / 2], [0.0, 80.0, h / 2], [0.0, 0.0, 1.0]])
    cameras = [
        [0.0, 0.0, -2.0],
        [0.8, 0.0, -1.8],
        [-0.8, 0.0, -1.8],
        [0.0, 0.0, 2.0],
        [0.8, 0.0, 1.8],
        [-0.8, 0.0, 1.8],
    ]
    depths, colors, valids, poses, Ks = [], [], [], [], []
    for i, cam in enumerate(cameras):
        pose = _look_at(cam)
        depth = _ray_box_depth(pose, K, w=w, h=h)
        depths.append(depth)
        colors.append(np.full((h, w, 3), [160, 80 + i * 30, 100], np.uint8))
        valids.append(depth > 0)
        poses.append(pose)
        Ks.append(K)
    return np.stack(depths), np.stack(colors), np.stack(valids), np.stack(poses), np.stack(Ks)


def test_meshmap_integrates_synthetic_plane():
    mm = MeshMap(MeshCfg(voxel_size=0.05, trunc_margin=0.15))
    submap = mm.integrate_views(*_plane_views(z=1.0, n=2), window_ids=[3, 4])

    assert submap.mesh.n_vertices > 0
    assert submap.mesh.n_faces > 0
    assert submap.window_ids == [3, 4]
    assert np.mean(submap.mesh.vertices[:, 2]) == pytest.approx(1.0, abs=0.04)


def test_meshmap_integrates_synthetic_box_depth():
    mm = MeshMap(MeshCfg(voxel_size=0.04, trunc_margin=0.12))
    submap = mm.integrate_views(*_box_views(), window_ids=[0, 1, 2, 3])

    assert submap.mesh.n_vertices > 100
    assert submap.mesh.n_faces > 100
    extent = submap.mesh.vertices.max(axis=0) - submap.mesh.vertices.min(axis=0)
    assert extent[0] > 0.5
    assert extent[1] > 0.3
    assert extent[2] > 0.4


def test_submap_rebuild_replaces_stale_surface():
    mm = MeshMap(MeshCfg(voxel_size=0.05, trunc_margin=0.15))
    submap = mm.integrate_views(*_plane_views(z=1.0), window_ids=[0])
    near_z = float(np.mean(submap.mesh.vertices[:, 2]))

    depths, colors, valids, poses, Ks = _plane_views(z=2.0)
    new_views = [
        MeshView(depths[i], colors[i], valids[i], Ks[i], poses[i], kf_id=i + 1)
        for i in range(2)
    ]
    rebuilt = mm.rebuild_submap(submap.submap_id, views=new_views)
    far_z = float(np.mean(rebuilt.mesh.vertices[:, 2]))

    assert near_z < 1.1
    assert far_z > 1.8
    assert rebuilt.version == 2


def test_mesh_support_filter_resists_single_bad_pass():
    good_depths, good_colors, good_valids, good_poses, good_Ks = _plane_views(z=1.0, n=5)
    bad_depths, bad_colors, bad_valids, bad_poses, bad_Ks = _plane_views(z=2.0, n=1)
    mm = MeshMap(MeshCfg(voxel_size=0.05, trunc_margin=0.15, min_surface_observations=2))

    submap = mm.integrate_views(
        np.concatenate([good_depths, bad_depths]),
        np.concatenate([good_colors, bad_colors]),
        np.concatenate([good_valids, bad_valids]),
        np.concatenate([good_poses, bad_poses]),
        np.concatenate([good_Ks, bad_Ks]),
    )

    z = submap.mesh.vertices[:, 2]
    assert np.count_nonzero(np.abs(z - 1.0) < 0.1) > 100
    assert np.count_nonzero(np.abs(z - 2.0) < 0.1) < 50


def test_submap_anchor_sim3_correction_moves_mesh_without_vertex_mutation():
    mm = MeshMap(MeshCfg(voxel_size=0.05, trunc_margin=0.15))
    submap = mm.integrate_views(*_plane_views(z=1.0), window_ids=[0])
    local_before = submap.mesh.vertices.copy()
    global_before = submap.global_vertices().copy()
    T = (1.2, np.eye(3), np.array([0.3, -0.1, 0.2]))

    mm.apply_sim3_to_anchors(T)

    assert np.allclose(submap.mesh.vertices, local_before)
    assert np.allclose(submap.global_vertices(), sim3_apply(T, global_before), atol=1e-6)
    assert submap.pose_version == 1


def test_meshmap_save_load_and_export_roundtrip(tmp_path):
    mm = MeshMap(MeshCfg(voxel_size=0.05, trunc_margin=0.15))
    mm.integrate_views(*_plane_views(z=1.0), window_ids=[0])

    state_path = tmp_path / "mesh_state.npz"
    out_path = tmp_path / "mesh.ply"
    assert save_meshmap(state_path, mm) == 1
    loaded = load_meshmap(state_path)
    exported = loaded.export_ply(out_path)

    assert exported.n_vertices > 0
    assert exported.n_faces > 0
    assert out_path.exists()
