# Stable Mesh Generation And Correction Goal

## Goal Summary
- Final outcome: `space-recognizer` keeps the existing point-cloud/object pipeline while adding stable surface mesh generation from RGB-D evidence, plus a correction/update path for previously generated mesh when poses, global Sim3, relocalization, or new free-space evidence changes.
- Scope: `/Users/chasoik/Projects/space-recognizer`, especially `src/spacerec/backend.py`, `src/spacerec/worldmap.py`, `src/spacerec/main.py`, `src/spacerec/viz.py`, `src/spacerec/persistence.py`, `src/spacerec/replay.py`, `src/spacerec/config.py`, tests, benchmarks, and docs.
- Non-goals: replacing the point-cloud/object/relocalization pipeline, large SLAM rewrite, external cloud services, broad recorder-app rewrite, a global Poisson-only mesh path, unrelated dirty-worktree cleanup, or self-directed target increases.

## Baseline And Assumptions
- Current baseline:
  - `GlobalMap` stores an evidence-weighted voxel-fused colored point cloud.
  - `ReconstructionBackend` emits `BackendResult(points, colors, view_origins, point_view_idx, poses/intrinsics/depth metadata)`, but no mesh.
  - Rerun visualization logs `Points3D` for map/object nodes.
  - Persistence saves points/colors/object metadata to `.npz` and uses Open3D ICP for point-cloud relocalization.
  - Recorded OAK-D-Lite replay provides metric RGB-aligned depth.
- Unknowns to verify:
  - Whether Open3D TSDF APIs work reliably in the current `.venv` on this macOS machine.
  - Practical mesh voxel/truncation defaults for recorded OAK sessions.
  - Mesh quality and scale behavior on non-metric DA3 video paths.
- Assumptions:
  - Start with Open3D CPU TSDF for robustness.
  - Treat mesh as a derivative cache regenerated from keyframe RGB-D evidence, not the source of truth.
  - Keep existing point-cloud path authoritative for object localization and ICP relocalization.
  - Use backend-window submaps first; spatial chunking can be added later if required.
  - Existing unrelated user/worktree changes must be preserved.

## Checkpoint Plan
| Step | Work | Stage Target | Verification | Done When |
|---|---|---|---|---|
| 1 | Review and patch this plan for the current mesh goal. | Plan is concrete, bounded, and verifiable. | Inspect `GOAL_PLAN.md`, `git status`, and current source. | Old replay plan is replaced or patched and progress log begins. |
| 2 | Add mesh configuration and core data model. | `MeshCfg`, `TriMesh`, `MeshSubmap`, and `MeshMap` contracts exist without breaking config load. | Config/unit tests. | Config defaults load and submap transform/export contracts pass tests. |
| 3 | Implement offline TSDF mesh generation. | Synthetic RGB-D views and recorded OAK replay frames can generate triangle meshes. | Synthetic plane/box tests and mesh smoke. | Mesh has vertices/faces and geometric checks pass within voxel tolerance. |
| 4 | Implement stale-surface correction/rebuild checks. | Old surfaces can be invalidated/rebuilt from newer evidence instead of persisting forever. | Mesh evidence tests. | A stale surface test shows old geometry decreases while stable geometry resists a single bad pass. |
| 5 | Extend backend output for mesh evidence. | `BackendResult` carries per-view calibrated depth, valid/static mask, K, poses, and window ids for mesh integration while preserving point output. | Backend fake-worker tests. | Existing backend tests pass and mesh payload shape/dtype contract is verified. |
| 6 | Wire `MeshMap` into the main replay/live pipeline. | Backend windows create/update mesh submaps while `GlobalMap.fuse` remains intact. | Main/replay smoke and unit tests. | Point cloud, objects, and mesh all run without crashing on recorded replay. |
| 7 | Add pose/global-Sim3/relocalization correction behavior. | Small corrections update submap transforms; large relocalization applies Sim3 to mesh anchors or rebuilds affected submaps. | Transform invariance and persistence/relocalization tests. | Mesh global positions stay consistent after Sim3 correction without direct cumulative vertex mutation. |
| 8 | Add Rerun visualization and persistence/export. | Mesh submaps can be logged as `Mesh3D`, saved/loaded, and exported as `.ply`. | Visualization smoke where practical, save/load/export tests. | Exported mesh can be read back and docs describe output paths. |
| 9 | Add benchmark/smoke and docs. | A mesh smoke reports vertices, faces, valid/depth frames, output path, and timing for recorded sessions. | Run benchmark on at least one recorded OAK session. | User can reproduce mesh generation from README/MANUAL. |
| 10 | Final verification and independent review. | All final criteria are proven from current state. | Full tests, mesh smoke, replay smoke, diff check, independent subagent/clean verification. | Evidence is reported; goal remains active unless every criterion passes. |

## Final Completion Criteria
- `.venv/bin/python -m pytest tests/ -q` passes.
- Synthetic TSDF mesh tests validate plane/box generation and stale-surface correction/rebuild behavior.
- A recorded OAK session mesh smoke generates a triangle mesh with nonzero vertices/faces and exports a readable `.ply`.
- Existing replay/backend/object smoke still passes, including a recorded-session path with backend map points and object observations.
- Existing point-cloud map, object localization, persistence/relocalization, video source, and OAK/replay source behavior are not intentionally broken.
- Rerun visualization has a mesh logging path while existing `world/points` and `world/objects` remain.
- Mesh correction design is implemented through submap anchor transforms and/or affected-submap rebuild, not cumulative ad hoc vertex warping.
- Docs explain mesh mode, limits, benchmark command, and export artifacts.
- Existing unrelated dirty worktree changes are preserved.

## Independent Verification Policy
- Independent final verification is required before marking the goal complete.
- Preferred verification: a separate subagent or clean worktree/fresh checkout reruns final criteria and reviews the diff.
- If unavailable, record why and run the strongest substitute: `git status`, `git diff --check`, full tests, mesh smoke, replay smoke, export reload smoke, docs check, and focused diff review.
- Final report must include exact commands, pass/fail result, mesh smoke metrics, replay smoke metrics, and generated artifact paths.

## Self-Directed Target Increase Policy
- User opt-in: no.
- After mandatory targets pass, stop without raising performance or quality targets.
- Do not expand scope into large SLAM rewrites, external integrations, broad refactors, or unrequested quality/performance target increases.

## Stop And Ask Conditions
- Open3D TSDF is unavailable or incompatible and choosing a different meshing backend would materially change scope.
- Required evidence persistence would create unexpectedly large storage costs and needs a user policy decision.
- Required changes would overwrite unrelated user work.
- Destructive commands, credentials, external service access, hardware-only validation, or large downloads beyond existing dependencies are required.
- Evidence shows the target is infeasible with the current recordings or repo constraints.

## Progress Log Rules
- After each checkpoint, log current step, changed files, verification command/result, remaining work, and blockers.
- Failed verification entries must include root cause, fix, and re-verification result.

## Progress Log
| Step | Status | Changed Files | Verification Result | Remaining / Blockers |
|---|---|---|---|---|
| 1 | Done | `GOAL_PLAN.md` | Current file inspected and replaced from the previous OAK replay goal to this mesh goal. | None. |
| 2 | Done | `src/spacerec/config.py`, `config.yaml`, `src/spacerec/mesh.py`, `tests/test_config.py`, `tests/test_mesh.py` | `.venv/bin/python -m pytest tests/test_mesh.py tests/test_config.py -q` passed. | None. |
| 3 | Done | `src/spacerec/mesh.py`, `benchmarks/mesh_smoke.py`, `tests/test_mesh.py` | Synthetic plane and ray-box TSDF tests generate non-empty vertices/faces; `benchmarks/mesh_smoke.py` generated recorded OAK mesh. | None. |
| 4 | Done | `src/spacerec/mesh.py`, `src/spacerec/config.py`, `tests/test_mesh.py` | `test_submap_rebuild_replaces_stale_surface` verifies affected submap rebuild; `test_mesh_support_filter_resists_single_bad_pass` verifies repeated good observations suppress a single bad surface. | None. |
| 5 | Done | `src/spacerec/backend.py`, `tests/test_backend.py` | `.venv/bin/python -m pytest tests/test_backend.py tests/test_mesh.py tests/test_config.py tests/test_persistence.py -q` passed. | None. |
| 6 | Done | `src/spacerec/main.py`, `src/spacerec/viz.py` | Main replay smoke with `--mesh-out` generated `artifacts/mesh_smoke/main_session_054320.ply`. | None. |
| 7 | Done | `src/spacerec/mesh.py`, `src/spacerec/persistence.py`, `src/spacerec/main.py`, `tests/test_mesh.py`, `tests/test_persistence.py` | Anchor Sim3 correction and saved mesh sidecar merge tests verify previous mesh anchors are transformed without vertex mutation; main loads/saves `<map>.mesh.npz` and merges saved mesh on relocalization. | None. |
| 8 | Done | `src/spacerec/viz.py`, `src/spacerec/persistence.py`, `tests/test_persistence.py` | Mesh state save/load test passed; exported `.ply` files were read back with Open3D and include colors/normals. | None. |
| 9 | Done | `benchmarks/mesh_smoke.py`, `README.md`, `docs/MANUAL.md` | Mesh smoke reported 99765 vertices / 155014 faces for `session_20260624_054320...`; docs updated with commands/config/sidecar notes. | None. |
| 10 | Done | Verification only | Full tests passed: `.venv/bin/python -m pytest tests/ -q` -> 46 passed. Mesh smoke reported 99765 vertices / 155014 faces; main `--mesh-out` reported 16472 vertices / 28766 faces; replay smoke with `--full-models --backend` passed for `session_20260624_054320...`; `git diff --check` passed. Independent subagent follow-up passed with mesh smoke 55838 vertices / 83078 faces and replay smoke object/backend checks. | Live OAK hardware was not physically exercised; code/import/shared-path coverage and replay smokes cover practical non-hardware criteria. |
