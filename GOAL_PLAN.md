# OAK Direct RGB-D Fusion Goal

## Goal Summary
- Final outcome: OAK-D-Lite live/replay metric depth can build an accumulated 3D environment point cloud and optional TSDF mesh without loading DA3 live depth, DA3 any-view backend, or DA3 metric anchor models.
- Scope: `/Users/chasoik/Projects/space-recognizer`, especially `src/spacerec/main.py`, `src/spacerec/config.py`, a new direct fusion module, tests, benchmarks, README, and manual docs.
- Non-goals: monocular RGB reconstruction without a metric depth source, replacing the existing DA3 backend, loop closure/pose graph/SLAM rewrite, object detection changes, broad VO/IMU changes, destructive cleanup of unrelated work, or self-directed target increases.

## Baseline And Assumptions
- Current baseline:
  - `worldmap.fuse()` is called only through backend result application, so `--no-backend` leaves the accumulated environment point cloud empty.
  - OAK live/replay frames already expose metric `depth_m`, camera `K`, RGB, optional depth confidence, and IMU metadata through the common `Frame` contract.
  - The live loop already has dynamic object masks and VO poses before backend keyframes are emitted.
  - Current `config.yaml` may contain unrelated user edits and must not be reverted.
- Unknowns to verify:
  - Which OAK replay session is available locally for final smoke verification.
  - Whether direct TSDF integration must run synchronously or needs a worker to avoid visible live-loop spikes.
- Assumptions:
  - Direct fusion is valid only when the source has metric depth aligned to the RGB frame.
  - Direct fusion should reuse the existing `BackendResult` consumption path where practical so `GlobalMap`, `MeshMap`, visualization, persistence, and export behavior stay shared.
  - DA3 backend and direct OAK fusion are mutually exclusive reconstruction sources for a single run.

## Checkpoint Plan
| Step | Work | Stage Target | Verification | Done When |
|---|---|---|---|---|
| 1 | Review the stale plan/current source and patch this plan to the active OAK direct fusion goal. | Plan is concrete, bounded, verifiable, and preserves unrelated dirty work. | Inspect `GOAL_PLAN.md`, `git status`, OAK/depth/backend/main/mesh/config sources. | Stale plan is replaced and progress log begins. |
| 2 | Add tests first for fusion config/mode resolution and direct RGB-D fusion behavior. | Tests cover direct-only metric source gating, default config loading, no DA3-needed policy, point backprojection, validity/mask filtering, RGB color order, BackendResult-compatible view arrays, and mesh-window output. | Run focused pytest and confirm new tests fail for missing implementation. | Red tests fail for the expected missing symbols/behavior. |
| 3 | Implement fusion config and direct fusion module. | Add `FusionCfg`, mode resolution, direct keyframe/result production, depth validity filters, dynamic mask dilation, optional edge filtering, point subsampling, RGB conversion, view buffering, and `BackendResult` output with `meters_per_unit=1.0`. | Focused direct fusion/config tests. | Red tests pass without loading DA3. |
| 4 | Wire main loop and CLI. | Add `--fusion`, `--direct-fusion`; instantiate DA3 backend only for backend mode; instantiate direct backend only for metric sources; keep `--fusion none` as reconstruction off; reject conflicting or invalid modes clearly. | Main/config policy tests and source review. | Direct OAK path reaches existing backend-result drain or equivalent shared application path. |
| 5 | Update docs and benchmarks. | README/MANUAL/config/benchmark help explain `backend`, `direct`, `none`, `auto`, `--no-backend`, OAK alignment requirements, DA3-free behavior, and drift/mesh limitations. | Docs grep/source review and benchmark help smoke where practical. | User-facing instructions match implementation. |
| 6 | Final verification and repair. | Related and full tests pass; strongest available OAK replay smoke proves non-empty accumulated map and, when mesh enabled, readable mesh output. | `git diff --check`, focused tests, full pytest, direct fusion smoke/benchmark on available OAK recording or documented substitute if unavailable. | Every final completion criterion is proven from current-state evidence. |

## Final Completion Criteria
- OAK live/replay metric depth can run with `--fusion direct` or `--direct-fusion` and produce accumulated `worldmap.points`.
- Direct mode does not instantiate or load `DepthEstimator`, `ReconstructionBackend` DA3 any-view, or DA3 metric anchor when `oak_fill_missing` is false.
- Direct mode rejects non-metric sources with a clear error.
- Existing `--fusion backend` or equivalent backend path preserves DA3 backend behavior.
- `--fusion none` or an explicit reconstruction-off path remains available for performance upper-bound runs.
- Depth validity includes range checks, optional confidence, dynamic object mask exclusion, mask dilation, and optional edge/flying-pixel filtering.
- RGB/BGR color conversion is tested and correct for point cloud and mesh data.
- Direct fusion can provide `view_depths`, `view_valid`, `view_colors`, `view_poses`, and `view_intrinsics` so existing `MeshMap` can build/export TSDF mesh when enabled.
- Existing DA3 backend, OAK depth policy, mesh, persistence, config, VO, replay, and object tests do not regress.
- README/MANUAL/config comments explain direct mode, DA3-free behavior, OAK depth alignment requirements, `--no-backend`/`--fusion none`, drift limitations, and mesh limitations.
- Existing unrelated dirty worktree changes and artifacts are preserved.

## Independent Verification Policy
- Independent final verification is required before marking the goal complete.
- Preferred verification: separate subagent or clean worktree/fresh checkout reruns final criteria and reviews the diff.
- If unavailable, record why and run the strongest substitute: `git status`, `git diff --check`, focused tests, full tests, direct OAK replay smoke, direct mesh smoke, artifact read-back, and focused diff review.
- Final report must include exact commands, pass/fail results, generated artifact paths if any, DA3-free/direct behavior evidence, and unresolved risks.

## Self-Directed Target Increase Policy
- User opt-in: no.
- After mandatory targets pass, stop without raising performance or quality targets.
- Do not expand scope into full SLAM, loop closure, pose graph, full spatial-block TSDF, broad refactors, or unrequested performance targets.

## Stop And Ask Conditions
- Correctness requires deleting unrelated user work or reverting dirty changes not made for this goal.
- Required validation needs physical OAK hardware and no recorded OAK session is available for a substitute smoke.
- Destructive commands, credentials, external services, or large new dependencies are required.
- Evidence shows metric-depth direct fusion cannot satisfy the goal without adding a broader SLAM/loop-closure system.
- Existing public behavior would need an incompatible CLI/config break instead of additive options.

## Progress Log Rules
- After each checkpoint, log current step, changed files, verification command/result, remaining work, and blockers.
- Failed verification entries must include root cause, fix, and re-verification result.

## Progress Log
| Step | Status | Changed Files | Verification Result | Remaining / Blockers |
|---|---|---|---|---|
| 1 | Done | `GOAL_PLAN.md` | Current `GOAL_PLAN.md` was a stale canonical mesh plan. Current source and `git status` reviewed; branch `codex/oak-direct-fusion` created to avoid implementing on `main`. | Continue with TDD red tests for direct fusion. |
| 2 | Done | `tests/test_config.py`, `tests/test_main_depth_policy.py`, `tests/test_directfusion.py` | Red check before implementation: `.venv/bin/python -m pytest tests/test_config.py tests/test_main_depth_policy.py tests/test_directfusion.py -q` failed for the expected missing `FusionCfg`, `resolve_fusion_mode`, and `directfusion` module. | None. |
| 3 | Done | `src/spacerec/config.py`, `src/spacerec/directfusion.py` | Added `FusionCfg` and direct RGB-D fusion. Focused check after implementation: `.venv/bin/python -m pytest tests/test_config.py tests/test_main_depth_policy.py tests/test_directfusion.py -q` -> 22 passed. | None. |
| 4 | Done | `src/spacerec/main.py`, `benchmarks/replay_smoke.py` | Added `--fusion`, `--direct-fusion`, direct/none/backend mode resolution, metric-source gating, RGB-depth alignment checks, explicit `--no-backend` direct conflict handling, DA3-free direct fill policy, and direct keyframe wiring into the shared result application path. Focused replay policy check: `.venv/bin/python -m pytest tests/test_config.py tests/test_main_depth_policy.py tests/test_directfusion.py tests/test_replay.py -q` -> 35 passed. Direct replay smoke: `.venv/bin/python benchmarks/replay_smoke.py sources/session_20260624_054320_194430108151D05A00 --frames 60 --direct-fusion` -> `direct_keyframes=11 direct_points=104863`. | None. |
| 5 | Done | `README.md`, `docs/MANUAL.md`, `docs/benchmarks.md`, `config.yaml`, `benchmarks/mesh_smoke.py`, `benchmarks/perf_matrix.py` | Docs/config/benchmarks now describe `backend`, `direct`, `none`, `auto`, OAK alignment requirements, `--no-backend` semantics, DA3-free direct mode, and drift/mesh limits. Main direct smoke: `.venv/bin/python -m spacerec.main --source sources/session_20260624_054320_194430108151D05A00 --fusion direct --no-realtime --max-frames 20 --no-viz --runtime-profile realtime --profile` -> direct fusion enabled, no backend wait, `map=40383pts`, `scale=1.000`, `done: 20 frames`. Direct mesh smoke: `.venv/bin/python benchmarks/mesh_smoke.py sources/session_20260624_054320_194430108151D05A00 --frames 60 --fusion direct --out-dir artifacts/mesh_direct_verify` -> readable PLY with `vertices=56302 faces=83309 read_vertices=56302 read_faces=83309`. | None. |
| 6 | Done | All changed files | Final local verification: `git diff --check` -> pass; `.venv/bin/python -m pytest tests/ -q` -> 92 passed; main direct smoke -> `map=40383pts`; direct replay smoke -> `direct_keyframes=11 direct_points=104863`; direct mesh smoke -> readable PLY at `artifacts/mesh_direct_verify/session_20260624_054320_194430108151D05A00.ply` with `56302` vertices and `83309` faces. Independent verifier `Russell` reran `git diff --check` -> pass and full tests -> `92 passed`, then reviewed that helper refactor preserved DA3-free direct mode, metric-depth gating, and RGB alignment checks. | No blocker. `config.yaml` had pre-existing `mesh.enabled: false`; preserved as unrelated dirty work while direct mesh is verified through `mesh_smoke.py`. |
