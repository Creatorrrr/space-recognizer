# Canonical Mesh Consolidation Goal

## Goal Summary
- Final outcome: OAK-D-Lite replay and live mesh output should not keep several stale copies of the same wall or surface in the default rendered/exported mesh. The default output must expose a confidence/recency-scored canonical mesh, while raw submaps remain available for debugging and future reintegration work.
- Scope: `/Users/chasoik/Projects/space-recognizer`, especially `src/spacerec/mesh.py`, `src/spacerec/viz.py`, `src/spacerec/main.py`, `src/spacerec/config.py`, `tests/test_mesh.py`, tests around config/persistence, and mesh docs/benchmarks.
- Non-goals: a full spatial-block TSDF rewrite, a full de-integration TSDF implementation, changing object detection behavior, changing VO/IMU behavior, destructive cleanup of unrelated files, or self-directed target increases.

## Baseline And Assumptions
- Current baseline:
  - `MeshMap.integrate_views()` creates a new `MeshSubmap` for each backend window when no explicit `submap_id` is given.
  - `MeshMap.combined_mesh()` concatenates every active submap, so overlapping windows can export several slightly offset copies of the same wall.
  - `Visualizer.log_mesh_submaps()` logs raw submaps under `world/mesh/submap_<id>` and does not clear deleted/superseded submaps unless an empty submap is logged.
  - `MeshCfg.max_active_submaps` is only a FIFO cap, not a confidence/latest canonicalization rule.
- Unknowns to verify:
  - How much canonical consolidation reduces duplicates on the recorded OAK sessions without over-deleting valid surfaces.
  - Whether the default Rerun view should show only canonical mesh or both canonical/raw during debugging.
- Assumptions:
  - Latest evidence is useful as a tie-breaker but must not override stronger support from older observations by itself.
  - A render/export-time canonicalization layer is the safest short-term fix and should not discard raw evidence needed for future spatial-block TSDF work.
  - Existing save/load format should remain compatible.

## Checkpoint Plan
| Step | Work | Stage Target | Verification | Done When |
|---|---|---|---|---|
| 1 | Review and patch this plan for the current canonical mesh goal. | Plan is concrete, bounded, and verifiable. | Inspect `GOAL_PLAN.md`, `git status`, mesh/viz/config/test source, and current mesh smoke commands. | Stale non-mesh plan is replaced and progress log begins. |
| 2 | Add tests that reproduce duplicate stale wall surfaces and config coverage. | Synthetic duplicate-plane submaps show raw combined output has multiple layers while canonical output keeps one layer. New config options load with safe defaults. | Focused pytest on `tests/test_mesh.py` and `tests/test_config.py`. | Duplicate reproduction and new option tests are in place. |
| 3 | Implement canonical mesh consolidation in `MeshMap`. | `canonical_mesh()` or equivalent scores overlapping face candidates by support, residual proxy, normal agreement, and recency, preserving high-confidence older surfaces over low-confidence newer noise. | Synthetic tests for duplicate removal, noisy latest pass, save/load/export compatibility. | Default export can return canonical mesh and tests prove duplicate planes collapse to one layer. |
| 4 | Track and clear removed/superseded submaps in visualization. | Default visualization logs canonical mesh, raw submaps remain optional/debug, and removed/superseded raw entities can be cleared. | Unit-level checks where practical plus focused source review. | Rerun default path no longer accumulates stale duplicate submap entities. |
| 5 | Update docs/config/benchmarks as needed. | Users can choose canonical/raw/both modes and understand limitations versus future spatial-block TSDF. | Docs grep/source review and config load test. | README/MANUAL/config describe canonical mesh behavior and options. |
| 6 | Run final verification and repair failures. | Related and full tests pass; recorded OAK mesh smoke still exports a readable mesh. | `git diff --check`, focused tests, full pytest, `benchmarks/mesh_smoke.py` on a recorded session. | Every final completion criterion is proven from current-state evidence. |

## Final Completion Criteria
- `GOAL_PLAN.md` matches this canonical mesh goal and includes progress evidence.
- Synthetic duplicate-wall tests prove raw submaps may contain several layers but the default canonical/export mesh keeps one selected surface layer.
- The scoring rule uses confidence/support evidence and recency; latest-only replacement is not the sole criterion.
- A high-support older surface is not blindly replaced by a single low-support newer noisy pass.
- Existing mesh TSDF generation, support filtering, Sim3 anchor correction, save/load, persistence merge, and export tests still pass.
- Default exported mesh uses canonical output unless explicitly configured otherwise.
- Default Rerun mesh display avoids stale duplicate raw submap accumulation; raw submaps remain inspectable via config/debug mode.
- Recorded OAK mesh smoke creates a readable PLY and does not regress to an empty mesh.
- Docs/config explain canonical mesh mode, raw debug mode, scoring limitations, and that full spatial-block TSDF/dirty-block rebuild is future work.
- Existing unrelated dirty worktree changes and untracked artifacts are preserved.

## Independent Verification Policy
- Independent final verification is required before marking the goal complete.
- Preferred verification: a separate subagent or clean worktree/fresh checkout reruns final criteria and reviews the diff.
- If unavailable, record why and run the strongest substitute: `git status`, `git diff --check`, full tests, focused mesh/config/persistence tests, recorded mesh smoke, artifact read-back, and focused diff review.
- Final report must include exact commands, pass/fail results, generated artifact paths if any, raw vs canonical behavior summary, and unresolved risks.

## Self-Directed Target Increase Policy
- User opt-in: no.
- After mandatory targets pass, stop without raising performance or quality targets.
- Do not expand scope into full spatial-block TSDF, full VIO/SLAM changes, broad refactors, or unrequested performance targets.

## Stop And Ask Conditions
- Correctness requires deleting raw evidence or breaking saved mesh compatibility.
- Required validation needs physical OAK hardware that is not connected.
- Required changes would overwrite unrelated user work.
- Destructive commands, credentials, external services, or large new dependencies are required.
- Evidence shows render/export-time canonicalization cannot satisfy the duplicate-wall requirement and a full spatial-block TSDF rewrite is required.

## Progress Log Rules
- After each checkpoint, log current step, changed files, verification command/result, remaining work, and blockers.
- Failed verification entries must include root cause, fix, and re-verification result.

## Progress Log
| Step | Status | Changed Files | Verification Result | Remaining / Blockers |
|---|---|---|---|---|
| 1 | Done | `GOAL_PLAN.md` | Stale IMU plan detected in the current worktree and replaced with this canonical mesh plan. Baseline focused tests before implementation: `.venv/bin/python -m pytest tests/test_mesh.py tests/test_config.py tests/test_persistence.py -q` -> 16 passed. | None. |
| 2 | Done | `tests/test_mesh.py`, `tests/test_config.py`, `src/spacerec/config.py`, `config.yaml` | Added duplicate-wall, high-support older surface, raw debug mode, normal group, residual proxy, export, removed-submap, and config option coverage. Focused check: `.venv/bin/python -m pytest tests/test_mesh.py tests/test_config.py -q` -> 18 passed after residual/normal additions. | None. |
| 3 | Done | `src/spacerec/mesh.py` | Implemented `raw_combined_mesh()`, `canonical_mesh()`, default canonical `combined_mesh()`, support/residual/normal/recency scoring, `export_ply(..., mode=None)`, and removed submap tracking. Synthetic tests prove five duplicate wall layers collapse to one canonical layer and high-support older surfaces resist single latest noisy passes. | None. |
| 4 | Done | `src/spacerec/viz.py`, `src/spacerec/main.py` | Added `log_canonical_mesh()` and `clear_mesh_submaps()`. `main._log_meshmap_update()` logs canonical mesh by default, keeps raw submaps for `raw|both`, and clears changed/removed raw submap entities in canonical mode. Independent verifier confirmed this by source review. | No Rerun mock test; covered by source review and default-mode code path. |
| 5 | Done | `README.md`, `docs/MANUAL.md`, `docs/benchmarks.md`, `benchmarks/mesh_smoke.py`, `config.yaml` | Docs/config describe `canonical`, `raw`, and `both`, scoring signals, limitations, and future spatial-block TSDF work. `mesh_smoke.py` now reports raw and canonical vertex/face counts. | None. |
| 6 | Done | All changed files | Final local verification: `git diff --check` -> pass; `.venv/bin/python -m pytest tests/ -q` -> 67 passed; `.venv/bin/python benchmarks/mesh_smoke.py sources/session_20260624_054320_194430108151D05A00 --frames 120 --out-dir artifacts/mesh_canonical_verify` -> readable PLY with `raw_vertices=99765 raw_faces=155014 vertices=99765 faces=155014 read_vertices=99765 read_faces=155014`. Independent verifier reran `git diff --check`, full pytest -> 67 passed, 120-frame mesh smoke -> readable PLY, duplicate/residual probes -> pass. | Recorded smoke has one submap, so real recorded multi-submap collapse evidence remains synthetic-test based. No blocker. |
