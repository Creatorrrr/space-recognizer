# IMU-Aided Visual Odometry And Reconstruction Goal

## Goal Summary
- Final outcome: `space-recognizer` uses OAK-D-Lite IMU data to improve 3D reconstruction quality by feeding gyro-derived rotation priors into the existing LK + PnP visual odometry path, while preserving the current point-cloud/object/backend pipeline.
- Scope: `/Users/chasoik/Projects/space-recognizer`, especially `src/spacerec/imu.py`, `src/spacerec/capture.py`, `src/spacerec/oak.py`, `src/spacerec/replay.py`, `src/spacerec/vo.py`, `src/spacerec/main.py`, `src/spacerec/config.py`, tests, benchmarks, and docs.
- Non-goals: full VIO/factor-graph SLAM, accelerometer double-integration for translation, replacing DA3/backend reconstruction, broad recorder-app rewrites, unrelated dirty-worktree cleanup, destructive git operations, or self-directed target increases.

## Baseline And Assumptions
- Current baseline:
  - `Frame.imu` carries at most a latest accel/gyro sample for diagnostics.
  - `VisualOdometry.process(gray, depth, ts, exclude_mask)` ignores IMU and uses LK optical flow plus `solvePnPRansac`.
  - `ReconstructionBackend` consumes visual VO poses through `BackendKeyframe.T_wc_live`.
  - Recorded OAK sessions contain about 10 IMU events per RGB frame interval at the default 100 Hz IMU / 10 FPS RGB cadence.
  - Baseline replay smoke examples:
    - `sources/session_20260624_054320_194430108151D05A00`: `lost=0`, `avg_tracked=200.0`, `avg_inlier=0.92`.
    - `sources/session_20260624_055321_194430108151D05A00`: `lost=1`, `avg_tracked=50.8`, `avg_inlier=0.79`.
- Unknowns to verify:
  - Live OAK RGB/depth/IMU timestamp-domain alignment.
  - Availability and direction of IMU-to-RGB-camera extrinsics in live DepthAI calibration and recorded metadata.
  - Whether gyro priors improve high-rotation replay segments without regressing stable segments.
- Assumptions:
  - Use DepthAI calibration APIs such as `getImuToCameraExtrinsics(CAM_A)` for live devices when available.
  - Use recorded metadata `imuExtrinsics` plus camera extrinsics as a replay fallback.
  - Use gyro only for rotation priors and gating. Use accelerometer only for optional stationary gravity diagnostics, not translation integration.
  - Keep every new IMU-assisted behavior behind config flags and preserve visual-only behavior when disabled.

## Checkpoint Plan
| Step | Work | Stage Target | Verification | Done When |
|---|---|---|---|---|
| 1 | Review and patch this plan for the current IMU goal. | Plan is concrete, bounded, and verifiable. | Inspect `GOAL_PLAN.md`, `git status`, current IMU/VO/replay source, and recorded session cadence. | Old mesh plan is replaced with this IMU plan and progress log begins. |
| 2 | Add IMU core data model and pure math utilities. | `ImuSample`, gyro integration, bias estimation, gravity direction, and camera-frame conversion exist without touching production VO behavior. | New `tests/test_imu.py` with red/green TDD. | Constant/nonuniform gyro, bias, gravity, and rotation-frame tests pass. |
| 3 | Preserve frame-interval IMU samples in live and replay frame contracts. | `Frame.imu_samples` carries all samples in the previous-frame-to-current-frame interval while legacy `Frame.imu` remains available. | Replay unit tests and a recorded-session cadence check. | Recorded sessions expose a median of about 10 IMU samples per RGB interval; existing replay tests still pass. |
| 4 | Load and validate IMU-to-camera extrinsics and timestamp metadata. | Live OAK metadata and recorded replay metadata expose a usable `R_cam_imu` or an explicit unavailable state. | Unit tests using recorded metadata and focused diagnostic output. | Extrinsics parsing is covered, and missing calibration safely disables IMU priors instead of guessing. |
| 5 | Add gyro-aided LK initial-flow and PnP prior to `VisualOdometry`. | Optional `R_delta_prev`, `R_since_keyframe`, and `omega_norm` inputs improve rotation tracking while falling back to visual-only when unsafe. | Synthetic VO tests and existing VO regression tests. | Rotation synthetic test improves with IMU prior; lateral translation and visual-only paths do not regress. |
| 6 | Wire IMU priors through `main.py` and replay smoke/benchmark. | Main pipeline computes gyro rotation windows and passes priors to VO when enabled; benchmark compares IMU off/on. | New or extended IMU replay smoke. | Off/on benchmark reports `lost`, `avg_tracked`, `avg_inlier`, keyframes, and prior/fallback counts. |
| 7 | Add gyro-based keyframe blur gating. | High-angular-rate frames can be excluded or delayed from backend keyframes with starvation protection. | Unit tests and replay smoke. | Blurry frames are not promoted to backend keyframes unless starvation fallback is triggered. |
| 8 | Update docs/config and run final verification. | README/MANUAL/config describe IMU modes, limits, benchmark commands, and safe defaults. | Full tests, replay smoke, IMU benchmark, diff check, independent verification or strongest substitute. | Every final criterion is proven from current-state evidence. |

## Final Completion Criteria
- `.venv/bin/python -m pytest tests/ -q` passes.
- IMU pure utility tests prove gyro integration, bias estimation, gravity direction, and camera-frame conversion.
- `Frame.imu_samples` is populated for recorded OAK replay without breaking legacy `Frame.imu`.
- Live/replay extrinsics handling never silently guesses a camera/IMU transform when unavailable.
- Visual-only VO remains available and compatible when IMU config flags are disabled.
- Gyro-assisted VO has synthetic test coverage showing improved pure-rotation tracking and no regression for existing lateral translation.
- An IMU off/on replay benchmark runs on at least `sources/session_20260624_054320_194430108151D05A00` and `sources/session_20260624_055321_194430108151D05A00`, reporting comparable metrics.
- On the difficult recorded session, IMU-assisted mode shows either fewer `lost` frames or improved `avg_tracked`/`avg_inlier` without a material regression on the stable session. If the data does not support improvement, the implementation must keep IMU priors off by default and document the evidence.
- Existing replay/backend/object/mesh smoke paths are not intentionally broken.
- Docs explain IMU mode, limits, benchmark command, and why accelerometer translation integration is out of scope.
- Existing unrelated dirty worktree changes and untracked artifacts are preserved.

## Independent Verification Policy
- Independent final verification is required before marking the goal complete.
- Preferred verification: a separate subagent or clean worktree/fresh checkout reruns final criteria and reviews the diff.
- If unavailable, record why and run the strongest substitute: `git status`, `git diff --check`, full tests, focused IMU tests, replay smoke, IMU off/on benchmark, docs check, and focused diff review.
- Final report must include exact commands, pass/fail results, benchmark metrics, generated artifact paths if any, and unresolved risks.

## Self-Directed Target Increase Policy
- User opt-in: no.
- After mandatory targets pass, stop without raising performance or quality targets.
- Do not expand scope into full VIO, new SLAM stacks, broad refactors, or unrequested performance targets.

## Stop And Ask Conditions
- Live or recorded IMU calibration is unavailable and any fallback would require guessing axis/sign conventions.
- IMU priors repeatedly worsen replay metrics and a design change beyond the stated approach is needed.
- Required validation needs physical OAK hardware that is not connected.
- Required changes would overwrite unrelated user work.
- Destructive commands, credentials, external services, or large new dependencies are required.
- Evidence shows the target is infeasible with the current recordings or repo constraints.

## Progress Log Rules
- After each checkpoint, log current step, changed files, verification command/result, remaining work, and blockers.
- Failed verification entries must include root cause, fix, and re-verification result.

## Progress Log
| Step | Status | Changed Files | Verification Result | Remaining / Blockers |
|---|---|---|---|---|
| 1 | Done | `GOAL_PLAN.md` | Stale mesh goal detected and replaced with this IMU-aided VO/reconstruction plan. Baseline: `pytest tests/test_vo.py tests/test_replay.py tests/test_config.py -q` -> 12 passed; `replay_smoke.py ... --frames 120` -> stable `lost=0 avg_tracked=200.0 avg_inlier=0.92`, difficult `lost=1 avg_tracked=50.8 avg_inlier=0.79`. | None. |
| 2 | Done | `src/spacerec/imu.py`, `tests/test_imu.py` | TDD red -> green. `pytest tests/test_imu.py -q` covered gyro integration, bias, gravity, camera-frame conversion, camera rotation prior, and keyframe gating helper. | None. |
| 3 | Done | `src/spacerec/capture.py`, `src/spacerec/replay.py`, `src/spacerec/oak.py`, `tests/test_replay.py` | Replay test now proves `Frame.imu_samples` carries frame-interval samples while legacy `Frame.imu` remains. Recorded cadence check showed median 10 IMU samples per RGB interval for both target sessions. | None. |
| 4 | Done | `src/spacerec/replay.py`, `src/spacerec/oak.py`, `tests/test_replay.py` | Replay metadata tests cover IMU-to-camera extrinsics path composition and missing-path fallback. Manual diagnostic found recorded `R_cam_imu` and frame metadata carries `imu_to_camera_rotation`. | Live hardware calibration remains untested on physical OAK in this run. |
| 5 | Done | `src/spacerec/vo.py`, `tests/test_vo.py` | Synthetic pure-rotation test: visual-only path loses tracking at 9 deg yaw while gyro prior path recovers `R_delta.T`; lateral translation regression still passes. | None. |
| 6 | Done | `src/spacerec/main.py`, `benchmarks/replay_smoke.py`, `src/spacerec/config.py`, `config.yaml`, tests | `pytest tests/test_imu.py tests/test_config.py tests/test_vo.py -q` -> 17 passed. After boundary integration fix, `replay_smoke.py ... --frames 120 --compare-imu` reports stable session unchanged and difficult session `lost=1 -> 0`, `avg_tracked=50.8 -> 50.9`, `avg_inlier=0.79 -> 0.79`. | IMU remains off by default until broader live/recorded evidence exists. |
| 7 | Done | `src/spacerec/imu.py`, `src/spacerec/main.py`, `benchmarks/replay_smoke.py`, `tests/test_imu.py` | Unit test proves high angular-rate backend keyframes are skipped until starvation fallback. Replay smoke reports `imu_blur_skipped_kf=0` and `imu_blur_forced_kf=0` on both provided sessions because they do not exceed the threshold. | Need faster-rotation recordings to observe nonzero gating in real data. |
| 8 | Done | `README.md`, `docs/MANUAL.md`, `docs/benchmarks.md`, `src/spacerec/oak.py`, `src/spacerec/imu.py`, `tests/test_oak_depth.py`, `tests/test_imu.py` | Independent verifier found live OAK IMU windowing was not timestamp bounded and replay priors under-integrated frame boundaries. Fixed with timestamp-aligned OAK IMU windows and optional estimator `t0/t1`. Final verification: `pytest tests/ -q` -> 60 passed; `git diff --check` -> pass; `replay_smoke.py ... --frames 120 --compare-imu` -> stable unchanged, difficult `lost=1 -> 0`; `mesh_smoke.py ... --frames 60` -> 56,302 vertices / 83,309 faces round-trip read. | Live OAK hardware was not connected, so physical calibration/timestamp behavior remains a field validation item. |
