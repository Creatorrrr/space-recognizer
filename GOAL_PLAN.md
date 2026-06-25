# OAK IMU Camera Pose Correction Goal

## Goal Summary
- Final outcome: improve `space-recognizer` camera pose estimation so OAK IMU rotation and visual LK + PnP complement each other, then prove the improvement on the recorded replay `sources/session_20260624_054320_194430108151D05A00`.
- Scope: `src/spacerec/vo.py`, `src/spacerec/imu.py`, `src/spacerec/main.py`, `src/spacerec/config.py`, perf/replay diagnostics, tests, benchmarks, and targeted docs/config comments if needed.
- Non-goals: making IMU an unconditional primary pose source, naively combining IMU rotation with an unverified visual translation, broad SLAM/pose-graph rewrites, unrelated direct-fusion/backend refactors, destructive cleanup of unrelated dirty work, or self-directed target increases.

## Baseline And Assumptions
- Baseline must be measured from the current code before behavior changes.
- The main problem window is the user-observed camera rotation around 45 seconds. Use `44.5s-46.5s` as the focused window unless current logs show a more precise interval.
- Current likely failure mode: high-inlier visual PnP can converge to a wrong keyframe-relative rotation during a rotation/low-parallax segment, while IMU is only used as LK/PnP initialization and does not veto or constrain the final visual solution.
- Existing OAK session health is assumed usable, but the run must verify the file exists and replay commands actually complete.
- Numeric targets must be derived from measured baseline. Do not invent unsupported thresholds as final success evidence before measuring.

## Checkpoint Plan
| Step | Work | Stage Target | Verification | Done When |
|---|---|---|---|---|
| 1 | Review and patch this plan, inspect worktree and relevant source. | Plan is current, bounded, verifiable, and unrelated dirty work is preserved. | Read `GOAL_PLAN.md`, `git status`, VO/IMU/main/config/perf/replay tests. | Stale plan is replaced and source map is recorded in the progress log. |
| 2 | Establish baseline on the real session. | Produce baseline artifacts with 45s focused metrics: visual-vs-IMU rotation residual, chosen pose source, inliers, tracked points, lost state, loop correction status if available. | Run the required focused replay and any additional diagnostic script needed to compute residuals. | Baseline CSV/artifact path and 45s before numbers are recorded. |
| 3 | Add RED tests for the intended behavior. | Tests cover rotation residual computation, bad-IMU protection, wrong-visual/right-IMU divergence handling, and no unsafe fusion of rejected poses. | Run focused pytest and confirm tests fail for the expected missing behavior. | Red tests fail for expected reasons, not syntax/import errors. |
| 4 | Implement IMU reliability checks, residual diagnostics, and candidate selection gate. | Visual pose remains accepted when it agrees with trusted IMU; visual pose is marked low-confidence/rejected or replaced by a constrained candidate when it diverges beyond threshold; bad IMU does not override good visual. | Focused tests pass; perf logs include residual/source/confidence fields. | Tests prove both correction and bad-IMU fallback behavior. |
| 5 | Add or wire a constrained fallback for divergent rotations. | When visual and trusted IMU disagree, attempt an IMU-rotation-constrained translation estimate and accept it only if reprojection/motion gates pass; otherwise hold/reject without polluting map/backend fusion. | Synthetic tests and focused replay diagnostics. | Divergent high-inlier wrong visual no longer silently updates global reconstruction. |
| 6 | Final real-session verification and regression checks. | Full and focused replays prove improvement at the 45s window and no major regressions. | Run required replay commands, focused tests, broader tests, `git diff --check`, and independent verification if available. | Every final completion criterion is proven from current-state evidence. |

## Final Completion Criteria
- Current baseline is measured and saved before implementation.
- The implementation exposes diagnostics equivalent to `imu_visual_rot_residual_deg`, `rotation_source` or `pnp_candidate_source`, low-confidence/rejected state, and fusion-skipped state where applicable.
- Visual LK + PnP remains the main source when it agrees with a trusted IMU signal.
- Trusted IMU can prevent or correct a high-inlier visual rotation solution that diverges from keyframe-since IMU by the configured threshold.
- A bad or unreliable IMU cannot forcibly override a good visual solution.
- Rejected or low-confidence poses do not get fused into the world map/backend/mesh path as if they were trustworthy.
- The focused real-session replay is run:
  ```powershell
  .\.venv\Scripts\python.exe -m spacerec.main --source "sources\session_20260624_054320_194430108151D05A00" --max-seconds 47 --no-viz --perf-log artifacts\imu_pose_after_45s.csv
  ```
- The full real-session replay is run:
  ```powershell
  .\.venv\Scripts\python.exe -m spacerec.main --source "sources\session_20260624_054320_194430108151D05A00" --no-viz --perf-log artifacts\imu_pose_after_full.csv
  ```
- If the measured baseline 45s keyframe-since IMU/visual rotation residual is at least 6 degrees, the after result must show at least 50% reduction or an absolute residual of at most 3 degrees for the problematic accepted/fused pose. If this is not possible because the IMU is rejected by reliability gates, report the goal as infeasible rather than passing.
- Related unit tests, replay/perf tests, and `python -m compileall src tests` pass.
- Final report includes exact commands, results/exit codes, artifact paths, before/after 45s metrics, completion-criteria pass/fail, and remaining risks.

## Independent Verification Policy
- Preferred final verification: a separate subagent or clean worktree/fresh checkout reruns final criteria and reviews the diff.
- If unavailable, record why and run the strongest substitute: `git status`, `git diff --check`, focused tests, broader tests, full and focused replay artifact regeneration, artifact read-back, and focused diff review.
- Do not mark the goal complete unless the verification evidence is surfaced in the final response.

## Self-Directed Target Increase Policy
- User opt-in: no.
- After mandatory targets pass, stop without raising performance or quality targets.
- Do not expand scope into full SLAM, loop closure, pose graph optimization, broad depth/backend redesign, or unrelated performance goals.

## Stop And Ask Conditions
- Correctness requires deleting or reverting unrelated user work.
- The recorded session is missing or unreadable and no equivalent user-approved replay is available.
- Required validation needs live OAK hardware rather than the recorded session.
- Destructive commands, credentials, external services, or large new dependencies are required.
- Evidence shows the IMU signal is not timestamp/extrinsic reliable enough to satisfy the goal under the stated constraints.

## Progress Log Rules
- After each checkpoint, log current step, changed files, verification command/result, remaining work, and blockers.
- Failed verification entries must include root cause, fix, and re-verification result.

## Progress Log
| Step | Status | Changed Files | Verification Result | Remaining / Blockers |
|---|---|---|---|---|
| 1 | Done | `GOAL_PLAN.md` | Existing plan was stale direct-fusion work and was replaced with the active IMU pose-correction goal. Worktree is on `codex/orbslam3-oak-pose-provider`; unrelated untracked benchmark/docs files were left untouched. | None. |
| 2 | Done | `artifacts/imu_pose_baseline_45s.csv`, `artifacts/imu_pose_baseline_nogate_config.yaml`, `artifacts/imu_pose_baseline_nogate_45s.csv`, `artifacts/imu_pose_45s_compare.json` | Current-code focused baseline command completed. A no-gate diagnostic baseline preserved pre-gate behavior while exposing residuals: in 44.5-46.5s, 6/6 keyframes were fused; 5 had residual >6 deg; max residual was 9.03 deg. | None. |
| 3 | Done | `tests/test_vo.py`, `tests/test_main_pose_safety.py` | RED checks failed for expected missing `rotation_residual_deg`, `_should_send_reconstruction_keyframe`, and then policy mismatches before implementation changes. | None. |
| 4 | Done | `src/spacerec/vo.py`, `src/spacerec/main.py`, `src/spacerec/perf.py`, `src/spacerec/config.py`, `config.yaml` | Added residual diagnostics, IMU-constrained translation candidate, bad-IMU reprojection guard, low-confidence/fusion-skip flags, perf CSV fields, reconstruction keyframe safety gate, and world-update safety gate. Focused tests: `.\.venv\Scripts\python.exe -m pytest tests\test_vo.py tests\test_main_pose_safety.py -q` -> 18 passed. | None. |
| 5 | Done | `artifacts/imu_pose_after_45s.csv`, `artifacts/imu_pose_45s_compare.json` | Focused after replay completed. In 44.5-46.5s, baseline fused 6/6 keyframes including 5 keyframes with residual >6 deg; after fused 1/6 keyframes and marked/skipped all 5 keyframes >6 deg for reconstruction and world updates. | None. |
| 6 | Done | `artifacts/imu_pose_after_full.csv`, `artifacts/imu_pose_after_full_summary.json` | Full replay completed: 750 frames, 0 lost rows, 168 low-confidence/fusion-skipped/world-update-skipped rows, final map_points 461618. Local checks completed: `git diff --check` pass, `.\.venv\Scripts\python.exe -m compileall src tests` pass, `.\.venv\Scripts\python.exe -m pytest tests -q` -> 124 passed. | Independent verification requested; final response must include its result or residual risk if unavailable. |
