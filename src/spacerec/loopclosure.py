"""Intra-session object-anchor loop correction.

This module estimates a live-to-global Sim(3) correction from currently visible
object observations to stable objects already stored in the global registry.
It intentionally leaves VO poses untouched; callers should apply accepted
results through GlobalMap.set_correction_target().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import LoopClosureCfg
from .geometry import (SIM3_IDENTITY, Sim3, sim3_apply, sim3_compose,
                       sim3_inverse, umeyama_sim3)
from .objects import ObjectRegistry, Observation, WorldObject

_INF = 1e9


@dataclass(frozen=True)
class LoopCorrectionResult:
    attempted: bool
    accepted: bool
    reason: str
    T_global_live: Sim3 | None = None
    matches: tuple[tuple[int, int], ...] = ()
    rms: float = 0.0
    spread: float = 0.0
    yaw_delta_deg: float = 0.0
    translation_delta: float = 0.0
    scale_delta: float = 0.0

    @property
    def match_count(self) -> int:
        return len(self.matches)


class LoopClosureEstimator:
    """Estimate smoothable live->global corrections from static object anchors."""

    def __init__(self, cfg: LoopClosureCfg):
        self.cfg = cfg
        self._last_accept_ts: float | None = None

    def estimate(
        self,
        frame_index: int,
        ts: float,
        observations_live: list[Observation],
        registry: ObjectRegistry,
        T_global_live_current: Sim3 = SIM3_IDENTITY,
    ) -> LoopCorrectionResult:
        if not self.cfg.enabled:
            return LoopCorrectionResult(False, False, "disabled")

        every = max(1, int(self.cfg.check_every_frames))
        if frame_index % every != 0:
            return LoopCorrectionResult(False, False, "skipped_frame")

        if (self._last_accept_ts is not None
                and ts - self._last_accept_ts < float(self.cfg.min_accept_interval_s)):
            return LoopCorrectionResult(False, False, "accept_interval")

        anchors = self._stable_anchors(registry)
        if (len(observations_live) < int(self.cfg.min_matches)
                or len(anchors) < int(self.cfg.min_matches)):
            return LoopCorrectionResult(True, False, "insufficient_inputs")

        pairs = self._match(observations_live, anchors, T_global_live_current)
        min_matches = int(self.cfg.min_matches)
        if len(pairs) < min_matches:
            return LoopCorrectionResult(True, False, "insufficient_matches",
                                        matches=self._pair_ids(pairs))

        distinct = {observations_live[i].det.cls_name for i, _, _ in pairs}
        if len(distinct) < int(self.cfg.min_distinct_classes):
            return LoopCorrectionResult(True, False, "insufficient_classes",
                                        matches=self._pair_ids(pairs))

        src = np.array([observations_live[i].position for i, _, _ in pairs],
                       dtype=np.float64)
        dst = np.array([obj.position for _, obj, _ in pairs], dtype=np.float64)
        spread = float(np.linalg.norm(dst - dst.mean(axis=0), axis=1).mean())
        if spread < float(self.cfg.min_spread):
            return LoopCorrectionResult(True, False, "low_spread",
                                        matches=self._pair_ids(pairs),
                                        spread=spread)

        T_raw = umeyama_sim3(src, dst, with_scale=bool(self.cfg.allow_scale))
        T_candidate = _yaw_only_sim3(T_raw, src, dst, allow_scale=bool(self.cfg.allow_scale))
        aligned = sim3_apply(T_candidate, src)
        rms = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
        if rms > float(self.cfg.max_rms):
            return LoopCorrectionResult(True, False, "rms_abs",
                                        T_global_live=T_candidate,
                                        matches=self._pair_ids(pairs),
                                        rms=rms, spread=spread)
        if rms > float(self.cfg.max_rms_frac) * max(spread, 1e-9):
            return LoopCorrectionResult(True, False, "rms_frac",
                                        T_global_live=T_candidate,
                                        matches=self._pair_ids(pairs),
                                        rms=rms, spread=spread)

        delta = sim3_compose(T_candidate, sim3_inverse(T_global_live_current))
        yaw_delta_deg = abs(_yaw_degrees(delta[1]))
        translation_delta = float(np.linalg.norm(delta[2]))
        scale_delta = abs(float(delta[0]) - 1.0)
        if yaw_delta_deg > float(self.cfg.max_yaw_delta_deg):
            return LoopCorrectionResult(True, False, "yaw_delta",
                                        T_global_live=T_candidate,
                                        matches=self._pair_ids(pairs),
                                        rms=rms, spread=spread,
                                        yaw_delta_deg=yaw_delta_deg,
                                        translation_delta=translation_delta,
                                        scale_delta=scale_delta)
        if translation_delta > float(self.cfg.max_translation_delta):
            return LoopCorrectionResult(True, False, "translation_delta",
                                        T_global_live=T_candidate,
                                        matches=self._pair_ids(pairs),
                                        rms=rms, spread=spread,
                                        yaw_delta_deg=yaw_delta_deg,
                                        translation_delta=translation_delta,
                                        scale_delta=scale_delta)
        if scale_delta > float(self.cfg.max_scale_delta):
            return LoopCorrectionResult(True, False, "scale_delta",
                                        T_global_live=T_candidate,
                                        matches=self._pair_ids(pairs),
                                        rms=rms, spread=spread,
                                        yaw_delta_deg=yaw_delta_deg,
                                        translation_delta=translation_delta,
                                        scale_delta=scale_delta)

        self._last_accept_ts = ts
        return LoopCorrectionResult(True, True, "accepted",
                                    T_global_live=T_candidate,
                                    matches=self._pair_ids(pairs),
                                    rms=rms, spread=spread,
                                    yaw_delta_deg=yaw_delta_deg,
                                    translation_delta=translation_delta,
                                    scale_delta=scale_delta)

    def _stable_anchors(self, registry: ObjectRegistry) -> list[WorldObject]:
        return [
            o for o in registry.objects.values()
            if o.n_obs >= int(self.cfg.min_observations) and not o.is_dynamic
        ]

    def _match(
        self,
        observations_live: list[Observation],
        anchors: list[WorldObject],
        T_global_live_current: Sim3,
    ) -> list[tuple[int, WorldObject, float]]:
        cost = np.full((len(observations_live), len(anchors)), _INF,
                       dtype=np.float64)
        for i, obs in enumerate(observations_live):
            predicted_global = sim3_apply(
                T_global_live_current,
                np.asarray(obs.position, dtype=np.float64)[None],
            )[0]
            for j, obj in enumerate(anchors):
                if obs.det.cls_name != obj.cls_name:
                    continue
                gate = min(
                    float(self.cfg.max_match_distance),
                    max(0.25, float(self.cfg.match_size_factor) * float(obj.size)),
                )
                dist = float(np.linalg.norm(predicted_global - obj.position))
                if dist > gate:
                    continue
                if obs.emb is not None and obj.embedding is not None:
                    cos = float(obs.emb @ obj.embedding)
                    if cos < float(self.cfg.min_cos):
                        continue
                    cost[i, j] = dist / gate + float(self.cfg.app_weight) * (1.0 - cos)
                elif self.cfg.require_appearance:
                    continue
                else:
                    cost[i, j] = dist / gate

        rows, cols = linear_sum_assignment(np.minimum(cost, _INF))
        pairs = [
            (int(r), anchors[int(c)], float(cost[r, c]))
            for r, c in zip(rows, cols)
            if cost[r, c] < _INF
        ]
        pairs.sort(key=lambda item: item[2])
        return pairs

    @staticmethod
    def _pair_ids(pairs: list[tuple[int, WorldObject, float]]) -> tuple[tuple[int, int], ...]:
        return tuple((int(obs_idx), int(obj.obj_id)) for obs_idx, obj, _ in pairs)


def _yaw_degrees(R: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))


def _yaw_only_sim3(T: Sim3, src: np.ndarray, dst: np.ndarray,
                   allow_scale: bool) -> Sim3:
    s_raw, R_raw, _ = T
    yaw = float(np.arctan2(R_raw[0, 2], R_raw[2, 2]))
    c, sn = float(np.cos(yaw)), float(np.sin(yaw))
    R_yaw = np.array([
        [c, 0.0, sn],
        [0.0, 1.0, 0.0],
        [-sn, 0.0, c],
    ], dtype=np.float64)
    s = float(s_raw) if allow_scale else 1.0
    t = dst.mean(axis=0) - s * R_yaw @ src.mean(axis=0)
    return s, R_yaw, t
