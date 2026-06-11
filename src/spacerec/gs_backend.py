"""Gaussian Splatting 품질 레이어 백엔드 (CUDA 전용, docs/upgrade-plan.md Tier 3).

별도 프로세스에서 키프레임을 받아 `gaussian.period_s` 주기로 정적 공간의
3D Gaussian 지도를 점진 최적화한다 (WildGS-SLAM 스타일의 가치 — 동적 물체가
제거된 렌더링 가능한 지도 — 를 gsplat + 기존 파이프라인 산출물로 획득).

설계 원칙:
- voxel 지도(기하·증거 레이어)와 완전히 독립 — 끄면 기존 동작과 동일.
- **live 좌표계에서 최적화**하고 표시 변환(T_global_live)은 메인 프로세스가
  담당한다 (라이브 포인트 미리보기와 동일한 정책 — Sim3 동기화 문제 회피).
- 동적 물체 픽셀(YOLOE dyn_mask)은 spawn과 손실 양쪽에서 제외 — WildGS의
  uncertainty 역할을 기존 검출 mask가 대신한다.
- anytime: 주기 내에 끝나는 만큼만 최적화하고 다음 주기로 이월.
- `holdout_every` 번째 키프레임은 학습에서 제외하고 PSNR 검증에만 사용.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from .backend import BackendKeyframe
from .config import GaussianCfg

_PRUNE_OPACITY = 0.05   # 이보다 투명한 gaussian은 주기마다 제거
_SPAWN_ALPHA = 0.5      # 렌더 알파가 이보다 낮은 픽셀에만 신규 spawn (중복 방지)
_MAX_TRAIN_KF = 48      # 학습 키프레임 링버퍼 (메모리 상한)
_VIZ_MAX_PTS = 200_000  # 미리보기로 보내는 최대 포인트 수
_DUTY = 0.5             # 주기 중 최적화에 쓰는 비율 — 0.8은 백엔드/라이브를
                        # GPU에서 굶겼다 (frames.mov 실측: 백엔드 윈도 97s)
_MIN_FREE_VRAM = 1.5e9  # 전역 VRAM 여유가 이보다 작으면 이번 주기 최적화 스킵
                        # (WDDM 스왑 진입 = 전 프로세스 10~30x 슬로다운 방지)


@dataclass
class GSResult:
    means: np.ndarray            # (N,3) live frame
    colors: np.ndarray           # (N,3) uint8
    n_gaussians: int = 0
    psnr: float | None = None    # held-out 키프레임 평균 PSNR (학습 미사용 뷰)
    render: np.ndarray | None = None   # 최신 키프레임 시점 렌더 (H,W,3 uint8)
    render_kf_id: int = -1
    runtime_s: float = 0.0
    error: str | None = None     # gsplat 사용 불가 등 — 메인은 경고 후 계속


@dataclass
class _TrainView:
    kf_id: int
    rgb: "object"                # torch (H,W,3) float, CPU pinned
    depth: "object"              # torch (H,W) float — 보정된 live depth
    static: "object"             # torch (H,W) bool — 동적 물체 밖 ∧ depth 유효
    viewmat: "object"            # torch (4,4) w2c
    K: "object"                  # torch (3,3)


class _GsWorker:
    def __init__(self, cfg: GaussianCfg, device: str = "cuda"):
        import torch
        from gsplat import rasterization  # JIT 컴파일은 첫 렌더에서 일어남

        self.torch = torch
        self.rasterize = rasterization
        self.cfg = cfg
        self.device = device
        self.params: dict[str, torch.Tensor] = {}
        self.train_views: list[_TrainView] = []
        self.holdout_views: list[_TrainView] = []
        self.opt = None

    # ---- keyframe 변환 ---------------------------------------------------
    def _to_view(self, kf: BackendKeyframe) -> _TrainView | None:
        torch = self.torch
        if kf.K is None or kf.raw_depth is None:
            return None
        a, b = kf.calib_ab
        depth = a * kf.raw_depth.astype(np.float32) + b
        static = depth > 1e-6
        if kf.dyn_mask is not None:
            static &= ~kf.dyn_mask
        T = kf.T_wc_live
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = T[:3, :3].T
        w2c[:3, 3] = -T[:3, :3].T @ T[:3, 3]
        return _TrainView(
            kf_id=kf.kf_id,
            rgb=torch.from_numpy(np.ascontiguousarray(kf.rgb)).float() / 255.0,
            depth=torch.from_numpy(depth),
            static=torch.from_numpy(static),
            viewmat=torch.from_numpy(w2c),
            K=torch.from_numpy(kf.K.astype(np.float32)))

    # ---- 렌더 ------------------------------------------------------------
    def _render(self, view: _TrainView, grad: bool = True):
        torch = self.torch
        p = self.params
        H, W = view.depth.shape
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out, alpha, _ = self.rasterize(
                p["means"],
                torch.nn.functional.normalize(p["quats"], dim=-1),
                p["log_scales"].exp(),
                torch.sigmoid(p["logit_op"]),
                torch.sigmoid(p["colors"]),
                view.viewmat.to(self.device)[None],
                view.K.to(self.device)[None], W, H,
                render_mode="RGB+ED")
        return out[0, ..., :3], out[0, ..., 3], alpha[0, ..., 0]

    # ---- spawn -----------------------------------------------------------
    def _spawn(self, view: _TrainView) -> None:
        torch = self.torch
        s = self.cfg.spawn_stride
        H, W = view.depth.shape
        need = view.static.clone()
        if self.params:
            _, _, alpha = self._render(view, grad=False)
            need &= alpha.cpu() < _SPAWN_ALPHA
        vs, us = torch.nonzero(need[::s, ::s], as_tuple=True)
        if len(vs) == 0:
            return
        vs, us = vs * s, us * s
        z = view.depth[vs, us]
        K = view.K
        x = (us.float() - K[0, 2]) / K[0, 0] * z
        y = (vs.float() - K[1, 2]) / K[1, 1] * z
        cam = torch.stack([x, y, z], dim=1)
        c2w = torch.linalg.inv(view.viewmat)
        means = cam @ c2w[:3, :3].T + c2w[:3, 3]
        n = len(means)
        rgb = view.rgb[vs, us].clamp(1e-4, 1 - 1e-4)
        new = {
            "means": means,
            "quats": torch.cat([torch.ones(n, 1), torch.zeros(n, 3)], dim=1),
            # 픽셀 풋프린트 비례 등방 스케일 (stride만큼 벌어진 간격을 덮음)
            "log_scales": (z / K[0, 0] * s * 0.6).clamp_min(1e-6).log()
                          .unsqueeze(1).expand(n, 3).contiguous(),
            "logit_op": torch.full((n,), -0.85),   # opacity ≈ 0.3
            "colors": torch.log(rgb / (1 - rgb)),  # inverse sigmoid
        }
        for k, v in new.items():
            v = v.to(self.device)
            if k in self.params:
                v = torch.cat([self.params[k].detach(), v])
            self.params[k] = v.requires_grad_(True)
        self._rebuild_optimizer()

    def _rebuild_optimizer(self) -> None:
        self.opt = self.torch.optim.Adam([
            {"params": [self.params["means"]], "lr": 2e-4},
            {"params": [self.params["quats"]], "lr": 1e-3},
            {"params": [self.params["log_scales"]], "lr": 5e-3},
            {"params": [self.params["logit_op"]], "lr": 5e-2},
            {"params": [self.params["colors"]], "lr": 2e-2},
        ])

    # ---- prune -----------------------------------------------------------
    def _prune(self) -> None:
        torch = self.torch
        op = torch.sigmoid(self.params["logit_op"].detach())
        keep = op > _PRUNE_OPACITY
        if len(op) > self.cfg.max_gaussians:
            thresh = op.topk(self.cfg.max_gaussians).values[-1]
            keep &= op >= thresh
        if keep.all():
            return
        for k in self.params:
            self.params[k] = (self.params[k].detach()[keep]
                              .contiguous().requires_grad_(True))
        self._rebuild_optimizer()

    # ---- 최적화 ----------------------------------------------------------
    def _optimize(self, deadline: float) -> int:
        torch = self.torch
        steps = 0
        for i in range(self.cfg.opt_steps):
            if time.monotonic() > deadline:
                break
            view = self.train_views[
                np.random.randint(len(self.train_views))]
            rgb, depth, alpha = self._render(view, grad=True)
            tgt = view.rgb.to(self.device, non_blocking=True)
            static = view.static.to(self.device, non_blocking=True)
            d_tgt = view.depth.to(self.device, non_blocking=True)
            loss = (rgb - tgt).abs()[static].mean()
            d_valid = static & (depth > 1e-6)
            if d_valid.any():
                loss = loss + self.cfg.depth_loss_w * (
                    (depth - d_tgt).abs()[d_valid].mean())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            steps += 1
        return steps

    def _eval_psnr(self) -> float | None:
        torch = self.torch
        if not self.holdout_views or not self.params:
            return None
        vals = []
        for view in self.holdout_views[-4:]:
            rgb, _, _ = self._render(view, grad=False)
            tgt = view.rgb.to(self.device)
            static = view.static.to(self.device)
            mse = ((rgb - tgt) ** 2)[static].mean().item()
            if mse > 0:
                vals.append(10 * np.log10(1.0 / mse))
        return float(np.mean(vals)) if vals else None

    # ---- 메인 루프 -------------------------------------------------------
    def run(self, in_q: mp.Queue, out_q: mp.Queue) -> None:
        torch = self.torch
        pending: list[BackendKeyframe] = []
        last_run = time.monotonic()
        while True:
            try:
                item = in_q.get(timeout=0.1)
                if item is None:
                    return
                pending.append(item)
            except queue.Empty:
                pass
            if (time.monotonic() - last_run < self.cfg.period_s
                    or not (pending or self.train_views)):
                continue
            last_run = time.monotonic()
            t0 = time.monotonic()
            try:
                for kf in pending:
                    view = self._to_view(kf)
                    if view is None:
                        continue
                    if (self.cfg.holdout_every > 0
                            and kf.kf_id % self.cfg.holdout_every
                            == self.cfg.holdout_every - 1):
                        self.holdout_views.append(view)
                        self.holdout_views = self.holdout_views[-8:]
                    else:
                        self._spawn(view)
                        self.train_views.append(view)
                        self.train_views = self.train_views[-_MAX_TRAIN_KF:]
                pending = []
                if not self.params or not self.train_views:
                    continue
                free, _ = torch.cuda.mem_get_info()
                if free < _MIN_FREE_VRAM:
                    print(f"[gs] VRAM 여유 부족({free / 2**30:.1f}GiB) — "
                          f"이번 주기 최적화 건너뜀", flush=True)
                    torch.cuda.empty_cache()
                    continue
                deadline = last_run + self.cfg.period_s * _DUTY
                self._optimize(deadline)
                self._prune()

                with torch.no_grad():
                    newest = self.train_views[-1]
                    rgb, _, _ = self._render(newest, grad=False)
                    render = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()
                    means = self.params["means"].detach()
                    cols = (torch.sigmoid(self.params["colors"].detach())
                            * 255).byte()
                    if len(means) > _VIZ_MAX_PTS:
                        idx = torch.randperm(len(means),
                                             device=means.device)[:_VIZ_MAX_PTS]
                        means, cols = means[idx], cols[idx]
                out_q.put(GSResult(
                    means=means.cpu().numpy(),
                    colors=cols.cpu().numpy(),
                    n_gaussians=len(self.params["means"]),
                    psnr=self._eval_psnr(),
                    render=render, render_kf_id=newest.kf_id,
                    runtime_s=time.monotonic() - t0))
                # 예약(reserved) VRAM을 OS에 반환 — 3개 CUDA 프로세스가
                # 캐시를 각자 쥐고 있으면 합산이 16GB를 넘는다
                torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc()


def _gs_worker_main(cfg: GaussianCfg, in_q: mp.Queue, out_q: mp.Queue) -> None:
    import spacerec  # noqa: F401  (env vars in the child process)

    try:
        worker = _GsWorker(cfg)
        # 첫 렌더에서 gsplat CUDA 커널이 JIT 컴파일된다 (캐시되면 수 초).
        # MSVC가 없거나 컴파일 실패 시 여기서 예외 → 메인에 에러 통지.
        import torch
        worker.params = {
            "means": torch.zeros(1, 3, device="cuda", requires_grad=True),
            "quats": torch.tensor([[1.0, 0, 0, 0]], device="cuda",
                                  requires_grad=True),
            "log_scales": torch.full((1, 3), -3.0, device="cuda",
                                     requires_grad=True),
            "logit_op": torch.zeros(1, device="cuda", requires_grad=True),
            "colors": torch.zeros(1, 3, device="cuda", requires_grad=True),
        }
        dummy = _TrainView(-1, torch.zeros(8, 8, 3), torch.ones(8, 8),
                           torch.ones(8, 8, dtype=torch.bool), torch.eye(4),
                           torch.tensor([[8.0, 0, 4], [0, 8.0, 4], [0, 0, 1]]))
        worker._render(dummy, grad=False)
        worker.params = {}
    except Exception as e:  # noqa: BLE001
        out_q.put(GSResult(means=np.empty((0, 3)), colors=np.empty((0, 3)),
                           error=f"{type(e).__name__}: {e}"))
        return
    out_q.put("ready")
    worker.run(in_q, out_q)


class GaussianBackend:
    """메인 프로세스 핸들 — ReconstructionBackend와 동일한 사용 패턴."""

    def __init__(self, cfg: GaussianCfg):
        ctx = mp.get_context("spawn")
        self._in_q: mp.Queue = ctx.Queue()
        self.results: mp.Queue = ctx.Queue()
        self._proc = ctx.Process(target=_gs_worker_main,
                                 args=(cfg, self._in_q, self.results),
                                 daemon=True)
        self.failed: str | None = None

    def start(self) -> None:
        self._proc.start()

    def wait_ready(self, timeout: float = 600.0) -> bool:
        """True면 사용 가능. False면 self.failed에 사유 (GS 레이어만 비활성)."""
        msg = self.results.get(timeout=timeout)
        if isinstance(msg, GSResult) and msg.error:
            self.failed = msg.error
            return False
        assert msg == "ready", f"unexpected gs backend message: {msg!r}"
        return True

    def add_keyframe(self, kf: BackendKeyframe) -> None:
        if self.failed is None:
            self._in_q.put(kf)

    def stop(self) -> None:
        if self._proc.is_alive():
            self._in_q.put(None)
            self._proc.join(timeout=15)
            if self._proc.is_alive():
                self._proc.terminate()
        from .backend import _drain_and_close
        _drain_and_close(self._in_q, self.results)
