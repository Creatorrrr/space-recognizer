# 3D 재구성 품질 업그레이드 작업 계획 (CUDA 전환 후속)

> 작성: 2026-06-11. 사전 검토(첨부 구현 계획안 + 코드 분석 + 기술 리서치)의 결론을
> 실행 가능한 작업 단계로 정리한 문서. 현재 구현 상태는 `README.md`,
> Mac 시절 실측·결정 근거는 `docs/benchmarks.md`, 최초 설계는 `docs/plan.md` 참고.

## 0. 배경과 전제

### 0.1 환경 전환

본 프로젝트는 M1 Max(MPS, CUDA 불가) 전제로 설계·튜닝되었으나, 실행 환경이
**Windows 11 + RTX 4080 16GB (torch 2.11+cu128)** 로 전환되었다
(커밋 25f33d0 "Support CUDA device selection"). 이로써 설계 당시 MPS 제약으로
배제했던 기술들이 재검토 대상이 되었고, 동시에 현재 설정이 새 하드웨어 대비
크게 보수적이라는 점이 확인되었다.

RTX 4080 실측 (`sample_720p_main.log`, sample_720p.mp4 1280×720):

| 항목 | M1 Max | RTX 4080 | 비고 |
|---|---|---|---|
| 백엔드 멀티뷰 윈도 (DA3-SMALL, 5~12뷰) | 2.3~2.7 s | **0.3~0.8 s** | 5초 주기 대비 6~16배 여유 |
| 라이브 depth DA3-SMALL @504px | 62 ms | 55 ms | fp32 그대로 — 미최적화 |
| YOLOE-11s-seg track | 65~85 ms | 26~30 ms | |
| VO (LK+PnP) | — | 12~26 ms | CPU |
| viz+rest | — | **60~88 ms** | 현재 e2e 병목 |
| e2e | 2.5~5 FPS | 6 FPS | |

### 0.2 업그레이드 원칙 (사용자 결정)

1. **기능 중복 금지**: 새 모델/기술이 기존 것과 역할이 겹치면 하나의 모델
   패밀리 최신 버전으로 통합한다 (유지보수성·메모리).
2. **성능·품질 비하락**: 모든 변경은 config 토글로 끌 수 있어야 하고, 끄면
   현재와 동일하게 동작해야 한다.
3. **백엔드 주기는 느슨해도 됨**: 품질 백엔드는 10초 이상 주기 허용.
4. 검증은 `benchmarks/headless_run.py` A/B 비교 + 기존 단위 테스트로 한다.

### 0.3 기술 리서치 결론 (2026-06-11 검증)

원안(FlashDepth/oVDA + DROID-W + WildGS-SLAM) 구성요소의 현재 상태:

| 후보 | 판정 | 핵심 사유 (검증일 2026-06-11) |
|---|---|---|
| FlashDepth | **탈락** | Windows 빌드 불가(vendored Mamba 컴파일 실패 이슈 1년 무응답), 2025-09 이후 방치, ~2K에서 VRAM 18~21GB(16GB OOM), 4080/4090 공개 벤치 없음. DA3와 depth 역할 중복 |
| oVDA | **탈락** | scale/shift-invariant 출력이라 별도 metric 보정 필수 — 현 DA3 보정 체계와 전면 중복. 도입 시 모델 패밀리 +1 |
| DROID-W (arXiv 2603.19076, CVPR 2026) | **탈락(시기상조)** | 코드는 2026-03-19 공개됐으나 4커밋 후 방치·재현 이슈 미응답. lietorch 등 커스텀 CUDA 확장 3종은 Windows 빌드 실패 이력 다수(WSL2 필요). DUSt3R(CC BY-NC-SA) 코드 번들로 라이선스 불투명. 외부 pose/depth 입력 모드 없음 — 풀 트래킹 파이프라인이라 VO·depth·dyn mask 전부와 중복 |
| WildGS-SLAM (CVPR 2025) | **탈락, 스타일만 차용** | 외부 pose 모드 없음(DROID 트래커와 공유 버퍼로 강결합). Windows 네이티브 사례 0건. 필수 rasterizer `diff-gaussian-rasterization-w-pose`가 Inria **비상업** 라이선스. RTX 4090에서 0.49 FPS(fast 1.96), 기본 해상도 VRAM 사실상 24GB급(V100 32GB OOM 보고) |
| DA3-Streaming (2025-12-11) | **직접 사용 탈락, 레시피 차용** | known pose 입력 불가(자체 추정 강제), 의미 있는 해상도에서 VRAM 18~28GB, 오프라인 배치 설계. 단 루프 클로저 레시피(DINO retrieval + Sim3 pose graph)는 Tier 4에서 차용 |
| 기성 GS-SLAM (SplaTAM/MonoGS/GS-ICP-SLAM/Photo-SLAM/RTG-SLAM/Splat-SLAM) | **전부 탈락** | 각각 Windows 불가, 외부 pose 불가, Inria 비상업 rasterizer, 방치 중 최소 1개 이상에 해당 |
| MapAnything v1.1.2 (2026-05-30) | **보류·주시** | Apache 가중치 변형 존재, pose/intrinsics/depth 입력 + metric 출력 — DA3 백엔드의 유력한 대체 후보지만 배치 전용이고 현 DA3로 충분. 교체는 패밀리 중복만 증가 |
| AMB3R (CVPR 2026) | **보류·주시** | 진정한 증분형 metric VO/SfM. 단 pytorch3d/flash-attn 의존(Windows 곤란), 라이선스·VRAM 미공표 |

활용 가능한 신규 사실 (설치된 `depth-anything-3` 0.1.1에서 직접 확인):

- **pose-conditioned 추론이 1급 API**: `inference(images, extrinsics=(N,4,4) w2c,
  intrinsics=(N,3,3), align_to_input_ext_scale=True)` — 외부 pose에 정합된
  멀티뷰 depth를 출력한다. benchmarks.md에서 확인한 "DA3 pose 헤드 병진
  과소추정" 문제를 우회하면서 멀티뷰 일관성을 얻는 경로.
- **DA3-LARGE-1.1 (0.35B)은 Apache-2.0** (2025-12-11 재학습판, 공식이 "이쪽
  권장"이라 명시). GIANT(1.15B)/NESTED-GIANT-LARGE(1.4B)는 CC-BY-NC.
- DA3 패키지에 GS head(`infer_gs=True`)가 있으나 GIANT 전용 → 라이선스·메모리
  문제로 미사용. Gaussian 레이어는 gsplat으로 자체 구현(Tier 3).
- **gsplat 1.5.3**: Windows 네이티브 공식 지원(INSTALL_WIN.md + win_amd64
  프리빌드 휠), Apache-2.0(Inria 코드 없음), 2026-06 현재 활발히 유지보수.
  조사 대상 중 유일하게 세 조건을 모두 만족하는 GS rasterizer.

### 0.4 종합 결론

> 새 모델 패밀리를 들이지 않는다. (1) CUDA 여유로 DA3 패밀리 내 상향,
> (2) pose-conditioned 추론으로 백엔드 일관성 개선, (3) gsplat 기반 자체
> Gaussian 품질 레이어(10~30초 주기, WildGS 스타일), (4) DINOv2 재활용
> 루프 클로저 — 의 4단계로 품질을 올린다. 기존 voxel 지도·객체 레지스트리·
> 증거 갱신 파이프라인은 그대로 유지한다 (비하락 보장의 기준선).

---

## Tier 1 — CUDA 여유 소진 (예상 0.5일, 위험 낮음) — ✅ 완료 (2026-06-11)

설정·소규모 코드 변경만으로 새 하드웨어의 여유를 품질로 전환한다.

> **결과 요약**: 백엔드 DA3-LARGE-1.1 @504px·16뷰·voxel 0.02 채택.
> 지도 밀도 4.9배(27.9k→137k pts), 윈도 0.6~1.0s(게이트 통과), 백엔드 프로세스
> VRAM peak 4.2GiB, 궤적 차이 중앙값 0.0001(비하락 확인). **672px는 게이트
> 위반으로 기각** — any-view 어텐션이 (뷰수×토큰수)²라 해상도 상향은 윈도
> 15s+/VRAM 16GB 소진. bf16 항목은 패키지 내장으로 판명되어 작업 불필요.
> 실측 상세는 `docs/benchmarks.md` RTX 4080 섹션.

### 작업 항목

1. **백엔드 모델 상향**: `config.yaml`의 백엔드 모델을 분리 지정 가능하게 하고
   기본을 `depth-anything/DA3-LARGE-1.1`로 상향.
   - 현재 `backend.py`는 라이브와 같은 `cfg.depth.model`을 사용 → `depth:`에
     `backend_model:` 키 추가 (`config.py` dataclass + `main.py` 전달부 수정).
   - 라이브는 DA3-SMALL 유지 (지연이 중요하므로).
   - HF 모델 id의 `-1.1` 표기는 다운로드 후 1회 검증 (가중치 ~1.4GB 예상).
2. **백엔드 해상도/윈도 상향**: `backend.period_s`는 5.0 유지,
   `process_res 504 → 672`(백엔드만), `window_size 12 → 16`, `overlap 6 → 8`.
   - 라이브 depth의 process_res는 504 유지 (지연 우선).
   - 백엔드 process_res 분리도 `backend_model`과 같은 방식으로 config 분리.
3. **지도 밀도 상향**: `voxel_size 0.03 → 0.02`, `max_points 800k → 2M`.
   - Rerun 메모리 한도(`viz.memory_limit`)와 `point_subsample` 동반 점검.
4. **CUDA bf16 autocast**: ~~depth.py/backend.py에 autocast 적용~~ →
   **불필요 판명 (2026-06-11)**: 공식 패키지 `api.py`의 `model_forward`가
   CUDA에서 이미 내부적으로 bf16 autocast를 적용하고 있음을 확인 (외부
   래퍼와 출력 동일, 오차 0%). 라이브 depth 55ms는 fp32 비용이 아니라
   모델+오버헤드 자체 — 추가 가속은 process_res 축소나 추후 torch.compile
   검토 영역으로 이관.
5. (부수) **viz 병목 완화**: 현재 e2e 병목은 viz+rest 60~88ms. 품질과 직접
   무관하므로 본 계획에서는 측정만 추가(`--profile` 세분화)하고 최적화는
   별도 작업으로 분리.

### 검증·게이트

- `benchmarks/headless_run.py`를 동일 영상·`--no-realtime`으로 변경 전/후 실행,
  npz 결과 비교: 지도 포인트 수, calib(a,b) 안정성, 객체 위치 분산.
- 게이트: 백엔드 윈도(16뷰@672px, LARGE) ≤ 4초 (5초 주기 내). 초과 시
  해상도→윈도 순으로 축소.
- bf16 검증: 같은 프레임 10장에서 fp32 대비 depth 상대 오차 중앙값 < 1%.
- VRAM: `nvidia-smi` 피크 기록. 라이브+백엔드 합산 < 10GB 목표
  (Tier 3 예산 확보).
- 단위 테스트 전체 통과 (`pytest tests/ -q`).

### 리스크

| 리스크 | 대응 |
|---|---|
| DA3-LARGE 멀티뷰가 SMALL과 conf 분포가 달라 `conf 30퍼센타일` 필터가 과/소제거 | percentile 값을 config로 빼고 A/B로 재튜닝 |
| voxel 0.02에서 carving 비용 증가 (`worldmap._carve`) | `_RAY_STRIDE`/`_MAX_STEPS` 조정. carving은 백엔드 결과 반영 시점에만 돌므로 여유 큼 |
| 윈도 16뷰로 첫 윈도 지연 증가 | 첫 윈도 `min_kf=5` 로직은 유지 — 영향 없음 |

---## Tier 2 — pose-conditioned 백엔드 (예상 1~2일, 위험 중간) — ⚠️ 구현 후 기각 (2026-06-11)

백엔드 멀티뷰 추론에 VO pose를 조건으로 입력해, 사후 정렬(α,β 스케일 정합 +
Sim3)에 의존하던 윈도 간 일관성을 모델 입력 단계에서 확보한다.

> **결과 요약**: 구현·A/B 결과 **기하 오염으로 기본 비활성 처리**
> (상호 커버리지가 sanity 기준 60%/38% 대비 10%/25%로 붕괴). 원인 분리
> 실험으로 확인된 근본 원인은 DA3 pose prior의 병진 과소추정(LARGE도
> VO 대비 1/10 스프레드)과 VO 베이스라인의 충돌 — 모델 자신의 pose를
> 재입력하면 4~5%만 변하므로 메커니즘이 아니라 입력 pose 불일치 문제.
> 코드 경로(`backend.pose_conditioned` 토글, 퇴화 게이트, 테스트, α/β
> 모니터링, 키프레임 K 동봉)는 유지 — 루프 클로저(Tier 4) 이후 또는 DA3
> 후속 버전에서 재평가. OFF 회귀는 tier1과 100%/100% 일치(비하락 확인).
> 상세는 `docs/benchmarks.md` Tier 2 섹션.

### 현재 구조의 한계

`backend.py:_run_window()`는 pose 없이 추론한 뒤:
- 멀티뷰 depth를 라이브 스케일로 affine 정합 (α,β — 최신 키프레임 1장 기준)
- 중첩 키프레임 기반 Sim3 정렬

이 사후 보정은 윈도마다 기준이 흔들릴 수 있고(특히 α,β가 단일 프레임 피팅),
윈도 경계에서 이중상(ghosting)의 원인이 된다.

### 작업 항목

1. `_run_window()`에서 `self.model.inference(...)`에
   `extrinsics=np.stack([kf.T_wc_live for kf in window])`(w2c 변환 주의 —
   `T_wc_live`는 camera-to-world이므로 역행렬), `intrinsics=` 고정 K(현
   `main.py`의 K_WARMUP 중앙값을 키프레임에 동봉)를 전달.
   `align_to_input_ext_scale=True`.
2. 출력 depth가 입력 pose 스케일에 정합되므로:
   - α,β 정합(`backend.py:177-186`)은 **안전망으로 유지**하되, 정합 결과가
     항등(α≈1, β≈0)에 수렴하는지 모니터링 지표로 전환.
   - Sim3 윈도 정렬(`robust_sim3`)도 유지 (live pose 사용 시 사실상 항등 —
     기존 주석과 동일한 역할).
3. config 토글: `backend.pose_conditioned: true|false` (false = 현행 동작).
4. intrinsics 전달 경로 정리: 현재 백엔드는 자체 추정 K를 반환만 하고
   (`res.intrinsics`) VO는 첫 프레임 고정 K를 쓴다. pose-conditioned 모드에서는
   **VO의 고정 K를 백엔드 입력으로 단일화** — K 이원화 제거 (중복 제거 원칙).

### 검증·게이트

- 동일 영상 A/B: (1) 윈도 경계 이중상 육안 비교(Rerun 톱다운),
  (2) α,β 시계열의 분산 감소, (3) calib inlier_frac 상승, (4) 객체 위치
  EMA 분산 감소.
- 단위 테스트 추가: 합성 키프레임(기지 pose·depth)으로 pose-conditioned 경로의
  출력 스케일이 입력 pose 스케일과 일치하는지.
- 실패 조건: pose 입력 시 depth 품질이 오히려 저하(모델이 잘못된 VO pose에
  과적합)되는 경우 — VO inlier_ratio 낮은 키프레임은 pose를 빼고 추론하는
  하이브리드 폴백을 검토.

### 리스크

| 리스크 | 대응 |
|---|---|
| VO pose 오차가 클 때 조건화가 depth를 오염 | inlier_ratio 게이트(예: <0.6이면 해당 윈도는 무조건화 폴백) |
| w2c/c2w 방향·정규화 실수 | 패키지 `_normalize_extrinsics` 동작을 단위 테스트로 고정 |
| `-1.1` 모델과 0.1.1 패키지 호환성 | Tier 1에서 선검증 완료 후 진입 |

---

## Tier 3 — Gaussian 품질 레이어 (예상 3~5일, 위험 중간) — ✅ 1차 완료 (2026-06-11)

WildGS-SLAM이 제공하는 가치(동적 물체가 제거된, 렌더링 가능한 고품질 정적
지도)를 **gsplat + 자체 매퍼**로 획득한다.

> **결과 요약**: gsplat git main 소스 빌드 성공 (PyPI 1.5.3은 torch 2.11
> 비호환 — 빌드 레시피와 함정 4개는 `docs/benchmarks.md` Tier 3 섹션).
> `gs_backend.py` 구현: 별도 프로세스, 15초 주기 anytime 최적화, live 좌표계
> 최적화 + 표시 시점 전역 변환, dyn_mask 제외 RGB L1 + depth L1, 렌더 알파
> 기반 중복 방지 spawn, held-out PSNR 검증. e2e: 13k gaussians/PSNR
> 20.5dB(3.6초 영상 2주기), 라이브 FPS 간섭 없음(6.0→6.2), gsplat 실패 시
> GS 레이어만 자동 비활성. 남은 개선 여지: SSIM 손실, gsplat MCMC
> densify 전략, GS 레이어 영속화(persistence) — 장기 세션 검증과 함께. 기존 voxel 지도는 기하·증거
레이어(객체 위치, free-space carving, 영속화)로 그대로 두고, GS는 시각 품질
레이어로 **추가**한다. 끄면 현재와 100% 동일 — 비하락 구조적 보장.

### 설계

```
[gs_backend.py — 별도 프로세스, 10~30초 주기 (backend.gs_period_s)]
  입력: 키프레임 묶음 {RGB, T_wc(global), 보정 depth, dyn_mask, K}
        — 기존 BackendKeyframe + 전역 pose 재사용, 신규 수집 경로 불필요
  1. spawn: 신규 영역의 depth 픽셀(동적 mask 제외, conf 통과)을
     역투영해 Gaussian 초기화 (위치=3D점, 색=RGB, scale=거리 비례,
     opacity 초기값) — SplaTAM식
  2. optimize: 해당 윈도 키프레임들로 N step(초기 100~300) 렌더 손실
     (L1+SSIM) 최적화. dyn_mask 픽셀은 손실에서 제외 — WildGS의
     uncertainty 역할을 기존 YOLOE mask가 대신함
  3. prune/densify: opacity 낮은 것 제거, gsplat MCMC 전략 활용
  4. 출력: Gaussian 집합(위치/공분산/SH색/opacity) → 메인 프로세스
  주기 내 미완료 시: 다음 주기로 이월 (anytime 설계 — 10초+ 허용이 전제)
```

- **시각화**: 1차는 Rerun에 Gaussian 중심점+색을 Points3D로 로깅(간이),
  2차는 gsplat 렌더 뷰(현재 카메라 시점 노벨뷰)를 2D 패널로 추가.
- **T_global_live 갱신 연동**: Gaussian은 global frame에 직접 생성하므로
  기존 보간 메커니즘과 자연 호환. 백엔드가 좌표계를 크게 보정하면 해당
  윈도 Gaussian만 Sim3로 이동 (voxel 지도와 동일 정책).
- **영속화**: `persistence.py`에 GS 레이어 저장(.npz 또는 .ply) 선택 추가.
  1차에서는 세션 내 휘발로 시작.

### 작업 항목

1. gsplat 설치 검증 스파이크 (0.5일 게이트): 프리빌드 휠은 torch 2.0~2.4
   대상이므로 **torch 2.11에서는 소스 빌드 필요 가능성 높음** — MSVC 143 +
   CUDA 12.8 조합으로 빌드 확인. 실패 시 torch 버전 핀 조정 또는 Tier 3 보류.
2. `gs_backend.py` 신규 모듈 (위 설계). config: `backend.gaussian:
   {enabled: false, period_s: 15, max_gaussians: 500k, opt_steps: 200}`.
3. `viz.py`에 GS 패널 추가 (`world/gaussians`).
4. VRAM 예산 검증: 50만 Gaussian 학습 시 ~2-4GB 예상. 라이브(~2GB) +
   백엔드 LARGE(~3-4GB) + GS(~4GB) < 12GB 확인. 초과 시 `max_gaussians` 축소.

### 검증·게이트

- 정량: 보류 키프레임(학습 미사용) 재투영 PSNR/SSIM — voxel 포인트 렌더
  대비 개선 확인.
- 정성: 동적 물체(person 등)가 GS 지도에 잔상으로 남지 않는지.
- 비간섭: GS 프로세스 가동 중 라이브 FPS 하락 < 10% (별도 프로세스 +
  CUDA 스트림 경합 측정). 초과 시 opt_steps 분할·주기 연장.
- `gaussian.enabled: false`에서 기존 전체 테스트·headless 결과 불변.

### 리스크

| 리스크 | 대응 |
|---|---|
| gsplat이 torch 2.11+cu128에서 빌드 실패 | 게이트에서 조기 판정. torch 다운핀은 ultralytics/DA3와 호환 범위 확인 후 |
| 3개 프로세스 GPU 경합 | GS 주기를 30초로 연장, opt_steps 동적 조절 (anytime) |
| VO drift가 GS 품질 한계로 직결 | 구조적 한계로 수용 — Tier 4가 해결 담당 |

---

## Tier 4 — 루프 클로저 (예상 1주+, 위험 중간~높음)

README 알려진 한계 1순위(장시간 drift)에 대한 대응. 신규 모델 없이
DA3-Streaming의 공개 레시피를 차용한다. 지도의 전역 기하 정확도에 가장 큰
효과 — 1~3분 이상 촬영에서 Tier 1~3의 품질을 유지하는 전제 조건.

### 설계

```
1. place recognition: 키프레임마다 DINOv2 전역 임베딩(기존 appearance.py
   재활용 — 객체 crop이 아닌 전체 프레임 1회 추가 추론) → 코사인 유사도
   상위 + 시간 갭 조건으로 루프 후보 쌍 검출
2. 루프 검증·상대 pose: 후보 쌍 ± 이웃 키프레임 묶음을 DA3 멀티뷰
   (pose-conditioned 없이) 추론 → 두 시점 간 상대 Sim3 추정, inlier 검증
3. pose graph: 키프레임 Sim3 그래프(순차 엣지 = VO, 루프 엣지 = 2의 결과)
   를 LM 최적화 (scipy 기반 자체 구현 또는 소형 의존성)
4. 지도 반영: 보정된 키프레임 pose로 (a) voxel 지도는 윈도 단위 재융합
   또는 포인트 Sim3 이동, (b) GS 레이어는 윈도별 Sim3 이동,
   (c) T_global_live는 기존 보간 메커니즘으로 무점프 갱신
```

### 작업 항목

1. `loop.py` 신규: 임베딩 링버퍼 + 후보 검출 (백엔드 프로세스에 동거).
2. 상대 Sim3 추정: 기존 `geometry.umeyama_sim3` + RANSAC 래퍼 재사용.
3. Sim3 pose graph 최적화 단위 테스트 (합성 루프 시나리오 — 사각형 궤적
   drift 보정).
4. 지도 재융합 정책: 1차는 "루프 검출 시 해당 구간 voxel 무효화 + 재융합
   대기열" 방식 (free-space carving과 동일한 증거 철학).

### 검증

- 합성: 단위 테스트 (위 3).
- 실측: 같은 지점으로 되돌아오는 1~3분 촬영 영상에서 (1) 루프 전후 지도
  이중상 제거, (2) 시작/종료 지점 카메라 위치 오차 감소, (3) 객체 노드
  중복 생성 감소.
- 오탐 루프가 지도를 파괴하지 않는지: inlier 임계 미달 루프는 기각 로그만.

---

## 실행 순서·의존 관계

```
Tier 1 (0.5일) ──► Tier 2 (1~2일) ──► Tier 4 (1주+)
                └─► Tier 3 게이트 스파이크 (0.5일) ──► Tier 3 본작업 (3~5일)
```

- Tier 1·2는 순차 (같은 backend.py를 수정).
- Tier 3은 Tier 1 완료 후 병행 가능 (신규 모듈이라 충돌 없음). 단 gsplat
  빌드 게이트를 먼저 통과해야 본작업 착수.
- Tier 4는 Tier 2의 pose 인프라 정리 이후가 깔끔하나, 독립 진행도 가능.

## 메모리 예산 (RTX 4080 16GB)

| 구성요소 | 상주 프로세스 | 예상 VRAM |
|---|---|---|
| 라이브: DA3-SMALL + YOLOE-11s + DINOv2-small (bf16) | main | ~2 GB |
| 백엔드: DA3-LARGE-1.1 + DA3METRIC-LARGE (bf16) | backend | ~3~4 GB |
| GS 레이어: gsplat 학습 (≤50만) | gs_backend | ~2~4 GB |
| 합계 | | **~8~10 GB** (여유 6+ GB) |

NESTED-GIANT-LARGE(1.4B) 통합안을 채택하지 않는 이유: CC-BY-NC 라이선스 +
VRAM 압박. METRIC-LARGE는 윈도당 1회 추론뿐이라 "any-view + metric 2모델"
유지가 중복이 아니라 역할 분담이다 (둘 다 Apache).

## 비하락 보장 장치 (전 Tier 공통)

1. 모든 신규 동작은 config 토글, 기본값은 검증 완료 후에만 변경.
2. 변경 전 기준선 고정: 현 main에서 `headless_run.py` 결과 npz를 커밋 직전
   상태로 보존 → 각 Tier 머지 전 동일 입력 비교.
3. `pytest tests/ -q` 전체 통과 + Tier별 신규 단위 테스트.
4. 실패 시 롤백 단위 = Tier (커밋 단위를 Tier별로 유지).

## 보류·주시 목록 (분기별 재확인)

- **MapAnything** (Meta/CMU, v1.1.2 2026-05-30, Apache 가중치 변형): DA3
  백엔드가 한계에 부딪히면 1순위 대체 후보. pose/intrinsics/depth 입력 +
  metric 출력. 배치 전용이므로 현 윈도 구조에 그대로 끼울 수 있음.
- **AMB3R** (CVPR 2026, 증분형 metric VO/SfM): 라이선스·VRAM 공표와 Windows
  사례가 나오면 VO+백엔드 통합 대체 후보로 재평가.
- **DROID-W**: 유지보수 재개 + 라이선스 정리되면 uncertainty 기반 BA만
  참고 구현으로 차용 검토.
- DA3 후속 릴리스 (2026-06 현재 신규 없음 — 2025-12-11 "-1.1"이 최신).
