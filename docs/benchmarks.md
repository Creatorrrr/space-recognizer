# Phase 0 벤치마크 (2026-06-10)

환경: MacBook Pro M1 Max 32GB, macOS Darwin 25.5.0, Python 3.12, torch 2.12.0 (MPS), fp32
입력: sources/10135156-uhd_3840_2160_30fps.mp4 → 1280×720 다운스케일

| 항목 | 측정값 | 게이트 | 판정 |
|---|---|---|---|
| DA3-SMALL mono depth 1뷰 (공식 패키지) | 62 ms/frame (16.1 FPS) | — | ✅ |
| DA3-SMALL 멀티뷰 8뷰 (pose+depth+intrinsics) | 2.3 s | ≤ 30 s | ✅ |
| DA3-SMALL 멀티뷰 16뷰 | 2.7 s | — | ✅ |
| yolo26n-seg predict | 33 ms/frame (30 FPS) | — | ✅ |
| yolo26n-seg track (ByteTrack) | 45 ms/frame (22 FPS) | — | ✅ |
| yolo26s-seg track (ByteTrack) | 50 ms/frame (20 FPS) | — | ✅ |
| yoloe-11s-seg track (오픈 보캐뷸러리, 45어휘) | 65~85 ms/frame | — | ✅ |
| yoloe-26s-seg track (오픈 보캐뷸러리, 45어휘) | 60 ms/frame | — | ✅ (기본 채택) |
| 라이브 계층 합계 (depth+track) | ~110 ms → 5.1 FPS e2e 실측 | ≤ 200 ms | ✅ |

## 결정 사항
- **YOLO26s-seg 채택** (당초 n-seg → s-seg로 상향). n(nano)은 바닥 카펫을
  'bed' conf 0.3대로 오인 — COCO에 rug 클래스가 없어 가장 비슷한 클래스로
  분류함. s부터는 같은 프레임에서 카펫 오검출이 사라지고 진짜 침대 conf가
  0.65→0.78로 상승. track 비용은 45→50ms로 +5ms뿐.
  남은 어휘 한계(카펫·장식장 등 COCO 밖 물체)는 YOLOE(오픈 보캐뷸러리,
  ultralytics 8.4.63에 포함)로 해소 가능 — 필요 시 옵션으로 추가.
- **YOLOE-11s-seg를 기본으로 채택** (config의 `detect.vocabulary`로 어휘
  지정). 카펫이 'rug', 장식장이 'cabinet/wardrobe' 등 올바른 라벨로 등록됨.
  텍스트 인코더(mobileclip_blt.ts, 572MB)는 최초 1회 다운로드.
  주의 1: `clip` 패키지가 토크나이저로 필요 — PyPI `clip-anytorch`로 충족
  (ultralytics가 시도하는 git 직설치 불필요).
  주의 2: 첫 추론이 커널 컴파일로 ~3초 → main이 페이싱 시작 전에 더미
  프레임으로 전 모델을 워밍업하도록 수정함 (짧은 영상에서 워밍업이 재생
  시간을 잠식하던 문제).
  트레이드오프: 검출이 풍부해진 만큼 중복/노이즈 노드도 늘어남 (예: 같은
  러그가 2노드로 분리될 수 있음). conf 상향(0.4~0.45)이나 어휘 축소로 조절.
- **공식 `depth-anything-3` 0.1.1 패키지 채택** — MPS에서 SDPA로 그대로 동작.
  멀티뷰 출력: depth (V,280,504), extrinsics (V,3,4) w2c, intrinsics (V,3,3) @504px
- ⚠️ **커뮤니티 래퍼 `awesome-depth-anything-3`는 사용 금지**: 체크포인트 로딩이 깨져
  depth가 ~1.1 상수 + 격자 노이즈로 출력됨 (mono/multi-view, CPU/MPS 모두). 공식
  패키지로 교차 검증해 확인. HF 모델 id는 `depth-anything/DA3-SMALL` 사용.
- 백엔드 윈도 8~16뷰 모두 5초 주기 내 처리 가능 (2.3~2.7s)
- **(2026-06-11) YOLOE-26s-seg로 상향, 최종 기본 채택.** 같은 어휘·동일 프레임
  비교에서 11s 대비: 핵심 가구 신뢰도 상승(침대 0.77→0.87, 램프 0.76→0.92),
  중복 검출 감소(NMS-free E2E), track 60ms로 더 빠름. recall은 소폭 감소
  (rug 신뢰도 0.85→0.6대 — conf 0.35 통과). e2e 회귀: 침실 bed 2노드+re-ID,
  rug 2노드(실제 개수 일치), 주방 refrigerator 0.7~0.9, --map 재위치추정
  매칭 7개 성공. realtime 모드는 프레임 샘플링 분산으로 실행별 노드 구성이
  다소 달라짐(11s도 동일).
  텍스트 인코더가 **MobileCLIP2**(mobileclip2_b.ts, 242MB)로 교체됨 — 자동
  다운로드 실패 시: `curl -L -o mobileclip2_b.ts
  https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts`
  (프로젝트 루트에 두면 됨). 구 인코더(mobileclip_blt.ts)와 yoloe-11s 가중치는
  더 이상 필요 없음.

## 환경 주의사항 (필수)
- `KMP_DUPLICATE_LIB_OK=TRUE` 필요 — pycolmap/torch가 각자 libomp를 번들해 OpenMP 중복 초기화 크래시 발생. 앱 엔트리포인트에서 import 전에 설정.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` 권장.
- `awesome-depth-anything-3`는 `numpy<2` 핀 때문에 `--no-deps`로 설치 (rerun-sdk 0.33이 numpy>=2 요구). 하위 의존성은 pyproject에 직접 명시. numpy 2.x에서 동작 확인됨.
- `moviepy==1.0.3`, `pycolmap`은 DA3 패키지 import 시점에 필요.

재실행: `KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python benchmarks/bench_models.py`

## Phase 3에서 확인된 사실 (설계 변경의 근거)

1. **PyTorch MPS는 멀티스레드 불안전**: 메인 스레드(라이브 추론)와 백엔드 스레드가
   동시에 MPS를 쓰면 `IOGPUMetalCommandBuffer` assertion으로 SIGABRT. →
   백엔드를 별도 프로세스(spawn)로 분리해 해결 (`backend.py`).
2. **DA3-small pose 헤드의 병진 과소추정**: 자기 depth와 회전(오차 1~2°)은 정확하지만
   병진이 4~8배 작게 나옴 (PnP 교차 검증, 본 영상 기준). DA3 extrinsics는 사용하지 않음.
3. **인접 키프레임 쌍 PnP 체인은 붕괴**: 키프레임 간 베이스라인(~0.01)이 PnP 노이즈
   바닥(reproj 2px ⇒ ~0.007)과 비슷해 병진이 무작위 보행으로 상쇄됨 (측정: 라이브
   대비 13~30x 작은 스프레드). 저해상도(504px) 윈도에서 LK 재추적도 간격이 크면 끊김.
4. **결론 — 윈도 pose는 라이브 VO pose를 사용**: 연속 고밀도 추적만이 신뢰 가능한
   병진 소스. 백엔드는 (a) 멀티뷰 일관 depth로 지도 융합, (b) 멀티뷰 depth를 라이브
   스케일로 정합(α,β), (c) mono depth 보정(a,b), (d) intrinsics 추정(fx≈800-860@1280px,
   60° FOV 가정 fx=1108 대비 상당한 차이)을 담당.
5. 전 구간 scale=1.000, calib a≈0.97-1.0으로 좌표계 일관성 확보. 10초 영상 기준
   지도 ~23k 포인트(voxel 0.03), 톱다운에서 단일 방 구조 확인.

## IMU-aided VO A/B (2026-06-24)

명령:

```bash
.venv/bin/python benchmarks/replay_smoke.py \
  sources/session_20260624_054320_194430108151D05A00 \
  sources/session_20260624_055321_194430108151D05A00 \
  --frames 120 --compare-imu
```

| 세션 | 모드 | lost | keyframes | avg_tracked | avg_inlier | imu_prior_frames | blur_skipped_kf |
|---|---|---:|---:|---:|---:|---:|---:|
| session_20260624_054320_194430108151D05A00 | off | 0 | 23 | 200.0 | 0.92 | 0 | 0 |
| session_20260624_054320_194430108151D05A00 | on | 0 | 23 | 200.0 | 0.92 | 119 | 0 |
| session_20260624_055321_194430108151D05A00 | off | 1 | 33 | 50.8 | 0.79 | 0 | 0 |
| session_20260624_055321_194430108151D05A00 | on | 0 | 33 | 50.9 | 0.79 | 119 | 0 |

해석:
- gyro prior는 두 기록 모두 119/120프레임에서 계산되어 LK/PnP 경로에 전달됐다.
- 안정 세션은 지표 변화가 없었고, 어려운 세션은 `lost`가 1에서 0으로 줄었다.
  다만 개선 폭이 현재 두 녹화 120프레임 smoke에 한정되므로 `imu.enabled` 기본값은
  보수적으로 `false`를 유지하고, 현장 녹화에서는 `--compare-imu`로 확인한 뒤 켠다.
- 두 기록 모두 `keyframe_blur_omega_rad_s=2.5`를 넘는 backend keyframe 후보가 없어
  blur gating skip은 0이었다. 빠른 회전 기록을 추가 확보하면 이 카운터를 먼저 본다.
- accelerometer translation 적분은 범위 밖이다. OAK-D-Lite급 MEMS accel bias를 이중
  적분하면 짧은 시간에도 position drift가 커지므로, 현재 IMU 사용은 gyro rotation
  prior와 backend keyframe blur gating으로 제한한다.

## Canonical TSDF mesh smoke (2026-06-24)

명령:

```bash
.venv/bin/python benchmarks/mesh_smoke.py \
  sources/session_20260624_054320_194430108151D05A00 --frames 120
```

해석:
- `mesh.render_mode: canonical`이 기본이므로 `.ply` export와 기본 Rerun mesh는
  raw submap concat이 아니라 canonical surface selection 결과다.
- smoke 출력은 `raw_vertices/raw_faces`와 `vertices/faces`를 함께 표시한다.
  raw 값은 backend-window submap을 그대로 합친 크기이고, `vertices/faces`는
  canonical export 크기다.
- canonical selection은 같은 world-space cell과 normal agreement group 안에서
  support score를 우선하고, depth residual proxy를 감점하며, recency를 약한 tie-break
  보너스로 사용한다. 최신 single noisy submap이 기존 high-support surface를 무조건
  덮지 않도록 하는 것이 의도다.
- 이 단계는 render/export-time 정리 계층이다. full spatial-block TSDF, dirty-block
  rebuild, free-space 기반 mesh reintegration은 별도 장기 과제로 남아 있다.
