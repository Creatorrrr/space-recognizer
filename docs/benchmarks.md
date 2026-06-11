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

# RTX 4080 벤치마크 — Tier 1 백엔드 상향 (2026-06-11)

환경: Windows 11, RTX 4080 16GB, torch 2.11.0+cu128, Python 3.12
입력: sources/sample_720p.mp4 (1280×720), headless stride 3, 검출 yoloe-26s

## CUDA 기본 실측 (Tier 1 이전)

| 항목 | 측정값 | M1 Max 대비 |
|---|---|---|
| DA3-SMALL mono depth (라이브) | 49~55 ms/frame | ≈동등* |
| yoloe-26s track | 26~30 ms/frame | 2~3x |
| 백엔드 DA3-SMALL 12뷰 @504px | 0.3~0.8 s | 3~8x |
| e2e (realtime, Rerun 포함) | 6.0 FPS | 병목이 viz+rest(60~88ms)로 이동 |

*공식 패키지가 **CUDA에서는 내부적으로 bf16 autocast를 이미 적용**함을 확인
(`api.py` model_forward, fp32 외부 래퍼와 출력 동일·오차 0%). 라이브 depth가
4080에서 빠르지 않은 것은 정밀도 문제가 아니라 모델·오버헤드 자체.

## 백엔드 상향 실측 (headless A/B, 동일 영상·stride)

| 구성 | 윈도 시간 | 지도 포인트 | VRAM (백엔드 프로세스) | 판정 |
|---|---|---|---|---|
| 기준선: SMALL @504, 12뷰, voxel 0.03 | 0.4~0.7 s | 27,944 | — | 기준 |
| LARGE-1.1 @672, 16뷰, voxel 0.02 | **14.8~17.5 s** | 51,474 | nvidia-smi 15.8GB 소진 | ❌ 게이트(≤4s) 위반 |
| **LARGE-1.1 @504, 16뷰, voxel 0.02 (채택)** | **0.6~1.0 s** | **136,979 (4.9x)** | peak alloc 4.2GiB / reserved 6.0GiB | ✅ |

- 672px 실패 원인: any-view 전역 어텐션 비용이 (뷰수×뷰당 토큰수)²로 증가 —
  672px·16뷰는 토큰 ~2.4배 → 어텐션 ~5.6배 + VRAM 16GB 소진(공유 메모리
  스왑)으로 시간이 비선형 폭증. 해상도가 뷰 수보다 민감하다.
- 라이브 계층 비하락 확인: 기준선 대비 **궤적 차이 중앙값 0.0001, 최대 0.0022**
  (장면 단위) — VO·calib 피드백 경로에 유의미한 변화 없음. calib a 1.10~1.18로
  양쪽 동일 범위.
- 16뷰 LARGE@504 윈도가 1초 내이므로 5초 주기 예산 대부분이 남는다 —
  Tier 3(gsplat 레이어)용 시간·VRAM 여유 확보.
- 기준선/Tier1 결과물: `baseline_tier0.npz` / `tier1.npz` (루트, headless 산출)

## Tier 2 — pose-conditioned 추론은 기각 (2026-06-11)

VO pose/K를 DA3 멀티뷰의 입력 조건(`inference(extrinsics=, intrinsics=,
align_to_input_ext_scale=True)`)으로 주는 방식을 구현·실측한 결과 **기하가
오염되어 기본 비활성(`backend.pose_conditioned: false`)으로 기각**. 코드
경로·퇴화 게이트·테스트는 향후 재평가용으로 유지.

| 비교 (r=0.04 상호 커버리지) | 결과 |
|---|---|
| 무조건화 sanity 기준 (baseline vs tier1) | 60.6% / 38.0% |
| VO pose 조건화 vs tier1 | **10.0% / 25.4%** + robust extent 0.6x 수축 |
| 조건화 OFF 회귀 (tier2_off vs tier1) | 100% / 100% (비하락 확인) |

원인 분리 실험:
1. **조건화 메커니즘 자체는 정상**: 모델이 예측한 pose를 그대로 재입력하면
   무조건화 출력 대비 depth 변화가 4~5%(중앙값)에 그침.
2. **DA3 pose 정규화 클램프는 부차적**: `_normalize_extrinsics`가 median
   카메라 거리(min 0.1 클램프)로 나누는데, 라이브 단위 베이스라인(~0.02)이
   클램프에 걸린다. median=1 사전 스케일로 무력화해도 결과 불변 (10%/25%).
3. **근본 원인 — VO 베이스라인과 DA3 pose prior의 충돌**: DA3-LARGE-1.1의
   자체 예측 pose 스프레드는 같은 영상에서 VO의 약 1/10 (0.046 vs 0.46) —
   Phase 3에서 SMALL로 확인한 "pose 헤드 병진 과소추정"이 LARGE에도 존재.
   조건화는 이 prior와 모순되는 VO 병진을 기하에 강제 주입해 depth shape를
   왜곡한다 (per-window α 0.65~0.73, mono calib a 0.86까지 요동).

재평가 조건: (a) 루프 클로저 등으로 pose 품질이 좋아진 뒤, (b) DA3 후속
버전이 pose 헤드 병진을 고친 뒤, (c) 베이스라인이 큰(빠른 이동) 촬영 패턴.
부산물: 키프레임에 VO 고정 K 동봉(`BackendKeyframe.K`), 윈도 정합 계수
α,β가 `[backend]` 로그에 노출(`pose-cond α= β=`), `benchmarks/compare_maps.py`.

## Tier 3 — gsplat Gaussian 품질 레이어 (2026-06-11)

### Windows 빌드 레시피 (필수 — PyPI 1.5.3은 torch 2.11 비호환)

PyPI gsplat 1.5.3은 torch 2.11에서 (a) 사설 API `_jit_compile` 시그니처
변경, (b) torch 헤더 C++ 표준 충돌로 JIT 컴파일이 깨진다. **git main을
소스 빌드**해야 하며, 다음 함정 4개를 모두 처리해야 한다:

```
git clone https://github.com/nerfstudio-project/gsplat.git && cd gsplat
git submodule update --init --recursive     # (1) glm 서브모듈 필수
# (2) setup.py 패치: return [gsplat_ext, inference_ext] → return [gsplat_ext]
#     experimental 렌더러 확장이 Linux 전용 `uint` 타입을 써서 MSVC 빌드 실패
# vcvars64.bat 환경에서, 환경 변수 4개 설정 후 설치:
#   VSLANG=1033 + TORCH_DONT_CHECK_COMPILER_ABI=1
#     (3) 한국어 MSVC의 cl 배너를 torch가 oem 코덱으로 디코드하다 크래시 — ABI
#         체크(경고용)를 끄면 우회됨
#   DISTUTILS_USE_SDK=1                      # (4) VC 환경에서 setup 빌드 요구사항
#   TORCH_CUDA_ARCH_LIST=8.9                 # RTX 4080 (빌드 시간 단축)
pip install . --no-deps --no-build-isolation
```

사전 컴파일 확장으로 설치되므로 **런타임에는 MSVC 불필요** (JIT 없음).

### 게이트 실측 (RTX 4080)

| 항목 | 측정값 |
|---|---|
| import + 첫 렌더 (사전 컴파일) | 0.1s + 0.1s |
| 최적화 스텝 (5k gaussians, 504×283, RGB+ED) | 4.0 ms/step |
| e2e (sample_720p, GS 2주기) | 13,148 gaussians, held-out PSNR 20.5dB |
| GS 주기 처리 시간 | 0.8~2.6 s (15초 주기 예산 내) |
| 라이브 FPS 간섭 | 6.0 → 6.2 FPS (간섭 없음, 별도 프로세스) |

- PSNR 20.5dB는 3.6초 영상·2주기(~300스텝) 기준 — 긴 세션에서 주기가
  쌓일수록 상승 여지. held-out 뷰(8번째 키프레임마다 학습 제외)로 측정.
- gsplat 미설치/빌드 실패 시 GS 레이어만 자동 비활성 (`[gs] gsplat 사용
  불가` 경고) — 나머지 파이프라인 무영향 확인.
- 재실행: `python benchmarks/bench_gsplat.py` (게이트), e2e는 main 실행 후
  `[gs]` 로그와 Rerun `GS Render` 패널 확인.
