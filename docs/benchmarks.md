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

## Tier 4 — 루프 클로저 (2026-06-11)

구성: DINOv2 키프레임 임베딩 place recognition → **ORB + 양쪽 depth 3D-3D
RANSAC Umeyama**로 상대 Sim3 검증 → Sim3 pose graph(LM, scipy) →
epoch별 voxel 지도 보정. 루프 검증에 DA3 멀티뷰를 쓰지 않는 이유:
pose 헤드 병진 과소추정(Phase 3/Tier 2) — 3D-3D 방식이 drift의 스케일
성분까지 직접 측정한다.

### 검증 (부메랑 영상 A/B)

검증 영상: sample_720p 정방향+역방향 연결(21.6s) — 마지막 프레임이 첫
프레임과 동일하므로 "종료 위치=시작 위치"라는 정답이 생긴다. 재생성:
`python -c "..."` (정방향 프레임 + 역순 프레임을 VideoWriter로 연결,
sources/sample_720p_loop.mp4).

| 항목 | loop OFF | loop ON |
|---|---|---|
| 지도 포인트 | 194,808 | **138,884 (−29%)** — drift 이중상 병합 |
| 복귀 오차 (traj 기준) | 0.1182 | 0.1076* |
| 루프 수락 | — | 3개 윈도에서 15쌍 (inl 82~1049) |
| 윈도 처리 시간 | 0.7~1.0s | 1.0~1.8s (ORB+pose graph 포함) |

*traj는 프레임 당시 기록이라 마지막 보정(드레인 중 도착)이 반영되지 않아
실제 개선을 과소측정한다. 지도 이중상 제거(−29%)가 더 직접적인 품질 신호.

### 튜닝 과정에서 확인된 함정 3개 (코드 주석에도 기록)

1. **pose graph 스케일 가중은 엣지별로 달라야 한다**: 단일 w_scale을
   높이면 노이즈가 큰 루프 엣지의 스케일 측정(mono depth 비율)까지 강하게
   믿어 역효과. 순차 엣지(scale=1)는 뻣뻣하게(×6), 루프 엣지는 보통으로.
2. **루프 직후 T_global_live는 재피팅하면 안 된다**: 보정이 윈도 내 카메라
   배치를 비균일 변형시켜 Umeyama 스케일이 폭주 (윈도 전체 1.4~1.5,
   신규 kf만 써도 정지 구간에서 2.6~9.8). **합성으로 갱신**:
   T_gl ← C_newest ∘ T_gl (pose graph의 최신 노드 보정을 그대로 적용).
3. **robust_sim3 폴백의 미세 베이스라인 비율 폭주**: 카메라 정지 구간에서
   두 미세 거리의 비율이 스케일로 직행 (실측 9.8x) → d>1e-3 요구 +
   [0.5, 2.0] 클램프 추가 (일반 견고성 수정).

## 업그레이드 전/후 종합 A/B (2026-06-11)

`config_legacy.yaml`(업그레이드 전 설정: SMALL 백엔드·12뷰·voxel 0.03·
루프/GS 없음) vs 현재 config.yaml. headless stride 3, 동일 영상.
재현: `python benchmarks/headless_run.py --config config_legacy.yaml ...`

| 영상 | 성격 | 지도 (legacy→new) | 궤적 차이 | 비고 |
|---|---|---|---|---|
| sample_720p (10.8s 거실) | 정상 도메인 | 27.6k → 133.5k (**4.8x**) | max 0.003 | 동일 윤곽, 표면 채움 |
| kitchen (9.2s 주방, 세로) | 정상 도메인 | 5.8k → 46.0k (**7.9x**) | max 0.035 | 점선 수준→연속 표면 |
| 8716784 (12.4s 회의실+보행자) | 동적 환경 | 4.8k → 13.2k (**2.7x**) | max 0.005 | 상호 커버리지 93/92% — 같은 기하의 고밀도판. 루프 오탐 후보 2건 게이트 기각 |
| 부메랑 (21.6s 재방문) | 루프 검증 | 이중상 −29% | — | Tier 4 섹션 |
| 10567193 (32s 책상 클로즈업+손) | **도메인 밖** | 양쪽 모두 불안정 | 발산 | 아래 참고 |

- 톱다운 투영 육안 확인: 세 정상 영상 모두 공간 윤곽·궤적 동일, 밀도만 증가
  (구조 왜곡 없음).
- e2e 실시간 모드 (sample_720p, 2회 반복): legacy 6.0~6.3 FPS → 신규(GS·
  루프·LARGE 백엔드 전부 켬) **5.4~5.7 FPS** — 신기능 총비용 약 8~10%.
  GS held-out PSNR은 주기마다 상승 (17.8 → 18.9dB).
- **도메인 밖 영상에서 확인된 기존 한계** (legacy도 동일 — 회귀 아님):
  매크로 클로즈업+손 영상에서 calib a가 0.6→0.01로 붕괴 (DA3 상대 depth
  정규화가 클로즈업 콘텐츠에서 요동 + VO 추적 끊김). 이때 루프 후보가
  다수 떴지만(지각적 유사 장면) 3D 검증 게이트가 30여 건 전부 기각 —
  스케일 붕괴 상황에서 잘못된 보정으로 지도를 망치지 않음을 확인.
  단, 같은 이유로 진짜 루프도 극단적 스케일 drift(게이트 0.2~5x 밖)에서는
  수락되지 못한다.

## 메모리 사건 분석 — frames.mov (2026-06-11/12)

3.4분 세로(720×1280@60fps) 야외 보행 영상에서 메모리 에러 발생 → 원인 3중
규명, 모두 수정:

1. **세로 영상 백엔드 크기 버그**: 백엔드 입력(bw,bh)을 가로변 기준으로
   계산해 세로 영상에서 키프레임 배열·GS 렌더 해상도가 의도(긴 변 504)의
   3.2배 픽셀 → VRAM 16GB 포화 → WDDM 스왑으로 백엔드 윈도가 30~97초로
   폭증. **긴 변 기준으로 수정** (main/headless).
2. **GS 레이어의 GPU 독식**: anytime 데드라인이 주기의 80%라 GPU 경합 시
   백엔드·라이브를 굶김. **50%로 양보 + 전역 VRAM 여유 <1.5GB면 해당 주기
   최적화 스킵 + 주기 후 empty_cache()** (backend 윈도 후에도 동일 —
   3개 CUDA 프로세스가 예약 캐시를 각자 쥐면 합산이 16GB를 넘는다).
3. **종료 행 → 좀비 프로세스의 VRAM 점유**: mp.Queue에 남은 대형 항목이
   feeder 스레드의 pipe 쓰기를 막아 인터프리터 exit가 영원히 멈춤. 행된
   실행이 CUDA 컨텍스트를 쥔 채 좀비로 남아(실측 6개 발견) 이후 모든
   실행의 VRAM을 잠식 — 메모리 에러의 숨은 공범. **stop()에서 큐 드레인 +
   cancel_join_thread()** (`_drain_and_close`).

수정 효과 (frames.mov 재실행): 1.9 → 3.3 FPS (+74%), 백엔드 윈도 97s →
대부분 3~7s, 에러 0. 단 이 영상 자체는 야외 보행이라 도메인 밖 — VO가
간헐적 LOST, GS PSNR 13~18dB에서 정체 (계속 새 영역이라 수렴 불가).

## 신규 영상 테스트 (2026-06-12, headless stride 3)

| 영상 | 결과 | 소견 |
|---|---|---|
| frames_tango (4.3s) | 61k pts | 스모크 통과 |
| dataset-desk (98.8s, TUM desk) | 327k pts, 980 poses | calib a 0.4~0.96 요동. **루프 후보 수십 건(sim 최대 0.90) 전량 3D 기각** — 책상 궤도 재방문은 진짜 루프일 가능성이 높아 기각 사유 진단 필요 (스케일 게이트 vs 인라이어 부족 구분 로깅 과제) |
| office-loop 루프 OFF (137.9s 사무실 워크스루) | 367k pts, 1095 poses | calib a 0.83→0.41 단조 drift — 루프 클로저가 필요한 전형 패턴 |
| office-loop 루프 ON | 220k pts, 루프 4건 수락 | 아래 진단 참고. 시작 행은 재실행에서 미재현 (좀비 경합으로 인한 일과성으로 판단) |

### 루프 기각 원인 진단 + 보정 효과 판정 (2026-06-12)

기각 사유 구분 로깅(`matches/best_inl/med_s`)으로 desk·office 재실행한 결과:

- **기각의 지배적 원인은 매칭 부족** — 후보 대부분이 ORB 원시 매칭 <25개로
  검증 시도조차 못 함 (early-out). 매칭이 충분한 경우(28~34개)에도 3D
  인라이어가 13~21개로 문턱(25) 직전 탈락. **스케일 게이트는 원인이 아님**
  (med_s≈0.9~1.5로 정상 범위) — 이전 추정 수정.
- **수락된 루프의 보정은 기하학적으로 옳음**: desk에서 루프 1건 수락 후
  궤적이 닫힌 궤도로 복원(톱다운 육안). 단 영상 끝에서 한 번에 발동해
  대형 보정(max|t|=0.8~4.7, scale 3.1)이 되고, 누적 중복 voxel이 대거
  병합되며 지도가 일시적으로 희박해짐 (desk 327k→14.8k — 라이브 세션이면
  재관측으로 재축적됨).
- **근본 질환은 mono depth 스케일 drift**: desk/office 모두 calib a가
  1~2분에 걸쳐 0.83→0.1대로 붕괴 (DA3 상대 depth 정규화가 시점 이동에
  따라 흐름). 루프 클로저는 말단 증상을 마지막에 당기는 역할에 그침.

후속 권장 3건 — **모두 구현·검증 완료 (2026-06-12, 아래 섹션)**.

## 스케일 서보 + 루프 강화 + 점진 보정 (2026-06-12)

### 구현

1. **스케일 서보** (`backend.scale_servo`): metric 앵커의 mpu(미터/단위)를
   최초 기준에 묶고, 편차를 윈도당 ±5%로 보정. **핵심 함정**: 백엔드가
   calib만 키우면 main의 frame_scale 피드백(키프레임 3D 예측 z vs 측정 z)이
   1/g로 즉시 상쇄한다 — 키프레임 3D가 옛 스케일이기 때문 (1차 구현 실측:
   효과 없음). 백엔드는 이득 g만 보내고(`BackendResult.servo_gain_g`),
   main이 **calib·VO 상태(vo.rescale)·T_global_live(rescale_live)에 일관
   적용**해야 작동한다.
2. **루프 매칭 강화**: ORB 1200→2500, 인라이어 임계값을 depth 비례로
   (mono 3D 오차 ∝ z — 고정 임계값이 원거리 매칭을 부당 기각하던 문제),
   min_inliers 25→15 + min_inlier_frac 0.45 (합의율이 오탐 방벽).
3. **점진 보정**: 윈도당 보정 상한 |t|≤0.5, scale≤1.5x — 초과분은 λ<1로
   부분 적용하고 다음 윈도 재검출로 반복 수렴.

### 검증 (office-loop 137.9s)

| | v0 (수정 전) | 수정 후 |
|---|---|---|
| mpu drift | 단조 발산 (~3x 함의) | **실행 대부분 3.6~4.6m 유지**, 말미 6.1 이탈 |
| 루프 발화 | 말미 1라운드 (대형 보정 한 방) | 2~6라운드 조기·반복 (λ=0.07~0.96 부분 적용) |
| 지도 | 367k pts, 번짐 심함 | 171k pts — 육안상 번짐 감소·구조 선명 |

desk: 루프 0~1건 → 2라운드 발화, 궤적이 조밀 궤도로 복원. 잔여 한계:
말미 무재방문 구간의 drift는 서보 권한(±5%/윈도)을 초과할 수 있음 —
이득 상향은 metric 추정 노이즈 증폭과 트레이드오프라 보수적으로 유지.

### 종료 행 2차 수정 (중요)

큐 드레인 수정(bc07624)이 **새로운 행 경로**를 만들었음을 py-spy로 실측:
Windows에서 죽은 자식이 쓰다 만 찢어진 메시지가 파이프에 남으면
`get_nowait()`의 poll()은 True인데 `_recv_bytes()`가 영원히 블록한다.
→ 드레인 제거, `cancel_join_thread()+close()`만으로 충분 (exit 행 방지
목적은 feeder join 포기로 달성). headless는 저장을 stop()보다 먼저 하도록
순서 변경 (종료 경로 문제가 결과를 잃지 않게). 일전의 office-loop "시작
행"도 사실은 이 종료 행이었음 (영상 처리는 완료된 상태였음 — stdout이
findstr 파이프라인에 버퍼링되어 시작 행처럼 보였던 것).

## 실사용 재검증 — office-loop 라이브 (2026-06-12, 사용자 보고 후속)

사용자 보고("기울어짐 미개선, 시작점 미복귀")를 라이브 실행 6회로 재현·진단.
**보고가 정확했음** — fix-plan 구현에 결함 3건이 있었고 수정:

1. **자세 서보가 0회 실행** (지배적 원인): "루프 보정이 없는 윈도에서만"
   게이트했는데 persist 재실행이 매 윈도 발화해 서보가 영구 기아.
   → 루프 보정과 *합성*해 매 윈도 실행으로 수정.
2. **persist 재실행 정체**: weight 감쇠(0.9ⁿ)로 약해진 루프 엣지가 순차
   엣지 강성에 밀려 잔차 0.483에서 18윈도 정체. → 감쇠를 수렴한 엣지에만
   적용 (수정 후 r 2.0→0.69 실수렴).
3. **포인트 기반 바닥 추정이 항상 기각**: 임계값 0.02×spread(≈0.009)가
   융합 점군 노이즈 대비 5~10배 빡빡 + 전체 점군에서 바닥 비중 3.4%뿐.
   → 높이 하위 45% 선필터 + 임계값 0.08×spread(바닥 0.03).

추가 발견 — **Y 침하의 주성분은 회전 drift가 아니라 높이 drift**: 윈도별
바닥 기울기 측정은 1~7°인데 카메라 Y가 단조 침하 (mono depth 바닥 편향이
VO 병진에 주입되는 계단식 패턴). → **바닥 높이 서보** 추가: 첫 윈도 바닥
평면 y를 기준으로 이번 epoch만 정렬 (과거 epoch은 들어올 때 이미 정렬돼
계단 비누적). 측정 게이트: 기준에서 0.45 이상 먼 평면은 책상면으로 기각.

| 라이브 실행 (138s) | 서보 발화 | 카메라 Y 침하 (이상적 0) |
|---|---|---|
| 수정 전 | 0회 | **+1.72 단조 증가** (끝까지 침하) |
| 전체 수정 후 | 5~13회 | **+1.1 부근에서 정지** (후반부 침하 멈춤) |

잔여 한계 (정직 기록): 초반 ~60초의 침하(+1.0)는 보수적 서보 예산(윈도당
높이 0.2·회전 3°)이 drift 속도를 따라잡기 전에 누적된 것 — 잔차는
점진 보정되나 0으로 완전 복귀는 못 한다. 근본 대응은 VO 단계의 mono depth
바닥 편향 보정(향후 과제). "시작점 미복귀"는 영상 검수 결과 **영상 자체가
시작 지점과 다른 통로에서 끝남**(프레임 비교 확인) — 중간 재방문 구간의
루프는 발화·수렴함.

### 키프레임 바닥 anchoring — VO단 근본 대응 (2026-06-12, 해결)

위 잔여 한계의 근본 수정: **키프레임마다 그 프레임 depth의 바닥(법선·높이)을
측정해 pose의 기울기·높이를 절대 기준에 부분 복원**(`vo.floor_anchor`,
`floor_anchor_correction` + `vo.apply_keyframe_correction`). drift가
키프레임 단위(0.5s)로 리셋되어 애초에 누적되지 않는다. 게이트: 기울기
>20° 또는 기준 바닥과 0.4+ 차이(책상면)는 기각, 복원은 β=0.5 부분
(회전 ≤2°/kf, 높이 ≤0.08/kf), 비용 ~10ms/kf.

| office-loop 라이브 (수평 보행) | 카메라 Y 침하 (이상적 0) |
|---|---|
| 수정 전 | +1.72 단조 침하 |
| 백엔드 서보 단계 | +1.15 (후반 정지) |
| **키프레임 바닥 anchoring** | **±0.2 전 구간 유지** (`_height_drift_compare.png`) |

백엔드 자세/높이 서보는 백스톱으로 유지 (anchoring이 잡으면 자연 휴면).
스케일 서보가 live 단위를 바꾸면 기준 높이도 함께 리스케일(scale_state).
주의: headless_run.py에는 이 main 전용 경로가 없어 A/B 시 참고.

### 한계 (문서화된 v1 범위)

- 보정 후 live→global 스케일이 1에서 벗어날 수 있다 (모노큘러 게이지 —
  부메랑 극단 케이스에서 0.7대). mpu(미터 환산)는 스케일로 나눠 보정함.
- GS 레이어의 기존 gaussian은 루프 보정으로 이동하지 않는다 (표시 변환만
  따라감) — 장기 세션에서 GS 내부 drift는 남는다. 차후 epoch별 gaussian
  이동 또는 재초기화로 확장 가능.
- 객체 노드는 재관측 시 EMA로 자가 보정된다 (루프 시 즉시 이동은 미구현).
