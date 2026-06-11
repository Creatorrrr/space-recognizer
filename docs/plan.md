# Space Recognizer — 웹캠 기반 실시간 3D 공간·오브젝트 인식 시스템 설계 및 작업 계획

> ⚠️ **이 문서는 구현 착수 시점의 설계 스냅샷입니다.** 구현 과정에서 다음이
> 변경·추가되었습니다 (현재 상태는 `README.md`, 변경 근거는 `docs/benchmarks.md`):
> - 검출: YOLO26n-seg → YOLOE-11s-seg → **YOLOE-26s-seg 오픈 보캐뷸러리** (COCO 어휘 한계 해소)
> - DA3: 커뮤니티 MPS 래퍼 → **공식 depth-anything-3 패키지** (래퍼 결함 발견)
> - 백엔드 pose: DA3 extrinsics 사용 불가 판명 → **라이브 VO pose 사용**
> - 추가된 기능: DINOv2 외형 re-ID, 증거 기반 지도 갱신(free-space carving),
>   객체 부재 처리, 세션 간 영속화+재위치추정(`--map`), 아이폰 Continuity Camera

## 1. 컨텍스트

웹캠(개발 단계에서는 `sources/`의 영상을 웹캠 입력으로 에뮬레이션)으로 주변 공간을 돌아다니며 촬영하면:

1. 실시간 depth 맵으로 현재 프레임의 **물체 3D 위치를 즉시 추정**하고,
2. 주기적인 멀티뷰 3D 재구성으로 **정적 공간 지도를 점점 완성**하며 depth를 캘리브레이션하고,
3. 별도 3D 뷰어에 **오브젝트 노드 + 관계 엣지 그래프**를 그리고, **가려지거나 화면 밖으로 나간 물체의 위치를 계속 기억**해서 표시한다.

원안은 FlashDepth + DROID-W + WildGS-SLAM 3계층 구조였으나, 실행 환경이 **MacBook Pro M1 Max(32GB, CUDA 불가, PyTorch 2.10 MPS)** 로 확정되어 세 기술 모두 사용 불가(커스텀 CUDA 커널 의존). **3계층 구조의 철학(빠른 측정 / 실시간 pose·기준 좌표계 / 주기적 지도 품질 개선)은 유지**하고 구현체를 Mac에서 동작하는 최신 기술로 교체한다.

### 사용자 결정 사항
| 항목 | 결정 |
|---|---|
| 실행 환경 | 이 Mac 단독 (M1 Max, MPS) |
| 실시간성 | 5~10 FPS면 충분 |
| 객체 인식 | 고정 클래스 + 세그멘테이션 (YOLO 계열 -seg) |
| 3D 시각화 | Rerun 뷰어 |
| 그래프 엣지 | 거리 기반 근접 관계 + 위상 관계(위/아래/옆) |
| 백엔드 | **A안: DA3(Depth Anything 3) 단일 패밀리** |
| 테스트 영상 | 기존 10.8초 영상으로 우선 진행 (부족 시 추가 요청) |
| 동적 물체 시간별 기억 | 후순위 (Phase 6) |

### 기술 스택 (리서치 검증 완료, 2026-06 기준 최신)
| 역할 | 기술 | 근거 |
|---|---|---|
| 라이브 depth (매 프레임) | **DA3-Small** (0.08B, Apache 2.0) | MPS 실측 ~46ms/frame(~22 FPS). `awesome-depth-anything-3` pip 래퍼가 MPS+SDPA 지원 |
| 5초 백엔드 멀티뷰 재구성 | **DA3-Small/Base any-view** (동일 모델로 멀티뷰 pose+pointmap 추정) | 단일 모델 패밀리·단일 버전 원칙. DA3-Streaming(VGGT-Long 방식)의 청크+Sim3 정렬 레시피 차용 |
| 절대 치수 앵커 (선택) | **DA3METRIC-LARGE** (Apache 2.0, ~265ms) | 저주기(수 초당 1회)로 metric scale 보정 |
| 실시간 카메라 pose | **광학 흐름(LK) + PnP RANSAC** (OpenCV, CPU 30+ FPS) | DROID-W의 실시간 pose 역할 대체. 키프레임 depth로 3D 점 역투영 → 추적 → solvePnPRansac |
| 객체 검출+추적 | **YOLO26n/s-seg + ByteTrack** (`device="mps"`, ultralytics) | 2026-01 최신, NMS-free. 문제 시 YOLO11-seg 폴백 |
| 3D 시각화 | **Rerun SDK 0.33** | Points3D/Pinhole/LineStrips3D/라벨/타임라인 내장 |
| 보조 | OpenCV, NumPy, SciPy, Open3D(voxel downsample) | |

MPS 공통 주의: fp32 강제(fp16 autocast 불안정), xformers/flash-attn → `F.scaled_dot_product_attention` 대체, `PYTORCH_ENABLE_MPS_FALLBACK=1`.

---

## 2. 아키텍처

```
[입력]  VideoSource (sources/*.mp4 → 웹캠 에뮬레이션 | 실제 웹캠)
          │  벽시계 기준 프레임 드롭으로 실시간성 모사, 4K→720p 다운스케일
          ▼
[실시간 계층 — 매 처리 프레임, 목표 5~10 FPS]
  DA3-Small mono depth (MPS, 518px)
  YOLO26-seg + ByteTrack (MPS, 640px)
  LK 광학흐름 + PnP RANSAC → 현재 카메라 pose (live 좌표계)
  depth 캘리브레이션 적용 (a·D_fast + b) → 객체 mask 내부 depth median
  → 객체 3D 위치 (live frame) → T_global_live 곱해 world frame
          ▼
[키프레임 매니저]
  선정: 시간(≥0.5s) + 시차(median flow > τ) + PnP inlier 비율 하락
  키프레임 = {RGB, depth, pose, 객체 mask, 타임스탬프}, 링버퍼 + 전역 커버리지용 앵커 보존
          ▼
[백엔드 계층 — 5초 주기, 별도 스레드]
  최근 키프레임 윈도(8~16뷰, 직전 윈도와 절반 중첩) → DA3 any-view 멀티뷰 추론
  → 키프레임 pose + pointmap (윈도 로컬 좌표계)
  → 중첩 키프레임으로 Sim(3) 정렬(Umeyama+RANSAC) → 전역 지도 좌표계에 누적
  → 정적 픽셀만 voxel hash로 전역 포인트클라우드 융합
  → T_global_live 부드럽게 갱신 (객체 위치 순간이동 방지)
  → 라이브 depth 캘리브레이션 계수(a, b) robust 재추정 (Huber/RANSAC, 정적 픽셀만)
  → (선택) DA3METRIC-LARGE로 metric scale 앵커
          ▼
[월드 모델]
  GlobalMap: 전역 정적 포인트클라우드 (voxel downsample)
  ObjectRegistry: track_id/class/world 위치(EMA)/최종 관측 시각/관측 수
    - 화면 밖·가림 → 마지막 world 위치 유지 (영속성)
    - 재등장 re-ID: 동일 class + 반경 내 → 기존 노드에 병합
    - 위치 분산 > τ → dynamic 플래그 (정적 지도 융합에서 픽셀 제외)
  SceneGraph: 거리 < τ_near 엣지(거리 라벨) + 위상 관계(위/아래/옆)
          ▼
[시각화 — Rerun 0.33]
  2D: RGB+검출박스/마스크, depth 컬러맵, 세그멘테이션
  3D: 전역 포인트클라우드, 카메라 궤적+현재 frustum,
      객체 노드(구+라벨, 미관측은 반투명), 그래프 엣지(LineStrips3D+거리 라벨)
  타임라인 스크럽, --memory-limit 설정
```

### 좌표계 설계 (원안의 핵심 유지)
- **live frame**: VO가 매 프레임 추정하는 좌표계 (마지막 키프레임 기준 연쇄)
- **global frame**: 백엔드가 5초마다 정렬·보정하는 누적 지도 좌표계
- `T_global_live`(Sim3)를 항상 유지. 백엔드가 drift를 고치면 이 변환만 수 프레임에 걸쳐 보간 갱신 → 라이브 객체 위치가 튀지 않음.

### depth 캘리브레이션 (원안 설계 채택)
```
정적 픽셀 S(t) = 객체 mask 밖 ∧ 흐름 잔차 낮음 ∧ 기준 depth 유효
기준 depth   = 최신 키프레임 pointmap을 현재 pose로 재투영한 D_ref
robust fit   : min over a,b  Σ_{p∈S} Huber(a·D_fast(p) + b − D_ref(p))
```
계수는 EMA로 평활. metric 앵커 사용 시 D_ref를 DA3METRIC 출력으로 스케일 정합.

### 카메라 intrinsics
미지수. 1차: DA3 any-view가 추정하는 카메라 ray에서 유도. 폴백: 수평 FOV 60° 가정 후 config로 조정. PnP/역투영에 동일 K 일관 사용.

---

## 3. 프로젝트 구조

```
space-recognizer/
├── pyproject.toml          # uv 관리, Python 3.12
├── config.yaml             # 해상도/주기/임계값/모델 변형/입력 소스
├── src/spacerec/
│   ├── config.py           # dataclass 설정 로더
│   ├── capture.py          # VideoSource: 파일(웹캠 에뮬) | 웹캠, 동일 인터페이스
│   ├── depth.py            # DA3-Small mono 래퍼 (MPS, fp32)
│   ├── detect.py           # YOLO26-seg + ByteTrack 래퍼
│   ├── vo.py               # LK+PnP VO, KeyframeManager
│   ├── calib.py            # depth affine robust fit (Huber/RANSAC)
│   ├── backend.py          # DA3 멀티뷰 워커(스레드), Sim3 윈도 정렬
│   ├── worldmap.py         # GlobalMap: voxel 융합, T_global_live 관리
│   ├── objects.py          # ObjectRegistry: 영속성, re-ID, dynamic 판정
│   ├── graph.py            # SceneGraph: 근접/위상 엣지 계산
│   ├── viz.py              # Rerun 로깅 + blueprint 레이아웃
│   └── main.py             # 오케스트레이터 (CLI: --source file|webcam)
├── tests/                  # calib/Sim3/registry 단위 테스트 (합성 데이터)
├── benchmarks/             # Phase 0 속도 측정 스크립트
├── sources/                # 테스트 영상
└── docs/                   # 설계 문서, 벤치마크 기록
```

각 모듈은 단일 책임 + 명시적 인터페이스(예: `DepthEstimator.infer(frame)->np.ndarray`)로 분리해 모델 교체(예: 추후 MapAnything 백엔드)가 가능하게 한다.

---

## 4. 작업 단계

### Phase 0 — 환경 구축 + 기술 스파이크 (결정 게이트)
- `uv venv` + 의존성 설치: torch 2.10, awesome-depth-anything-3, ultralytics, rerun-sdk, opencv-python, open3d, scipy
- 벤치마크 스크립트로 **실측**: DA3-Small mono(518px) ms/frame, DA3 멀티뷰 8뷰 1회 추론 시간·메모리, YOLO26-seg(640px) ms/frame, 동시 실행 시 MPS 경합
- 샘플 영상 프레임으로 DA3 멀티뷰 품질 육안 확인 (Rerun에 pointmap 출력)
- **게이트**: 라이브 계층 합계 ≤ 200ms/frame(5 FPS), 백엔드 8뷰 ≤ 30s. 미달 시 해상도/뷰 수/모델 크기 조정 후 docs에 기록
- YOLO26 MPS 회귀 발견 시 YOLO11-seg로 폴백 결정

### Phase 1 — 입력 파이프라인 + 라이브 2D 인식
- `capture.py`: 영상 파일을 벽시계 기준으로 읽어 웹캠처럼 프레임 드롭하는 에뮬레이터 + 실제 웹캠 모드
- `depth.py`, `detect.py` 래퍼 구현
- `viz.py` 1차: Rerun 2D 패널(RGB+박스+마스크, depth 컬러맵)
- 검증: 샘플 영상 재생 시 Rerun에서 검출·추적·depth가 5 FPS 이상으로 갱신

### Phase 2 — 실시간 카메라 pose + 단일 프레임 3D
- `vo.py`: goodFeaturesToTrack → LK(전후방 검증) → solvePnPRansac, 키프레임 선정 로직
- 객체 mask depth median → 카메라 좌표 3D → live frame 변환
- Rerun 3D 뷰: 카메라 frustum 궤적 + 현재 프레임 포인트클라우드 + 객체 위치 점
- 검증: 카메라가 움직일 때 궤적이 연속적이고 객체 점이 공간에 안정적으로 찍힘. calib/vo 단위 테스트(합성 데이터) 통과

### Phase 3 — 5초 백엔드 재구성 + 전역 지도
- `backend.py`: 키프레임 윈도 → DA3 멀티뷰 → pose+pointmap, 중첩 키프레임 Umeyama Sim3 정렬
- `worldmap.py`: voxel hash 융합, `T_global_live` 보간 갱신
- `calib.py`: 정적 픽셀 robust affine fit → 라이브 depth 보정 적용
- 검증: 10초 영상 종료 시 전역 포인트클라우드가 일관된 단일 공간으로 누적(이중상 없음), 백엔드 갱신 시 객체 위치 점프 없음. Sim3 정렬 단위 테스트 통과

### Phase 4 — 오브젝트 월드 모델 + 그래프
- `objects.py`: EMA 위치, 영속성(미관측 유지), class+반경 re-ID 병합, 위치 분산 기반 dynamic 플래그(정적 융합 제외)
- `graph.py`: 거리 < τ 근접 엣지(거리 라벨), 위상 관계(수직 오프셋 우세+수평 중첩→위/아래, 그 외 옆)
- Rerun: 노드(구+클래스 라벨, 미관측 반투명), 엣지(LineStrips3D), 관계 라벨
- 검증: 물체가 화면 밖으로 나가도 3D 뷰에 위치 유지, 재등장 시 같은 노드로 병합, 그래프 관계가 육안과 일치

### Phase 5 — 통합·튜닝·실사용 준비
- DA3METRIC-LARGE 저주기 metric 앵커 (config 토글)
- 웹캠 라이브 모드 e2e 확인, config.yaml 정리, README/실행 가이드
- 성능 튜닝: 포인트 수 상한(≤1M), `send_columns` 일괄 로깅, 메모리 한도
- 검증: `python -m spacerec.main --source sources/...mp4` 한 줄로 전체 데모 동작

### Phase 6 — (후순위) 동적 물체 시간별 기억
- dynamic 객체의 타임스탬프 궤적 기록, Rerun 타임라인에서 시간별 재생

---

## 5. 리스크와 대응
| 리스크 | 대응 |
|---|---|
| DA3 멀티뷰가 MPS에서 미검증 (mono만 실측 존재) | Phase 0 스파이크에서 최우선 검증. SDPA 패치·fp32. 실패 시 뷰 수/해상도 축소, 최후엔 MapAnything 시도(백엔드 인터페이스 분리로 교체 용이) |
| 라이브(MPS)와 백엔드(MPS) GPU 경합 | 백엔드 추론 중 라이브 depth 프레임 스킵 허용(검출은 지속). Phase 0에서 경합 실측 |
| 10.8초 영상으로는 영속성·재등장 검증 한계 | Phase 4 시점에 사용자에게 1~3분 실내 촬영 영상 요청 |
| 단안 scale drift | 키프레임 중첩 Sim3 정렬 + metric 앵커 + calib EMA |
| 회전 위주 움직임에서 VO 불안정 | depth 기반 PnP는 순수 회전에도 동작. inlier 급락 시 즉시 키프레임 재설정 |

## 6. 검증 방법 (전체)
- 단위: `pytest` — calib affine fit, Umeyama Sim3, registry 병합/영속성 (합성 데이터로 정답 비교)
- 통합: 샘플 영상 e2e 실행 → Rerun에서 (1) 5 FPS 이상 라이브 갱신 (2) 전역 지도 단일 공간 누적 (3) 객체 노드·그래프·영속성 동작 육안 확인
- 성능: benchmarks/ 스크립트로 각 단계 ms 기록, docs에 회귀 추적
