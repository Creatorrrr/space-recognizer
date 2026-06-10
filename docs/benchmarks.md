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
| 라이브 계층 합계 (depth+track) | ~110 ms → 5.1 FPS e2e 실측 | ≤ 200 ms | ✅ |

## 결정 사항
- **YOLO26n-seg 채택** (YOLO11 폴백 불필요, MPS 정상 동작, mask+track id 확인)
- **공식 `depth-anything-3` 0.1.1 패키지 채택** — MPS에서 SDPA로 그대로 동작.
  멀티뷰 출력: depth (V,280,504), extrinsics (V,3,4) w2c, intrinsics (V,3,3) @504px
- ⚠️ **커뮤니티 래퍼 `awesome-depth-anything-3`는 사용 금지**: 체크포인트 로딩이 깨져
  depth가 ~1.1 상수 + 격자 노이즈로 출력됨 (mono/multi-view, CPU/MPS 모두). 공식
  패키지로 교차 검증해 확인. HF 모델 id는 `depth-anything/DA3-SMALL` 사용.
- 백엔드 윈도 8~16뷰 모두 5초 주기 내 처리 가능 (2.3~2.7s)

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
