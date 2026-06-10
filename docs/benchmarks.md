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
