# Space Recognizer

웹캠(또는 영상 파일)으로 주변 공간을 돌아다니며 촬영하면:

- 실시간으로 물체를 검출·추적하고 3D 위치를 추정하며 (YOLOE 오픈 보캐뷸러리 + DA3-Small depth)
- 5초 주기 멀티뷰 재구성으로 정적 공간의 3D 지도를 점점 완성하고 (DA3-Large any-view)
- 별도 3D 뷰어(Rerun)에 **오브젝트 노드 + 관계 그래프**를 그리고,
- 화면 밖으로 나가거나 가려진 물체의 위치를 **계속 기억**해서 표시합니다.

Apple Silicon Mac(M1 Max, 32GB)에서 CUDA 없이 MPS만으로 동작하도록 설계되었습니다.

> 📖 **상세 사용법은 [docs/MANUAL.md](docs/MANUAL.md)** — 설치, 뷰어 조작,
> 촬영 요령, 설정 튜닝, 문제 해결(FAQ)을 다룹니다.

## 실행

```bash
# 의존성 설치 (최초 1회)
uv venv --python 3.12
uv pip install -p .venv -e ".[dev]"
uv pip install -p .venv --no-deps depth-anything-3   # numpy<2/xformers 핀 우회

# 샘플 영상으로 실행 (config.yaml의 source 사용)
.venv/bin/python -m spacerec.main

# 특정 영상 / 웹캠으로 실행
.venv/bin/python -m spacerec.main --source sources/sample_720p.mp4
.venv/bin/python -m spacerec.main --source 0          # 웹캠 (카메라 권한 필요)
.venv/bin/python -m spacerec.main --source 0 --map maps/room.npz  # 세션 간 누적

# 옵션
#   --max-seconds 8     : 앞 8초만 처리
#   --no-realtime       : 벽시계 페이싱 없이 모든 프레임 처리 (오프라인 분석)
#   --profile           : 단계별 처리 시간 출력
```

첫 실행 시 모델 가중치(총 약 3.6GB: DA3-SMALL 0.3GB, DA3-LARGE-1.1 1.4GB(CUDA 백엔드), DA3METRIC-LARGE 1.4GB,
YOLOE-26s-seg + MobileCLIP2 텍스트 인코더 0.3GB, DINOv2-small)가 자동
다운로드됩니다. Rerun 뷰어 창이 자동으로 열립니다.

- 웹캠 모드는 터미널 앱에 macOS 카메라 권한이 필요합니다
  (시스템 설정 → 개인정보 보호 및 보안 → 카메라).
- 영상 파일 모드는 기본적으로 **웹캠처럼 벽시계 기준으로 프레임을 드롭**하며
  재생합니다 (`config.yaml`의 `realtime: false`로 전체 프레임 처리 가능).

## Rerun 뷰어 보는 법

- **좌상단 Live RGB**: 검출 박스 + 추적 id
- **좌중단 Depth**: 보정된 실시간 depth
- **좌하단 Depth Calibration**: depth 보정 계수(a, b, frame_scale) 시계열 —
  캘리브레이션이 동작 중인지 확인용
- **우측 3D World**:
  - 컬러 포인트클라우드 = 누적된 정적 공간 지도
  - 파란 선 = 카메라 이동 궤적, 피라미드 = 현재 카메라
  - 구 + 라벨 = 기억된 오브젝트 (반투명 = 지금 안 보이는 물체의 기억된 위치)
  - 회색 선 = 근접 관계, 주황 선 = 위/아래 관계 (라벨에 거리, metric 앵커
    활성 시 미터 단위)
- 하단 타임라인으로 과거 시점 스크럽 가능

## 아키텍처 (3계층)

```
[실시간 계층 — 매 프레임, MPS]
  YOLOE-26s-seg + ByteTrack → 객체 mask/추적 (config의 vocabulary로 어휘 지정)
  DINOv2-small 외형 임베딩  → 재등장 물체 re-ID
  DA3-Small mono depth     → 보정된 depth → 객체 3D 위치
  LK 광학흐름 + PnP RANSAC → 카메라 pose (live 좌표계)

[백엔드 계층 — 5초 주기, 별도 프로세스]
  키프레임 윈도(16뷰, 절반 중첩) → DA3-LARGE-1.1 any-view 멀티뷰 추론 (CUDA; MPS 등은 backend_model을 비워 SMALL 폴백)
  → 멀티뷰 depth를 라이브 스케일로 정합(α,β) → 전역 지도 voxel 융합
  → mono depth 보정(a,b), intrinsics 추정, (옵션) metric 앵커
  → T_global_live(Sim3)를 부드럽게 갱신 — 객체 위치가 점프하지 않음

[월드 모델 — 증거 기반 갱신]
  GlobalMap: 가중치 voxel + free-space carving — 새 관측의 시선이 옛 표면을
    관통하면 빈 공간 증거로 깎아 제거. 잘못된 재구성은 재촬영으로 지워지고,
    단발성 불량 관측이 좋은 지도를 바꾸려면 반복 증거가 필요 (양방향 견고)
  ObjectRegistry: 영속 위치(EMA), 재등장 re-ID 병합, dynamic 판정,
    부재 증거(시야 안·비가림인데 미검출 누적 → 노드 제거)
  SceneGraph: 거리 기반 근접 엣지 + 위상 관계(위/아래/옆)
```

설계 근거와 실측 데이터는 `docs/plan.md`, `docs/benchmarks.md` 참고.
특히 benchmarks.md에는 Mac/MPS 환경에서 확인된 함정들(MPS 멀티스레드 크래시,
DA3-small pose 헤드의 병진 과소추정, 커뮤니티 래퍼 결함 등)이 기록되어 있습니다.

## 테스트

```bash
.venv/bin/python -m pytest tests/ -q          # 단위 테스트 (기하/보정/레지스트리/증거 갱신/영속화)
.venv/bin/python benchmarks/bench_models.py   # 모델 속도 벤치마크
.venv/bin/python benchmarks/headless_run.py   # 헤드리스 전체 파이프라인 → /tmp/map.npz
```

## 알려진 한계 / 다음 단계

- 처리율은 백엔드 추론과 GPU를 공유하는 동안 약 2.5~4 FPS (목표 5~10 FPS의 하한).
  `backend.metric_anchor: false`로 끄면 다소 빨라집니다.
- 전역 좌표계는 라이브 VO 궤적을 따르므로 장시간 사용 시 drift가 누적됩니다.
  (DA3-small의 pose 헤드가 병진을 과소추정해 독립적 drift 보정원으로 쓸 수 없음 —
  루프 클로저는 향후 과제)
- 대형 가구는 카메라가 궤도를 돌 때 mask 중심이 이동해 dynamic으로 오판될 수 있음.
- 동적 물체의 시간별 궤적은 기록·표시되지만(`world/objects/dyn_traj`) 검증은 후순위.
- YOLOE 오픈 보캐뷸러리는 검출이 풍부한 대신 중복/노이즈 노드가 다소 늘 수 있음
  (`detect.conf` 상향이나 어휘 축소로 조절).
- `--map` 재위치추정은 같은 물체가 3개 이상 다시 보여야 성공 — 이전 세션과
  완전히 다른 각도로만 촬영하면 정렬에 실패할 수 있음 (데이터는 보존됨).
- 10~12초 샘플 영상 3종(거실·침실·주방)으로 개발·검증됨 — 1~3분짜리 실내
  촬영 영상으로 장시간 영속성/재등장 검증 권장.
