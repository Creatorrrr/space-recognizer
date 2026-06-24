# Space Recognizer

웹캠(또는 영상 파일)으로 주변 공간을 돌아다니며 촬영하면:

- 실시간으로 물체를 검출·추적하고 3D 위치를 추정하며 (YOLOE 오픈 보캐뷸러리 + DA3-Small depth)
- 5초 주기 멀티뷰 재구성으로 정적 공간의 3D 지도를 점점 완성하고 (DA3 any-view)
- TSDF submap으로 정적 표면 mesh를 생성/export하고 (recorded OAK metric depth 권장)
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
uv pip install -p .venv -e ".[oak]"       # OAK-D-Lite를 쓸 때만 필요

# 샘플 영상으로 실행 (config.yaml의 source 사용)
.venv/bin/python -m spacerec.main

# 특정 영상 / 웹캠으로 실행
.venv/bin/python -m spacerec.main --source sources/sample_720p.mp4
.venv/bin/python -m spacerec.main --source 0          # 웹캠 (카메라 권한 필요)
.venv/bin/python -m spacerec.main --source oak        # OAK-D-Lite RGB + metric stereo depth
.venv/bin/python -m spacerec.main --source sources/session_20260624_054320_194430108151D05A00 --no-realtime
.venv/bin/python -m spacerec.main --source sources/session_20260624_054320_194430108151D05A00 --no-realtime --mesh-out artifacts/mesh/session.ply
.venv/bin/python -m spacerec.main --source 0 --map maps/room.npz  # 세션 간 누적

# 옵션
#   --max-seconds 8     : 앞 8초만 처리
#   --no-realtime       : 벽시계 페이싱 없이 모든 프레임 처리 (오프라인 분석)
#   --profile           : 단계별 처리 시간 출력
```

첫 실행 시 모델 가중치(총 약 2.2GB: DA3-SMALL 0.3GB, DA3METRIC-LARGE 1.4GB,
YOLOE-26s-seg + MobileCLIP2 텍스트 인코더 0.3GB, DINOv2-small)가 자동
다운로드됩니다. Rerun 뷰어 창이 자동으로 열립니다.

- 웹캠 모드는 터미널 앱에 macOS 카메라 권한이 필요합니다
  (시스템 설정 → 개인정보 보호 및 보안 → 카메라).
- OAK-D-Lite 모드는 DepthAI가 RGB와 RGB에 정합된 미터 단위 stereo depth를
  가져오고, stereo가 비는 픽셀만 DA3로 metric 보정해 채웁니다. USB 상태가
  `HIGH`면 USB 2.0 연결이므로 USB 3 케이블/포트를 먼저 확인하세요.
  IMU가 노출되는 장치에서는 accel/gyro도 함께 읽어 smoke 출력과 프레임
  metadata에 남깁니다. 현재 pose 추정에는 아직 직접 융합하지 않습니다.
- OAK 녹화 세션 디렉터리(`metadata.json`, `events.jsonl`, `streams/`)도
  `--source`로 바로 재생할 수 있습니다. `capture.replay_depth_mode` 기본값
  `calibrated`는 녹화된 stereo depth를 RGB 카메라 좌표계로 재투영하고,
  부족한 픽셀은 기존 OAK depth 보정 경로처럼 DA3 fallback으로 채웁니다.
  빠른 포맷 확인만 필요하면 `resize`로 바꿔 단순 리사이즈 smoke를 돌릴 수
  있습니다.
- 영상 파일 모드는 기본적으로 **웹캠처럼 벽시계 기준으로 프레임을 드롭**하며
  재생합니다 (`config.yaml`의 `realtime: false`로 전체 프레임 처리 가능).
- `--map maps/room.npz`로 world state를 저장할 때 mesh가 활성화되어 있으면
  sidecar `maps/room.mesh.npz`도 함께 저장됩니다. 다음 세션에서 relocalization에
  성공하면 이전 mesh submap anchor에 같은 Sim3 보정을 적용해 현재 meshmap에
  병합합니다.

## Rerun 뷰어 보는 법

- **좌상단 Live RGB**: 검출 박스 + 추적 id
- **좌중단 Depth**: 보정된 실시간 depth
- **좌하단 Depth Calibration**: depth 보정 계수(a, b, frame_scale) 시계열 —
  캘리브레이션이 동작 중인지 확인용
- **우측 3D World**:
  - 컬러 포인트클라우드 = 누적된 정적 공간 지도
  - mesh/submap = TSDF로 추출된 정적 표면 mesh (mesh enabled 시)
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
  OAK stereo depth(선택)   → 미터 단위 depth, DA3-Small은 hole-fill 보조
  DA3-Small mono depth     → 웹캠/영상 기본 depth → 객체 3D 위치
  LK 광학흐름 + PnP RANSAC → 카메라 pose (live 좌표계)

[백엔드 계층 — 5초 주기, 별도 프로세스]
  키프레임 윈도(12뷰, 절반 중첩) → DA3 any-view 멀티뷰 추론
  → 멀티뷰 depth를 라이브 스케일로 정합(α,β) → 전역 지도 voxel 융합
  → mono depth 보정(a,b), intrinsics 추정, (옵션) metric 앵커
  → T_global_live(Sim3)를 부드럽게 갱신 — 객체 위치가 점프하지 않음

[월드 모델 — 증거 기반 갱신]
  GlobalMap: 가중치 voxel + free-space carving — 새 관측의 시선이 옛 표면을
    관통하면 빈 공간 증거로 깎아 제거. 잘못된 재구성은 재촬영으로 지워지고,
    단발성 불량 관측이 좋은 지도를 바꾸려면 반복 증거가 필요 (양방향 견고)
  MeshMap: backend window를 TSDF submap으로 통합하고 triangle mesh를 추출.
    mesh는 RGB-D keyframe evidence에서 재생성 가능한 파생 캐시로 취급하며,
    pose/Sim3 보정은 submap anchor transform 또는 affected submap rebuild로 처리
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
.venv/bin/python benchmarks/oak_smoke.py      # OAK USB/K/depth stream 확인
.venv/bin/python benchmarks/replay_smoke.py sources/session_20260624_054320_194430108151D05A00 --frames 60 --full-models
.venv/bin/python benchmarks/mesh_smoke.py sources/session_20260624_054320_194430108151D05A00 --frames 120
```

## 알려진 한계 / 다음 단계

- 처리율은 백엔드 추론과 GPU를 공유하는 동안 약 2.5~4 FPS (목표 5~10 FPS의 하한).
  `backend.metric_anchor: false`로 끄면 다소 빨라집니다.
- 전역 좌표계는 라이브 VO 궤적을 따르므로 장시간 사용 시 drift가 누적됩니다.
  (DA3-small의 pose 헤드가 병진을 과소추정해 독립적 drift 보정원으로 쓸 수 없음 —
  루프 클로저는 향후 과제)
- 대형 가구는 카메라가 궤도를 돌 때 mask 중심이 이동해 dynamic으로 오판될 수 있음.
- 동적 물체의 시간별 궤적은 기록·표시되지만(`world/objects/dyn_traj`) 검증은 후순위.
- mesh는 현재 TSDF submap 기반의 표시/export 레이어입니다. object localization과
  relocalization의 기준 표현은 기존 point cloud이고, DA3 video 입력에서는 metric
  scale drift가 남을 수 있어 recorded OAK metric depth가 더 안정적입니다.
- YOLOE 오픈 보캐뷸러리는 검출이 풍부한 대신 중복/노이즈 노드가 다소 늘 수 있음
  (`detect.conf` 상향이나 어휘 축소로 조절).
- `--map` 재위치추정은 같은 물체가 3개 이상 다시 보여야 성공 — 이전 세션과
  완전히 다른 각도로만 촬영하면 정렬에 실패할 수 있음 (데이터는 보존됨).
- 10~12초 샘플 영상 3종(거실·침실·주방)으로 개발·검증됨 — 1~3분짜리 실내
  촬영 영상으로 장시간 영속성/재등장 검증 권장.
