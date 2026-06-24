# Space Recognizer 사용 매뉴얼

웹캠(또는 영상 파일)으로 주변 공간을 촬영하면서 물체를 인식하고, 3D 공간 지도 위에
물체들의 위치와 관계를 실시간으로 그려주는 프로그램입니다.

---

## 1. 시스템 요구사항

| 항목 | 요구사항 |
|---|---|
| 하드웨어 | Apple Silicon Mac (M1 이상 권장, 개발·검증 환경은 M1 Max 32GB) |
| OS | macOS |
| Python | 3.12 (uv로 가상환경 생성) |
| GPU | 불필요 — Apple MPS만 사용 (CUDA 불필요) |
| 디스크 | 모델 가중치 약 2.2GB (최초 실행 시 자동 다운로드) |
| 네트워크 | 최초 실행 시에만 필요 (Hugging Face 가중치 다운로드) |

---

## 2. 설치 (최초 1회)

프로젝트 루트(`space-recognizer/`)에서:

```bash
# 1) 가상환경 생성
uv venv --python 3.12

# 2) 의존성 설치
uv pip install -p .venv -e ".[dev]"

# 3) DA3(Depth Anything 3)는 의존성 핀 충돌 때문에 --no-deps로 별도 설치
uv pip install -p .venv --no-deps depth-anything-3

# 4) OAK-D-Lite를 쓸 때만 DepthAI extra 설치
uv pip install -p .venv -e ".[oak]"
```

> **왜 3번이 따로 있나요?** 공식 `depth-anything-3` 패키지가 `numpy<2`와
> `xformers`(CUDA 전용)를 요구하지만 실제로는 둘 다 없어도 동작합니다.
> 필요한 하위 의존성은 이미 `pyproject.toml`에 들어 있습니다.

설치 확인:

```bash
.venv/bin/python -m pytest tests/ -q     # 전체 테스트가 통과해야 합니다
```

---

## 3. 빠른 시작

### 3-1. 샘플 영상으로 실행 (권장 첫 실행)

```bash
.venv/bin/python -m spacerec.main
```

- `config.yaml`의 `source`(기본: `sources/sample_720p.mp4`)를 입력으로 사용합니다.
- 첫 실행은 모델 다운로드(약 2.2GB) 때문에 수 분 걸릴 수 있습니다.
  콘솔에 `waiting for backend process...`가 길게 보여도 정상입니다.
- Rerun 뷰어 창이 자동으로 열립니다.

### 3-2. 웹캠으로 실행

```bash
.venv/bin/python -m spacerec.main --source 0
```

- `0`은 기본 카메라 인덱스입니다 (외장 캠은 `1`, `2`, ...).
- **아이폰을 웹캠으로** (Continuity Camera): 아이폰이 같은 Apple ID로 근처에
  있으면 자동으로 카메라 장치로 잡힙니다. 인덱스 확인:
  `ffmpeg -f avfoundation -list_devices true -i ""` → 보통 `0`=내장 FaceTime,
  `1`=아이폰, `2`=아이폰 데스크뷰. `--source 1`로 실행하면 됩니다.
  아이폰이 잠자고 있으면 첫 프레임까지 1~2초 걸립니다(자동 재시도).
- **최초 실행 시 macOS 카메라 권한이 필요합니다**:
  시스템 설정 → 개인정보 보호 및 보안 → 카메라 → 사용 중인 터미널 앱 허용.
  권한이 없으면 `cannot open video source: 0` 오류가 납니다.
- 천천히 걸으면서 방을 둘러보듯 촬영하면 지도가 점점 완성됩니다(촬영 요령은 6장).

### 3-3. OAK-D-Lite로 실행

```bash
.venv/bin/python benchmarks/oak_smoke.py
.venv/bin/python -m spacerec.main --source oak
```

- `oak_smoke.py`로 장치 MXID, USB 속도, RGB intrinsics, depth stream 유효
  비율을 먼저 확인하세요.
- 콘솔 metadata의 USB 속도가 `HIGH`면 USB 2.0 연결입니다. OAK-D-Lite는 USB 3
  연결이 가능하므로 케이블/허브/포트를 먼저 바꾸는 것이 FPS와 depth 안정성에
  가장 큽니다.
- OAK 모드는 RGB에 정합된 미터 단위 stereo depth를 우선 사용하고, stereo가
  비는 픽셀만 DA3를 OAK depth에 affine 정합해 채웁니다.
- IMU가 노출되는 OAK-D-Lite에서는 accel/gyro stream도 읽습니다.
  `imu.enabled: true`로 켜면 gyro 적분 회전을 LK 초기 flow와 PnP 초기값으로
  넘기고, 고속 회전 구간의 backend keyframe 승격을 지연시킵니다.
  현재 제공된 녹화 세션 A/B에서는 어려운 세션의 `lost`가 줄었지만 검증 폭이
  아직 좁아 기본값은 off입니다.
- `config.yaml`의 `capture.*`에서 OAK RGB 크기, FPS, LR-check, subpixel,
  depth min/max, IMU stream 활성화 여부를 조정할 수 있습니다.

### 3-4. OAK 녹화 세션 재생

OAK-D-Lite 입력을 디렉터리로 녹화해 둔 세션도 입력으로 사용할 수 있습니다.
세션 디렉터리는 `metadata.json`, `events.jsonl`, `streams/<stream>/*.npy`를
포함해야 합니다.

```bash
.venv/bin/python benchmarks/replay_smoke.py sources/session_20260624_054320_194430108151D05A00 --frames 60
.venv/bin/python benchmarks/replay_smoke.py sources/session_20260624_054320_194430108151D05A00 --frames 60 --full-models
.venv/bin/python benchmarks/replay_smoke.py sources/session_20260624_054320_194430108151D05A00 sources/session_20260624_055321_194430108151D05A00 --frames 120 --compare-imu
.venv/bin/python benchmarks/mesh_smoke.py sources/session_20260624_054320_194430108151D05A00 --frames 120 --out-dir artifacts/mesh
.venv/bin/python -m spacerec.main --source sources/session_20260624_054320_194430108151D05A00 --max-seconds 3 --no-realtime
.venv/bin/python -m spacerec.main --source sources/session_20260624_054320_194430108151D05A00 --no-realtime --mesh-out artifacts/mesh/session.ply
```

- 기본 `capture.replay_depth_mode: calibrated`는 녹화된 stereo depth를
  RGB 카메라 intrinsics/extrinsics로 재투영한 뒤 낮은 해상도 때문에 생기는
  구멍을 작은 z-buffer splat으로 보강합니다. 이후 기존 OAK 경로와 동일하게
  DA3 fallback을 metric fit해 빈 픽셀을 채웁니다.
- `capture.replay_depth_mode: resize`는 depth를 RGB 해상도로 단순 nearest
  resize합니다. 빠른 포맷 smoke에는 유용하지만 RGB/depth 좌표계가 다를 수
  있어 정밀 3D 검증용 기본값은 아닙니다.
- 녹화 replay는 파일 입력이지만 metric depth와 RGB intrinsics를 가진
  source로 취급됩니다. 따라서 DA3 mono-only 영상 경로가 아니라 OAK metric
  depth 경로를 탑니다.
- `--compare-imu`는 각 세션을 visual-only와 IMU-assisted로 두 번 돌려
  `lost`, `avg_tracked`, `avg_inlier`, `imu_prior_frames`,
  `imu_blur_skipped_kf`를 함께 출력합니다. 이 결과가 개선을 보이지 않으면
  `imu.enabled`는 끈 상태로 두는 것이 안전합니다.

### 3-5. TSDF mesh 생성/export

기본 지도(`world/points`)는 여전히 evidence-weighted point cloud입니다. Mesh는
이 지도와 병존하는 표시/export 레이어이며, backend window를 TSDF submap으로
통합한 뒤 triangle mesh를 추출합니다.

- recorded OAK replay처럼 RGB-aligned metric depth가 있는 입력이 가장 안정적입니다.
- mesh는 source of truth가 아니라 RGB-D keyframe evidence에서 재생성 가능한
  캐시입니다. pose/Sim3 보정은 submap anchor transform 또는 affected submap
  rebuild로 처리합니다.
- `benchmarks/mesh_smoke.py`는 모델을 로드하지 않고 recorded depth와 VO pose만으로
  `.ply`를 생성하고 다시 읽어 vertices/faces를 검증합니다.
- `--mesh-out <path.ply>`를 주면 main pipeline 종료 시 현재 mesh submap들을 global
  좌표로 합친 `.ply`를 저장합니다.
- `--map maps/room.npz`로 world state를 저장하면 mesh sidecar
  `maps/room.mesh.npz`도 함께 저장됩니다. 이후 relocalization에 성공하면 이전
  mesh submap anchor에 같은 Sim3 보정을 적용해 현재 meshmap에 병합합니다.

### 3-6. 종료

터미널에서 `Ctrl+C`. 영상 파일 모드는 영상이 끝나면 백엔드 결과를 마저 반영한 뒤
(`video ended; draining backend...`) 자동 종료됩니다. Rerun 뷰어 창은 따로 닫으면
됩니다 — 뷰어를 닫아도 데이터는 다시 열기 전까지 메모리에 유지됩니다.

---

## 4. 명령행 옵션

```bash
.venv/bin/python -m spacerec.main [옵션]
```

| 옵션 | 설명 |
|---|---|
| `--source <경로\|숫자>` | 입력 소스. 영상 파일 경로 또는 웹캠 인덱스. config보다 우선 |
| `--config <경로>` | 설정 파일 (기본 `config.yaml`) |
| `--max-seconds <초>` | 앞부분 N초만 처리하고 종료 (빠른 동작 확인용) |
| `--no-realtime` | 벽시계 페이싱 없이 모든 프레임을 처리 (오프라인 분석·검증용) |
| `--profile` | 10프레임마다 단계별 처리 시간(ms) 출력 (성능 진단용) |
| `--map <경로>` | **세션 간 누적**: 종료 시 지도+객체를 .npz로 저장하고, 다음 실행에서 불러와 이어서 누적. 시작 시 이전 세션 객체들과 외형 매칭으로 좌표계를 자동 정렬(재위치추정) |

예시:

```bash
# 내 영상의 앞 30초만 빠르게 확인
.venv/bin/python -m spacerec.main --source sources/my_room.mp4 --max-seconds 30

# 프레임을 하나도 빠뜨리지 않고 정밀 처리 (재생 속도 무시)
.venv/bin/python -m spacerec.main --source sources/my_room.mp4 --no-realtime

# 같은 공간을 여러 번 돌며 지도를 점점 완성 (세션 간 누적)
.venv/bin/python -m spacerec.main --source 0 --map maps/my_room.npz
```

> **`--map` 동작 방식**: 같은 공간이면 이전 세션에서 본 물체(침대, 의자 등)를
> 외형으로 알아보고 좌표계를 자동 정렬한 뒤 이전 지도를 병합합니다
> (`[reloc] 이전 지도 정렬 성공 ...` 로그). 같은 물체가 3개 이상 다시
> 보여야 정렬이 가능합니다. 다른 공간이라 정렬에 실패하면 이전 파일을
> 보존하고 이번 세션은 `*.unmerged.npz`로 따로 저장합니다.

> **realtime 모드란?** 영상 파일도 웹캠처럼 동작하도록, 처리가 느리면 그만큼
> 프레임을 건너뜁니다. 실사용(웹캠) 조건과 동일한 동작을 재현하기 위한 기본값입니다.

---

## 5. Rerun 뷰어 보는 법

실행하면 화면이 세 영역으로 나뉩니다.

```
┌──────────────┬──────────────────────────┐
│  Live RGB    │                          │
│ (검출 박스)   │       3D World           │
├──────────────┤  (지도·궤적·그래프)        │
│  Depth       │                          │
├──────────────┤                          │
│ Depth Calib. │                          │
└──────────────┴──────────────────────────┘
                [타임라인]
```

### Live RGB (좌상단)
- 실시간 영상 위에 검출 박스와 `클래스#추적id` 라벨이 표시됩니다.

### Depth (좌중단)
- 보정된 실시간 depth 맵 (밝을수록 멂, Viridis 컬러맵).

### Depth Calibration (좌하단)
- depth 보정 계수의 시계열: `a`/`b`(5초 주기 멀티뷰 기준 affine 보정),
  `frame_scale`(프레임 단위 스케일 보정). 값이 시간에 따라 움직이면
  캘리브레이션이 정상 동작 중인 것입니다 (`a`는 보통 1.0 부근).

### 3D World (우측) — 핵심 화면
| 요소 | 의미 |
|---|---|
| 컬러 점들 | 지금까지 누적된 **정적 공간의 3D 지도** (움직이는 물체는 제외됨). 증거 기반으로 갱신됨 — 잘못 만들어진 표면은 그 자리를 다시 비추면 지워지고, 단발성 오류로는 좋은 지도가 망가지지 않음 |
| 옅은 점들 (`world/live_preview`) | 최신 키프레임의 실시간 depth 미리보기 (지도에 융합되기 전 모습) |
| 파란 선 | 카메라가 지나온 궤적 |
| 피라미드(frustum) | 현재 카메라 위치와 시선 방향 |
| **구 + 라벨** | 인식된 오브젝트 노드 (`chair#5` 등) |
| 불투명 구 | 지금 화면에 보이는 오브젝트 |
| **반투명 구** | 화면 밖/가려진 오브젝트의 **기억된 위치** |
| 라벨 앞 `~` | 움직이는 물체로 판정된 오브젝트 |
| 회색 선 | 근접 관계 엣지 (가까운 물체끼리 연결) |
| 주황 선 | 위/아래 관계 엣지 (예: 컵이 테이블 위) |

- 마우스 드래그로 회전, 스크롤로 줌, 우클릭 드래그로 이동.
- 엣지를 클릭하면 거리 라벨이 보입니다. `metric_anchor`가 켜져 있으면
  `0.85m`처럼 **미터 단위**, 꺼져 있으면 상대 단위입니다.
- 좌측 엔티티 트리에서 `world/points`(지도), `world/objects`(그래프) 등을
  체크박스로 켜고 끌 수 있습니다.

### 타임라인 (하단)
- 슬라이더를 드래그하면 과거 시점의 화면·지도·오브젝트 상태로 되감아 볼 수 있습니다.
- 동적 물체의 이동 궤적(`world/objects/dyn_traj`)도 시간에 따라 재생됩니다.

### 콘솔 출력 읽는 법

```
t=  9.1s processed=30 avg 3.4 FPS | pos=(-0.30,+0.07,+0.31) inliers=0.75 n=270 fscale=0.987
[backend] window=12kf 3.6s map=24471pts calib a=0.998 b=0.002 scale=1.000 1unit=4.03m
[obj] new rug#4 size=0.41
[reid] bed#0 재획득 (공백 6.0s, cost=0.44)
```

- `inliers`: 카메라 추적 품질 (0.5 이상이면 양호, `LOST` 표시는 추적 실패),
  `fscale`: 프레임 단위 depth 스케일 보정값 (1.0 부근이 정상)
- `[backend] ...`: 5초 주기 재구성 결과 — 지도 포인트 수, depth 보정 계수,
  `1unit=...m`은 상대 단위→미터 환산 계수
- `[obj] new ...`: 새 물체 등록 / `[reid] ... 재획득`: 화면 밖에 있다 돌아온
  물체를 같은 노드로 복원 / `[obj] ... 제거 — 부재 증거`: 보여야 하는 자리에
  계속 없는 물체(치워졌거나 오인식) 정리 / `[reloc] ...`: `--map` 이전 세션
  지도 정렬
- 종료 시 `world objects (N):` 목록으로 기억된 모든 물체와 위치가 출력됩니다.

---

## 6. 좋은 결과를 얻는 촬영 요령

1. **천천히 움직이세요.** 빠른 회전·이동은 추적 실패(`LOST`)의 주원인입니다.
   1초에 30° 이내 회전, 보통 걸음의 절반 속도가 적당합니다.
2. **회전만 하지 말고 옆으로도 이동하세요.** 3D 재구성은 시차(parallax)에서
   나옵니다. 제자리 회전만 하면 깊이를 알 수 없습니다.
3. **물체를 여러 각도에서 보여주세요.** 같은 물체를 2~3개 시점에서 비추면
   위치 추정이 안정됩니다.
4. **조명이 충분하고 질감 있는 환경**이 유리합니다. 새하얀 벽만 보이면
   특징점이 없어 추적이 끊길 수 있습니다.
5. 시작 후 첫 5~10초는 지도의 기준 좌표계를 만드는 구간입니다. 이때는 특히
   천천히 움직여 주세요.

---

## 7. 설정 파일 (config.yaml)

자주 바꿀 만한 항목만 정리합니다. 수정 후 재실행하면 적용됩니다.

```yaml
source: sources/sample_720p.mp4   # 입력: 파일 경로 또는 웹캠 인덱스(정수)
realtime: true                    # false면 모든 프레임 처리 (오프라인)
proc_width: 1280                  # 처리 해상도. 낮추면 빨라짐 (예: 960)

capture:
  source_kind: video              # video 또는 oak
  oak_fps: 15.0                   # USB2 연결에서도 안정적인 기본값
  oak_align_depth_to_rgb: true    # 객체 mask와 depth 좌표계를 맞춤
  oak_lr_check: true
  oak_subpixel: true
  oak_depth_min_m: 0.3
  oak_depth_max_m: 8.0
  oak_enable_imu: true
  oak_imu_rate_hz: 100
  replay_depth_mode: calibrated   # recorded OAK: calibrated 또는 resize
  replay_pair_tolerance_ms: 20.0  # RGB-depth nearest pairing 허용 오차

detect:
  conf: 0.35                      # 검출 신뢰도 임계값. 오검출 많으면 ↑ (0.45)
  dynamic_classes: [person, ...]  # 항상 '움직임'으로 간주해 지도에서 제외할 클래스
  vocabulary: [bed, rug, ...]     # 인식할 물체 어휘 (자유 텍스트, YOLOE 모드).
                                  # 원하는 물체 이름을 영어로 추가/삭제하면 됨

vo:
  keyframe_interval_s: 0.5        # 키프레임 간격. 줄이면 정밀↑ 부하↑

imu:
  enabled: false                  # true면 gyro-derived LK/PnP rotation prior 사용
  use_lk_prior: true
  use_pnp_prior: true
  min_rotation_samples: 2
  max_rotation_deg: 35.0          # 이상치 rotation prior는 버리고 visual-only fallback
  keyframe_blur_omega_rad_s: 2.5  # 고속 회전 frame은 backend keyframe 승격 지연
  keyframe_max_delay_s: 1.0       # 지연이 길어지면 starvation 방지로 승격 허용

backend:
  period_s: 5.0                   # 3D 재구성 주기 (초)
  window_size: 12                 # 재구성에 쓰는 키프레임 수
  voxel_size: 0.03                # 지도 해상도. 줄이면 촘촘↑ 무거움↑
  metric_anchor: true             # 미터 단위 추정. 끄면 FPS가 다소 오름

mesh:
  enabled: true                   # TSDF mesh submap 생성/표시/export
  voxel_size: 0.05                # mesh 해상도. 줄이면 정밀↑ 무거움↑
  trunc_margin: 0.15              # 보통 voxel_size의 3배
  depth_trunc_m: 8.0              # TSDF에 넣을 최대 depth
  min_surface_observations: 2      # 단발 bad depth로 생긴 표면 제거
  max_active_submaps: 32          # live mesh submap cap
  export_on_exit: false           # true면 artifacts/mesh/latest.ply 저장

objects:
  merge_radius: 0.5               # 재등장 병합 반경의 상한 (기본은 물체 크기 비례)
  dynamic_var_thresh: 0.3         # 움직임 판정 민감도. 오판 많으면 ↑
  appearance: true                # DINOv2 외형 임베딩 re-ID (같은 클래스 이웃 구분)
  app_gate: 0.4                   # 외형 유사도 게이트. 낮추면 병합 관대해짐
  absence_limit: 12               # '보여야 하는데 안 보임' 누적 시 노드 제거

graph:
  near_dist: 1.2                  # 근접 엣지 거리 임계값. 엣지가 너무 많으면 ↓
```

### 성능 튜닝 가이드

증상별 처방 (효과 큰 순):

| 증상 | 처방 |
|---|---|
| FPS가 너무 낮다 | `backend.metric_anchor: false` → `proc_width: 960` → `backend.window_size: 8` |
| 지도가 듬성듬성하다 | `backend.voxel_size: 0.02`, `viz.point_subsample: 2` |
| mesh가 너무 무겁다 | `mesh.voxel_size: 0.07`, `mesh.max_active_submaps: 16`, smoke `--frames` 축소 |
| mesh가 거칠다 | recorded OAK replay 사용, `mesh.voxel_size: 0.03`, 더 느린 카메라 이동 |
| 카메라 추적이 자주 끊긴다 | 더 천천히 촬영, `vo.keyframe_interval_s: 0.3` |
| 빠른 회전 때 backend mesh/point cloud가 흐릿하다 | OAK/replay에서 `imu.enabled: true`로 A/B 측정 후 개선될 때만 사용 |
| 같은 물체가 여러 노드로 등록된다 | `objects.merge_radius: 0.8`, `objects.app_gate: 0.3` |
| 정지한 물체에 `~`(동적) 표시가 붙는다 | `objects.dynamic_var_thresh: 0.5` |
| 오검출/중복 노드가 많다 (YOLOE) | `detect.conf: 0.45`, `vocabulary`에서 불필요 어휘 제거 |
| 치워진 물체 노드가 너무 오래 남는다 | `objects.absence_limit: 6` |
| 멀쩡한 물체 노드가 사라진다 | `objects.absence_limit: 24` |

---

## 8. 보조 도구

```bash
# 모델 추론 속도 벤치마크 (내 머신 성능 확인)
.venv/bin/python benchmarks/bench_models.py

# 뷰어 없이 전체 파이프라인 실행 → 지도를 /tmp/map.npz로 저장
.venv/bin/python benchmarks/headless_run.py --stride 3

# OAK-D-Lite 연결/캘리브레이션/depth stream 점검
.venv/bin/python benchmarks/oak_smoke.py

# recorded OAK replay에서 IMU off/on VO 비교
.venv/bin/python benchmarks/replay_smoke.py sources/session_20260624_054320_194430108151D05A00 sources/session_20260624_055321_194430108151D05A00 --frames 120 --compare-imu

# recorded OAK replay에서 TSDF mesh export smoke
.venv/bin/python benchmarks/mesh_smoke.py sources/session_20260624_054320_194430108151D05A00 --frames 120
```

`headless_run.py` 결과는 `numpy.load("/tmp/map.npz")`로 열어 `pts`(지도 포인트),
`cols`(색), `traj`(카메라 궤적)를 후처리·분석에 쓸 수 있습니다.

---

## 9. 문제 해결 (FAQ)

**Q. `cannot open video source: 0` — 웹캠이 안 열립니다.**
터미널 앱에 카메라 권한이 없습니다. 시스템 설정 → 개인정보 보호 및 보안 →
카메라에서 사용 중인 터미널(iTerm, Terminal 등)을 허용하고 터미널을 재시작하세요.

**Q. 첫 실행이 몇 분째 멈춰 있습니다.**
모델 가중치(약 2GB)를 다운로드 중입니다. 한 번 받으면 캐시되어 다음 실행부터
빠릅니다. 15분 이상 멈춰 있으면 네트워크를 확인하세요.

**Q. Rerun 뷰어 창이 안 열립니다.**
`.venv/bin/rerun`이 있는지 확인하세요 (`uv pip install -p .venv rerun-sdk`).
이미 떠 있는 뷰어가 있으면 새 창 대신 기존 창에 연결됩니다.

**Q. `OMP: Error #15 ... libomp.dylib` 크래시가 납니다.**
환경변수 `KMP_DUPLICATE_LIB_OK=TRUE`가 필요합니다. `spacerec` 패키지를 import하면
자동 설정되지만, 별도 스크립트에서 모듈을 직접 쓸 때는 `import spacerec`를
가장 먼저 하세요.

**Q. 콘솔에 `LOST`가 계속 찍힙니다.**
카메라 추적 실패입니다. 너무 빠른 움직임, 모션 블러, 특징 없는 벽면이 원인입니다.
천천히, 질감 있는 장면을 비추면 자동으로 복구됩니다.

**Q. IMU를 켜면 위치 추정도 좋아지나요?**
현재 IMU는 gyro로 회전 prior만 제공합니다. accelerometer translation 적분은 bias와
노이즈가 이중 적분되어 몇 초 안에 크게 드리프트하므로 쓰지 않습니다. 녹화 세션
A/B에서 개선이 확인될 때만 `imu.enabled: true`를 사용하세요.

**Q. 지도가 두 겹으로 어긋나 보입니다.**
장시간 사용으로 drift가 누적된 경우입니다. 어긋난 옛 표면은 그 자리를 다시
천천히 비추면 빈 공간 증거(carving)로 점차 지워지지만, drift 자체를 되돌리는
루프 클로저는 아직 없습니다. 재시작(또는 `--map`으로 재시작 후 재위치추정)이
가장 빠른 해결책입니다.

**Q. 잘못 인식됐던 물체/표면이 안 사라집니다.**
그 자리를 카메라로 몇 초간 똑바로 비추세요. 표면은 시선 관통 증거로 지워지고
(5초 주기 백엔드가 돌 때마다), 물체 노드는 "보여야 하는데 안 보임"이
`absence_limit`(기본 12회)만큼 누적되면 제거됩니다. 가려져 있거나 화면
가장자리에 있으면 부재로 세지 않으므로 정면으로 비춰야 합니다.

**Q. 물체 위치가 미터로 안 나오고 이상한 단위입니다.**
`backend.metric_anchor: true`인지 확인하세요. 켜져 있어도 첫 백엔드 결과
(시작 후 ~10초)가 나오기 전에는 상대 단위로 표시됩니다.

**Q. 같은 영상인데 실행할 때마다 처리 프레임 수가 다릅니다.**
`realtime: true`(기본)는 실제 시간 기준으로 프레임을 드롭하므로 머신 부하에 따라
달라집니다. 재현 가능한 결과가 필요하면 `--no-realtime`을 쓰세요.

---

## 10. 더 읽을 것

- `README.md` — 프로젝트 개요와 아키텍처 요약
- `docs/plan.md` — 설계 문서 (3계층 구조, 좌표계 설계)
- `docs/benchmarks.md` — 이 머신에서의 실측 성능과 Mac/MPS 환경의 함정 기록
