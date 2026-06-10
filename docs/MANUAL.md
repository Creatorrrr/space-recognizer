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
| 디스크 | 모델 가중치 약 2GB (최초 실행 시 자동 다운로드) |
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
```

> **왜 3번이 따로 있나요?** 공식 `depth-anything-3` 패키지가 `numpy<2`와
> `xformers`(CUDA 전용)를 요구하지만 실제로는 둘 다 없어도 동작합니다.
> 필요한 하위 의존성은 이미 `pyproject.toml`에 들어 있습니다.

설치 확인:

```bash
.venv/bin/python -m pytest tests/ -q     # 13개 테스트가 통과해야 합니다
```

---

## 3. 빠른 시작

### 3-1. 샘플 영상으로 실행 (권장 첫 실행)

```bash
.venv/bin/python -m spacerec.main
```

- `config.yaml`의 `source`(기본: `sources/sample_720p.mp4`)를 입력으로 사용합니다.
- 첫 실행은 모델 다운로드(약 2GB) 때문에 수 분 걸릴 수 있습니다.
  콘솔에 `waiting for backend process...`가 길게 보여도 정상입니다.
- Rerun 뷰어 창이 자동으로 열립니다.

### 3-2. 웹캠으로 실행

```bash
.venv/bin/python -m spacerec.main --source 0
```

- `0`은 기본 카메라 인덱스입니다 (외장 캠은 `1`, `2`, ...).
- **최초 실행 시 macOS 카메라 권한이 필요합니다**:
  시스템 설정 → 개인정보 보호 및 보안 → 카메라 → 사용 중인 터미널 앱 허용.
  권한이 없으면 `cannot open video source: 0` 오류가 납니다.
- 천천히 걸으면서 방을 둘러보듯 촬영하면 지도가 점점 완성됩니다(촬영 요령은 6장).

### 3-3. 종료

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

예시:

```bash
# 내 영상의 앞 30초만 빠르게 확인
.venv/bin/python -m spacerec.main --source sources/my_room.mp4 --max-seconds 30

# 프레임을 하나도 빠뜨리지 않고 정밀 처리 (재생 속도 무시)
.venv/bin/python -m spacerec.main --source sources/my_room.mp4 --no-realtime
```

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
└──────────────┴──────────────────────────┘
                [타임라인]
```

### Live RGB (좌상단)
- 실시간 영상 위에 검출 박스와 `클래스#추적id` 라벨이 표시됩니다.

### Depth (좌하단)
- 보정된 실시간 depth 맵 (밝을수록 멂, Viridis 컬러맵).

### 3D World (우측) — 핵심 화면
| 요소 | 의미 |
|---|---|
| 컬러 점들 | 지금까지 누적된 **정적 공간의 3D 지도** (움직이는 물체는 제외됨) |
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
t=  9.1s processed=30 avg 3.4 FPS | pos=(-0.30,+0.07,+0.31) inliers=0.75 n=270
[backend] window=12kf 3.6s map=24471pts calib a=0.998 b=0.002 scale=1.000 1unit=4.03m
```

- `inliers`: 카메라 추적 품질 (0.5 이상이면 양호, `LOST` 표시는 추적 실패)
- `[backend] ...`: 5초 주기 재구성 결과 — 지도 포인트 수, depth 보정 계수,
  `1unit=...m`은 상대 단위→미터 환산 계수
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

detect:
  conf: 0.35                      # 검출 신뢰도 임계값. 오검출 많으면 ↑ (0.45)
  dynamic_classes: [person, ...]  # 항상 '움직임'으로 간주해 지도에서 제외할 클래스

vo:
  keyframe_interval_s: 0.5        # 키프레임 간격. 줄이면 정밀↑ 부하↑

backend:
  period_s: 5.0                   # 3D 재구성 주기 (초)
  window_size: 12                 # 재구성에 쓰는 키프레임 수
  voxel_size: 0.03                # 지도 해상도. 줄이면 촘촘↑ 무거움↑
  metric_anchor: true             # 미터 단위 추정. 끄면 FPS가 다소 오름

objects:
  merge_radius: 0.5               # 재등장 병합 반경. 같은 물체가 둘로 나뉘면 ↑
  dynamic_var_thresh: 0.3         # 움직임 판정 민감도. 오판 많으면 ↑

graph:
  near_dist: 1.2                  # 근접 엣지 거리 임계값. 엣지가 너무 많으면 ↓
```

### 성능 튜닝 가이드

증상별 처방 (효과 큰 순):

| 증상 | 처방 |
|---|---|
| FPS가 너무 낮다 | `backend.metric_anchor: false` → `proc_width: 960` → `backend.window_size: 8` |
| 지도가 듬성듬성하다 | `backend.voxel_size: 0.02`, `viz.point_subsample: 2` |
| 카메라 추적이 자주 끊긴다 | 더 천천히 촬영, `vo.keyframe_interval_s: 0.3` |
| 같은 물체가 여러 노드로 등록된다 | `objects.merge_radius: 0.8` |
| 정지한 물체에 `~`(동적) 표시가 붙는다 | `objects.dynamic_var_thresh: 0.5` |

---

## 8. 보조 도구

```bash
# 모델 추론 속도 벤치마크 (내 머신 성능 확인)
.venv/bin/python benchmarks/bench_models.py

# 뷰어 없이 전체 파이프라인 실행 → 지도를 /tmp/map.npz로 저장
.venv/bin/python benchmarks/headless_run.py --stride 3
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

**Q. 지도가 두 겹으로 어긋나 보입니다.**
장시간 사용으로 drift가 누적된 경우입니다. 현재 버전은 루프 클로저가 없으므로
긴 세션에서는 어쩔 수 없습니다. 재시작하면 새 지도로 시작합니다.

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
