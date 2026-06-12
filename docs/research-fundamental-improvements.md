# 근본 개선 방안 연구 — drift 문제군에 대한 구조적 대응 (2026-06-12)

> 배경: office-loop 실검증에서 드러난 3대 문제 — (P1) mono depth 바닥
> 편향이 VO 병진에 주입되는 침하/기울기 drift, (P2) 장시간 스케일 붕괴,
> (P3) drift 누적 후 루프 검증 실패. P1은 키프레임 바닥 anchoring으로
> 1차 해결(`benchmarks.md`), 본 문서는 웹 리서치(2026-06 검증) 기반의
> **다음 단계 구조적 개선안**이다. 출처·라이선스·Windows 호환성 검증 포함.

## 핵심 발견: 우리 문제는 알려진 문제다

1. **mono depth의 바닥 곡면/기울기 편향은 문헌으로 확인된 계통 결함**이다.
   - arXiv 2510.19814 (2025-10): 표준 depth 지표(AbsRel, δ1)가 "평면을
     굽히는" 곡률 오차에 둔감함을 정량 증명 — DA 계열이 지표상 우수해도
     바닥이 굽을 수 있는 구조적 이유.
   - Depth-Anything issue #186 / V2 #128: 역투영 시 바닥이 깔때기/사발형으로
     왜곡되는 현상 다수 보고 — 우리의 "가라앉음"과 동일 계열.
   - GenDepth (2312.06021): 원인을 학습 카메라의 높이/pitch prior가
     네트워크에 박제된 것으로 진단.
2. **"느리고 정확한 depth가 빠른 odometry를 교정"하는 패턴은 출판된
   정석**이다 — Metric3Dv2+DROID(2404.15506, metric prior가 스케일 drift
   제거 입증), GlORIE-SLAM(2403.19549, mono prior로 BA 정규화),
   AnchorD(2605.02667, 2026-05: factor graph 기반 patch-affine depth 정합,
   DA3-Mono로 실험, 코드 공개). 우리의 "백엔드 DA3-LARGE depth로 라이브를
   교정"하는 방향이 옳고, 더 체계화할 수 있다.

## 제안 (우선순위순)

### 제안 1 — GeoCalib로 중력을 '학습된 절대 측정'으로 (P1 완성)

현재 키프레임 바닥 anchoring은 RANSAC 평면에 의존 — **바닥이 안 보이는
구간(클로즈업, 책상 사이)에서는 무보정**이다. GeoCalib(ECCV 2024,
cvg/GeoCalib)은 단일 RGB에서 **중력 방향+intrinsics를 불확실도와 함께**
추정하는 학습 모델로:

- 코드 Apache-2.0 / 가중치 CC-BY-4.0, 순수 PyTorch, pip 설치 — Windows
  적합성 최상 (검증됨).
- 키프레임당 1회(~수십 ms 추정, 4080 실측 필요) 실행해 중력 벡터를
  pitch/roll의 **절대 측정**으로 사용 — 바닥 가시성과 무관하게 동작.
- 통합 위치: `floor_anchor` 경로에서 RANSAC 실패 시 폴백, 또는 두 측정의
  불확실도 가중 융합. intrinsics 추정은 현재 K 워밍업(DA3 추정 중앙값)의
  교차 검증으로도 활용 가능.
- 부수 효과: 시작 중력 정렬(`gravity_align`)도 바닥 미노출 시작에서
  동작하게 됨 (sample_720p에서 생략됐던 케이스).

### 제안 2 — 백엔드 depth로 키프레임 depth를 재정합 (P1+P2 통합, AnchorD 패턴)

현재 라이브 depth 교정은 전역 affine(a,b) 1쌍 — **공간적으로 균일한
보정만 가능**해 바닥 곡면(저주파 휨)을 못 편다. AnchorD(2026-05)의
patch-affine 정합을 차용:

- 키프레임 depth를 패치 격자로 나눠, 같은 시점의 **백엔드 DA3-LARGE 융합
  depth**(이미 보유)를 앵커로 패치별 (a,b)를 log-slope 일관성 제약과 함께
  풀면 상대 구조를 보존하며 저주파 휨 제거.
- 적용 시점: 백엔드 윈도 결과가 도착하면 해당 키프레임들의 depth 보정장을
  추정 → 이후 키프레임의 곡면 편향을 사전 보정(편향은 장면별로 느리게
  변함) → PnP에 들어가는 3D 자체가 평평해져 anchoring 부담이 줄어든다.
- 구현 난도 중상 (factor graph는 scipy least_squares로 대체 가능, 자체
  구현 ~수일). 코드 참고: https://anchord.cs.uni-freiburg.de

### 제안 3 — 루프 검증을 '정류된 기하'에서 수행 (P3)

루프 검증 실패의 원인은 기하 자체가 아니라 **검증에 쓰는 키프레임 3D가
drift로 오염**된 것. 이미 Sim3 RANSAC(스케일 변수 흡수)은 쓰고 있으므로:

- 키프레임 저장 시점의 depth 대신 **백엔드 재정합 depth(제안 2)** 를
  저장하면 스케일·휨이 정류된 3D로 매칭 → 합의율 상승.
- 중력 정류(제안 1)로 두 키프레임을 같은 자세 기준에 놓고 매칭하면
  ORB 시점 변화 부담도 감소.
- P1/P2가 잡히면 누적 drift 자체가 줄어 검증 성공률이 따라 오른다
  (office-loop에서 이미 부분 관찰).

### 제안 4 — 촬영 단계 side-channel: ARKit/ARCore pose 인제스트 (최강·저비용)

사용자가 촬영 앱만 바꾸면 중력·스케일 문제가 **원천 소멸**한다:

| 플랫폼 | 도구 | 제공 데이터 | 비고 |
|---|---|---|---|
| iOS (LiDAR 불필요) | **NeRFCapture** (무료, 오픈소스) | ARKit 중력 정렬·미터 단위 pose + RGB | 최저 마찰, 검증됨 |
| iOS (LiDAR) | Stray Scanner / Polycam dev mode | + depth + IMU CSV | polyform 파서 존재 |
| Android | ARCore Recording API | MP4 안에 IMU·depth 트랙 | 소형 커스텀 앱 필요 |

- 파이프라인 통합: `VideoSource` 옆에 pose 트랙 리더 추가 → VO를 "ARKit
  pose 사용 + 미보유 구간만 자체 VO" 하이브리드로. 중력·스케일이 GT로
  고정되므로 서보/anchoring은 일반 영상용 폴백으로 강등.
- 권장: **차기 촬영부터 NeRFCapture 사용을 표준 워크플로로** 하고,
  일반 영상은 제안 1~3 경로로 처리.

### 제안 5 — (대규모 대안) VO 전면 교체 후보 현황

당장 권장하지 않지만 (제안 1~4가 더 싸고 위험이 낮음) 임계점 도달 시:

| 후보 | 상태 (2026-06 검증) | 결격/보류 사유 |
|---|---|---|
| MASt3R-SLAM | **공식 windows 브랜치 존재** (유일), RTX4090 ~15FPS | 체크포인트 CC-BY-NC(비상업), 커스텀 CUDA 빌드 |
| AMB3R-VO | 2026-02 코드 공개, feed-forward metric VO | 라이선스 미표기, flash-attn Windows 빌드, VRAM 미확인 |
| VGGT-SLAM 2.0 | RSS 2026, BSD-2, drift 제거 설계 | 실시간 코드 미공개 — 관찰 대상 |
| DPVO/DPV-SLAM | MIT, 저VRAM | Linux 전용 문서(WSL2), 모노 스케일 모호 잔존 |
| FoundationSLAM | 2512.25008, 4090 18FPS | 코드 미공개 |

## 권장 실행 순서

1. **제안 1 (GeoCalib)** — 1~2일, 위험 낮음, 현 anchoring의 사각(바닥
   미노출)을 즉시 메움. 첫 단계로 4080 지연 실측 게이트.
2. **제안 4 (NeRFCapture 인제스트)** — 1~2일, 사용자 촬영 협조 시 가장
   확실. 제안 1과 독립적으로 병행 가능.
3. **제안 2 (patch-affine 재정합)** — 3~5일, P1·P2의 구조적 마무리.
4. **제안 3** — 제안 2의 부산물로 대부분 달성, 잔여는 미세 조정.
5. 제안 5는 분기별 재평가 (VGGT-SLAM 2.0 실시간 코드 공개 모니터링).

## 출처 (리서치 에이전트 검증, 2026-06-12)

- 곡률 편향: arXiv 2510.19814 · DA issue #186 · GenDepth 2312.06021
- 정합/교정: AnchorD 2605.02667 (https://anchord.cs.uni-freiburg.de) ·
  GlORIE-SLAM 2403.19549 · Metric3Dv2+DROID 2404.15506
- 중력 추정: GeoCalib https://github.com/cvg/GeoCalib (ECCV 2024) ·
  MoGe-2 https://github.com/microsoft/MoGe (normal 출력, MIT)
- SLAM 후보: MASt3R-SLAM https://github.com/rmurai0610/MASt3R-SLAM ·
  AMB3R https://github.com/HengyiWang/amb3r · VGGT-SLAM 2.0 arXiv 2601.19887
- 촬영 도구: NeRFCapture https://github.com/jc211/NeRFCapture ·
  Stray Scanner · Polycam polyform · ARCore Recording & Playback API
