# 3D Gaussian Splatting (3DGS) 학습 가이드 🇰🇷

> **원본 논문**: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"  
> **원본 저자**: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis (INRIA)  
> **원본 레포**: https://github.com/graphdeco-inria/gaussian-splatting  
> **논문 링크**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf

---

## 📌 개요

3D Gaussian Splatting(3DGS)은 2023년 발표된 실시간 Novel-View Synthesis(새로운 시점 합성) 방법입니다.  
기존 NeRF 계열 방법과 달리 **명시적(explicit) 표현 방식**을 사용하여 다음 특징을 가집니다:

| 특징 | 내용 |
|------|------|
| 표현 방식 | 수백만 개의 3D Gaussian 점 |
| 렌더링 속도 | **30fps 이상** (실시간, 1080p) |
| 학습 시간 | 약 **30분 ~ 1시간** (A6000 기준) |
| 입력 | COLMAP SfM 포인트 클라우드 + 멀티뷰 이미지 |
| 출력 | `.ply` 형식의 3D Gaussian 모델 |

---

## 🗂️ 프로젝트 구조

```
gaussian-splatting/
│
├── train.py                  # ★ 메인 학습 스크립트
├── render.py                 # 학습된 모델로 이미지 렌더링
├── metrics.py                # PSNR, SSIM, LPIPS 평가
├── convert.py                # 커스텀 이미지 → COLMAP 변환
├── full_eval.py              # 전체 평가 파이프라인
├── environment.yml           # Conda 환경 설정
│
├── arguments/
│   └── __init__.py           # ★ 모든 하이퍼파라미터 정의
│                             #   ModelParams, PipelineParams, OptimizationParams
│
├── gaussian_renderer/
│   ├── __init__.py           # ★ 렌더링 파이프라인 (Rasterization 호출)
│   └── network_gui.py        # 실시간 학습 시각화 GUI 서버
│
├── scene/
│   ├── __init__.py           # Scene 클래스 (데이터셋 로딩 통합)
│   ├── gaussian_model.py     # ★ GaussianModel 클래스 (핵심 모델)
│   ├── cameras.py            # 카메라 파라미터 처리
│   ├── colmap_loader.py      # COLMAP 데이터 파싱
│   └── dataset_readers.py    # 데이터셋 읽기 (COLMAP / NeRF Synthetic)
│
├── utils/
│   ├── loss_utils.py         # L1 Loss, SSIM Loss
│   ├── image_utils.py        # PSNR 계산
│   ├── camera_utils.py       # 카메라 유틸
│   ├── graphics_utils.py     # 회전행렬, FOV 등 3D 수학 유틸
│   ├── sh_utils.py           # 구면 조화 함수(Spherical Harmonics) 유틸
│   ├── general_utils.py      # 학습률 스케줄러, 로깅
│   └── system_utils.py       # 파일 시스템 유틸
│
├── lpipsPyTorch/             # LPIPS 손실 함수 모듈
│
└── submodules/
    ├── diff-gaussian-rasterization/  # ★ CUDA 래스터라이저 (핵심 C++/CUDA 코드)
    ├── fused-ssim/                   # 빠른 SSIM 계산 (학습 가속화용)
    └── simple-knn/                   # KNN 탐색 (포인트 초기화용)
```

---

## 🔑 핵심 개념 설명

### 1. 3D Gaussian이란?

각 Gaussian은 다음 속성을 가집니다:

```
μ  (평균, Position)      : 3D 공간 좌표 (x, y, z)
Σ  (공분산, Covariance)  : 크기 + 방향 (타원체 모양)
α  (불투명도, Opacity)   : 0~1 투명도
c  (색상, Color/SH)      : 구면 조화 함수 계수 (시점 의존적 색상)
```

실제 코드에서는 `Σ`를 안정적인 최적화를 위해 분해합니다:
- `s` (scale): 타원체의 3축 스케일 → `log` 공간에서 최적화
- `r` (rotation): 쿼터니언(quaternion) 표현

### 2. Adaptive Density Control

학습 중 Gaussian의 수가 자동으로 조절됩니다:

```
Densification (밀도 증가):
  - Under-reconstruction: 큰 Gaussian → 2개로 분리(Split)
  - Over-reconstruction:  작은 Gaussian → 복제(Clone)
  조건: 2D 위치 기울기(gradient) > threshold (기본: 0.0002)

Pruning (정리):
  - Opacity가 너무 낮은 Gaussian 제거
  - 주기: 매 3000 iteration마다 opacity를 리셋
```

### 3. 렌더링 파이프라인

```
3D Gaussian → [View Frustum Culling] → [Tile-based Sorting] 
→ [Alpha Compositing (Front-to-Back)] → 2D 이미지
```

CUDA로 구현된 Tile-based 래스터라이저를 사용하여 실시간 속도 달성

---

## 🚀 환경 설정 (Setup)

### 필수 요구사항
- CUDA 지원 GPU (Compute Capability 7.0+, 권장 VRAM 24GB)
- CUDA SDK 11.x
- Conda

### 설치

```bash
# 1. Conda 환경 생성
conda env create -f environment.yml
conda activate gaussian_splatting

# 2. CUDA 서브모듈 빌드
# (환경 활성화 후 자동으로 pip install이 실행됨)
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# 3. (선택) 학습 가속화용 rasterizer 설치
pip install submodules/fused-ssim
```

---

## 🏃 학습 실행

### 기본 학습

```bash
python train.py -s <데이터셋 경로>
```

### 주요 하이퍼파라미터

```bash
python train.py \
  -s ./data/garden \          # 데이터셋 경로 (COLMAP 형식)
  -m ./output/garden \        # 모델 저장 경로
  --iterations 30000 \        # 총 학습 반복 횟수 (기본: 30,000)
  --sh_degree 3 \             # 구면 조화 차수 (0~3, 높을수록 정교한 색상)
  --eval \                    # train/test 분리 평가 모드
  --densify_from_iter 500 \   # Densification 시작 iteration
  --densify_until_iter 15000 \ # Densification 종료 iteration
  --lambda_dssim 0.2          # SSIM 손실 가중치
```

### 빠른 학습 (가속화 버전)

```bash
python train.py -s ./data/garden --optimizer_type sparse_adam
```

---

## 📊 평가

```bash
# 렌더링 생성
python render.py -m ./output/garden

# 메트릭 계산 (PSNR, SSIM, LPIPS)
python metrics.py -m ./output/garden
```

---

## 📁 데이터셋 구조

### COLMAP 형식 (실제 촬영 데이터)

```
<dataset>/
├── images/           # 입력 이미지들
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin    # 카메라 내부 파라미터
        ├── images.bin     # 카메라 외부 파라미터 (포즈)
        └── points3D.bin   # SfM 초기 포인트 클라우드
```

### 나만의 데이터 만들기

```bash
# 직접 찍은 사진/동영상으로 COLMAP 데이터 생성
# 먼저 images/ 폴더에 사진들을 넣고:
python convert.py -s <위치> [--resize]
```

---

## 📈 손실 함수

```
L_total = (1 - λ) * L1 + λ * L_SSIM

λ = lambda_dssim = 0.2 (기본값)

# 깊이 정규화 사용 시 (선택):
L_total += w(t) * L1_depth
```

---

## 💡 학습 팁

| 상황 | 해결책 |
|------|--------|
| VRAM 부족 | `--data_device cpu` 또는 `-r 2` (해상도 절반) |
| 넓은 야외 씬 | `position_lr`과 `scaling_lr`을 0.1~0.3배로 줄이기 |
| Floater(떠있는 점) 발생 | `--depth -d <depth_map_path>`로 깊이 정규화 |
| 노출 변화가 심한 데이터 | `--exposure_lr_init` 파라미터 추가 |
| 품질 향상 | `--antialiasing` 플래그 추가 |

---

## 🔬 핵심 코드 읽기 순서 (학습용)

1. **`arguments/__init__.py`** → 모든 하이퍼파라미터 이해
2. **`scene/gaussian_model.py`** → GaussianModel 클래스 (핵심!)
   - `__init__()`: Gaussian 속성 초기화
   - `training_setup()`: 옵티마이저 설정
   - `densify_and_clone()`, `densify_and_split()`: Adaptive Density Control
3. **`gaussian_renderer/__init__.py`** → 렌더링 파이프라인
4. **`train.py`** → 전체 학습 루프
5. **`scene/__init__.py`** → 데이터셋 로딩
6. **`utils/loss_utils.py`** → 손실 함수

---

## 📚 참고 자료

| 자료 | 링크 |
|------|------|
| 원본 논문 (PDF) | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf |
| 프로젝트 페이지 | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ |
| 논문 발표 영상 | https://youtu.be/T_kXY43VZnk |
| 공식 GitHub | https://github.com/graphdeco-inria/gaussian-splatting |
| Colab 튜토리얼 | https://github.com/camenduru/gaussian-splatting-colab |
| 사전 학습 모델 | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip |
| T&T+DB 데이터셋 | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip |

---

## 📝 BibTeX

```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

---

*이 레포지토리는 3DGS를 학습하기 위한 목적으로 공식 코드를 가져온 것입니다.  
원본 코드의 라이선스는 `LICENSE.md`를 참고하세요 (비상업적 연구/평가 목적으로만 사용 가능).*
