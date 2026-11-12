# Apple Silicon (MPS) 로컬 실행 가이드

`app.py`는 이제 Hugging Face ZeroGPU Space와 **로컬 Mac(Apple Silicon, MPS)** 양쪽에서
같은 파일로 돌아갑니다. ZeroGPU가 아니면 자동으로 `spaces`/ZeroGPU 계층을 건너뛰고
프로세스 내에서 직접 실행하며, 백엔드는 PyTorch가 자동 감지하는 **MPS**를 사용합니다.

> ⚠️ MLX가 아니라 **MPS(Metal)**입니다. 이 프로젝트는 PyTorch + ComfyUI 기반이고
> MLX는 사용하지 않습니다.

---

## 사전 요구사항

- Apple Silicon Mac (M1/M2/M3/M4 계열)
- **통합 메모리 32GB 이상** — 36GB면 아래 저메모리 모드로 동작하도록 자동 설정됩니다
  (64GB 이상이면 훨씬 여유 있습니다)
- Python 3.11~3.13 (3.12 권장)
- 디스크 여유 공간 약 **35GB 이상** (모델 가중치 28 GiB + ComfyUI/노드팩)

## 1. 설치

```bash
cd synth-id-remover

# 가상환경 생성 (권장)
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치 (torch는 MPS 빌드가 자동 설치됨)
pip install -r requirements.txt
```

## 2. 실행 (첫 실행 = 다운로드)

```bash
python app.py
```

첫 실행 시 `install.py`가 다음을 수행합니다:

1. ComfyUI v0.7.0 + 커스텀 노드팩 6개를 git clone (수십 MB)
2. 모델 가중치 **28.1 GiB**를 Hugging Face 캐시로 다운로드 후 심링크
   (SAM은 Meta CDN에서 다운로드)

다운로드는 네트워크에 따라 수 분~수십 분 걸립니다. 완료되면
`http://127.0.0.1:7860` 에 Gradio UI가 열립니다.

## 3. 환경 변수

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `SYNTHID_LOW_MEMORY` | Mac에서 자동 `1` | `1`이면 Qwen/Z-Image 스택을 동시에 올리지 않고 단계마다 교체(36GB 대응) |

```bash
# 저메모리 모드 강제/해제 예시
SYNTHID_LOW_MEMORY=1 python app.py
SYNTHID_LOW_MEMORY=0 python app.py   # 64GB 이상에서 두 스택 상주
```

## 4. 예상 성능

- **첫 장**: 다운로드 + 모델 로드 때문에 수 분 이상.
- **이후 한 장**: MPS에서 20B Qwen 모델 4스텝 + Z-Image 얼굴 정리라
  **수 분** 수준. 해상도가 낮고 얼굴이 없으면 더 빠릅니다.

## 4-1. 예상 메모리 사용량

| 스택 | 가중치 |
| --- | ---: |
| Qwen-Image (unet+CLIP+VAE+ControlNet+LoRA) | 20.68 GiB |
| Z-Image (unet+CLIP+VAE) | 7.06 GiB |
| 디텍터 (YOLOv8n + SAM) | 0.36 GiB |
| **전체 가중치** | **28.10 GiB** |

활성화 ~4–8 GiB + PyTorch 오버헤드 ~2–3 GiB 추가.

- **저메모리 모드(기본)**: 피크 ≈ Qwen 20.68 + 활성화 ~5 + 오버헤드 ~2 ≈ **27–29 GiB**
  → 36GB 통합메모리에서 빡빡하지만 동작(작은 이미지 권장).
- **상주 모드(`SYNTHID_LOW_MEMORY=0`)**: 28.1 + 활성화 ~6 + 오버헤드 ~2 ≈ **35–39 GiB**
  → 64GB 이상에서 권장.

## 5. 주의 / 검증 필요

- **ComfyUI-GGUF의 MPS 지원은 CUDA만큼 성숙하지 않습니다.** 일부 연산이 CPU로
  폴백되거나 특정 연산에서 에러가 날 수 있습니다. 실행 시 로그에
  `[setup] ... backend=mps ...` 가 찍히는지 먼저 확인하세요.
- GGUF 경로가 MPS에서 막히면, 원본 워크플로(`Synthid-Bypass-v2.0.json`)를
  ComfyUI 데스크톱(MPS)에서 직접 돌리는 쪽이 안정적입니다.
- 36GB에서는 이미지 해상도를 낮춰서 넣는 게 안전합니다 (네이티브 해상도 처리라
  큰 입력일수록 메모리·시간이 급증).

## 6. 문제 해결

- **SAM 다운로드 SSL 에러**: `install.py`가 인증서 검증 실패 시 `-k`로 한 번
  재시도하도록 되어 있습니다. 그래도 실패하면 `dl.fbaipublicfiles.com`에서 수동
  다운로드 후 `ComfyUI/models/sams/sam_vit_b_01ec64.pth`에 두세요.
- **`ModuleNotFoundError: No module named 'spaces'`**: `pip install spaces` 후 재시도
  (코드는 `spaces`가 없어도 동작하지만 `requirements.txt`에 포함돼 있습니다).
- **OOM**: `SYNTHID_LOW_MEMORY=1` 확인 + 더 작은 입력 이미지 사용.
