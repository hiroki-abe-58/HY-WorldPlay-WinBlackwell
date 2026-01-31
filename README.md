# HY-WorldPlay-WinBlackwell

[![Windows](https://img.shields.io/badge/Windows-10%2F11-blue)](https://www.microsoft.com/windows)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.12-yellow)](https://www.python.org/)
[![Blackwell](https://img.shields.io/badge/GPU-Blackwell%20(sm__120)-red)](https://www.nvidia.com/)

A Windows Native + NVIDIA Blackwell GPU (RTX 6000 Blackwell) compatible fork of [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay).

## Overview

This fork adapts the official HY-WorldPlay (HY-World 1.5) for:
- **Windows 10/11** native execution (no WSL required)
- **NVIDIA Blackwell architecture** GPUs (sm_120, e.g., RTX 6000 Blackwell)
- **Single-GPU inference** optimization
- **Gradio WebUI** for easy interaction

## Key Changes from Official Version

| Feature | Official | This Fork |
|---------|----------|-----------|
| OS | Linux | Windows 10/11 |
| GPU Architecture | sm_90 (Hopper) | sm_120 (Blackwell) |
| flash-attn | Required | Optional (auto-fallback to PyTorch SDPA) |
| Distributed | torchrun (multi-GPU) | Single-GPU optimized |
| Environment | conda | Python venv |
| Launch | Command line | Double-click batch file |
| UI | None | Gradio WebUI |

## Requirements

- Windows 10/11
- NVIDIA Blackwell GPU (RTX 6000 Blackwell, etc.)
- Python 3.12
- PyTorch Nightly with CUDA 13.0 (cu130)
- ~72GB VRAM for single-GPU inference (125 frames)

## Quick Start

### 1. Create Virtual Environment

```powershell
cd C:\HYWorldPlay
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu130
pip install -r requirements.txt
pip install gradio loguru
```

### 3. Download Models

**Option A: Using download-models.bat**

```powershell
# Set HuggingFace token first
$env:HF_TOKEN = "your_hf_token"
.\download-models.bat
```

**Option B: Using huggingface-cli**

```powershell
# HY-WorldPlay Action Models
huggingface-cli download tencent/HY-WorldPlay --local-dir ckpts\HY-WorldPlay

# HunyuanVideo-1.5 Base Model
huggingface-cli download tencent/HunyuanVideo-1.5 --include "vae/*" "scheduler/*" "transformer/480p_i2v/*" --local-dir ckpts\HunyuanVideo-1.5

# Text Encoders
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ckpts\HunyuanVideo-1.5\text_encoder\llm
huggingface-cli download google/byt5-small --local-dir ckpts\HunyuanVideo-1.5\text_encoder\byt5-small

# Vision Encoder (requires HF token with FLUX access)
huggingface-cli download black-forest-labs/FLUX.1-Redux-dev --local-dir ckpts\HunyuanVideo-1.5\vision_encoder\siglip
```

**Note:** Vision encoder requires access approval at [FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev).

### 4. Launch

**Option A: Double-click (Recommended)**

Double-click `run-wp.bat`

**Option B: Command line**

```powershell
.\venv\Scripts\activate
$env:MODEL_BASE = "C:\HYWorldPlay\ckpts"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python app.py --port 7860
```

### 5. Open Browser

Navigate to http://localhost:7860

## Model Directory Structure

```
C:\HYWorldPlay\
├── ckpts\
│   ├── HY-WorldPlay\
│   │   ├── ar_model\
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   ├── ar_distilled_action_model\
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   └── bidirectional_model\
│   │       └── diffusion_pytorch_model.safetensors
│   └── HunyuanVideo-1.5\
│       ├── vae\
│       ├── scheduler\
│       ├── transformer\
│       │   └── 480p_i2v\
│       ├── text_encoder\
│       │   ├── llm\ (Qwen2.5-VL-7B-Instruct)
│       │   ├── byt5-small\
│       │   └── Glyph-SDXL-v2\ (from ModelScope)
│       └── vision_encoder\
│           └── siglip\ (from FLUX.1-Redux-dev)
├── app.py              # Gradio WebUI
├── run-wp.bat          # Launch script
└── ...
```

## Camera Control

Use pose strings to control camera movement:

| Action | Key | Description |
|--------|-----|-------------|
| Forward | `w` | Move camera forward |
| Backward | `s` | Move camera backward |
| Left | `a` | Strafe left |
| Right | `d` | Strafe right |
| Look Up | `up` | Pitch camera up |
| Look Down | `down` | Pitch camera down |
| Turn Left | `left` | Yaw camera left |
| Turn Right | `right` | Yaw camera right |

**Format:** `action-duration` (e.g., `w-31` = forward for 31 latents)

**Examples:**
- `w-31` - Move forward (generates 125 frames)
- `w-15,d-16` - Forward then right
- `a-10,w-5,right-16` - Complex trajectory

## Troubleshooting

### "No module named 'flash_attn'"

This is expected on Windows. The code automatically falls back to PyTorch's native `scaled_dot_product_attention`.

### CUDA out of memory

- Enable `--offloading true` in the UI
- Reduce video length

### Slow download

```powershell
pip install "huggingface_hub[hf_xet]"
```

### Port in use

`run-wp.bat` automatically finds an available port between 7860-7900.

## Technical Notes

### flash-attn Fallback

Since `flash-attn` doesn't compile on Windows, this fork implements automatic fallback to PyTorch's native SDPA (Scaled Dot Product Attention). Quality is equivalent, with slightly reduced speed.

### Single-GPU Optimization

The distributed processing code (`torchrun`, `init_device_mesh`) has been adapted to work with single-GPU setups by:
- Setting `WORLD_SIZE=1` and `LOCAL_RANK=0` by default
- Skipping device mesh initialization for single GPU
- Adding guards to distributed communication functions

### Windows Path Handling

- Font paths adapted for Windows (`C:/Windows/Fonts/`)
- Model paths use Windows-compatible separators
- Temporary directories use `tempfile.gettempdir()`

## License

This project is licensed under the [Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE).

## Acknowledgements

- [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) - Original repository
- [Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) - Base video model
- [Gradio](https://gradio.app/) - WebUI framework

---

## Japanese / 日本語

詳細な日本語ドキュメントは [README_Windows.md](README_Windows.md) を参照してください。
