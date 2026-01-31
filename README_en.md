<p align="right">
  <strong>Language / 言語:</strong>
  <a href="README.md"><img src="https://img.shields.io/badge/日本語-gray?style=flat-square" alt="日本語"></a>
  <a href="README_en.md"><img src="https://img.shields.io/badge/English-0078D6?style=flat-square" alt="English"></a>
</p>

# HY-WorldPlay-WinBlackwell

**Windows Native + Blackwell GPU Compatible Fork**

[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6?logo=windows)](https://www.microsoft.com/windows)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-WebUI-FF7C00?logo=gradio)](https://gradio.app/)

A fork of [Tencent HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) optimized for **Windows native execution** and **NVIDIA Blackwell architecture GPUs** (RTX 6000 Blackwell, etc.).

---

## Why This Fork?

### 1. No WSL Required - Full Windows Native

The official version is Linux-only, but this fork **runs directly on Windows without WSL (Windows Subsystem for Linux)**.

- No path conversion issues
- No filesystem overhead
- Full compatibility with Windows native tools

### 2. Latest Blackwell GPU (sm_120) Support

Full support for NVIDIA's latest **Blackwell architecture (sm_120)**.

- RTX 6000 Blackwell (96GB VRAM)
- PyTorch Nightly cu130 for latest GPU features
- Single-GPU inference leveraging large VRAM

### 3. No flash-attn Required - Automatic Fallback

Building `flash-attn` on Windows is difficult, but this fork **automatically falls back to PyTorch's native Scaled Dot Product Attention (SDPA)**.

- No manual build required
- Equivalent quality maintained
- Starts working immediately without errors

### 4. Intuitive Gradio WebUI

The official version is command-line only, but this fork provides a **browser-based WebUI**.

- Image upload
- Prompt input
- Camera trajectory settings
- Model selection
- One-click generation

### 5. Double-Click Launch

Just double-click `run-wp.bat` to:

- Auto-activate virtual environment
- Auto-set environment variables
- Auto-detect available port
- Auto-launch browser

---

## Technical Differences

| Feature | Official | This Fork |
|---------|----------|-----------|
| **OS** | Linux only | Windows 10/11 native |
| **GPU Architecture** | sm_90 (Hopper) | sm_120 (Blackwell) |
| **flash-attn** | Required (build needed) | Optional (auto-fallback) |
| **Distributed** | torchrun (multi-GPU) | Single-GPU optimized |
| **Environment** | conda | Python venv |
| **Launch** | Command line | Double-click / WebUI |
| **UI** | None | Gradio WebUI |
| **Paths** | POSIX paths | Windows paths |
| **Fonts** | Linux fonts | Windows fonts |

### Code Modifications

<details>
<summary>Click to expand</summary>

#### 1. `hyvideo/generate.py`
- Default values for `WORLD_SIZE` / `LOCAL_RANK` (single-GPU support)
- Windows font path support (`C:/Windows/Fonts/`)

#### 2. `hyvideo/commons/parallel_states.py`
- Skip `init_device_mesh` for single GPU
- Added `world_size == 1` check

#### 3. `hyvideo/utils/flash_attn_no_pad.py`
- try-except for `flash_attn` import
- PyTorch SDPA fallback implementation

#### 4. `hyvideo/utils/communications.py`
- Guard for distributed communication functions
- Pass-through when not initialized

#### 5. `download_models.py`
- Windows temp directory usage (`tempfile.gettempdir()`)

#### 6. `app.py` (new)
- Gradio WebUI implementation
- Auto frame count calculation from pose string
- Long generation timeout prevention

</details>

---

## Requirements

| Item | Requirement |
|------|-------------|
| OS | Windows 10 / 11 |
| GPU | NVIDIA Blackwell architecture (RTX 6000 Blackwell, etc.) |
| VRAM | 72GB+ (for single-GPU inference) |
| Python | 3.12 |
| PyTorch | Nightly (cu130) |

---

## Quick Start

### Step 1: Clone Repository

```powershell
git clone https://github.com/hiroki-abe-58/HY-WorldPlay-WinBlackwell.git
cd HY-WorldPlay-WinBlackwell
```

### Step 2: Setup Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies

```powershell
# PyTorch Nightly (cu130)
pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu130

# Other dependencies
pip install -r requirements.txt
pip install gradio loguru
```

### Step 4: Download Models

```powershell
# Set HuggingFace token
$env:HF_TOKEN = "your_huggingface_token"

# Run download script
.\download-models.bat
```

> **Note:** Vision Encoder requires access approval at [FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev).

### Step 5: Launch

```powershell
# Option A: Batch file (recommended)
.\run-wp.bat

# Option B: Direct launch
.\venv\Scripts\activate
python app.py --port 7860
```

### Step 6: Open Browser

http://localhost:7860

---

## Camera Control

Control camera movement with WASD-style pose strings.

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

---

## Model Structure

```
ckpts/
├── HY-WorldPlay/
│   ├── ar_model/                      # AR model (50 steps)
│   ├── ar_distilled_action_model/     # AR distilled model (4 steps, fast)
│   └── bidirectional_model/           # Bidirectional model
└── HunyuanVideo-1.5/
    ├── vae/                           # VAE
    ├── scheduler/                     # Scheduler
    ├── transformer/480p_i2v/          # Transformer
    ├── text_encoder/
    │   ├── llm/                       # Qwen2.5-VL-7B-Instruct
    │   ├── byt5-small/                # ByT5
    │   └── Glyph-SDXL-v2/             # Glyph encoder
    └── vision_encoder/siglip/         # SigLIP Vision Encoder
```

---

## Troubleshooting

<details>
<summary><strong>"No module named 'flash_attn'" error</strong></summary>

**This is expected behavior.** The code automatically falls back to PyTorch SDPA. You can safely ignore this.

</details>

<details>
<summary><strong>CUDA out of memory</strong></summary>

- Enable "CPU Offloading" in WebUI
- Reduce video length

</details>

<details>
<summary><strong>Slow download</strong></summary>

```powershell
pip install "huggingface_hub[hf_xet]"
```

</details>

<details>
<summary><strong>Port in use</strong></summary>

`run-wp.bat` automatically finds an available port between 7860-7900.

</details>

---

## Acknowledgements

- [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) - Original repository
- [Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) - Base video model
- [Gradio](https://gradio.app/) - WebUI framework

---

## License

[Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE)

---

## Related Links

- [日本語 README](README.md)
- [Original HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)
- [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
