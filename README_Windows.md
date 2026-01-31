# HY-World 1.5 (WorldPlay) - Windows + Blackwell GPU Setup

このドキュメントは、HY-WorldPlay を Windows ネイティブ環境 + Blackwell GPU (RTX 6000 Blackwell) で動作させるための設定手順を説明します。

## 動作環境

- Windows 10/11
- NVIDIA RTX 6000 Blackwell (96GB VRAM)
- Python 3.12
- PyTorch Nightly cu130

## フォークで行った修正

| 項目 | 公式版 | Windows版 |
|------|--------|-----------|
| OS | Linux | Windows 10/11 |
| GPU | sm_90 | sm_120 (Blackwell) |
| flash-attn | 必須 | 不要（自動フォールバック） |
| 分散処理 | torchrun（マルチGPU） | シングルGPU最適化 |
| 環境 | conda | Python venv |
| 起動 | コマンドライン | ダブルクリック (bat) |
| UI | なし | Gradio WebUI |

## クイックスタート

### 1. 仮想環境の有効化

```powershell
cd C:\HYWorldPlay
.\venv\Scripts\activate
```

### 2. モデルのダウンロード

**方法A: huggingface-cli を使用**

```powershell
# HY-WorldPlay (Action Models)
huggingface-cli download tencent/HY-WorldPlay --local-dir ckpts\HY-WorldPlay

# HunyuanVideo-1.5 (Base Model)
huggingface-cli download tencent/HunyuanVideo-1.5 --include "vae/*" "scheduler/*" "transformer/480p_i2v/*" --local-dir ckpts\HunyuanVideo-1.5

# Qwen2.5-VL-7B-Instruct (Text Encoder)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ckpts\HunyuanVideo-1.5\text_encoder\llm

# ByT5 (Text Encoder)
huggingface-cli download google/byt5-small --local-dir ckpts\HunyuanVideo-1.5\text_encoder\byt5-small

# Vision Encoder (要HuggingFaceトークン)
huggingface-cli download black-forest-labs/FLUX.1-Redux-dev --local-dir ckpts\HunyuanVideo-1.5\vision_encoder\siglip --token YOUR_HF_TOKEN
```

**方法B: download_models.py を使用**

```powershell
python download_models.py --hf_token YOUR_HF_TOKEN
```

### 3. 起動

**方法A: バッチファイル（推奨）**

`run-wp.bat` をダブルクリック

**方法B: コマンドライン**

```powershell
.\venv\Scripts\activate
python app.py --port 7860
```

### 4. ブラウザで開く

http://localhost:7860

## モデルパスの構成

```
C:\HYWorldPlay\
├── ckpts\
│   ├── HY-WorldPlay\
│   │   ├── ar_model\
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   ├── ar_distilled_action_model\
│   │   │   └── diffusion_pytorch_model.safetensors (model.safetensorsをリネーム)
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
│       │   └── Glyph-SDXL-v2\ (ModelScopeから)
│       └── vision_encoder\
│           └── siglip\ (FLUX.1-Redux-dev)
├── app.py                 # Gradio WebUI
├── run-wp.bat             # ダブルクリック起動
└── ...
```

## 環境変数

`run-wp.bat` で設定される環境変数:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - CUDAメモリ最適化
- `MODEL_BASE=%~dp0ckpts` - モデルパスのベースディレクトリ

## 注意事項

### flash-attn について

Windows では flash-attn がインストールできないため、PyTorch ネイティブの `scaled_dot_product_attention` に自動フォールバックします。速度は若干低下しますが、品質は同等です。

### SageAttention / FP8 GEMM について

Blackwell GPU での互換性問題が報告されているため、以下のオプションは無効化推奨:

- `--use_sageattn false`
- `--use_fp8_gemm false`

### VRAM 要件

| 構成 | VRAM |
|------|------|
| sp=8（推奨、マルチGPU） | 28GB |
| sp=4 | 34GB |
| sp=1（シングルGPU） | 72GB |

RTX PRO 6000 Blackwell (96GB) であれば sp=1 でも十分に動作可能です。

## トラブルシューティング

### "No module named 'flash_attn'" エラー

これは想定内の動作です。自動的に PyTorch SDPA にフォールバックします。

### ダウンロードが遅い場合

```powershell
pip install "huggingface_hub[hf_xet]"
```

### CUDA out of memory

`--offloading true` を使用するか、`video_length` を減らしてください。

### ポートが使用中

`run-wp.bat` は自動的に空きポートを探します（7860-7900）。

## CLI での推論（上級者向け）

```powershell
.\venv\Scripts\activate
python -m hyvideo.generate `
    --prompt "A paved pathway leads towards a stone arch bridge..." `
    --image_path "assets/bridge.png" `
    --pose "w-31" `
    --model_path "ckpts/HunyuanVideo-1.5" `
    --action_ckpt "ckpts/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors" `
    --model_type ar `
    --resolution 480p `
    --video_length 125 `
    --output_path "outputs"
```

## ライセンス

HY-WorldPlay は [Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE) の下で提供されています。
