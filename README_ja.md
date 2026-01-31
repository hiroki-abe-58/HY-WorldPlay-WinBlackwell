# HY-WorldPlay-WinBlackwell

**Windows ネイティブ + Blackwell GPU 対応フォーク**

[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6?logo=windows)](https://www.microsoft.com/windows)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-WebUI-FF7C00?logo=gradio)](https://gradio.app/)

[Tencent HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) を **Windows ネイティブ環境** と **NVIDIA Blackwell アーキテクチャ GPU** (RTX 6000 Blackwell 等) で動作させるためのフォークです。

---

## このフォークのメリット

### 1. WSL不要 - 完全Windows ネイティブ動作

公式版は Linux 専用ですが、このフォークは **WSL (Windows Subsystem for Linux) なしで Windows 上で直接動作** します。

- パス変換の問題なし
- ファイルシステムのオーバーヘッドなし
- Windows ネイティブツールとの完全な互換性

### 2. 最新 Blackwell GPU (sm_120) 対応

NVIDIA の最新アーキテクチャ **Blackwell (sm_120)** に完全対応。

- RTX 6000 Blackwell (96GB VRAM)
- PyTorch Nightly cu130 で最新GPU機能を活用
- 大容量 VRAM を活かしたシングルGPU推論

### 3. flash-attn 不要 - 自動フォールバック

Windows では `flash-attn` のビルドが困難ですが、このフォークは **PyTorch ネイティブの Scaled Dot Product Attention (SDPA) に自動フォールバック** します。

- 手動でのビルド作業不要
- 品質は同等を維持
- エラーなく即座に動作開始

### 4. 直感的な Gradio WebUI

公式版はコマンドラインのみですが、このフォークは **ブラウザベースの WebUI** を提供。

- 画像アップロード
- プロンプト入力
- カメラ軌道設定
- モデル選択
- ワンクリック生成

### 5. ダブルクリック起動

`run-wp.bat` をダブルクリックするだけで:

- 仮想環境の自動有効化
- 環境変数の自動設定
- 空きポートの自動検出
- ブラウザの自動起動

---

## 技術的な違い（詳細）

| 項目 | 公式版 | このフォーク |
|------|--------|-------------|
| **対応OS** | Linux のみ | Windows 10/11 ネイティブ |
| **GPU アーキテクチャ** | sm_90 (Hopper) | sm_120 (Blackwell) |
| **flash-attn** | 必須（ビルド必要） | 不要（自動フォールバック） |
| **分散処理** | torchrun (マルチGPU) | シングルGPU最適化 |
| **環境構築** | conda | Python venv |
| **起動方法** | コマンドライン | ダブルクリック / WebUI |
| **ユーザーインターフェース** | なし | Gradio WebUI |
| **パス処理** | POSIX パス | Windows パス対応 |
| **フォント** | Linux フォント | Windows フォント |

### コード修正箇所

#### 1. `hyvideo/generate.py`
- `WORLD_SIZE` / `LOCAL_RANK` のデフォルト値設定（シングルGPU対応）
- Windows フォントパスへの対応 (`C:/Windows/Fonts/`)

#### 2. `hyvideo/commons/parallel_states.py`
- シングルGPU時の `init_device_mesh` スキップ処理
- `world_size == 1` の判定追加

#### 3. `hyvideo/utils/flash_attn_no_pad.py`
- `flash_attn` インポートの try-except 処理
- PyTorch SDPA へのフォールバック実装

#### 4. `hyvideo/utils/communications.py`
- 分散通信関数のガード処理追加
- 非初期化時のパススルー

#### 5. `download_models.py`
- Windows 一時ディレクトリの使用 (`tempfile.gettempdir()`)

---

## 動作環境

| 項目 | 要件 |
|------|------|
| OS | Windows 10 / 11 |
| GPU | NVIDIA Blackwell アーキテクチャ (RTX 6000 Blackwell 等) |
| VRAM | 72GB 以上（シングルGPU推論時） |
| Python | 3.12 |
| PyTorch | Nightly (cu130) |

---

## クイックスタート

### Step 1: リポジトリのクローン

```powershell
git clone https://github.com/hiroki-abe-58/HY-WorldPlay-WinBlackwell.git
cd HY-WorldPlay-WinBlackwell
```

### Step 2: 仮想環境のセットアップ

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: 依存関係のインストール

```powershell
# PyTorch Nightly (cu130)
pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu130

# その他の依存関係
pip install -r requirements.txt
pip install gradio loguru
```

### Step 4: モデルのダウンロード

```powershell
# HuggingFace トークンを設定
$env:HF_TOKEN = "your_huggingface_token"

# ダウンロードスクリプトを実行
.\download-models.bat
```

**注意:** Vision Encoder は [FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) へのアクセス承認が必要です。

### Step 5: 起動

```powershell
# 方法A: バッチファイル（推奨）
.\run-wp.bat

# 方法B: 直接起動
.\venv\Scripts\activate
python app.py --port 7860
```

### Step 6: ブラウザでアクセス

http://localhost:7860

---

## カメラ操作

WASD スタイルのポーズ文字列でカメラを制御できます。

| アクション | キー | 説明 |
|-----------|------|------|
| 前進 | `w` | カメラを前方に移動 |
| 後退 | `s` | カメラを後方に移動 |
| 左移動 | `a` | カメラを左にストレイフ |
| 右移動 | `d` | カメラを右にストレイフ |
| 上を向く | `up` | カメラを上にピッチ |
| 下を向く | `down` | カメラを下にピッチ |
| 左を向く | `left` | カメラを左にヨー |
| 右を向く | `right` | カメラを右にヨー |

**形式:** `アクション-持続時間` （例: `w-31` = 31 latent 分前進）

**例:**
- `w-31` - 前進（125フレーム生成）
- `w-15,d-16` - 前進してから右移動
- `a-10,w-5,right-16` - 複雑な軌道

---

## モデル構成

```
ckpts/
├── HY-WorldPlay/
│   ├── ar_model/                      # AR モデル（50ステップ）
│   ├── ar_distilled_action_model/     # AR 蒸留モデル（4ステップ、高速）
│   └── bidirectional_model/           # 双方向モデル
└── HunyuanVideo-1.5/
    ├── vae/                           # VAE
    ├── scheduler/                     # スケジューラ
    ├── transformer/480p_i2v/          # Transformer
    ├── text_encoder/
    │   ├── llm/                       # Qwen2.5-VL-7B-Instruct
    │   ├── byt5-small/                # ByT5
    │   └── Glyph-SDXL-v2/             # Glyph エンコーダ
    └── vision_encoder/siglip/         # SigLIP Vision Encoder
```

---

## トラブルシューティング

### "No module named 'flash_attn'" エラー

**想定内の動作です。** 自動的に PyTorch SDPA にフォールバックするため、無視して問題ありません。

### CUDA out of memory

- WebUI で「CPU Offloading」を有効化
- 動画の長さを短く設定

### ダウンロードが遅い

```powershell
pip install "huggingface_hub[hf_xet]"
```

### ポートが使用中

`run-wp.bat` は自動的に 7860〜7900 の範囲で空きポートを探します。

---

## 謝辞

- [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) - オリジナルリポジトリ
- [Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) - ベース動画モデル
- [Gradio](https://gradio.app/) - WebUI フレームワーク

---

## ライセンス

[Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE)

---

## 関連リンク

- [English README](README.md)
- [オリジナル HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)
- [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
