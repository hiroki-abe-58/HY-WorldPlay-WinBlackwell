@echo off
setlocal enabledelayedexpansion

REM ===================================================
REM  HY-World 1.5 (WorldPlay) Model Download Script
REM  Supports resume from interrupted downloads
REM ===================================================

REM ===== Proxy Clear =====
set ALL_PROXY=
set HTTP_PROXY=
set HTTPS_PROXY=

REM ===== UTF-8 =====
chcp 65001 >nul

REM ===== Change to script directory =====
cd /d %~dp0

REM ===== Activate virtual environment =====
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

REM ===== Create ckpts directory =====
if not exist "ckpts" mkdir ckpts

echo.
echo ===================================================
echo  HY-World 1.5 Model Download
echo ===================================================
echo  Downloads will resume from where they left off.
echo ===================================================
echo.

REM ===== Step 1: HY-WorldPlay Action Models =====
echo [1/5] Downloading HY-WorldPlay Action Models...
huggingface-cli download tencent/HY-WorldPlay --local-dir ckpts\HY-WorldPlay
if %errorlevel% neq 0 (
    echo WARNING: HY-WorldPlay download may have failed. Will retry later.
)

REM ===== Step 2: HunyuanVideo-1.5 Base Model =====
echo.
echo [2/5] Downloading HunyuanVideo-1.5 Base Model (vae, scheduler, transformer)...
huggingface-cli download tencent/HunyuanVideo-1.5 --include "vae/*" "scheduler/*" "transformer/480p_i2v/*" --local-dir ckpts\HunyuanVideo-1.5
if %errorlevel% neq 0 (
    echo WARNING: HunyuanVideo-1.5 download may have failed. Will retry later.
)

REM ===== Step 3: Text Encoder (Qwen2.5-VL) =====
echo.
echo [3/5] Downloading Text Encoder (Qwen2.5-VL-7B-Instruct)...
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ckpts\HunyuanVideo-1.5\text_encoder\llm
if %errorlevel% neq 0 (
    echo WARNING: Qwen download may have failed. Will retry later.
)

REM ===== Step 4: ByT5 Text Encoder =====
echo.
echo [4/5] Downloading ByT5 Text Encoder...
huggingface-cli download google/byt5-small --local-dir ckpts\HunyuanVideo-1.5\text_encoder\byt5-small
if %errorlevel% neq 0 (
    echo WARNING: byt5 download may have failed. Will retry later.
)

REM ===== Step 5: Vision Encoder (Optional - requires HF token) =====
echo.
echo [5/5] Vision Encoder (FLUX.1-Redux-dev)
echo.
echo NOTE: This model requires access approval from HuggingFace.
echo If you have a token, set HF_TOKEN environment variable before running.
echo.

if defined HF_TOKEN (
    echo Downloading with token...
    huggingface-cli download black-forest-labs/FLUX.1-Redux-dev --local-dir ckpts\HunyuanVideo-1.5\vision_encoder\siglip --token %HF_TOKEN%
) else (
    echo Skipping vision encoder (no HF_TOKEN set).
    echo To download later, run:
    echo   set HF_TOKEN=your_token
    echo   huggingface-cli download black-forest-labs/FLUX.1-Redux-dev --local-dir ckpts\HunyuanVideo-1.5\vision_encoder\siglip --token %%HF_TOKEN%%
)

REM ===== Fix ar_distilled_action_model filename =====
echo.
echo Checking ar_distilled_action_model...
if exist "ckpts\HY-WorldPlay\ar_distilled_action_model\model.safetensors" (
    if not exist "ckpts\HY-WorldPlay\ar_distilled_action_model\diffusion_pytorch_model.safetensors" (
        echo Copying model.safetensors to diffusion_pytorch_model.safetensors...
        copy "ckpts\HY-WorldPlay\ar_distilled_action_model\model.safetensors" "ckpts\HY-WorldPlay\ar_distilled_action_model\diffusion_pytorch_model.safetensors"
    )
)

echo.
echo ===================================================
echo  Download Complete!
echo ===================================================
echo.
echo Model paths:
echo   HY-WorldPlay:    ckpts\HY-WorldPlay
echo   HunyuanVideo-1.5: ckpts\HunyuanVideo-1.5
echo.
echo You can now run: run-wp.bat
echo.

pause
