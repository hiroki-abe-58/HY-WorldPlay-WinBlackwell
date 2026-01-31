@echo off
setlocal enabledelayedexpansion

REM ===================================================
REM  HY-World 1.5 (WorldPlay) Launcher
REM  Windows + Blackwell GPU
REM ===================================================

REM ===== Proxy Clear =====
set ALL_PROXY=
set HTTP_PROXY=
set HTTPS_PROXY=
set GIT_HTTP_PROXY=
set GIT_HTTPS_PROXY=

REM ===== CUDA Optimization =====
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM ===== Model Path (fix Linux default path) =====
set MODEL_BASE=%~dp0ckpts

REM ===== UTF-8 =====
chcp 65001 >nul

REM ===== Change to script directory =====
cd /d %~dp0

REM ===== Activate virtual environment =====
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run the setup first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

REM ===== Find available port (starting from 7860) =====
set PORT=7860
:find_port
netstat -an | findstr "LISTENING" | findstr ":%PORT% " >nul 2>&1
if %errorlevel%==0 (
    echo Port %PORT% is in use, trying next...
    set /a PORT+=1
    if !PORT! gtr 7900 (
        echo ERROR: No available port found between 7860-7900
        pause
        exit /b 1
    )
    goto find_port
)

echo.
echo ===================================================
echo  HY-World 1.5 (WorldPlay)
echo ===================================================
echo  Starting on port %PORT%
echo  URL: http://localhost:%PORT%
echo ===================================================
echo.

REM ===== Launch browser after delay =====
start "" cmd /c "timeout /t 15 >nul && start http://localhost:%PORT%"

REM ===== Run application =====
python app.py --port %PORT% --host 127.0.0.1

pause
