@echo off
REM ===========================================================================
REM MULTIMODAL ROOT CAUSE ANALYSIS - DEMO LAUNCHER
REM ===========================================================================
REM
REM Usage:
REM   demo.bat              - Interactive menu
REM   demo.bat evaluate     - Evaluate model
REM   demo.bat inference    - Run inference with explanations
REM   demo.bat speed        - Benchmark inference speed
REM   demo.bat quick        - Quick demo (3 samples, no LLM)
REM   demo.bat all          - Run all demos
REM
REM Authors: Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan
REM ===========================================================================

setlocal enabledelayedexpansion

REM Change to project directory
cd /d "%~dp0"

echo.
echo ======================================================================
echo                MULTIMODAL ROOT CAUSE ANALYSIS DEMO
echo ======================================================================
echo.
echo   Authors: Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan
echo   Course:  B.Tech Major Project - November 2025
echo.
echo ======================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please ensure Python is installed and in PATH.
    pause
    exit /b 1
)

REM Check if we have a conda environment active
if defined CONDA_DEFAULT_ENV (
    echo Using conda environment: %CONDA_DEFAULT_ENV%
) else (
    echo Note: No conda environment detected. Using system Python.
)
echo.

REM Parse arguments
if "%1"=="" (
    set MODE=interactive
) else (
    set MODE=%1
)

REM Run the appropriate demo
if "%MODE%"=="interactive" (
    echo Starting interactive demo...
    python scripts/demo.py --mode interactive
) else if "%MODE%"=="evaluate" (
    echo Running model evaluation...
    python scripts/demo.py --mode evaluate
) else if "%MODE%"=="inference" (
    echo Running inference demo with explanations...
    python scripts/demo.py --mode inference --samples 5
) else if "%MODE%"=="speed" (
    echo Running speed benchmark...
    python scripts/demo.py --mode speed
) else if "%MODE%"=="architecture" (
    echo Showing model architecture...
    python scripts/demo.py --mode architecture
) else if "%MODE%"=="quick" (
    echo Running quick demo...
    python scripts/demo.py --mode quick --no-llm
) else if "%MODE%"=="all" (
    echo Running all demos...
    python scripts/demo.py --mode all
) else if "%MODE%"=="help" (
    echo.
    echo Usage: demo.bat [mode]
    echo.
    echo Modes:
    echo   ^(none^)       - Interactive menu
    echo   evaluate     - Evaluate model accuracy on test set
    echo   inference    - Run inference with LLM explanations
    echo   speed        - Benchmark inference speed
    echo   architecture - Show model architecture diagram
    echo   quick        - Quick demo ^(3 samples, no LLM^)
    echo   all          - Run all demos sequentially
    echo   help         - Show this help message
    echo.
) else (
    echo Unknown mode: %MODE%
    echo Run 'demo.bat help' for usage information.
    exit /b 1
)

echo.
pause
