@echo off
REM ============================================================================
REM LAUNCH MULTIMODAL RCA DASHBOARD
REM ============================================================================
REM 
REM This script launches the interactive Streamlit dashboard for visualizing
REM Root Cause Analysis predictions, causal graphs, and model explanations.
REM
REM Usage:
REM     launch_dashboard.bat
REM
REM Authors: Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan
REM Date: November 2025
REM ============================================================================

echo.
echo  ============================================================
echo    MULTIMODAL ROOT CAUSE ANALYSIS DASHBOARD
echo  ============================================================
echo.

cd /d %~dp0

echo  [INFO] Starting Streamlit dashboard...
echo  [INFO] The dashboard will open in your default browser.
echo  [INFO] Press Ctrl+C to stop the server.
echo.

python -m streamlit run dashboard/app.py --theme.base dark --server.port 8501

pause
