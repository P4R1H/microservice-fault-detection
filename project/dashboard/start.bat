@echo off
echo ========================================
echo  Multimodal RCA Dashboard Launcher
echo ========================================
echo.

:: Check if we're in the right directory
if not exist "package.json" (
    echo Error: package.json not found!
    echo Please run this script from the dashboard directory.
    pause
    exit /b 1
)

:: Start the FastAPI backend
echo Starting FastAPI backend on http://localhost:8000...
start "RCA API Server" cmd /k "cd api && python server.py"

:: Wait a moment for the backend to start
timeout /t 3 /nobreak > nul

:: Start the Vite dev server
echo Starting Vite dev server on http://localhost:5173...
start "Vite Dev Server" cmd /k "npm run dev"

:: Wait and open browser
timeout /t 3 /nobreak > nul
start http://localhost:5173

echo.
echo ========================================
echo  Dashboard is now running!
echo  - Frontend: http://localhost:5173
echo  - Backend:  http://localhost:8000
echo ========================================
echo.
echo Press any key to close this window...
pause > nul
