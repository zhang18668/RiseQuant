@echo off
REM ===== 一键启动 RiseQuant 前后端开发环境 =====
REM 后端: FastAPI on :8000
REM 前端: Vite on :5173

set ROOT=%~dp0
set BACKEND=%ROOT%limit_up_project
set FRONTEND=%ROOT%webui

echo [start_dev] 启动后端 (FastAPI)...
start "RiseQuant Backend" cmd /k "cd /d %BACKEND% && set PYTHONPATH=%BACKEND% && uvicorn api.main:app --reload --port 8000"

timeout /t 3 /nobreak > nul

echo [start_dev] 启动前端 (Vite)...
start "RiseQuant Frontend" cmd /k "cd /d %FRONTEND% && npm run dev"

echo.
echo [start_dev] 两个服务已在新窗口启动:
echo   - 后端:  http://127.0.0.1:8000        (API 文档: /docs)
echo   - 前端:  http://127.0.0.1:5173
echo.
echo 关闭这个窗口不会关闭后端/前端, 请直接关闭它们对应的 cmd 窗口.
pause
