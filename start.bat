@echo off
chcp 65001 >nul
echo ========================================
echo   Seedance 视频处理工具 - 一键启动
echo ========================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

:: 创建虚拟环境
if not exist "venv" (
    echo [1/3] 正在创建虚拟环境...
    python -m venv venv
)

:: 激活虚拟环境并安装依赖
echo [2/3] 正在检查依赖...
call venv\Scripts\activate.bat
pip install -r requirements.txt -q

:: 启动应用
echo [3/3] 正在启动应用...
echo.
echo 浏览器打开 http://127.0.0.1:7860
echo 按 Ctrl+C 停止
echo.
python app.py
pause
