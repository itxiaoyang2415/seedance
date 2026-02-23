@echo off
chcp 65001 >nul
echo ========================================
echo   Seedance - 打包为 EXE
echo ========================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python
    pause
    exit /b 1
)

:: 创建虚拟环境
if not exist "venv" (
    echo [1/4] 创建虚拟环境...
    python -m venv venv
)

call venv\Scripts\activate.bat

:: 安装依赖 + pyinstaller
echo [2/4] 安装依赖...
pip install -r requirements.txt -q
pip install pyinstaller -q

:: 打包
echo [3/4] 正在打包（需要几分钟）...
pyinstaller ^
    --name Seedance ^
    --onedir ^
    --noconfirm ^
    --add-data "core;core" ^
    --hidden-import=rembg ^
    --hidden-import=insightface ^
    --hidden-import=onnxruntime ^
    --hidden-import=cv2 ^
    --hidden-import=gradio ^
    --hidden-import=PIL ^
    --collect-all rembg ^
    --collect-all insightface ^
    --collect-all gradio ^
    --collect-all onnxruntime ^
    app.py

:: 复制模型目录
echo [4/4] 复制资源文件...
if not exist "dist\Seedance\models" mkdir "dist\Seedance\models"
if exist "models\inswapper_128.onnx" copy "models\inswapper_128.onnx" "dist\Seedance\models\"

echo.
echo ========================================
echo   打包完成！
echo   输出目录: dist\Seedance\
echo   运行: dist\Seedance\Seedance.exe
echo ========================================
pause
