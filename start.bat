@echo off
rem ============================================================
rem  A股行情分析与选股软件 - 一键启动
rem  双击本文件即可启动（使用项目自带 venv，无需手动装依赖）
rem ============================================================
cd /d "%~dp0"

set PYTHON=D:\Stock\.venv\Scripts\python.exe
if not exist "%PYTHON%" (
    echo [错误] 未找到 venv 解释器：%PYTHON%
    echo 请确认 D:\Stock\.venv 存在；否则先运行 D:\Stock\setup_venv.bat
    pause
    exit /b 1
)

echo 正在启动 A股行情分析软件（首次启动会联网拉数据，请耐心等待）...
"%PYTHON%" main.py
if errorlevel 1 (
    echo.
    echo [程序异常退出] 请查看日志：data\app.log
    pause
)
