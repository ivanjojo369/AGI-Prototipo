@echo off
echo 🚀 Iniciando AGI Insight Logger...
cd /d "%~dp0"
call C:\Users\PC\miniconda3\Scripts\activate.bat llama-env-py310
python utils\start_insight_logger.py
pause
