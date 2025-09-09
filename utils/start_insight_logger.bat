@echo off
:: Activar entorno y ejecutar el script de inicio de Insight Logger
cd /d "C:\Users\PC\Documents\AGI-Prototipo"
call C:\Users\PC\miniconda3\Scripts\activate.bat llama-env-py310
python utils\start_insight_logger.py
pause
