@echo off
call C:\Users\PC\miniconda3\Scripts\activate.bat llama-env-py310
cd /d "C:\Users\PC\Documents\AGI-Prototipo"
start "AGI Clipboard Exporter" cmd.exe /k "python -m utils.clipboard_exporter"
