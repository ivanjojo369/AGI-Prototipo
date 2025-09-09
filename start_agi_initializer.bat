@echo off
cd /d "C:\Users\PC\Documents\AGI-Prototipo"

:: Nombre de archivo de log con fecha y hora
set LOG_FILE=logs\agi_start_%DATE:~6,4%-%DATE:~3,2%-%DATE:~0,2%_%TIME:~0,2%-%TIME:~3,2%-%TIME:~6,2%.log

:: Crear carpeta de logs si no existe
if not exist logs mkdir logs

:: Ejecutar AGI y guardar log
start cmd /k "conda activate llama-env-py310 && python agi_initializer.py >> %LOG_FILE% 2>&1"
