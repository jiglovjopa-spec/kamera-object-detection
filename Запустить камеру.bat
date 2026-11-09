@echo off
chcp 65001 >nul
title Камера: предметы, номера, овощи
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

"%PY%" -c "import cv2, onnxruntime, PIL, open_image_models, tkinter" >nul 2>&1
if not errorlevel 1 goto models

echo Первая настройка: будет скачано около 1,1 ГБ библиотек (на диске займут 1,7 ГБ). Нужен интернет, 5-15 минут.
if exist ".venv" (
    echo Папка .venv не работает на этом компьютере. Закройте программу и удалите папку .venv, затем запустите снова.
    goto fail
)
py -3 -m venv .venv
if errorlevel 1 python -m venv .venv
if errorlevel 1 (
    echo Не найден Python. Установите Python 3.14 с сайта python.org и запустите этот файл снова.
    goto fail
)
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r app\requirements.txt
if errorlevel 1 goto netfail
"%PY%" -m pip install --no-deps open-image-models==0.6.0
if errorlevel 1 goto netfail

:models
"%PY%" app\setup_models.py
if errorlevel 1 goto fail

echo Запуск… Окно с камерой откроется через несколько секунд. Это окно не закрывайте.
"%PY%" app\main.py %*
if errorlevel 1 goto fail
exit /b 0

:netfail
echo Не удалось скачать библиотеки. Проверьте интернет и запустите файл снова.
:fail
echo.
echo Программа остановилась с ошибкой. Текст ошибки выше.
pause
exit /b 1