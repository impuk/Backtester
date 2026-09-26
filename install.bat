@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Установка бэктестера ===
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m venv .venv
) else (
    python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Не удалось создать окружение. Проверьте, что установлен Python 3.12, 3.13 или 3.14
    echo и при установке отмечена галочка "Add python.exe to PATH".
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Ошибка установки библиотек. Проверьте интернет и запустите install.bat ещё раз.
    pause
    exit /b 1
)
echo.
echo Компилирую движок, это может занять до минуты...
".venv\Scripts\python.exe" warmup.py
echo.
echo Готово. Запускайте start.bat
pause
