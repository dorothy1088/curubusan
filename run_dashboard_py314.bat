@echo off
setlocal
cd /d "%~dp0"
echo [1/3] Python 3.14 venv readying...
if not exist ".venv314\Scripts\python.exe" (
  py -3.14 -m venv .venv314
  if errorlevel 1 goto :error
)
echo [2/3] Installing dependencies...
".venv314\Scripts\python.exe" -m ensurepip --upgrade
".venv314\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto :error
echo [3/3] Opening Streamlit dashboard...
".venv314\Scripts\python.exe" -m streamlit run dash2.py
if errorlevel 1 goto :error
exit /b 0
:error
echo.
echo [ERROR] Please copy the full error text from this window.
pause
exit /b 1
