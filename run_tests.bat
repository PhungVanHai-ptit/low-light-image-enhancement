@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0

echo ======================================================================
echo    CHAY KIEM THU TU DONG (UNIT TESTS) - NHOM XLA PTIT
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m unittest discover -s tests
) else (
    python -m unittest discover -s tests
)

echo.
pause

