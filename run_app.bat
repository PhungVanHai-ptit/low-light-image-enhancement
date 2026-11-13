@echo off
REM Chuyen den thu muc chua file bat
cd /d "%~dp0"

REM Dat bien moi truong PYTHONPATH
set PYTHONPATH=%~dp0

echo ======================================================================
echo    UNG DUNG CAI THIEN CHAT LUONG ANH THIEU SANG - NHOM XLA PTIT
echo    Thuat toan: Gamma Correction, Global HE, CLAHE
echo ======================================================================
echo.
echo [*] Dang khoi dong ung dung...
echo.

REM Uu tien moi truong ao .venv neu ton tai, neu khong dung python he thong
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m source.app %*
) else (
    python -m source.app %*
)

REM Neu xay ra loi thi dung man hinh de kiem tra, neu thanh cong thi tu dong dong CMD
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Ung dung da dung voi ma loi: %ERRORLEVEL%
    pause
) else (
    exit /b 0
)
