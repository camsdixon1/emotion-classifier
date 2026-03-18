@echo off
REM ============================================================
REM  Lifelogger — Add to Windows Startup (runs on login)
REM ============================================================
REM  This creates a scheduled task that starts lifelogger
REM  when you log in. Run this script once as Administrator.
REM ============================================================

set PYTHON_PATH=python
set SCRIPT_PATH=%~dp0main.py

echo Creating scheduled task: Lifelogger
echo Python: %PYTHON_PATH%
echo Script: %SCRIPT_PATH%
echo.

REM Create a scheduled task that runs at logon
schtasks /create /tn "Lifelogger" /tr "\"%PYTHON_PATH%\" \"%SCRIPT_PATH%\"" /sc onlogon /rl highest /f

if %ERRORLEVEL% equ 0 (
    echo.
    echo SUCCESS: Lifelogger will start automatically on login.
    echo.
    echo To start it now:   python "%SCRIPT_PATH%"
    echo To remove:         schtasks /delete /tn "Lifelogger" /f
) else (
    echo.
    echo FAILED: Run this script as Administrator.
)

pause
