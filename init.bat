@echo off
REM AAC Model Testing Framework - Windows Batch Initialization
REM Simple wrapper to run the PowerShell initialization script

echo.
echo ================================================================================
echo AAC MODEL TESTING FRAMEWORK - WINDOWS SETUP
echo ================================================================================
echo.
echo This will run the PowerShell initialization script to set up everything needed
echo for AAC model testing.
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul

echo.
echo Running PowerShell initialization script...
echo.

powershell -ExecutionPolicy Bypass -File "init.ps1"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo Setup completed successfully!
    echo ================================================================================
    echo.
    echo You can now run: uv run python model_test.py
    echo.
) else (
    echo.
    echo ================================================================================
    echo Setup encountered errors. Please check the output above.
    echo ================================================================================
    echo.
)

echo Press any key to exit...
pause >nul
