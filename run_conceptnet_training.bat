@echo off
echo ConceptNet Training Options
echo -------------------------
echo 1. Run basic training (no integration phase)
echo 2. Run full training with Prolog fixes
echo.

set /p choice=Enter your choice (1 or 2): 

if "%choice%"=="1" (
    echo.
    echo Running basic ConceptNet training (no integration phase)...
    python run_conceptnet.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Running full ConceptNet training with Prolog fixes...
    python run_fixed_conceptnet.py
    goto end
)

echo Invalid choice. Please run again and select 1 or 2.

:end
echo.
echo ConceptNet training complete! 