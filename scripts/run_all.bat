@echo off
setlocal
echo ====================================================================
echo  RetinaGuard End-to-End Automated Pipeline Execution
echo ====================================================================

python scripts\run_pipeline.py --stage all
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline execution failed.
    exit /b 1
)

echo ====================================================================
echo  RetinaGuard Pipeline Execution Completed Successfully!
echo ====================================================================
endlocal
