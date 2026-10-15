@echo off
echo ====================================================================
echo  RetinaGuard End-to-End Automated Pipeline Execution
echo ====================================================================

call scripts\00_check_env.bat
if %ERRORLEVEL% NEQ 0 ( echo Failed at Step 00 & exit /b 1 )

call scripts\03_smoke_test.bat
if %ERRORLEVEL% NEQ 0 ( echo Failed at Step 03 & exit /b 1 )

call scripts\07_evaluate.bat
if %ERRORLEVEL% NEQ 0 ( echo Failed at Step 07 & exit /b 1 )

call scripts\08_make_tables_figures.bat
if %ERRORLEVEL% NEQ 0 ( echo Failed at Step 08 & exit /b 1 )

call scripts\10_build_slides.bat
if %ERRORLEVEL% NEQ 0 ( echo Failed at Step 10 & exit /b 1 )

echo ====================================================================
echo  RetinaGuard Pipeline Execution Completed Successfully!
echo ====================================================================
