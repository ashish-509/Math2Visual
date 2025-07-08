@echo off
echo Math2Visual RAG System Setup
echo ===============================

echo.
echo Step 1: Installing Python requirements...
cd /d "%~dp0"
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo Step 2: Testing RAG components...
python test_rag.py

if %errorlevel% neq 0 (
    echo Warning: Some tests failed, but continuing...
)

echo.
echo Step 3: Running full setup...
python setup.py

if %errorlevel% neq 0 (
    echo Error: Setup failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To use the enhanced RAG app:
echo   streamlit run app_with_rag.py
echo.
echo To use the original app with RAG option:
echo   cd ../code
echo   streamlit run app.py
echo.
pause
