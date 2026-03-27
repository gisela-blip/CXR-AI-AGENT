@echo off
REM ============================================================
REM  Medical X-Ray AI Chatbot — One-click Installer (Windows)
REM  Run this from the AI AGENT folder:
REM      install.bat
REM ============================================================

echo.
echo ============================================================
echo   Medical X-Ray AI Chatbot -- Dependency Installer
echo ============================================================
echo.

REM Detect python
where py >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python 3 not found. Please install from https://python.org
    pause
    exit /b 1
)

echo [Step 1/4] Upgrading pip...
py -3 -m pip install --upgrade pip -q

echo.
echo [Step 2/4] Installing PyTorch (CPU version)...
echo   NOTE: For GPU support, cancel and run:
echo   py -3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
py -3 -m pip install torch torchvision torchaudio -q

echo.
echo [Step 3/4] Installing all other dependencies from requirements.txt...
py -3 -m pip install -r requirements.txt -q

echo.
echo [Step 4/4] Building FAISS RAG index from PPK COMPILE.xlsx...
py -3 rag_builder.py

echo.
echo ============================================================
echo   Installation complete!
echo   Run the chatbot with:  py -3 app.py
echo ============================================================
echo.
pause
