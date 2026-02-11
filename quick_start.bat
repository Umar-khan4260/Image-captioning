@echo off
REM Quick Start Script for Image Captioning Deployment
REM For Windows Users

echo ================================================
echo Image Captioning - Quick Start
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if Git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed!
    echo Please install Git from https://git-scm.com/downloads
    pause
    exit /b 1
)

echo Python and Git are installed. Great!
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo Next steps:
echo 1. Place your model file (complete_model_package.pkl) in this folder
echo 2. Run: streamlit run app.py
echo 3. Your app will open in the browser
echo.
echo To deploy to Streamlit Cloud:
echo 1. Create GitHub repo
echo 2. Push your code: git init, git add ., git commit, git push
echo 3. Go to share.streamlit.io and deploy
echo.
echo See DEPLOYMENT_GUIDE.md for detailed instructions
echo.
pause
