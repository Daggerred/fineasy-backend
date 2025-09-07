@echo off
REM FinEasy AI Backend Development Setup Script for Windows

echo 🚀 Setting up FinEasy AI Backend development environment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python installation found

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Download spaCy model
echo 🧠 Downloading spaCy English model...
python -m spacy download en_core_web_sm

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚠️  Please update .env file with your actual credentials!
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist ml_models mkdir ml_models
if not exist logs mkdir logs

echo ✅ Development environment setup complete!
echo.
echo Next steps:
echo 1. Update .env file with your Supabase credentials
echo 2. Activate virtual environment: venv\Scripts\activate.bat
echo 3. Start development server: uvicorn app.main:app --reload
echo 4. Visit http://localhost:8000/docs for API documentation
echo.
echo Happy coding! 🎉
pause