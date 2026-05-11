@echo off
echo ========================================
echo Industrial IoT Predictive Maintenance
echo ========================================
echo.

 REMOVE OLD PYTHON CACHE
echo Cleaning cache...
if exist __pycache__ rmdir /s /q __pycache__ 2>nul
if exist src\__pycache__ rmdir /s /q src\__pycache__ 2>nul
if exist backend\__pycache__ rmdir /s /q backend\__pycache__ 2>nul
if exist models\__pycache__ rmdir /s /q models\__pycache__ 2>nul
if exist data\__pycache__ rmdir /s /q data\__pycache__ 2>nul
echo.

 REMOVE OLD DATABASE
echo Resetting database...
if exist predictive_maintenance.db del predictive_maintenance.db 2>nul
if exist backend\predictive_maintenance.db del backend\predictive_maintenance.db 2>nul
echo.

 INSTALL DEPENDENCIES IF NEEDED
echo Checking dependencies...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Flask...
    pip install flask pandas numpy scikit-learn
)

echo.
echo ========================================
echo Starting server...
echo ========================================
echo.
echo Open your browser to:
echo http://localhost:5050/
echo.

python backend\app.py

pause