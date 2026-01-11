@echo off
setlocal

echo ===============================================
echo        ROCKET STARTING ULTIMATE RUN ROCKET             
echo ===============================================

REM 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH.
    echo Please install Python and add it to PATH.
    pause
    exit /b 1
)

REM Check for VENV
if not exist "venv" (
    echo Creating Virtual Environment ^(venv^)... DO NOT CLOSE.
    echo This may take 1-2 minutes...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
    echo Venv created.
)

REM Activate VENV
call venv\Scripts\activate.bat

echo Checking environment integrity...
python -c "import numpy; import scipy; import pandas; import matplotlib; import sklearn; import pytest" >nul 2>&1

if %errorlevel% equ 0 (
    echo ✔ Dependencies already installed. Skipping download.
) else (
    echo ⚠ Missing dependencies detected.
    echo Launching Smart Installer...
    python utils/install_requirements.py
    if %errorlevel% neq 0 (
        echo WARNING: Failed to install dependencies.
    )
)

echo Using Python:
where python

REM 2. Add current directory to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%CD%

echo.
echo Cleaning ENTIRE data directory for fresh run...
if exist "data" rd /s /q "data"
mkdir "data"

echo.
echo Step 0: Part A - Algorithm Validation (Statistical Study)
python experiments/run_part_a.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo -----------------------------------------------
echo Step 1: Part B - Reference Optimization (PSO)
echo Cleaning old Part B results...
if exist "data\PartB\pso\results" rd /s /q "data\PartB\pso\results"
if exist "data\PartB\pso\figures" rd /s /q "data\PartB\pso\figures"

echo Running PSO (Pop 20, Evals 400 ~20 Gen)... (FAST MODE)
python experiments/run_airfoil.py --part B --solver pso --pop 20 --evals 400 --seed 123 --jobs 8
if %errorlevel% neq 0 exit /b %errorlevel%

REM Find LATEST_REF (Best PSO) - Windows Batch is tricky with ls -t
REM We will rely on python to define it or just find it later.
REM For simpler batch, we'll just let the plotting script find the latest default if we don't pass explicit path, OR use a small python helper.
REM Actually, run_airfoil.py saves to data/PartB/pso/results/...
REM The plotting script (Step 5) can auto-find the latest. I will update the python call there to rely on auto-detection if argument missing, OR assumes latest.
REM But since we want to be explicit for 3-way, let's use a tiny python one-liner to get the path.

for /f "delims=" %%I in ('python -c "import glob, os; lists=sorted(glob.glob('data/PartB/pso/results/*_best.json'), key=os.path.getmtime); print(lists[-1] if lists else '')"') do set LATEST_REF=%%I

echo.
echo -----------------------------------------------
echo Step 1.5: GA Parameter Study (Sensitivity Analysis)

echo Run A: Low Fidelity GA (Pop 10, Evals 100)... (FAST MODE)
python experiments/run_airfoil.py --part B --solver ga --pop 10 --evals 100 --seed 123 --jobs 8

echo Run B: Medium Fidelity GA (Pop 10, Evals 200)... (FAST MODE)
python experiments/run_airfoil.py --part B --solver ga --pop 10 --evals 200 --seed 123 --jobs 8

echo Run C: High Fidelity GA (Pop 20, Evals 400)... (FAST MODE)
python experiments/run_airfoil.py --part B --solver ga --pop 20 --evals 400 --seed 123 --jobs 8

for /f "delims=" %%I in ('python -c "import glob, os; lists=sorted(glob.glob('data/PartB/ga/results/*_best.json'), key=os.path.getmtime); print(lists[-1] if lists else '')"') do set LATEST_GA=%%I

echo Reference Design Found (PSO): %LATEST_REF%
echo Reference Design Found (GA):  %LATEST_GA%

echo.
echo -----------------------------------------------
echo Step 2: Part C - Robust Data Generation
echo Cleaning old data...
if exist "data\PartC\training_data.csv" del "data\PartC\training_data.csv"

echo Generating 50 Samples with 8 Workers... (FAST MODE)
python experiments/part_c_data_gen.py --samples 50 --jobs 8
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo -----------------------------------------------
echo Step 3: Part C - Surrogate Training
python experiments/part_c_surrogate.py

echo.
echo -----------------------------------------------
echo Step 4: Part C - Surrogate Optimization
python experiments/part_c_opt_surrogate.py

echo.
echo -----------------------------------------------
echo Step 5: Visualization and Comparison (3-Way)
python experiments/part_c_plot_geometry.py --ref "%LATEST_GA%" --pso "%LATEST_REF%"

echo.
echo -----------------------------------------------
echo Step 6: Uncertainty Quantification
python experiments/part_c_uq_opt.py --json data/PartC/surrogate_comparison.json --samples 50

echo.
echo ===============================================
echo        ULTIMATE RUN COMPLETE             
echo ===============================================
echo Results:
echo  - Report: data/PartC/surrogate_metrics.txt
echo  - Plots:  data/PartC/figures/
pause
