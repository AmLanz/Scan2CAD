@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Scan2CAD
set "VENV=%~dp0.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "MARK=%VENV%\installed_requirements.txt"

rem  Aufruf:  start.bat               -> einrichten (falls noetig) und starten
rem           start.bat datei.stl     -> direkt mit Netz starten (auch Drag&Drop auf die .bat)
rem           start.bat --reinstall   -> Abhaengigkeiten neu installieren
rem           start.bat --lang=en     -> englische Oberflaeche (--lang=de deutsch)

if /i "%~1"=="--reinstall" (
    if exist "%MARK%" del "%MARK%"
)

if exist "%VPY%" if exist "%MARK%" (
    fc /b "requirements.txt" "%MARK%" >nul 2>&1 && goto run
)

echo ================================================================
echo  Scan2CAD - Einrichtung der Python-Umgebung im Ordner .venv
echo  Einmalig ca. 400 MB Download, danach startet nur das Programm.
echo ================================================================
if not exist "%VPY%" (
    call :findpython
    if not defined PYEXE (
        echo [FEHLER] Kein 64-bit Python 3.11 bis 3.14 gefunden.
        echo          Bitte von https://www.python.org/downloads/ installieren.
        pause
        exit /b 1
    )
    echo Verwende Python: !PYEXE!
    !PYEXE! -m venv "%VENV%"
    if errorlevel 1 (
        echo [FEHLER] venv konnte nicht erstellt werden.
        pause
        exit /b 1
    )
)

"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARNUNG] Feste Versionen nicht installierbar - versuche aktuelle Versionen ...
    "%VPY%" -m pip install -r requirements-loose.txt
    if errorlevel 1 (
        echo [FEHLER] Installation fehlgeschlagen. Ausgabe oben pruefen.
        pause
        exit /b 1
    )
)

"%VPY%" -c "import OCP, vtk, pyvista, pyvistaqt, scipy, qtpy; from OCP.STEPControl import STEPControl_Writer; print('Abhaengigkeiten OK')"
if errorlevel 1 (
    echo [FEHLER] Import-Test fehlgeschlagen.
    pause
    exit /b 1
)
copy /y "requirements.txt" "%MARK%" >nul

:run
"%VPY%" "%~dp0app\main.py" %*
if errorlevel 1 (
    echo.
    echo Programm mit Fehler beendet - Details oben bzw. in scan2cad.log
    pause
)
exit /b 0

:findpython
set "PYEXE="
for %%V in (3.13 3.12 3.14 3.11) do (
    if not defined PYEXE (
        py -%%V -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
        if not errorlevel 1 set "PYEXE=py -%%V"
    )
)
if not defined PYEXE (
    python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,14) and sys.maxsize > 2**32 else 1)" >nul 2>&1
    if not errorlevel 1 set "PYEXE=python"
)
exit /b 0
