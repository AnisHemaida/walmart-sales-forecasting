@echo off
title SIPV Walmart — Interface Flask
echo.
echo  ================================================
echo   SIPV Walmart — Lancement de l'interface web
echo  ================================================
echo.
echo  Demarrage du serveur Flask...
echo  Ouvre ton navigateur sur : http://localhost:5000
echo.
echo  Pour arreter : ferme cette fenetre
echo  ================================================
echo.

cd /d "%~dp0"
python app.py

pause