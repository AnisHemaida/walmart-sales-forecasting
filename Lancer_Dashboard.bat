@echo off
chcp 65001 >nul
title SIPV Walmart — Tableau de bord
echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║   SIPV Walmart — Tableau de bord Streamlit          ║
echo  ╚══════════════════════════════════════════════════════╝
echo.
echo  Demarrage du tableau de bord...
echo  Le navigateur va s'ouvrir automatiquement.
echo  Pour arreter : fermez cette fenetre ou appuyez sur Ctrl+C
echo.

cd /d "%~dp0"
streamlit run dashboard_walmart.py

pause
