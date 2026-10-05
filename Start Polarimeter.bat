@echo off
rem Starts the Ramp Polarimeter with the system Python (numpy, matplotlib, pyvisa, pyserial).
cd /d "%~dp0"
start "" pythonw polarimeter.pyw
