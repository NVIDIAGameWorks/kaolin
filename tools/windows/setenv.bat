@echo off
set KAOLIN_HOME=%~dp0..\..\
set PYTHONPATH=%PYTHONPATH%;%KAOLIN_HOME%\_build\target-deps\nv_usd\release\lib\python
set PATH=%PATH%;%KAOLIN_HOME%\_build\target-deps\nv_usd\release\lib\
:Success
exit /b 0
