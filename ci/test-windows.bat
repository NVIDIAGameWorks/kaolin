call conda activate kaolin
mkdir test
cd test

python -m pip install --force-reinstall --find-links=..\artifacts kaolin
IF NOT "%ERRORLEVEL%"=="0" EXIT /b %ERRORLEVEL%

python -c "import kaolin; print(kaolin.__version__)"
IF NOT "%ERRORLEVEL%"=="0" EXIT /b %ERRORLEVEL%
