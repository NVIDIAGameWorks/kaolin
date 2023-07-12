# Note: Keep C:\Program Files (x86)\Microsoft Visual Studio\Installer directory.
#       disutils/setuptools calls vswhere.exe from that location unconditionally.

FROM mcr.microsoft.com/dotnet/framework/sdk:4.8-windowsservercore-ltsc2019

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# set up conda and python environment
ARG CUDA_VERSION=11.1
ENV CUDA_VERSION=${CUDA_VERSION}
ARG PYTHON_VERSION=3.7
ENV PYTHON_VERSION=${PYTHON_VERSION}
ARG PYTORCH_VERSION=1.8
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

ARG CUDA_URL="http://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_456.81_win10.exe"
ENV CUDA_URL=${CUDA_URL}
ARG CUDA_FILENAME=cuda_${CUDA_VERSION}_win10.exe
ENV CUDA_FILENAME=${CUDA_FILENAME}


#### BUILD TOOLS ####

# pulling downloads to top to not repeat
# download & install VS build tools 2019
RUN Invoke-WebRequest "https://aka.ms/vs/16/release/vs_buildtools.exe" -OutFile vs_buildtools.exe -UseBasicParsing; \
    Start-Process -FilePath 'vs_buildtools.exe' -Wait -ArgumentList '--quiet', '--norestart', '--downloadThenInstall', '--nocache', '--wait', '--installPath', 'C:\BuildTools', '--add', 'Microsoft.VisualStudio.Workload.VCTools'; \
    Start-Process -FilePath 'vs_buildtools.exe' -Wait -ArgumentList '--quiet', '--norestart', '--downloadThenInstall', '--nocache', '--wait', '--installPath', 'C:\BuildTools', '--add', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'; \
    Start-Process -FilePath 'vs_buildtools.exe' -Wait -ArgumentList '--quiet', '--norestart', '--downloadThenInstall', '--nocache', '--wait', '--installPath', 'C:\BuildTools', '--add', 'Microsoft.VisualStudio.Workload.MSBuildTools'; \
    Start-Process -FilePath 'vs_buildtools.exe' -Wait -ArgumentList '--quiet', '--norestart', '--downloadThenInstall', '--nocache', '--wait', '--installPath', 'C:\BuildTools', '--add', 'Microsoft.Component.VC.Runtime.UCRTSDK'; \
    Start-Process -FilePath 'vs_buildtools.exe' -Wait -ArgumentList '--quiet', '--norestart', '--downloadThenInstall', '--nocache', '--wait', '--installPath', 'C:\BuildTools', '--add', 'Microsoft.VisualStudio.Component.Windows10SDK.18362'; \
    Remove-Item .\vs_buildtools.exe


# The msvc version (subdir) has been shown to change over time.
# The winsdk version (subdir) may change over time either through changes by MS or our own arguments to vs_buildtools.exe above.
# Keeping that in mind, let's dynamically pick up the paths which get created.
RUN setx /M PATH $('c:\buildtools\vc\tools\msvc\' + $(Get-ChildItem -Path 'c:\buildtools\vc\tools\msvc' -Force -Directory | Select-Object -First 1).Name + '\bin\hostx64\x64;' \
    + 'c:\buildtools\vc\tools\msvc\' + $(Get-ChildItem -Path 'c:\buildtools\vc\tools\msvc' -Force -Directory | Select-Object -First 1).Name + '\lib\x64;' \
    + 'c:\buildtools\vc\tools\msvc\' + $(Get-ChildItem -Path 'c:\buildtools\vc\tools\msvc' -Force -Directory | Select-Object -First 1).Name + '\include;' \
    + 'C:\Program Files (x86)\Windows Kits\10\bin\' + $(Get-ChildItem -Path 'C:\Program Files (x86)\Windows Kits\10\bin\' -Force -Directory | Select-Object -First 1).Name + '\x64;' \
    + 'C:\BuildTools\VC\Auxiliary\Build;' \
    + $Env:PATH)

#### CUDA ####

# needed:
# nvcc
# cusparse
# cusparse_dev
# cublas
# cublas_dev
# curand
# curand_dev
# cudart_11.1  # for vector_types.h
# cusolver_11.1
# cusolver_dev_11.1
# cuda_thrust_11.3 # if CUDA >= 11.3, needed to build

# install CUDA toolkit
# TODO: may need to add more packages or alter names for later CUDA versions...
# CUDA 11.3+ adds 'thrust' to the set of requirements we need to install
RUN Invoke-WebRequest $Env:CUDA_URL -OutFile $Env:CUDA_FILENAME -UseBasicParsing; \
    if ([convert]::ToDouble(${Env:CUDA_VERSION}) -ge 11.3) { Start-Process -FilePath \"$Env:CUDA_FILENAME\" -Wait -ArgumentList '-s', \"nvcc_${Env:CUDA_VERSION}\", \"cusparse_${Env:CUDA_VERSION}\", \"cusparse_dev_${Env:CUDA_VERSION}\", \"cublas_${Env:CUDA_VERSION}\", \"cublas_dev_${Env:CUDA_VERSION}\", \"curand_${Env:CUDA_VERSION}\", \"curand_dev_${Env:CUDA_VERSION}\", \"cudart_${Env:CUDA_VERSION}\", \"cusolver_${Env:CUDA_VERSION}\", \"cusolver_dev_${Env:CUDA_VERSION}\", \"thrust_${Env:CUDA_VERSION}\" } \
    else { Start-Process -FilePath \"$Env:CUDA_FILENAME\" -Wait -ArgumentList '-s', \"nvcc_${Env:CUDA_VERSION}\", \"cusparse_${Env:CUDA_VERSION}\", \"cusparse_dev_${Env:CUDA_VERSION}\", \"cublas_${Env:CUDA_VERSION}\", \"cublas_dev_${Env:CUDA_VERSION}\", \"curand_${Env:CUDA_VERSION}\", \"curand_dev_${Env:CUDA_VERSION}\", \"cudart_${Env:CUDA_VERSION}\", \"cusolver_${Env:CUDA_VERSION}\", \"cusolver_dev_${Env:CUDA_VERSION}\"; } \
    Remove-Item $Env:CUDA_FILENAME

# TODO: can we use conda cuda install instead? only once we need cuda 11.3.0+
# RUN conda install -y cuda cuda-nvcc -c nvidia/label/cuda-11.3.1
# RUN setx /M PATH $('C:\Users\Administrator\miniconda3\bin;C:\Users\Administrator\miniconda3\include;' + $Env:Path)


#### ANACONDA ####
RUN Invoke-WebRequest "https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile miniconda3.exe -UseBasicParsing;

# install conda & python packages
# PATH must be updated before attempting to run "conda" or it will error out strangely on web requests
RUN setx /M PATH $('C:\Users\Administrator\miniconda3\Library\bin;C:\Users\Administrator\miniconda3\Scripts;C:\Users\Administrator\miniconda3;' + $Env:PATH)
RUN Start-Process -FilePath 'miniconda3.exe' -Wait -ArgumentList '/S', '/D=C:\Users\Administrator\miniconda3'; \
    Remove-Item .\miniconda3.exe; \
    conda install -y python=$Env:PYTHON_VERSION

# NOTE: update the install list (2 lines here) if SSL_CERTIFICATE errors crop up to pip install the related dependency
# conditional: bespoke logic for 1.8.2 LTS install, otherwise generic wheel lookup
RUN $Env:TORCH_STR = ${Env:PYTORCH_VERSION} + '+cu' + ${Env:CUDA_VERSION}.Replace('.',''); \
    $Env:TORCH_URL='https://download.pytorch.org/whl/cu' + ${Env:CUDA_VERSION}.Replace('.',''); \
    pip install --no-cache-dir torch==${Env:TORCH_STR} ninja --extra-index-url ${Env:TORCH_URL};

RUN pip install brotli

# CUDA setup for running & testing on CI machine
COPY tools/windows/SetupDriver.ps1 C:/data/
COPY tools/windows/Test.ps1 C:/data/
COPY tools/windows/main.ps1 C:/data/
COPY tools/windows/deviceQuery.exe c:/data/
COPY tools/windows/bandwidthTest.exe c:/data/

# #### Kaolin ####

WORKDIR /kaolin

RUN conda list > conda_build.txt

COPY . .

# NOTE: pin setuptools to avoid directory copy bug
RUN pip install --upgrade --no-cache-dir setuptools==58.0.0 certifi
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org -r tools\ci_requirements.txt \
    -r tools\viz_requirements.txt -r tools\requirements.txt -r tools/build_requirements.txt

ENV TORCH_CUDA_ARCH_LIST "7.0 7.5+PTX"
RUN python setup.py develop

# fix for paging memory issue on CI machines
#   see: https://gist.github.com/cobryan05/7d1fe28dd370e110a372c4d268dcb2e5
RUN pip install pefile
COPY tools/fixNvPe.py c:/data/

# pip3 install --upgrade certifi
