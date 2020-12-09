# Note: Keep C:\Program Files (x86)\Microsoft Visual Studio\Installer directory.
#       disutils/setuptools calls vswhere.exe from that location unconditionally.

# Build Kaolin Windows Base
FROM mcr.microsoft.com/dotnet/framework/sdk:4.8-windowsservercore-ltsc2019

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]


RUN Invoke-WebRequest "https://aka.ms/vs/16/release/vs_buildtools.exe" -OutFile vs_buildtools.exe -UseBasicParsing; \
    Start-Process -FilePath 'vs_buildtools.exe' -Wait -ArgumentList '--quiet', '--norestart', '--nocache', '--wait', '--installPath', 'C:\BuildTools', '--add', 'Microsoft.VisualStudio.Workload.VCTools', '--add', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '--add', 'Microsoft.VisualStudio.Workload.MSBuildTools', '--add', 'Microsoft.Component.VC.Runtime.UCRTSDK', '--add', 'Microsoft.VisualStudio.Component.Windows10SDK.18362'; \
    Remove-Item .\vs_buildtools.exe

ARG CUDA_VERSION=10.1
ENV CUDA_VERSION=${CUDA_VERSION}
ARG PYTHON_VERSION=3.6
ENV PYTHON_VERSION=${PYTHON_VERSION}
ARG PYTORCH_VERSION=1.6
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

# PATH must be updated before attempting to run "conda" or it will error out strangely on web requests
RUN setx /M PATH $('C:\Users\Administrator\miniconda3\Library\bin;C:\Users\Administrator\miniconda3\Scripts;C:\Users\Administrator\miniconda3;' + $Env:PATH)
RUN Invoke-WebRequest "https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile miniconda3.exe -UseBasicParsing; \
    Start-Process -FilePath 'miniconda3.exe' -Wait -ArgumentList '/S', '/D=C:\Users\Administrator\miniconda3'; \
    Remove-Item .\miniconda3.exe; \
    conda install -y python=3.6 7za pytorch=$Env:PYTORCH_VERSION cudatoolkit=$Env:CUDA_VERSION cython -c pytorch

ARG CUDA_FILENAME=cuda_10.1.243_426.00_win10.exe
RUN Invoke-WebRequest "http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/$Env:CUDA_FILENAME" -OutFile $Env:CUDA_FILENAME -UseBasicParsing; \
    Start-Process -FilePath "$Env:CUDA_FILENAME" -Wait -ArgumentList '-s', 'nvcc_10.1', 'cusparse_10.1', 'cusparse_dev_10.1', 'cublas_10.1', 'cublas_dev_10.1', 'curand_10.1', 'curand_dev_10.1'; \
    \
    7za x $Env:CUDA_FILENAME universal\Display.Driver\nvcuda64.dl_ universal\Display.Driver\nvfatbinaryloader64.dl_; \
    \
    expand universal\Display.Driver\nvcuda64.dl_ c:\windows\system32\nvcuda.dll; \
    expand universal\Display.Driver\nvfatbinaryloader64.dl_ c:\windows\system32\nvfatbinaryloader.dll; \
    Remove-Item -Path universal -Recurse -Confirm:$false -Force; \
    del $Env:CUDA_FILENAME

# The msvc version (subdir) has been shown to change over time.
# The winsdk version (subdir) may change over time either through changes by MS or our own arguments to vs_buildtools.exe above.
# Keeping that in mind, let's dynamically pick up the paths which get created.
RUN setx /M PATH $('c:\buildtools\vc\tools\msvc\' + $(Get-ChildItem -Path 'c:\buildtools\vc\tools\msvc' -Force -Directory | Select-Object -First 1).Name + '\bin\hostx64\x64;' \
    + 'C:\Program Files (x86)\Windows Kits\10\bin\' + $(Get-ChildItem -Path 'C:\Program Files (x86)\Windows Kits\10\bin\' -Force -Directory | Select-Object -First 1).Name + '\x64;' \
    + 'C:\BuildTools\VC\Auxiliary\Build;' \
    + $Env:PATH)

WORKDIR /kaolin

RUN conda list > conda_build.txt

#################################### Build Kaolin ####################################

COPY . .

RUN pip install -r tools\ci_requirements.txt

ENV TORCH_CUDA_ARCH_LIST "7.0 7.5+PTX"
RUN python setup.py bdist_wheel

ENTRYPOINT ["powershell.exe", "-Command"]
