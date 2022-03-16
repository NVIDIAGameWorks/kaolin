ARG CUDA_VERSION=10.1
ARG CUDNN_VERSION=7
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04

# used for cross-compilation in docker build
ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

# Note: removing this line as it seems to cause a lot of problems.
# RUN sed -i -e 's|^deb http://[^.]*[.]ubuntu[.]com/ubuntu|deb  https://urm.nvidia.com/artifactory/ubuntu-remote|' /etc/apt/sources.list \
RUN echo "Acquire { https::Verify-Peer false }" > /etc/apt/apt.conf.d/99verify-peer.conf \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ca-certificates \
    && rm /etc/apt/apt.conf.d/99verify-peer.conf \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         build-essential \
         curl \
         git \
         unzip \
         libgtk2.0-0 libgtk-3-0 libgbm-dev libnotify-dev libgconf-2-4 libnss3 libxss1 libasound2 libxtst6 xauth xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION=3.7
ENV PYTHON_VERSION=${PYTHON_VERSION}
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     sh ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

ARG PYTORCH_VERSION=1.6
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
RUN conda install -y pytorch==$PYTORCH_VERSION cudatoolkit=${CUDA_VERSION} -c pytorch \
    && conda clean --all --force-pkgs-dirs

RUN conda install -c conda-forge nodejs==16.13.1 \
    && conda clean --all --force-pkgs-dirs

WORKDIR /kaolin

RUN conda list > conda_build.txt

### Install CI Requirements ###
COPY tools/ci_requirements.txt tools/ci_requirements.txt
RUN pip install --no-cache-dir -r tools/ci_requirements.txt

### Install Dash3D CI Requirements ###
RUN npm install -g npm@8.5.4
COPY package.json package-lock.json ./
RUN npm install
# Note: do not run npm cache clean

#################################### Build Kaolin ####################################
COPY . .

ENV TORCH_CUDA_ARCH_LIST "5.2 6.0 6.1 7.0 7.5+PTX"
ENV KAOLIN_INSTALL_EXPERIMENTAL "1"
RUN python setup.py develop
