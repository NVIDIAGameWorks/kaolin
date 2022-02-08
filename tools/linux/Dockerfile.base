ARG CUDA_VERSION=11.3.1
ARG CUDNN_VERSION=8
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04

ARG PYTHON_VERSION=3.9
ENV PYTHON_VERSION=${PYTHON_VERSION}
ARG PYTORCH_VERSION=1.10.2
ENV PYTORCH_VERSION=${PYTORCH_VERSION}

# Note: removing this line as it seems to cause a lot of problems.
# RUN sed -i -e 's|^deb http://[^.]*[.]ubuntu[.]com/ubuntu|deb  https://urm.nvidia.com/artifactory/ubuntu-remote|' /etc/apt/sources.list \
RUN echo "Acquire { https::Verify-Peer false }" > /etc/apt/apt.conf.d/99verify-peer.conf \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ca-certificates \
    && rm /etc/apt/apt.conf.d/99verify-peer.conf \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         build-essential curl git unzip gfortran \
         libgtk2.0-0 libgtk-3-0 libgbm-dev libnotify-dev libgconf-2-4 \
         libnss3 libxss1 libasound2 libxtst6 xauth xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN export CUDA_MAJOR="$(echo ${CUDA_VERSION} | cut -d'.' -f1)" \
    && export CUDA_MINOR="$(echo ${CUDA_VERSION} | cut -d'.' -f2)" \
    && conda install -y pytorch==${PYTORCH_VERSION} cudatoolkit=${CUDA_MAJOR}.${CUDA_MINOR} -c pytorch -c conda-forge \
    && conda clean -y --all --force-pkgs-dirs

RUN conda install -c conda-forge nodejs==16.13.0 \
    && conda clean --all --force-pkgs-dirs

RUN conda list > conda_build.txt

### Install Dash3D Requirements ###
WORKDIR /tmp
RUN npm install -g npm@8.5.4
COPY package.json package-lock.json ./
RUN npm install

RUN pip install --no-cache-dir ninja setuptools==46.4.0 numpy==1.18.0
