ARG CUDA_VERSION=10.1
ARG CUDNN_VERSION=7
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04

ARG PYTHON_VERSION=3.7
ENV PYTHON_VERSION=${PYTHON_VERSION}
ARG PYTORCH_VERSION=1.6
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
# used for cross-compilation in docker build
ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates

RUN sed -i -e 's|http[s]*://.*ubuntu.com/ubuntu/ |https://urm.nvidia.com/artifactory/ubuntu-remote |' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         curl \
         git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     sh ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN conda install -y pytorch==$PYTORCH_VERSION cudatoolkit=${CUDA_VERSION} -c pytorch

WORKDIR /kaolin

RUN conda list > conda_build.txt

#################################### Build Kaolin ####################################

COPY . .

RUN pip install -r tools/ci_requirements.txt

ENV TORCH_CUDA_ARCH_LIST "5.2 6.0 6.1 7.0 7.5+PTX"
RUN python setup.py develop
