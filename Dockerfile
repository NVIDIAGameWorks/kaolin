ARG CUDA_VERSION=10.1
FROM nvidia/cuda:$CUDA_VERSION-cudnn7-devel

ARG PYTHON_VERSION=3.6
ARG PYTORCH_VERSION=1.2

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         vim && \
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH


RUN pip install torch==$PYTORCH_VERSION && \
    pip install torchvision==0.5.0

WORKDIR /kaolin
COPY . .

ENV KAOLIN_HOME "/kaolin"

RUN pip install -r requirements.txt
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
    python setup.py install
