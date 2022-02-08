ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# used for cross-compilation in docker build
ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

WORKDIR /kaolin

COPY . .


### Install Dash3D Requirements ###
RUN conda install -c conda-forge nodejs==16.13.0 \
    && conda clean --all --force-pkgs-dirs
RUN npm install -g npm@8.5.4
RUN npm install 

RUN pip install --no-cache-dir setuptools==46.4.0 ninja cython==0.29.20

ENV KAOLIN_INSTALL_EXPERIMENTAL "1"
ENV IGNORE_TORCH_VER "1"
RUN python setup.py develop
