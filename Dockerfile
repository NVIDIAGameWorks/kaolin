FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

WORKDIR /kaolin
COPY . .

ENV KAOLIN_HOME "/kaolin"
ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX"

RUN apt-get update && \
    apt-get install -y vim

RUN pip install -r requirements.txt && \
    python setup.py develop
