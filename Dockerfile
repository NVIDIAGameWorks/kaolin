FROM nvcr.io/nvidia/pytorch:19.11-py3

WORKDIR /kaolin
COPY . .

ENV KAOLIN_HOME "/kaolin"

RUN pip install -r requirements.txt && \
    python setup.py install
