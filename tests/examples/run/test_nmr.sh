#!/bin/bash
python ${KAOLIN_HOME}/examples/renderers/NMR/example1.py \
    || exit 1
python ${KAOLIN_HOME}/examples/renderers/NMR/example2.py \
       --epochs 1 \
    || exit 1
python ${KAOLIN_HOME}/examples/renderers/NMR/example3.py \
       --epochs 1 \
    || exit 1
python ${KAOLIN_HOME}/examples/renderers/NMR/example4.py \
       --epochs 1 \
    || exit 1
