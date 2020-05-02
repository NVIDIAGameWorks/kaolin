#!/bin/bash
python ${KAOLIN_HOME}/examples/renderers/softras/softras_simple_render.py \
    --stride 360 \
    --no-viz \
&& \
python ${KAOLIN_HOME}/examples/renderers/softras/softras_vertex_optimization.py \
    --iters 2\
    --no-viz \
&& \
python ${KAOLIN_HOME}/examples/renderers/softras/softras_texture_optimization.py \
    --iters 2 \
    --no-viz
