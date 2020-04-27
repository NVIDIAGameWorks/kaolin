#!/bin/bash
python ${KAOLIN_HOME}/examples/Classification/pointcloud_classification.py \
    --modelnet-root /data/ModelNet10/ \
    --categories desk dresser \
    --epochs 1 --transforms-device cuda \
&& \
python ${KAOLIN_HOME}/examples/Classification/pointcloud_classification.py \
    --modelnet-root /data/ModelNet10/ \
    --categories desk dresser \
    --epochs 1 --transforms-device cpu \
&& \
python ${KAOLIN_HOME}/examples/Classification/mesh_classification.py \
    --shrec-root /data/SHREC16/ \
    --categories ants cat \
    --epochs 1
