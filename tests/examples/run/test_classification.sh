#!/bin/bash
python ${KAOLIN_HOME}/examples/Classification/pointcloud_classification.py \
    --modelnet-root /data/ModelNet10/ \
    --categories desk dresser \
    --epochs 1 --transforms-device cuda

python ${KAOLIN_HOME}/examples/Classification/pointcloud_classification.py \
    --modelnet-root /data/ModelNet10/ \
    --categories desk dresser \
    --epochs 1 --transforms-device cpu
