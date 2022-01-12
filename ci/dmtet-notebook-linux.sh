#!/usr/bin/env sh
set -e
set -u

scriptroot="$( cd "$(dirname "$0")" ; pwd -P )"
cd $scriptroot/..

IMAGE_NAME=kaolin
if [ ! -z "${CI_REGISTRY_IMAGE}" ]; then
    IMAGE_NAME="${CI_REGISTRY_IMAGE}/kaolin:${CI_COMMIT_REF_SLUG}-${CI_PIPELINE_ID}"
    docker pull $IMAGE_NAME
fi

docker run \
    --runtime=nvidia \
    --rm \
    "$IMAGE_NAME" \
    /bin/bash -c "cd examples/tutorial/ && ipython dmtet_tutorial.ipynb"
