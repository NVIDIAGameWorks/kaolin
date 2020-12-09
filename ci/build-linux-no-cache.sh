#!/usr/bin/env sh
set -e

script_root="$( cd "$(dirname "$0")" ; pwd -P )"

image_name=kaolin
if [ ! -z "${CI_REGISTRY_IMAGE}" ]; then
    image_name="${CI_REGISTRY_IMAGE}/kaolin:${CI_COMMIT_REF_SLUG}-${CI_PIPELINE_ID}"
fi

$script_root/../tools/linux/build_docker.sh $image_name --no-cache

if [ ! -z "${CI_REGISTRY_IMAGE}" ]; then
    docker push $image_name
fi
