#!/usr/bin/env sh
set -e

script_root="$( cd "$(dirname "$0")" ; pwd -P )"

cd "$script_root/../.."

image_name=kaolin
if [ $# -gt 0 ]; then
    image_name=$1
    shift
fi

docker build \
    --network host \
    -f tools/linux/Dockerfile \
    -t ${image_name} \
    $@ \
    .

docker run --rm "${image_name}" cat conda_build.txt
