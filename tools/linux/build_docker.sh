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
[ -d artifacts ] && rm -rf artifacts

mkdir dist/
docker run --rm --network=host --ipc=host -v `pwd`/dist:/dist "${image_name}" \
    /bin/bash -c "export KAOLIN_INSTALL_EXPERIMENTAL=1; python setup.py bdist_wheel && python -m pip install --find-links=dist kaolin"

mv -v dist artifacts
