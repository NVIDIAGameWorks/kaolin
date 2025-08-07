# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# Exit on error
set -e

CONDA_ENV=${1:-"base"}

# parse an optional second arg WITH_GCC11 to also manually use gcc-11 within the environment
WITH_GCC11=false
if [ $# -ge 2 ]; then
    if [ "$2" = "WITH_GCC11" ]; then
        WITH_GCC11=true
    fi
fi

# Verify user arguments
echo "Arguments:"
echo "  CONDA_ENV: $CONDA_ENV"
echo "  WITH_GCC11: $WITH_GCC11"

# Test if we have GCC<=11, and early-out if not
if [ ! "$WITH_GCC11" = true ]; then
    # Make sure gcc is at most 11 for nvcc compatibility
    gcc_version=$(gcc -dumpversion)
    if [ "$gcc_version" -gt 11 ]; then
        echo "Default gcc version $gcc_version is higher than 11. See note about installing gcc-11 (you may need 'sudo apt-get install gcc-11 g++-11') and rerun with ./install_env.sh 3dgrut WITH_GCC11"
        exit 1
    fi
fi

# If we're going to set gcc11, make sure it is available
if [ "$WITH_GCC11" = true ]; then
    # Ensure gcc-11 is on path
    if ! command -v gcc-11 2>&1 >/dev/null
    then
        echo "gcc-11 could not be found. Perhaps you need to run 'sudo apt-get install gcc-11 g++-11'?"
        exit 1
    fi
    if ! command -v g++-11 2>&1 >/dev/null
    then
        echo "g++-11 could not be found. Perhaps you need to run 'sudo apt-get install gcc-11 g++-11'?"
        exit 1
    fi

    GCC_11_PATH=$(which gcc-11)
    GXX_11_PATH=$(which g++-11)
fi
GCC_VERSION=$($GCC_11_PATH -dumpversion | cut -d '.' -f 1)

# Create and activate conda environment
eval "$(conda shell.bash hook)"

# Finds the path of the environment if the environment already exists
CONDA_ENV_PATH=$(conda env list | sed -E -n "s/^${CONDA_ENV}[[:space:]]+\*?[[:space:]]*(.*)$/\1/p")
if [ -z "${CONDA_ENV_PATH}" ]; then
  echo "Conda environment '${CONDA_ENV}' not found, creating it"
  conda create --name ${CONDA_ENV} -y python=3.11
else
  echo "NOTE: Conda environment '${CONDA_ENV}' already exists at ${CONDA_ENV_PATH}, skipping environment creation"
fi
conda activate $CONDA_ENV

# Set CC and CXX variables to gcc11 in the conda env
if [ "$WITH_GCC11" = true ]; then
    echo "Setting CC=$GCC_11_PATH and CXX=$GXX_11_PATH in conda environment"

    conda env config vars set CC=$GCC_11_PATH CXX=$GXX_11_PATH

    conda deactivate
    conda activate $CONDA_ENV

    # Make sure it worked
    gcc_version=$($CC -dumpversion | cut -d '.' -f 1)
    echo "gcc_version=$gcc_version"
    if [ "$gcc_version" -gt 11 ]; then
        echo "gcc version $gcc_version is still higher than 11, setting gcc-11 failed"
        exit 1
    fi
fi

conda deactivate
conda activate $CONDA_ENV

CUDA_VERSION=`python -c "import torch; print(torch.version.cuda)"`
IFS=. read -r CUDA_MAJOR CUDA_MINOR <<< ${CUDA_VERSION}
echo "CUDA_VERSION=${CUDA_VERSION}"

# Make sure TORCH_CUDA_ARCH_LIST matches the pytorch wheel setting.
# Reference: https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_cuda.sh#L54
if [ -z ${TORCH_CUDA_ARCH_LIST+x} ]; then
    if [[ "${CUDA_MAJOR}" == "11" ]]; then
        if [[ "${CUDA_MINOR}" == "0" ]]; then
            TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0"
        elif [[ "${CUDA_MINOR}" -lt 8 ]]; then
            TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6"
        else
            TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0"
        fi
    elif [[ "${CUDA_MAJOR}" == "12" ]]; then
        if [[ "${CUDA_MINOR}" -lt 8 ]]; then
            TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
        elif [[ "${CUDA_MINOR}" == 8 ]]; then
            TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0;12.0"
        else
            TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0;12.0;12.1"
        fi
    fi
fi

echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

conda env config vars set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
conda deactivate
conda activate $CONDA_ENV


# Initialize git submodules and install Python requirements
pip install -r requirements.txt
pip install -e .

echo "Setup completed successfully!"
