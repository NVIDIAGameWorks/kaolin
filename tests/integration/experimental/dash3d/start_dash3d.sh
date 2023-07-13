#!/bin/bash -e
set -o nounset

# Note: when run as a subprocess something is setting this
# variable, which causes issues; printing for debug information
# and unsetting
if [[ -v MKL_THREADING_LAYER ]];
then
  echo "Unsetting MKL_THREADING_LAYER=$MKL_THREADING_LAYER"
  unset MKL_THREADING_LAYER
fi

# Get the directory where current script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
KAOLIN_ROOT=$SCRIPT_DIR/../../../..

DASH3D=kaolin-dash3d

USAGE="$0 [log_directory] (optional: port)

Runs dash3d in the background using script:
$DASH3D
"
if [ $# -lt 1 ]; then
    echo "$USAGE"
    exit
fi

FLAGS="--logdir=$1 --log_level=10"  # DEBUG
if [ $# -gt 1 ]; then
  FLAGS="$FLAGS --port=$2"
fi

echo "Running Dash3D in the background using command: "
echo "$DASH3D $FLAGS"

$DASH3D $FLAGS &
PID=$!

sleep 2
set +e
kill -0 $PID  # Check still runs
if [ "$?" -ne "0" ]; then
  echo "Failed to start dash3d"
  exit 1
fi
