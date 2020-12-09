#${KAOLIN_ROOT}="$( cd "$(dirname "$0")" ; pwd -P ).."
KAOLIN_ROOT="$( cd "$(dirname "$0")/.." ; pwd -P )"
# files ending by *.so are custom cuda/c++/cython generated files
# not to be used by end users
# on /kaolin/ops/mesh/ we only want to documented the main module,
# so we exclude all modules except the __init__.py containing __all__
EXCLUDE_PATHS="${KAOLIN_ROOT}/**.so \
${KAOLIN_ROOT}/kaolin/ops/conversions/[!_][!_]*.py \
${KAOLIN_ROOT}/kaolin/ops/mesh/[!_][!_]*.py \
${KAOLIN_ROOT}/kaolin/render/mesh/[!_][!_]*.py"


# Those files are unused since we already have index.rst and conf.py
EXCLUDE_GEN_RST="${KAOLIN_ROOT}/docs/modules/setup.rst \
${KAOLIN_ROOT}/docs/modules/kaolin.rst \
${KAOLIN_ROOT}/docs/modules/kaolin.version.rst"

sphinx-apidoc -eT -d 2 --templatedir=${KAOLIN_ROOT}/docs/modules/ -o ${KAOLIN_ROOT}/docs/modules/ ${KAOLIN_ROOT} ${EXCLUDE_PATHS}

rm ${EXCLUDE_GEN_RST}
