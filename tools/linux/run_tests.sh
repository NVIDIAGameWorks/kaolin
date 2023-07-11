#!/bin/bash
set -o nounset

USAGE="$0 <type of test(optional)>

Run some or all Kaolin tests, saving logs to file. Summary
will be printed at the end.

To ensure everything passes, export variables such as:
export KAOLIN_TEST_SHAPENETV2_PATH=/path/to/local/shapenet

To run all tests:
bash $0 all

To run only pytest tests:
bash $0 pytest

To run only notebooks:
bash $0 notebook

To run only recipes:
bash $0 recipes

To build the docs:
bash $0 docs
"

if [ $# -ne 1 ]; then
    echo -e "$USAGE"
    exit 1
fi

export CLI_COLOR=1
RED='\033[1;31m'
GREEN='\033[1;32m'
NOCOLOR='\033[0m'


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
KAOLIN_ROOT=$SCRIPT_DIR/../..
cd $KAOLIN_ROOT
KAOLIN_ROOT=`pwd`

LOG_DIR=$KAOLIN_ROOT/.test_logs
mkdir -p $LOG_DIR

RUN_PYTEST=0
RUN_NOTEBOOK=0
RUN_RECIPES=0
BUILD_DOCS=0
if [ $1 == "all" ]; then
    RUN_PYTEST=1
    RUN_NOTEBOOK=1
    RUN_RECIPES=1
    BUILD_DOCS=1
elif [ $1 == "pytest" ]; then
    RUN_PYTEST=1
elif [ $1 == "notebook" ]; then
    RUN_NOTEBOOK=1
elif [ $1 == "recipes" ]; then
    RUN_RECIPES=1
elif [ $1 == "docs" ]; then
    BUILD_DOCS=1
else
    echo "$RED Unknown argument type $1 $NOCOLOR"
    echo -e "$USAGE"
    exit 1
fi

start_test_info() {
    echo "***********************************************"
    echo "             Running $1 Tests              "
    echo "***********************************************"
    echo
    echo " ...running, see log: $LOG_DIR/log_$1.txt"
}

STATUS=0
end_test_info() {
    if [ $1 -ne 0 ]; then
        STATUS=1
        echo -e "$RED FAILED: $NOCOLOR $2"
    else
        echo -e "$GREEN SUCCESS: $NOCOLOR $2"
    fi
    echo
}

maybe_open_url() {
    which xdg-open
    if [ $? -eq 0 ]; then
        xdg-open $1
    fi
}

PYTEST_LOG=$LOG_DIR/log_pytest.txt
if [ $RUN_PYTEST -eq "1" ]; then
    echo "" > $PYTEST_LOG
    start_test_info "pytest"

    CMDLINE="pytest --import-mode=importlib --cov=kaolin -s --cov-report=html --cov-report term-missing  tests/python/"
    $CMDLINE >> $PYTEST_LOG 2>&1
    RES=$?
    COV_URL=".test_coverage/index.html"
    echo "                      HTML line-by-line test coverage available in $COV_URL"
    end_test_info $RES "$CMDLINE"
    maybe_open_url $COV_URL >> $PYTEST_LOG 2>&1
fi


NOTEBOOK_LOG=$LOG_DIR/log_notebook.txt
if [ $RUN_NOTEBOOK -eq "1" ]; then
    echo "" > $NOTEBOOK_LOG
    start_test_info "notebook"

    CMDLINE="pytest --nbmake --nbmake-timeout=3000 examples/**/*.ipynb"
    $CMDLINE >> $NOTEBOOK_LOG 2>&1

    end_test_info $? "$CMDLINE"
fi

RECIPES_LOG=$LOG_DIR/log_recipes.txt
if [ $RUN_RECIPES -eq "1" ]; then
    echo "" > $RECIPES_LOG
    start_test_info "recipes"

    NFAIL=0
    NPASS=0

    cd $KAOLIN_ROOT/examples/recipes
    for F in $(find . -name "*.py" | grep -v "ipynb_checkpoints"); do

        echo "Executing python $F" >> $RECIPES_LOG
        python $F >> $RECIPES_LOG 2>&1
        RES=$?
        if [ $RES -ne 0 ]; then
            echo -e "$RED     failed : $NOCOLOR python $F"
            NFAIL=$((NFAIL+1))
        else
            echo -e "$GREEN     success: $NOCOLOR python $F"
            NPASS=$((NPASS+1))
        fi
    done

    end_test_info $NFAIL "python examples/recipes/**/*.py"
fi


DOCS_LOG=$LOG_DIR/log_docs.txt
if [ $BUILD_DOCS -eq "1" ]; then
    echo "" > $DOCS_LOG
    start_test_info "docs"

    cd $KAOLIN_ROOT
    rm -rf $KAOLIN_ROOT/docs/_build

    echo " ...copying docs/ to build_docs/ to avoid git confusion" >> $DOCS_LOG 2>&1
    mkdir -p build_docs
    cp -r docs/* build_docs/.
    cd build_docs
    echo " ...replacing DOCS_MODULE_PATH in build_docs/kaolin_ext.py" >> $DOCS_LOG 2>&1
    sed -i 's/"docs"/"build_docs"/g' kaolin_ext.py >> $DOCS_LOG 2>&1

    echo " ...building docs in build_docs dir" >> $DOCS_LOG 2>&1
    CMDLINE="python -m sphinx -T -E -W --keep-going -b html -d _build/doctrees -D language=en . _build/html"
    $CMDLINE >> $DOCS_LOG 2>&1
    RES=$?

    cd $KAOLIN_ROOT
    DOCS_URL="build_docs/_build/html/index.html"
    echo "    HTML written to $DOCS_URL"

    end_test_info $RES "$CMDLINE"
    maybe_open_url $DOCS_URL >> $DOCS_LOG 2>&1
fi
