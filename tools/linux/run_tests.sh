#!/bin/bash
set -o nounset

USAGE="$0 <type of test> [--no-open] [--v] [--q]

Run some or all Kaolin tests, saving logs to file. Summary
will be printed at the end.

Pass --no-open to skip auto-opening any generated HTML report
(coverage / docs) in the browser, handy when rerunning until green.

By default, the log file of any stage that fails is dumped to stdout.
This is useful in CI, where the per-stage logs aren't otherwise
visible in the pipeline output.

Pass --v (verbose) to dump the log file of every stage (whether it
passes or fails) to stdout.

Pass --q (quiet) to suppress dumping logs to stdout entirely; logs are
still written to their per-stage files.

--v and --q are mutually exclusive.

To ensure everything passes, export variables such as:
export KAOLIN_TEST_SHAPENETV2_PATH=/path/to/local/shapenet

To run all tests:
bash $0 all

To run only pytest tests:
bash $0 pytest

To run only TypeScript (npm) tests:
bash $0 ts

To run only notebooks:
bash $0 notebook

To run only recipes:
bash $0 recipes

To build the docs:
bash $0 docs
"

if [ $# -lt 1 ]; then
    echo -e "$USAGE"
    exit 1
fi

export CLI_COLOR=1
RED='\033[1;31m'
GREEN='\033[1;32m'
NOCOLOR='\033[0m'

# First positional arg is the test type; the rest are optional flags (any order).
TYPE=$1
shift

OPEN_URLS=1
VERBOSE=0
QUIET=0
while [ $# -gt 0 ]; do
    case "$1" in
        --no-open)
            OPEN_URLS=0
            ;;
        --v)
            VERBOSE=1
            ;;
        --q)
            QUIET=1
            ;;
        *)
            echo -e "$RED Unknown option $1 $NOCOLOR"
            echo -e "$USAGE"
            exit 1
            ;;
    esac
    shift
done

if [ $VERBOSE -eq 1 ] && [ $QUIET -eq 1 ]; then
    echo -e "$RED --v and --q are mutually exclusive $NOCOLOR"
    echo -e "$USAGE"
    exit 1
fi


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
KAOLIN_ROOT=$SCRIPT_DIR/../..
cd $KAOLIN_ROOT
KAOLIN_ROOT=`pwd`

LOG_DIR=$KAOLIN_ROOT/.test_logs
mkdir -p $LOG_DIR

RUN_PYTEST=0
RUN_TS=0
RUN_NOTEBOOK=0
RUN_RECIPES=0
BUILD_DOCS=0
if [ $TYPE == "all" ]; then
    RUN_PYTEST=1
    RUN_TS=1
    RUN_NOTEBOOK=1
    RUN_RECIPES=1
    BUILD_DOCS=1
elif [ $TYPE == "pytest" ]; then
    RUN_PYTEST=1
elif [ $TYPE == "ts" ]; then
    RUN_TS=1
elif [ $TYPE == "notebook" ]; then
    RUN_NOTEBOOK=1
elif [ $TYPE == "recipes" ]; then
    RUN_RECIPES=1
elif [ $TYPE == "docs" ]; then
    BUILD_DOCS=1
else
    echo "$RED Unknown argument type $TYPE $NOCOLOR"
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
# Usage: end_test_info <exit_code> <description> [log_file]
# Dumps log_file (if given) to stdout BEFORE the SUCCESS/FAILED message, so the
# result line -- which includes the log location -- is the last thing printed.
# Log printing rules: --v (verbose) dumps the log of every stage; otherwise
# (the default) only a failing stage's log is dumped, so failures are visible
# where per-stage logs aren't (e.g. CI pipeline output); --q (quiet) suppresses
# all log dumping.
end_test_info() {
    local code=$1
    local desc=$2
    local log_file=${3:-}

    local print_log=0
    if [ $VERBOSE -eq 1 ]; then
        print_log=1
    elif [ $QUIET -eq 0 ] && [ $code -ne 0 ]; then
        print_log=1
    fi
    if [ $print_log -eq 1 ] && [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo "---------------- begin log: $log_file ----------------"
        cat "$log_file"
        echo "----------------- end log: $log_file -----------------"
    fi

    if [ $code -ne 0 ]; then
        STATUS=1
        echo -e "$RED FAILED: $NOCOLOR $desc"
    else
        echo -e "$GREEN SUCCESS: $NOCOLOR $desc"
    fi
    if [ -n "$log_file" ]; then
        echo "         see log: $log_file"
    fi
    echo
}

maybe_open_url() {
    if [ $OPEN_URLS -eq 0 ]; then
        return 0
    fi
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
    end_test_info $RES "$CMDLINE" "$PYTEST_LOG"
    maybe_open_url $COV_URL >> $PYTEST_LOG 2>&1
fi


TS_LOG=$LOG_DIR/log_ts.txt
if [ $RUN_TS -eq "1" ]; then
    echo "" > $TS_LOG
    start_test_info "ts"

    cd $KAOLIN_ROOT
    CMDLINE="npm run test:coverage"
    $CMDLINE >> $TS_LOG 2>&1
    RES=$?
    TS_COV_URL=".test_coverage_web/index.html"
    echo "                      HTML line-by-line test coverage available in $TS_COV_URL"
    end_test_info $RES "$CMDLINE" "$TS_LOG"
    maybe_open_url $TS_COV_URL >> $TS_LOG 2>&1
fi


NOTEBOOK_LOG=$LOG_DIR/log_notebook.txt
if [ $RUN_NOTEBOOK -eq "1" ]; then
    echo "" > $NOTEBOOK_LOG
    start_test_info "notebook"

    CMDLINE="pytest --nbmake --nbmake-timeout=3000 examples/**/*.ipynb"
    $CMDLINE >> $NOTEBOOK_LOG 2>&1

    end_test_info $? "$CMDLINE" "$NOTEBOOK_LOG"
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

    end_test_info $NFAIL "python examples/recipes/**/*.py" "$RECIPES_LOG"
fi


DOCS_LOG=$LOG_DIR/log_docs.txt
if [ $BUILD_DOCS -eq "1" ]; then
    echo "" > $DOCS_LOG
    start_test_info "docs"

    cd $KAOLIN_ROOT
    rm -rf $KAOLIN_ROOT/docs/_build

    echo " ...copying docs/ to build_docs/ to avoid git confusion" >> $DOCS_LOG 2>&1
    mkdir -p build_docs
    rm -rf build_docs/*
    cp -r docs/* build_docs/.
    cd build_docs
    echo " ...replacing DOCS_MODULE_PATH in build_docs/kaolin_ext.py" >> $DOCS_LOG 2>&1
    sed -i 's/"docs"/"build_docs"/g' kaolin_ext.py >> $DOCS_LOG 2>&1

    echo " ...building docs in build_docs dir" >> $DOCS_LOG 2>&1
    # -W --keep-going: treat Sphinx warnings as errors (collecting them all) to match
    # readthedocs' fail_on_warning, so a warning fails the build.
    CMDLINE='make html SPHINXOPTS="-W --keep-going"'
    export PYTORCH_JIT=0
    make html SPHINXOPTS="-W --keep-going" >> $DOCS_LOG 2>&1
    RES=$?
    export PYTORCH_JIT=1

    cd $KAOLIN_ROOT
    DOCS_URL="build_docs/_build/html/index.html"
    echo "    HTML written to $DOCS_URL"

    end_test_info $RES "$CMDLINE" "$DOCS_LOG"
    maybe_open_url $DOCS_URL >> $DOCS_LOG 2>&1
fi

# Propagate failure: STATUS is set to 1 by end_test_info whenever any stage failed,
# so callers (e.g. CI) get a non-zero exit code.
exit $STATUS
