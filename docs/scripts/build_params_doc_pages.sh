#!/bin/bash

set -e

TMP=$(mktemp -d)

APPLICATION=$1
SRC_DIR=$2

shift;

INPUT_FILE="${SRC_DIR}/docs/input_files/${APPLICATION}_example.inp"
HTML_OUTPUT="html/${APPLICATION}_params.html"
# We need to copy the "output" file, since we're really modifying it in place
# and that doesn't let `make` pick up on the rebuilds correctly.
#
# It has to be modified in place because of the way that the doxygen refs work.
HTML_COPY_OUTPUT="html/${APPLICATION}_params_copy.html"

./warpii --print-parameters $INPUT_FILE > $TMP/params.json

cat $TMP/params.json | ${SRC_DIR}/docs/scripts/process_params_json.py $INPUT_FILE > $TMP/contents.html
python3 ${SRC_DIR}/docs/scripts/replace_doxygen_html_contents.py $HTML_OUTPUT $TMP/contents.html > $TMP/result.html 
cat $TMP/result.html > $HTML_OUTPUT

cp $HTML_OUTPUT $HTML_COPY_OUTPUT
