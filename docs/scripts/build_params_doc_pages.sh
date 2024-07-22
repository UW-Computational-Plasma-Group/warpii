#!/bin/bash

set -e

TMP=$(mktemp -d)

./warpii --print-parameters docs/input_files/five_moment_example.inp > /dev/null

./warpii --print-parameters docs/input_files/five_moment_example.inp | docs/scripts/process_params_json.py docs/input_files/five_moment_example.inp > $TMP/contents.html
python3 docs/scripts/replace_doxygen_html_contents.py html/five_moment_params.html $TMP/contents.html > $TMP/five_moment_params.html
mv $TMP/five_moment_params.html html/five_moment_params.html

./warpii --print-parameters docs/input_files/phmaxwell_example.inp > /dev/null

./warpii --print-parameters docs/input_files/phmaxwell_example.inp | docs/scripts/process_params_json.py docs/input_files/phmaxwell_example.inp > $TMP/contents.html
python3 docs/scripts/replace_doxygen_html_contents.py html/phmaxwell_params.html $TMP/contents.html > $TMP/phmaxwell_params.html
mv $TMP/phmaxwell_params.html html/phmaxwell_params.html
