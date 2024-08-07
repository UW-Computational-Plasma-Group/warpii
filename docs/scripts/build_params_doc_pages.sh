#!/bin/bash

set -e

TMP=$(mktemp -d)

./warpii --print-parameters ../../examples/five-moment/shu_osher_shock_1d.inp > $TMP/params.json

cat $TMP/params.json

cat $TMP/params.json | docs/scripts/process_params_json.py ../../examples/five-moment/shu_osher_shock_1d.inp > $TMP/contents.html
#./warpii --print-parameters ../../examples/five-moment/shu_osher_shock_1d.inp | docs/scripts/process_params_json.py ../../examples/five-moment/shu_osher_shock_1d.inp > $TMP/contents.html
#./warpii --print-parameters ../../examples/five-moment/shu_osher_shock_1d.inp
python3 docs/scripts/replace_doxygen_html_contents.py html/five_moment_params.html $TMP/contents.html > $TMP/five_moment_params.html

mv $TMP/five_moment_params.html html/five_moment_params.html

touch param_doc_pages
