#!/bin/bash

TMP=$(mktemp -d)


#./warpii --print-parameters ../../examples/five-moment/shu_osher_shock_1d.inp 
./warpii --print-parameters ../../examples/five-moment/shu_osher_shock_1d.inp | docs/scripts/process_params_json.py > $TMP/contents.html
python3 docs/scripts/replace_doxygen_html_contents.py html/reference.html $TMP/contents.html > $TMP/reference.html

mv $TMP/reference.html html/reference.html

