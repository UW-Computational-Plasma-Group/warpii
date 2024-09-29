#!/bin/bash

set -ex

WARPII_BINARY=$1
INPUT_FILE=$2
T_END=$3

TMP=$(mktemp -d)
pushd $TMP

echo $INPUT_FILE

cp $INPUT_FILE example.inp
cat >> example.inp <<EOF
set t_end = ${T_END}
set n_writeout_frames = 1
EOF

cat example.inp

$WARPII_BINARY --enable-fpe example.inp

popd
