#!/bin/bash

set -ex

./check-warpii-env.sh

set +e
cmake --find-package -DNAME=Boost -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST
set -e
if [[ $? -eq 0 ]]; then
    echo "Boost exists, no need to install it."
else
    if [[ "$unamestr" == 'Darwin' ]]; then
        brew install boost
    else
        echo "We don't know how to install openmpi automatically on this system."
    fi
fi

cmake --find-package -DNAME=Boost -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE
cmake --find-package -DNAME=Boost -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=LINK

rm -rf CMakeFiles
