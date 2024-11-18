#!/bin/bash

set -ex

./check-warpii-env.sh

DEALII_BUILD_MODE=RelWithDebInfo

DEALII_SRCDIR=$WARPIISOFT/deps/dealii/src/dealii-${DEALII_VERSION}

mkdir -p $WARPIISOFT/deps
pushd $WARPIISOFT/deps
(cd candi && git pull) || git clone https://github.com/dealii/candi.git
pushd candi

sed -i "" "s/^DEAL_II_VERSION.*$/DEAL_II_VERSION=v${DEALII_VERSION}/" candi.cfg

grep DEAL_II_VERSION candi.cfg

./candi.sh -y --prefix="${WARPIISOFT}/deps/dealii" --packages="dealii" -j $WARPII_MAKE_PARALLELISM

popd # candi
popd # $WARPIISOFT
