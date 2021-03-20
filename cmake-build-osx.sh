#!/bin/sh
-euo pipefail

mkdir build
cd build
cmake -DAPR_BUILD_STATIC_LIB=ON -DAPR_BUILD_SHARED_LIB=ON -DAPR_BUILD_EXAMPLES=ON -DAPR_TESTS=ON -DAPR_USE_CUDA=OFF -DAPR_PREFER_EXTERNAL_BLOSC=ON ..
cmake --build . -j 4

ctest -j 4 --output-on-failure
