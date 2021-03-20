#!/bin/sh
-euo pipefail

mkdir build
cd build
cmake -DAPR_BUILD_STATIC_LIB=ON -DAPR_BUILD_SHARED_LIB=OFF -DAPR_BUILD_EXAMPLES=ON -DAPR_TESTS=ON -DAPR_USE_CUDA=OFF -DAPR_PREFER_EXTERNAL_BLOSC=ON -DCODE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug ..

cmake --build . -j 4 --config Debug
ctest -j 4 --output-on-failure

lcov --directory . --capture --output-file coverage.info
lcov --remove coverage.info '/usr/*' "${HOME}"'/.cache/*' --output-file coverage.info
lcov --list coverage.info
bash <(curl -s https://codecov.io/bash) -f coverage.info || echo "Codecov did not collect coverage reports"