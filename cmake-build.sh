#!/bin/sh
mkdir build
cd build
cmake -DAPR_BUILD_STATIC_LIB=1 -DAPR_BUILD_DYNAMIC_LIB=1 -DAPR_EXAMPLES=1 -DAPR_BUILD_EXAMPLES=1 ..
cmake --build .

# TODO: Add unit test execution
