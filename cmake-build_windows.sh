#!/bin/sh

git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat
.\vcpkg\vcpkg.exe install blosc:x64-windows gtest:x64-windows tiff:x64-windows hdf5:x64-windows szip:x64-windows

mkdir build
cd build
cmake -DAPR_BUILD_STATIC_LIB=ON -DAPR_BUILD_SHARED_LIB=ON -DAPR_BUILD_EXAMPLES=ON -DAPR_TESTS=ON -DAPR_USE_CUDA=OFF -DAPR_PREFER_EXTERNAL_BLOSC=ON -DAPR_PREFER_EXTERNAL_GTEST=ON ..
cmake --build .

CTEST_OUTPUT_ON_FAILURE=1 ctest -V
