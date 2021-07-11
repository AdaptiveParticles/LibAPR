#!/bin/sh
set -x
mkdir build
cd build
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg.exe install blosc:x64-windows gtest:x64-windows tiff:x64-windows hdf5:x64-windows szip:x64-windows
cd ..

cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE="vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -T ClangCL -DAPR_BUILD_EXAMPLES=ON -DAPR_BUILD_STATIC_LIB=ON -DAPR_BUILD_SHARED_LIB=OFF -DAPR_TESTS=ON -DAPR_DENOISE=ON ..
cmake --build . --config Release

CTEST_OUTPUT_ON_FAILURE=1 ctest -V
