# Install instructions for LibAPR.

The build is configured it via several CMake options (APR_* variables), which can be set to ON/OFF depending on your needs.
The following options control the build type and installation:

- APR_BUILD_STATIC_LIB=ON
- APR_BUILD_SHARED_LIB=OFF
- APR_INSTALL=ON

See [README](README.md) for the full list of build options.

## OSX / UNIX Installation

The full command-line to install the library may look like:

```
mkdir build
cd build
cmake -DAPR_INSTALL=ON -DAPR_BUILD_STATIC_LIB=ON -DAPR_BUILD_SHARED_LIB=OFF ..
make
make install
```
with other build options turned on/off as required. Non-default install locations can be set via 
`-DCMAKE_INSTALL_PREFIX=/tmp/APR`. Depending on the location, you may need file-permissions (sudo) for the install command.

## Windows Installation Clang

On Windows the install commands may look like:

```
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE="PATH_TO_VCPKG\vcpkg\scripts\buildsystems\vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -T ClangCL -DAPR_BUILD_STATIC_LIB=ON -DAPR_BUILD_SHARED_LIB=OFF -DAPR_INSTALL=ON ..
cmake --build . --config Release
cmake --install .
```

(Probably need to be in a console running as administrator)

## Minimal CMake example

A minimalistic CMakeLists.txt file for using LibAPR in another C++ project can be found here: https://github.com/AdaptiveParticles/APR_cpp_project_example

### Notes:

If shared version is preferred then `APR::sharedLib` should be used (and of course APR_BUILD_SHARED_LIB=ON during the build step).

Use of `APR::staticLib` has been tested across Windows, Linux, and Mac.
`APR::sharedLib` has currently only been tested on Linux.

If APR is installed in a non-standard location then some hint for cmake must be provided by adding the install directory 
to CMAKE_PREFIX_PATH, for example:
```
export CMAKE_PREFIX_PATH=/tmp/APR:$CMAKE_PREFIX_PATH
```