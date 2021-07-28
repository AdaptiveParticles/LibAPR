# Install instructions for LibAPR.

The way to use it is configure it via some new APR_* variables (can be set to ON/OFF depending on needs) which describes what to build and if it should be installed through cmake commands:

APR_BUILD_STATIC_LIB=ON
APR_BUILD_SHARED_LIB=OFF
APR_INSTALL=ON
(all other configuration possibilities are now in the top of CMakeLists.txt file)

## OSX / UNIX Installation

so full command line would look like: (-DCMAKE_INSTALL_PREFIX=/tmp/APR can be used for a non-default location)

```
mkdir build
cd build
cmake -DAPR_INSTALL=ON -DAPR_BUILD_STATIC_LIB=ON -DAPR_BUILD_SHARED_LIB=OFF ..
make
make install
```

You may need file-permissions (sudo for the install)

## Windows Installation Clang

``
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE="PATH_TO_VCPKG\vcpkg\scripts\buildsystems\vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -T ClangCL -DAPR_BUILD_STATIC_LIB=ON -DAPR_BUILD_SHARED_LIB=OFF -DAPR_INSTALL=ON ..
cmake --build . --config Release
``

Need to be in a console running as administrator. 

``
cmake --install .
``

## Minimal example CMAKE

To use APR the minimalistic CMakeLists.txt file can be found here: https://github.com/AdaptiveParticles/APR_cpp_project_example

##

APR::staticLib (Note, tested across Windows, Linux, and Mac)
APR::sharedLib (Note, not tested)

If shared version is preferred then APR::sharedLib should be used (and of course APR_BUILD_SHARED_LIB=ON during lib build step).

NOTICE: if APR is installed in not standard directory then some hint for cmake must be provided by adding install dir to CMAKE_PREFIX_PATH like for above example:

```
export CMAKE_PREFIX_PATH=/tmp/APR:$CMAKE_PREFIX_PATH
```