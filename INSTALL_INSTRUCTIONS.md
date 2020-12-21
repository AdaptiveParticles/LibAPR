# Install instructions for LibAPR.

The way to use it is configure it via some new APR_* variables (can be set to ON/OFF depending on needs) which describes what to build and if it should be installed through cmake commands:

APR_BUILD_STATIC_LIB=ON
APR_BUILD_SHARED_LIB=OFF
APR_INSTALL=ON
(all other configuration possibilities are now in the top of CMakeLists.txt file)

so full command line would look like:

```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/tmp/APR -DAPR_INSTALL=ON -DAPR_BUILD_STATIC_LIB=ON -DAPR_BUILD_SHARED_LIB=OFF ..
make
make install
```

To use APR the minimalistic CMakeLists.txt file would look like:

```
cmake_minimum_required(VERSION 3.2)
project(myAprProject)
set(CMAKE_CXX_STANDARD 14)

#external libraries needed for APR
find_package(HDF5 REQUIRED)
find_package(TIFF REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR} )

find_package(APR REQUIRED)

add_executable(HelloAPR helloWorld.cpp)
target_link_libraries(HelloAPR  ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} apr::staticLib)
```

if shared version is preferred then apr::sharedLib should be used (and of course APR_BUILD_SHARED_LIB=ON during lib build step).

NOTICE: if APR is isntalled in not standard directory then some hint for cmake must be provided by adding install dir to CMAKE_PREFIX_PATH like for above example:

```
export CMAKE_PREFIX_PATH=/tmp/APR:$CMAKE_PREFIX_PATH
```