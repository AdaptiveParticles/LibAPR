###############################################################################
# APR - Adaptive Particle Representation
###############################################################################
cmake_minimum_required(VERSION 3.2)
project(LibAPR)
set(CMAKE_CXX_STANDARD 14)

###############################################################################
# Generate configuration file
###############################################################################
set (APR_VERSION_MAJOR 1)
set (APR_VERSION_MINOR 0)
set (APR_VERSION_PATCH 0)
execute_process(COMMAND git rev-parse HEAD 
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} 
		OUTPUT_VARIABLE APR_GIT_HASH)
string(REGEX REPLACE "\n$" "" APR_GIT_HASH "${APR_GIT_HASH}")
configure_file (
        "${PROJECT_SOURCE_DIR}/src/ConfigAPR.h.in"
        "${PROJECT_BINARY_DIR}/ConfigAPR.h"
)
include_directories("${PROJECT_BINARY_DIR}")

###############################################################################
# Find all required libraries
###############################################################################
find_package(HDF5 REQUIRED)
find_package(TIFF REQUIRED)
find_package(OpenMP)
if(NOT OPENMP_FOUND OR DISABLE_OPENMP)
    message("OpenMP support not found with current compiler. While APR can compile like this, performance might not be optimal. Please see README.md for instructions.")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_OPENMP ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OPENMP ${OpenMP_CXX_FLAGS}")
endif()
include_directories(${HDF5_INCLUDE_DIR} ${TIFF_INCLUDE_DIR} )
add_subdirectory("external/c-blosc")
set_target_properties(blosc_shared PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set_target_properties(blosc_shared PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})
include_directories("external/glm")
include_directories("external/c-blosc/blosc")

###############################################################################
# Configure compiler options
###############################################################################
# If you ever want to compile with Intel's icc (or any other compiler) provide
# compiler names/paths in cmake command like this:
# CC="icc" CXX="icc" CXXFLAGS="-O3" cmake -DAPR_TESTS=1
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic ")
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O4 -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Bdynamic -ldl ")
endif(CMAKE_COMPILER_IS_GNUCC)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
endif()

###############################################################################
# Generate APR library
###############################################################################
set(SOURCE_FILES src/io/blosc_filter.c src/io/hdf5functions_blosc.cpp)
set(SOURCE_FILES_RAYCAST src/vis/Camera.cpp src/vis/Object.cpp src/vis/RaytracedObject.cpp)

add_library(libapr ${SOURCE_FILES} ${SOURCE_FILES_RAYCAST})
set_target_properties(libapr PROPERTIES PREFIX "")
target_include_directories(libapr PUBLIC . src external/glm ${PROJECT_BINARY_DIR})
target_link_libraries(libapr PUBLIC ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} blosc_shared)

###############################################################################
# Examples
###############################################################################
option (APR_BUILD_EXAMPLES "Build APR examples" ON)
if(APR_BUILD_EXAMPLES)
    message(STATUS "APR: Building examples")
    add_subdirectory(examples)
endif(APR_BUILD_EXAMPLES)

###############################################################################
# Tests
###############################################################################
option (APR_TESTS "Build tests wrappers" OFF)
if(APR_TESTS)
    message(STATUS "APR: Building tests")
    add_subdirectory("external/gtest")
    set(GTEST_LIBRARIES gtest)
    add_subdirectory(test)
endif(APR_TESTS)

###############################################################################
# JAVA wrappers
###############################################################################
option (APR_BUILD_JAVA_WRAPPERS "Build JAVA wrappers" OFF)
if(APR_BUILD_JAVA_WRAPPERS)
    message(STATUS "APR: Building JAVA wrappers")
    find_package(SWIG 3.0 REQUIRED)
    find_package(JNI REQUIRED)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/natives")
    include_directories(${JNI_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
    include(${SWIG_USE_FILE})
    set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/src/main/java/de/mpicbg/mosaic/apr")
    set(CMAKE_SWIG_FLAGS -package de.mpicbg.mosaic.apr -Wall)
    set_source_files_properties(libapr.i PROPERTIES CPLUSPLUS ON)
    swig_add_library(apr LANGUAGE java SOURCES libapr.i ${SOURCE_FILES})
    swig_link_libraries(apr ${HDF5_LIBRARIES} blosc_static ${TIFF_LIBRARIES})
endif(APR_BUILD_JAVA_WRAPPERS)
