###############################################################################
# APR - Adaptive Particle Representation
###############################################################################
cmake_minimum_required(VERSION 3.9)
project(LibAPR DESCRIPTION "Adaptive Particle Representation library")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# APR build options:
option(APR_INSTALL "Install APR library" OFF)
option(APR_BUILD_SHARED_LIB "Builds shared library" ON)
option(APR_BUILD_STATIC_LIB "Builds static library" OFF)
option(APR_BUILD_EXAMPLES "Build APR examples" OFF)
option(APR_USE_LIBTIFF "Use LibTIFF" ON)
option(APR_TESTS "Build APR tests" OFF)
option(APR_PREFER_EXTERNAL_GTEST "When found, use the installed GTEST libs instead of included sources" OFF)
option(APR_PREFER_EXTERNAL_BLOSC "When found, use the installed BLOSC libs instead of included sources" OFF)
option(APR_BUILD_JAVA_WRAPPERS "Build APR JAVA wrappers" OFF)
option(APR_BUILD_PYTHON_WRAPPERS "Build APR PYTHON wrappers (experimental - under development)" OFF)
option(APR_USE_CUDA "should APR use CUDA? (experimental - under development)" OFF)
option(APR_USE_OPENMP "should APR use OpenMP?" ON)
option(APR_BENCHMARK "add benchmarking code" OFF)

# Validation of options
if (NOT APR_BUILD_SHARED_LIB AND NOT APR_BUILD_STATIC_LIB)
    message(FATAL_ERROR "At least one target: APR_BUILD_SHARED_LIB or APR_BUILD_STATIC_LIB must be build!")
endif()
if (NOT APR_USE_LIBTIFF)
    if (APR_TESTS OR APR_BUILD_EXAMPLES OR APR_BUILD_JAVA_WRAPPERS)
        message(FATAL_ERROR "Building tests, examples or java wrappers not possible when APR_USE_LIBTIFF=OFF!")
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/)

###############################################################################
# Generate configuration file
###############################################################################
set (APR_VERSION_MAJOR 1)
set (APR_VERSION_MINOR 0)
set (APR_VERSION_PATCH 1)
set (APR_VERSION_STRING ${APR_VERSION_MAJOR}.${APR_VERSION_MINOR}.${APR_VERSION_PATCH})
execute_process(COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE APR_GIT_HASH)
string(REGEX REPLACE "\n$" "" APR_GIT_HASH "${APR_GIT_HASH}")
configure_file (
        "${PROJECT_SOURCE_DIR}/src/ConfigAPR.h.in"
        "${PROJECT_BINARY_DIR}/ConfigAPR.h"
)
include_directories("${PROJECT_BINARY_DIR}")
message("Configuring for APR version: " ${APR_VERSION_STRING})


###############################################################################
# Find all required libraries
###############################################################################
find_package(HDF5 REQUIRED)
find_package(ZLIB REQUIRED)
if(APR_USE_LIBTIFF)
    find_package(TIFF)
    set(CMAKE_C_FLAGS "${CMAKE_CFLAGS} -DHAVE_LIBTIFF")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -DHAVE_LIBTIFF")
endif()

# Handle OpenMP
find_package(OpenMP)
if(NOT OPENMP_FOUND OR NOT APR_USE_OPENMP)
    message(WARNING "OpenMP support not found or disabled with current compiler. While APR can compile like this, performance might not be optimal. Please see README.md for instructions.")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_OPENMP ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OPENMP ${OpenMP_CXX_FLAGS}")
endif()
include_directories(${HDF5_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR})

if(APR_PREFER_EXTERNAL_BLOSC)
    find_package(BLOSC)
endif()
if(BLOSC_FOUND)
    message(STATUS "APR: blosc found, using external blosc")
    include_directories(${BLOSC_INCLUDE_DIR})
else()
    message(STATUS "APR: blosc not found, using internal blosc")
    # needed here for blosc library
    SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # Configure and add submodule BLOSC
    set(BLOSC_IS_SUBPROJECT ON)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(PREFER_EXTERNAL_SNAPPY OFF CACHE BOOL "" FORCE)
    add_subdirectory("external/c-blosc")
    include_directories(external/c-blosc/blosc)
    set(BLOSC_LIBRARIES blosc_static)
endif()

# Add submodule GLM (include files only)
include_directories("external/glm")

message(STATUS "---------------------- VARS BEG ---------------------")
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
message(STATUS "---------------------- VARS END ---------------------")

###############################################################################
# Configure compiler options
###############################################################################
# If you ever want to compile with Intel's icc (or any other compiler) provide
# compiler names/paths in cmake command like this:
# CC="icc" CXX="icc" CXXFLAGS="-O3" cmake -DAPR_TESTS=1
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic ")
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O4 -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Bdynamic")
    if(NOT WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl -lz")
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()

if (APR_BENCHMARK)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DAPR_BENCHMARK")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPR_BENCHMARK")
endif()


###############################################################################
# Generate APR library
###############################################################################

set(LIBRARY_NAME apr)

include_directories(src)

set(SOURCE_FILES src/io/blosc_filter.c src/io/hdf5functions_blosc.cpp)
#set(SOURCE_FILES_RAYCAST src/numerics/APRRaycaster.cpp src/vis/Camera.cpp src/vis/Object.cpp src/vis/RaytracedObject.cpp)

add_library(aprObjLib OBJECT ${SOURCE_FILES} ${SOURCE_FILES_RAYCAST})

if(APR_USE_CUDA)
    message(STATUS "APR: Building CUDA for APR")
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS "--default-stream per-thread --cudart shared -g -lineinfo -Xptxas -O3,-v -use_fast_math -DAPR_USE_CUDA")
    if(APR_BENCHMARK)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DAPR_BENCHMARK")
    endif()
    enable_language(CUDA)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DAPR_USE_CUDA")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPR_USE_CUDA")
    set(APR_CUDA_SROUCE_FILES
	        src/algorithm/ComputeGradientCuda.cu
            src/data_structures/Mesh/PixelData.cu
            src/algorithm/LocalIntensityScale.cu
            )
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(APR_BUILD_STATIC_LIB)
    # generate static library used as a intermediate step in generating fat lib
    set(STATIC_TARGET_NAME staticLib)
    add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:aprObjLib> ${APR_CUDA_SROUCE_FILES})
    target_compile_features(${STATIC_TARGET_NAME} PUBLIC cxx_std_14)
    set_target_properties(${STATIC_TARGET_NAME} PROPERTIES OUTPUT_NAME ${LIBRARY_NAME})
    set_target_properties(${STATIC_TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_include_directories(${STATIC_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)

    # generate fat static library by adding dependencies
    if(NOT BLOSC_FOUND)
        include(cmake/AddStaticLibs.cmake)
        addStaticLibs(${STATIC_TARGET_NAME} blosc_static)
    endif()
endif()

if(APR_BUILD_SHARED_LIB)
# generate fat shared library
    set(SHARED_TARGET_NAME sharedLib)
    add_library(${SHARED_TARGET_NAME} SHARED $<TARGET_OBJECTS:aprObjLib> ${APR_CUDA_SROUCE_FILES})

    target_include_directories(${SHARED_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
    set_target_properties(${SHARED_TARGET_NAME} PROPERTIES OUTPUT_NAME ${LIBRARY_NAME})
    set_target_properties(${SHARED_TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_NAME ${LIBRARY_NAME})
    set_target_properties(${SHARED_TARGET_NAME} PROPERTIES ARCHIVE_OUTPUT_NAME ${LIBRARY_NAME})
    set_property(TARGET ${SHARED_TARGET_NAME} PROPERTY VERSION ${APR_VERSION_STRING})
    set_property(TARGET ${SHARED_TARGET_NAME} PROPERTY SOVERSION ${APR_VERSION_MAJOR})

    target_link_libraries(${SHARED_TARGET_NAME} PRIVATE ${HDF5_LIBRARIES} ${TIFF_LIBRARIES}  ${CUDA_CUDART_LIBRARY} ${ZLIB_LIBRARIES})
    if(BLOSC_FOUND)
        target_link_libraries(${SHARED_TARGET_NAME} PRIVATE ${BLOSC_LIBRARIES} ${ZLIB_LIBRARIES})
    else()
        add_dependencies(${SHARED_TARGET_NAME} blosc_static)
        if (APPLE)
            target_link_libraries(${SHARED_TARGET_NAME} PRIVATE -Wl,-force_load,$<TARGET_FILE:blosc_static>)
        else()
            if(WIN32)
                target_link_libraries(${SHARED_TARGET_NAME} PRIVATE blosc_static)
            else()
                target_link_libraries(${SHARED_TARGET_NAME} PRIVATE -Wl,--whole-archive $<TARGET_FILE:blosc_static> -Wl,--no-whole-archive)
            endif()
        endif()
    endif()
endif()

# choose one of the build libraries to be used later for tests and/or examples
if (APR_BUILD_SHARED_LIB)
    set(APR_BUILD_LIBRARY ${SHARED_TARGET_NAME})
else()
    set(APR_BUILD_LIBRARY ${STATIC_TARGET_NAME})
endif()

###############################################################################
# Install APR library
###############################################################################
if(APR_INSTALL)
    set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
    set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
    set(INSTALL_INCLUDE_PROJECT_DIR ${INSTALL_INCLUDE_DIR}/${LIBRARY_NAME} CACHE PATH "Installation directory for header files")
    set(INSTALL_CMAKE_DIR ${INSTALL_LIB_DIR}/cmake/${LIBRARY_NAME} CACHE PATH "Installation directory for cmake files")

    message(STATUS "APR: Install library in [${CMAKE_INSTALL_PREFIX}]")
    set(INC_DIRS src/algorithm src/data_structures src/io src/misc src/numerics src/vis)
    install(DIRECTORY ${INC_DIRS} DESTINATION ${INSTALL_INCLUDE_PROJECT_DIR} COMPONENT DEV FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
    # TODO: It is installed twice since it is used by different compoonents - it must be fixed
    install(FILES ${PROJECT_BINARY_DIR}/ConfigAPR.h COMPONENT DEV DESTINATION ${INSTALL_INCLUDE_PROJECT_DIR}/io)
    install(FILES ${PROJECT_BINARY_DIR}/ConfigAPR.h COMPONENT DEV DESTINATION ${INSTALL_INCLUDE_PROJECT_DIR})

    set(APR_TARGETS LibAprTargets)
    install(TARGETS ${SHARED_TARGET_NAME} ${STATIC_TARGET_NAME}
            EXPORT ${APR_TARGETS}
            LIBRARY DESTINATION ${INSTALL_LIB_DIR}
            ARCHIVE DESTINATION ${INSTALL_LIB_DIR}
            INCLUDES DESTINATION ${INSTALL_INCLUDE_DIR} ${INSTALL_INCLUDE_DIR}/${LIBRARY_NAME})

    # pkg-config
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/apr.pc.in" "${CMAKE_CURRENT_BINARY_DIR}/apr.pc" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/apr.pc" DESTINATION lib/pkgconfig COMPONENT DEV)

    # AprTargets
    set(APR_NAMESPACE ${LIBRARY_NAME}::)
    export( EXPORT ${APR_TARGETS} FILE "${CMAKE_CURRENT_BINARY_DIR}/AprTargets.cmake" NAMESPACE ${APR_NAMESPACE})
    install(EXPORT ${APR_TARGETS} FILE AprTargets.cmake NAMESPACE ${APR_NAMESPACE} DESTINATION ${INSTALL_CMAKE_DIR})

    # AprConfig
    configure_file(cmake/AprConfig.cmake "${CMAKE_CURRENT_BINARY_DIR}/AprConfig.cmake" @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AprConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/AprConfigVersion.cmake DESTINATION ${INSTALL_CMAKE_DIR} COMPONENT DEV)

    # AprConfigVersion
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/AprConfigVersion.cmake" VERSION ${APR_VERSION_STRING} COMPATIBILITY AnyNewerVersion)

    # uninstall
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake-uninstall.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/cmake-uninstall.cmake"
                   IMMEDIATE @ONLY)
    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake-uninstall.cmake)
endif(APR_INSTALL)


###############################################################################
# Examples
###############################################################################
if(APR_BUILD_EXAMPLES)
    message(STATUS "APR: Building examples")
    add_subdirectory(examples)
endif(APR_BUILD_EXAMPLES)

###############################################################################
# Tests
###############################################################################
if(APR_TESTS)
    message(STATUS "APR: Building tests")
    if(APR_PREFER_EXTERNAL_GTEST)
        find_package(GTest 1.8.0)
    endif()
    if(GTEST_FOUND)
        include_directories(${GTEST_INCLUDE_DIRS})
    else(GTEST_FOUND)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(BUILD_GTEST ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        message(STATUS "APR: GTest not found, using internal gtest")
        add_subdirectory("external/gtest")
        set(GTEST_LIBRARIES gtest)
    endif(GTEST_FOUND)

    enable_testing()
    add_subdirectory(test)
endif(APR_TESTS)


###############################################################################
# JAVA wrappers
###############################################################################
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
    swig_link_libraries(apr ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${BLOSC_LIBRARIES} ${ZLIB_LIBRARIES})
endif(APR_BUILD_JAVA_WRAPPERS)


###############################################################################
# PYTHON wrappers
###############################################################################
if(APR_BUILD_PYTHON_WRAPPERS)
    message(STATUS "APR: Building PYTHON wrappers")
    set(PYBIND11_PYTHON_VERSION 2.7)

    add_subdirectory("external/pybind11")

    set(APR_PYTHON_MODULE_NAME pyApr)
    add_definitions(-DAPR_PYTHON_MODULE_NAME=${APR_PYTHON_MODULE_NAME})
    add_library(${APR_PYTHON_MODULE_NAME} MODULE src/wrapper/pythonBind.cpp $<TARGET_OBJECTS:aprObjLib>)
    target_include_directories(${APR_PYTHON_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${APR_PYTHON_MODULE_NAME} PRIVATE pybind11::module ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${BLOSC_LIBRARIES} ${ZLIB_LIBRARIES})
    set_target_properties(${APR_PYTHON_MODULE_NAME} PROPERTIES OUTPUT_NAME ${APR_PYTHON_MODULE_NAME})
    set_target_properties(${APR_PYTHON_MODULE_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                                               SUFFIX "${PYTHON_MODULE_EXTENSION}")

endif(APR_BUILD_PYTHON_WRAPPERS)
