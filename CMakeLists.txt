###############################################################################
# APR - Adaptive Particle Representation
###############################################################################
cmake_minimum_required(VERSION 3.2)
project(LibAPR DESCRIPTION "Adaptive Particle Representation library")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# force blosc to use local snappy and zstd
set(PREFER_EXTERNAL_ZSTD OFF CACHE BOOL "Prefer included Zstd")
set(PREFER_EXTERNAL_LZ4 OFF CACHE BOOL "Prefer included LZ4")
set(PREFER_EXTERNAL_SNAPPY OFF CACHE BOOL "Prefer included Snappy")

# APR build options:
option(APR_INSTALL "Install APR library" ON)
option(APR_BUILD_SHARED_LIB "Builds shared library" ON)
option(APR_BUILD_STATIC_LIB "Builds shared library" ON)
option(APR_BUILD_EXAMPLES "Build APR examples" OFF)
option(APR_TESTS "Build APR tests" OFF)
option(APR_PREFER_EXTERNAL_GTEST "When found, use the installed GTEST libs instead of included sources" OFF)
option(APR_BUILD_JAVA_WRAPPERS "Build APR JAVA wrappers" OFF)
option(APR_USE_CUDA "should APR use CUDA?" ON)

# Validation of options
if (NOT APR_BUILD_SHARED_LIB AND NOT APR_BUILD_STATIC_LIB)
    message(FATAL_ERROR "At least one target: APR_BUILD_SHARED_LIB or APR_BUILD_STATIC_LIB must be build!")
endif()


###############################################################################
# Generate configuration file
###############################################################################
set (APR_VERSION_MAJOR 1)
set (APR_VERSION_MINOR 0)
set (APR_VERSION_PATCH 0)
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
find_package(TIFF REQUIRED)

# Handle OpenMP
find_package(OpenMP)
if(NOT OPENMP_FOUND OR DISABLE_OPENMP)
    message("OpenMP support not found with current compiler. While APR can compile like this, performance might not be optimal. Please see README.md for instructions.")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_OPENMP ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OPENMP ${OpenMP_CXX_FLAGS}")
endif()
include_directories(${HDF5_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR})

# needed here for blosc library
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configure and add submodule BLOSC
set(BLOSC_IS_SUBPROJECT ON)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC ON CACHE BOOL "" FORCE)
add_subdirectory("external/c-blosc")
include_directories(external/c-blosc/blosc)

# Add submodule GLM (include files only)
include_directories("external/glm")

if(APR_USE_CUDA)

endif()

###############################################################################
# Configure compiler options
###############################################################################
# If you ever want to compile with Intel's icc (or any other compiler) provide
# compiler names/paths in cmake command like this:
# CC="icc" CXX="icc" CXXFLAGS="-O3" cmake -DAPR_TESTS=1
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic")
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O4 -ffast-math -lrt -fPIC")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -lrt -fPIC")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Bdynamic -ldl ")
endif(CMAKE_COMPILER_IS_GNUCC)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -fPIC")
    set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -fPIC")
endif()


###############################################################################
# Generate APR library
###############################################################################

set(LIBRARY_NAME apr)

set(SOURCE_FILES src/io/blosc_filter.c src/io/hdf5functions_blosc.cpp)
set(SOURCE_FILES_RAYCAST src/numerics/APRRaycaster.cpp src/vis/Camera.cpp src/vis/Object.cpp src/vis/RaytracedObject.cpp )

add_library(aprObjLib OBJECT ${SOURCE_FILES} ${SOURCE_FILES_RAYCAST})

set(CUDA_TARGET "")
set(CUDA_APR_TARGET "")
if(APR_USE_CUDA)
  enable_language(CUDA)
    message(STATUS "APR: Building CUDA for APR")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    find_package(CUDA REQUIRED)
    message (STATUS "CUDA LIBS: ${CUDA_LIBRARIES}")
    add_library(cudaLib STATIC IMPORTED)
    set_property(TARGET cudaLib PROPERTY IMPORTED_LOCATION ${CUDA_cudart_static_LIBRARY})
    set(CUDA_TARGET cudaLib)
    set(CUDA_APR_TARGET cudaAprLib)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
    #list(APPEND CUDA_NVCC_FLAGS "-O0; -Xcompiler -fPIC")#"-std=c++14;-O0; -Xcompiler -fPIC")
    list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

    cuda_add_library(${CUDA_APR_TARGET} STATIC src/algorithm/number.cu)
    target_link_libraries(${CUDA_APR_TARGET} ${CUDA_LIBRARIES})
    set_target_properties(${CUDA_APR_TARGET} PROPERTIES
      CUDA_STANDARD 14
      POSITION_INDEPENDENT_CODE True
      COMPILE_FLAGS "-DAPR_USE_CUDA"
      )
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DAPR_USE_CUDA")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPR_USE_CUDA")
endif()

if(APR_BUILD_STATIC_LIB)
    # generate static library used as a intermediate step in generating fat lib
    set(STATIC_TARGET_NAME staticLib)
    add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:aprObjLib>)
    set_target_properties(${STATIC_TARGET_NAME} PROPERTIES OUTPUT_NAME ${LIBRARY_NAME})
    target_include_directories(${STATIC_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
    target_link_libraries(${STATIC_TARGET_NAME} PRIVATE ${CUDA_CUDART_LIBRARY})
    if(APR_USE_CUDA)
        add_dependencies(${STATIC_TARGET_NAME} ${CUDA_APR_TARGET})
    endif()
    # generate fat static library by adding dependencies
    include(cmake/AddStaticLibs.cmake)
    addStaticLibs(${STATIC_TARGET_NAME} blosc_static ${CUDA_APR_TARGET})
endif()

if(APR_BUILD_SHARED_LIB)
# generate fat shared library
    set(SHARED_TARGET_NAME sharedLib)
    add_library(${SHARED_TARGET_NAME} SHARED $<TARGET_OBJECTS:aprObjLib>)
    target_include_directories(${SHARED_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
    set_target_properties(${SHARED_TARGET_NAME} PROPERTIES OUTPUT_NAME ${LIBRARY_NAME})
    set_target_properties(${SHARED_TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_NAME ${LIBRARY_NAME})
    set_target_properties(${SHARED_TARGET_NAME} PROPERTIES ARCHIVE_OUTPUT_NAME ${LIBRARY_NAME})
    add_dependencies(${SHARED_TARGET_NAME} blosc_static)
    set_property(TARGET ${SHARED_TARGET_NAME} PROPERTY VERSION ${APR_VERSION_STRING})
    set_property(TARGET ${SHARED_TARGET_NAME} PROPERTY SOVERSION ${APR_VERSION_MAJOR})
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	    if(APR_USE_CUDA)
            add_dependencies(${SHARED_TARGET_NAME} ${CUDA_APR_TARGET})
		    set(EXTRA_APR_CUDA_LIBS "-Wl,-force_load,$<TARGET_FILE:${CUDA_APR_TARGET}>")
	    endif()
	    target_link_libraries(${SHARED_TARGET_NAME} PRIVATE ${CUDA_CUDART_LIBRARY} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} -Wl,-force_load,$<TARGET_FILE:blosc_static> ${EXTRA_APR_CUDA_LIBS})
    else()
	    if(APR_USE_CUDA)
            add_dependencies(${SHARED_TARGET_NAME} ${CUDA_APR_TARGET})
		    set(EXTRA_APR_CUDA_LIBS "$<TARGET_FILE:${CUDA_APR_TARGET}>")
	    endif()
	    target_link_libraries(${SHARED_TARGET_NAME} PRIVATE ${CUDA_CUDART_LIBRARY} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} -Wl,--whole-archive,$<TARGET_FILE:blosc_static>,${EXTRA_APR_CUDA_LIBS},--no-whole-archive)
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
    include_directories(src)
    message(STATUS "APR: Building examples")
    add_subdirectory(examples)
endif(APR_BUILD_EXAMPLES)

include_directories(src)
message(STATUS "APR: GPUHACK Examples")
add_subdirectory(gpuhack)

###############################################################################
# Tests
###############################################################################
if(APR_TESTS)
    include_directories(src)
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
    swig_link_libraries(apr ${HDF5_LIBRARIES} blosc_static ${TIFF_LIBRARIES})
endif(APR_BUILD_JAVA_WRAPPERS)
