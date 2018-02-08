cmake_minimum_required(VERSION 3.2)
project(libAPR)

set(CMAKE_CXX_STANDARD 14)

if(TESTS)
    #find_package(GTest REQUIRED)
    add_subdirectory("external/gtest")
    set(GTEST_LIBRARIES gtest)
endif(TESTS)


find_package(OpenMP)
if(NOT OPENMP_FOUND OR DISABLE_OPENMP)
    message("OpenMP support not found with current compiler. While APR can compile like this, performance might not be optimal. Please see README.md for instructions.")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_OPENMP ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -DHAVE_OPENMP ${OpenMP_CXX_FLAGS}")
endif()

find_package(HDF5 REQUIRED)
#find_package(Threads)
find_package(SWIG 3.0 REQUIRED)
find_package(JNI REQUIRED)
INCLUDE(${SWIG_USE_FILE})

set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/src/main/java/de/mpicbg/mosaic/apr")
set(CMAKE_SWIG_FLAGS -package de.mpicbg.mosaic.apr -Wall)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/natives")

SET_SOURCE_FILES_PROPERTIES(libapr.i PROPERTIES CPLUSPLUS ON)

find_package(TIFF REQUIRED)

include_directories(${HDF5_INCLUDE_DIRS} ${HDF5_INCLUDE_DIR} ${TIFF_INCLUDE_DIR} ${GTEST_INCLUDE_DIRS} sources tests ${JNI_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

SET(BUILD_TESTS OFF CACHE BOOL "")
SET(BUILD_BENCHMARKS OFF CACHE BOOL "")
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory("external/c-blosc")
include_directories("external/c-blosc/blosc")

include_directories(${PROJECT_SOURCE_DIR}/external/glm)

set(SOURCE_FILES
        benchmarks/development/old_io/readimage.h
        benchmarks/development/old_io/writeimage.h
        benchmarks/development/old_io/parameters.cpp
        src/data_structures/Mesh/MeshData.hpp
        src/data_structures/APR/APR.hpp
        src/data_structures/APR/APRIterator.hpp
        src/data_structures/APR/APRAcess.hpp
	src/numerics/APRNumerics.hpp
        )

set(SOURCE_FILES_COMPILED
        src/io/blosc_filter.c
        src/io/hdf5functions_blosc.cpp
        )

# If you ever want to compile with icc it should look more or less like this:
#set(CMAKE_CXX_COMPILER /opt/intel/compilers_and_libraries_2016/linux/bin/intel64/icc)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -ip -ansi-alias -xCORE-AVX2 ")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -qopenmp")

if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O4 -ffast-math -I /usr/include/hdf5/serial/ -Bdynamic -ldl ")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++14 -O0 -g -I /usr/include/hdf5/serial/")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Bdynamic -ldl ")
endif(CMAKE_COMPILER_IS_GNUCC)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    MESSAGE(STATUS "CLANG DETECTED")

    if(CMAKE_BUILD_TYPE MATCHES "Release")
        MESSAGE(STATUS "RELEASE BUILD")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -ffast-math")
    endif()

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        MESSAGE(STATUS "DEBUG BUILD")
        set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS} -std=c++14 -O0 -g")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()


if(TESTS)
    add_executable(testMeshData
            test/MeshDataTest.cpp
            )

    add_executable(testTiff
            test/TiffTest.cpp
            )

    add_executable(testAPR
            test/APRTest.cpp
            ${SOURCE_FILES_COMPILED}
            src/io/hdf5functions_blosc.cpp
            )

    target_link_libraries(testMeshData ${GTEST_LIBRARIES} ${TIFF_LIBRARIES})
    target_link_libraries(testTiff ${GTEST_LIBRARIES} ${TIFF_LIBRARIES})
    target_link_libraries(testAPR ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TIFF_LIBRARIES} blosc_shared)

endif(TESTS)

add_executable(Example_get_apr
        examples/Example_get_apr.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_apr_iterate
        examples/Example_apr_iterate.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_reconstruct_image
        examples/Example_reconstruct_image.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_ray_cast
        examples/Example_ray_cast.cpp
        ${SOURCE_FILES_COMPILED}
        src/vis/Camera.cpp src/vis/Camera.h src/vis/Object.cpp src/vis/Object.h src/vis/RaytracedObject.cpp src/vis/RaytracedObject.h)

add_executable(Example_compress
        examples/Example_compress_apr.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_produce_paraview_file
        examples/Example_produce_paraview_file.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_apr_neighbour_access
        examples/Example_apr_neighbour_access.cpp

        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_compute_gradient
        examples/Example_compute_gradient.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_random_access
        examples/Example_random_access.cpp
        ${SOURCE_FILES_COMPILED}
        )

if(BUILD_WRAPPERS)
  SWIG_ADD_MODULE(apr java libapr.i

        ${SOURCE_FILES_COMPILED}
        )
  SWIG_LINK_LIBRARIES(apr ${HDF5_LIBRARIES} blosc_static ${TIFF_LIBRARIES})
endif()

target_link_libraries(Example_get_apr ${HDF5_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TIFF_LIBRARIES} blosc_shared)
target_link_libraries(Example_apr_iterate ${HDF5_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TIFF_LIBRARIES} blosc_shared)
target_link_libraries(Example_ray_cast ${HDF5_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TIFF_LIBRARIES} blosc_shared)
target_link_libraries(Example_reconstruct_image ${HDF5_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TIFF_LIBRARIES} blosc_shared)
target_link_libraries(Example_apr_neighbour_access ${HDF5_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TIFF_LIBRARIES} blosc_shared)
target_link_libraries(Example_produce_paraview_file ${HDF5_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TIFF_LIBRARIES} blosc_shared)
target_link_libraries(Example_compress ${HDF5_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TIFF_LIBRARIES} blosc_shared)
target_link_libraries(Example_compute_gradient ${HDF5_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TIFF_LIBRARIES} blosc_shared)
target_link_libraries(Example_random_access ${HDF5_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TIFF_LIBRARIES} blosc_shared)

set_target_properties(blosc_shared PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set_target_properties(blosc_shared PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})

list( APPEND ExternalSharedLibraries "${TIFF_LIBRARIES}" )
message( STATUS "\tCopying shared libraries..." )
file( COPY ${ExternalSharedLibraries} DESTINATION "${PROJECT_BINARY_DIR}")

