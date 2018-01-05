cmake_minimum_required(VERSION 2.8)
project(PartPlay)
# Run it out of source as cmake -H. -Bbuild ..

if(TESTS)
    find_package(GTest REQUIRED)
endif(TESTS)


find_package(OpenMP)
find_package(HDF5 REQUIRED)
find_package(SWIG 3.0 REQUIRED)
find_package(JNI REQUIRED)
INCLUDE(${SWIG_USE_FILE})

set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/src/main/java/de/mpicbg/mosaic/apr")
set(CMAKE_SWIG_FLAGS -package de.mpicbg.mosaic.apr -Wall)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/natives")

SET_SOURCE_FILES_PROPERTIES(libapr.i PROPERTIES CPLUSPLUS ON)

find_package(TIFF REQUIRED)

include_directories(${HDF5_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR} ${GTEST_INCLUDE_DIRS} sources tests ${JNI_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

SET(BUILD_TESTS OFF CACHE BOOL "")
SET(BUILD_BENCHMARKS OFF CACHE BOOL "")
add_subdirectory("external/c-blosc")
include_directories("external/c-blosc/blosc")

include_directories(${PROJECT_SOURCE_DIR}/external/glm)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES

        test/tests.cpp

        test/k_test.cpp
        test/alg_cells_test.cpp

        test/test_tree/level_iterator_test.cpp
        test/test_tree/tree_fixtures.cpp

        src/io/readimage.h
        src/io/writeimage.h
        src/io/parameters.cpp

        src/data_structures/meshclass.h
        src/data_structures/structure_parts.cpp

        src/data_structures/APR/APR.hpp
        src/data_structures/APR/APR_iterator.hpp

        src/algorithm/pipeline.cpp

        )

set(SOURCE_FILES_COMPILED
        src/data_structures/structure_parts.cpp
        src/data_structures/Tree/PartCellData.cpp
        src/io/parameters.cpp
        src/io/hdf5functions.cpp
        src/io/blosc_filter.c
        src/io/hdf5functions_blosc.cpp
        src/io/write_parts.cpp
        )


if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)


# If you ever want to compile with icc it should look more or less like this:
#set(CMAKE_CXX_COMPILER /opt/intel/compilers_and_libraries_2016/linux/bin/intel64/icc)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -ip -ansi-alias -xCORE-AVX2 ")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -qopenmp")

if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -ffast-math -I /usr/include/hdf5/serial/ -Bdynamic -ldl")
   #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O1  -I /usr/include/hdf5/serial/ ")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g -I /usr/include/hdf5/serial/")
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

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -ffast-math -Bdynamic")

    if (NOT NO_OPENMP)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
    endif()
endif()


if(TESTS)

    add_executable(testBspline
            test/tests.cpp
            test/bspline_test.cpp
            ${SOURCE_FILES_COMPILED}
            test/utils.cpp
            )
    add_executable(testVariance
            test/tests.cpp
            test/variance_test.cpp
            ${SOURCE_FILES_COMPILED}
            test/utils.cpp
            )
    add_executable(testLevel
            test/k_test.cpp
            ${SOURCE_FILES_COMPILED}
            test/utils.cpp
            )
    add_executable(testResult
            test/alg_cells_test.cpp
            src/io/write_parts.cpp
            ${SOURCE_FILES_COMPILED}
            test/utils.cpp
            )

    add_executable(testLevelIterator
            test/test_tree/tree_fixtures.cpp
            test/test_tree/level_iterator_test.cpp
            )

    add_executable(testTreeNeighbours
            test/test_tree/tree_fixtures.cpp
            test/test_tree/neighbours_test.cpp
	    test/utils.cpp
	    ${SOURCE_FILES_COMPILED}
            )

    add_executable(testTreeNeighboursFace
            test/test_tree/tree_fixtures.cpp
            test/test_tree/face_neighbours_test.cpp
            )

    add_executable(testSparsePartStruc
            test/test_tree/tree_fixtures.cpp

            #test/test_tree/create_part_cell_structure.cpp
            test/test_tree/sparse_rep_test.cpp
            ${SOURCE_FILES_COMPILED}
            test/utils.cpp
	        src/io/hdf5functions_blosc.cpp
            )

    target_link_libraries(testBspline ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc_shared)
    target_link_libraries(testVariance ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc_shared)
    target_link_libraries(testLevel ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc_shared)
    target_link_libraries(testResult ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc_shared)
    target_link_libraries(testLevelIterator ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc_shared)
    target_link_libraries(testTreeNeighbours ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc_shared)
    target_link_libraries(testTreeNeighboursFace ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc_shared)
    target_link_libraries(testSparsePartStruc ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc_shared)

endif(TESTS)

add_executable(Example_get_apr
        test/Examples/get_apr.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
	    #test/utils.cpp
        )

add_executable(Example_apr_iterate
        test/Examples/Example_apr_iterate.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_convert_full_format
        test/Examples/write_full_apr_format.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_ray_cast
        test/Examples/Example_ray_cast.cpp
        #test/utils.cpp
        ${SOURCE_FILES_COMPILED}
        src/vis/Camera.cpp src/vis/Camera.h src/vis/Object.cpp src/vis/Object.h src/vis/RaytracedObject.cpp src/vis/RaytracedObject.h)

add_executable(Example_vis
        test/Examples/vis_example.cpp
        #test/utils.cpp
        ${SOURCE_FILES_COMPILED}
        src/vis/Camera.cpp src/vis/Camera.h src/vis/Object.cpp src/vis/Object.h src/vis/RaytracedObject.cpp src/vis/RaytracedObject.h)

#add_executable(pipeline
 #   src/algorithm/pipeline.cpp
 #   src/io/write_parts.cpp
 #   ${SOURCE_FILES_COMPILED}
#)

add_executable(Example_filter
        test/Examples/filter_apr.cpp
        #test/utils.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_enhance_apr
        test/Examples/Example_enhance_apr.cpp
        #test/utils.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
        )


add_executable(Example_segment_gc
    test/Examples/segmentation_apr.cpp
    external/maxflow-v3.04.src/maxflow.cpp
    external/maxflow-v3.04.src/graph.cpp
    #test/utils.cpp
    src/io/write_parts.cpp
    ${SOURCE_FILES_COMPILED} src/vis/Camera.cpp src/vis/Camera.h src/vis/Object.cpp src/vis/Object.h src/vis/RaytracedObject.cpp src/vis/RaytracedObject.h
)


add_executable(Example_neigh
        test/Examples/Example_neigh.cpp
        test/utils.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
        )



SWIG_ADD_MODULE(apr java libapr.i
#        src/io/partcell_io.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
#        src/algorithm/pipeline.cpp
        )

SWIG_LINK_LIBRARIES(apr ${HDF5_LIBRARIES} blosc_shared tiff)


#target_link_libraries(pipeline ${HDF5_LIBRARIES} pthread tiff blosc_shared)
target_link_libraries(Example_get_apr ${HDF5_LIBRARIES} pthread ${TIFF_LIBRARIES} blosc_shared)
target_link_libraries(Example_apr_iterate ${HDF5_LIBRARIES} pthread tiff blosc_shared)
target_link_libraries(Example_convert_full_format ${HDF5_LIBRARIES} pthread tiff blosc_shared)
target_link_libraries(Example_ray_cast ${HDF5_LIBRARIES} pthread tiff blosc_shared)
target_link_libraries(Example_vis ${HDF5_LIBRARIES} pthread tiff blosc_shared)
target_link_libraries(Example_filter ${HDF5_LIBRARIES} pthread tiff blosc_shared)
target_link_libraries(Example_segment_gc ${HDF5_LIBRARIES} pthread tiff blosc_shared)
target_link_libraries(Example_enhance_apr ${HDF5_LIBRARIES} pthread tiff blosc_shared)
target_link_libraries(Example_neigh ${HDF5_LIBRARIES} pthread tiff blosc_shared)

set_target_properties(blosc_shared PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set_target_properties(blosc_shared PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})

list( APPEND ExternalSharedLibraries "${TIFF_LIBRARIES}" )
message( STATUS "\tCopying shared libraries..." )
file( COPY ${ExternalSharedLibraries} DESTINATION "${PROJECT_BINARY_DIR}")

if(BENCHMARKS)

    #   Benchmark codes, requries the SynImage Code


    if(SynImage_PATH)

    FIND_PACKAGE(ArrayFire REQUIRED)
    # If ArrayFire is found, the following variables will be defined:
    #
    # ArrayFire_INCLUDE_DIRS    - Location of ArrayFire's include directory.
    # ArrayFire_LIBRARIES       - Location of ArrayFire's libraries. This will default
    #                             to a GPU backend if one is found.
    # ArrayFire_FOUND           - True if ArrayFire has been located
    #
    # You may provide a hint to where ArrayFire's root directory may be located
    # by setting ArrayFire_DIR.
    #
    # ----------------------------------------------------------------------------
    #
    # ArrayFire_CPU_FOUND        - True of the ArrayFire CPU library has been found.
    # ArrayFire_CPU_LIBRARIES    - Location of ArrayFire's CPU library, if found
    # ArrayFire_CUDA_FOUND       - True of the ArrayFire CUDA library has been found.
    # ArrayFire_CUDA_LIBRARIES   - Location of ArrayFire's CUDA library, if found
    # ArrayFire_OpenCL_FOUND     - True of the ArrayFire OpenCL library has been found.
    # ArrayFire_OpenCL_LIBRARIES - Location of ArrayFire's OpenCL library, if found

    SET(INSTALL_SYNIMAGE_DIR ${SynImage_PATH})

    MESSAGE(STATUS ${SynImage_PATH})

    set(BENCHMARK_SOURCE_FILES  external/maxflow-v3.04.src/maxflow.cpp
            external/maxflow-v3.04.src/graph.cpp src/vis/Camera.cpp src/vis/Camera.h src/vis/Object.cpp src/vis/Object.h src/vis/RaytracedObject.cpp src/vis/RaytracedObject.h)

    # Include the ArrayFire hreaders
    INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
    INCLUDE_DIRECTORIES(${SynImage_PATH}/src/)

    # ArrayFire OpenCL backend
    FIND_PACKAGE(OpenCL)
    IF(${ArrayFire_OpenCL_FOUND} AND ${OpenCL_FOUND})
        # We need to find OpenCL as transitive linking is disabled on some OSes
        MESSAGE(STATUS "ArrayFire OpenCL backend found.")
        ADD_EXECUTABLE(SynImageGen benchmarks/test_syn_gen.cpp  ${SOURCE_FILES_COMPILED}  ${BENCHMARK_SOURCE_FILES})
        ADD_EXECUTABLE(TestBenchmark benchmarks/test_benchmark.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(IncreaseDomain benchmarks/benchmark_increase_domain.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(IncreaseInfo benchmarks/benchmark_increase_info.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(IncreaseRelError benchmarks/benchmark_increase_rel_error.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(IncreaseZoom benchmarks/benchmark_increase_zoom.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(IncreaseInfoandDomain benchmarks/increase_info_and_domain.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(RealData benchmarks/benchmark_real_data.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(FixRatio benchmarks/benchmark_fix_ratio.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(GenSingleImage benchmarks/generate_single_image.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(GenAPRTime benchmarks/Generate_APR_Time.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(TestLocalScale benchmarks/benchmark_local_scale.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(GenAnisoImage benchmarks/generate_anisotropic_single.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )

        ADD_EXECUTABLE(BenchAnisoImage benchmarks/benchmark_aniso.cpp  ${SOURCE_FILES_COMPILED} ${BENCHMARK_SOURCE_FILES} )


        TARGET_LINK_LIBRARIES(SynImageGen ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)
        TARGET_LINK_LIBRARIES(TestBenchmark ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(IncreaseDomain ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(IncreaseInfo ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(IncreaseRelError ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(IncreaseZoom ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(IncreaseInfoandDomain ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(RealData ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(FixRatio ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(GenSingleImage ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(GenAPRTime ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(TestLocalScale ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(GenAnisoImage ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        TARGET_LINK_LIBRARIES(BenchAnisoImage ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

        add_executable(Example_analysis
                ${BENCHMARK_SOURCE_FILES}
                test/utils.cpp
                src/io/write_parts.cpp
                ${SOURCE_FILES_COMPILED}
                test/Examples/analyze_apr.cpp )

        TARGET_LINK_LIBRARIES(Example_analysis ${ArrayFire_OpenCL_LIBRARIES}
                ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${HDF5_LIBRARIES} pthread tiff blosc)

    ENDIF()

    if(${ArrayFire_CPU_FOUND})
        MESSAGE(STATUS "ArrayFire CPU backend found.")
        ADD_EXECUTABLE(SynImageGen_cpu benchmarks/test_syn_gen.cpp ${BENCHMARK_SOURCE_FILES}  ${SOURCE_FILES_COMPILED})
        TARGET_LINK_LIBRARIES(SynImageGen_cpu ${ArrayFire_CPU_LIBRARIES} ${HDF5_LIBRARIES} pthread tiff blosc
                ${CMAKE_THREAD_LIBS_INIT})
    ENDIF()

    else()
        MESSAGE(STATUS " Need to set SynImage_PATH to syn image repository using -DSynImage_PATH")

    endif(SynImage_PATH)


endif(BENCHMARKS)

