cmake_minimum_required(VERSION 2.8)
project(PartPlay)
# Run it out of source as cmake -H. -Bbuild ..
set(TESTS 1)
if(TESTS)
    find_package(GTest REQUIRED)
endif(TESTS)
find_package(OpenMP)
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS} sources tests)


set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES

        test/tests.cpp
        test/utils.cpp
        test/k_test.cpp
        test/alg_cells_test.cpp

        test/test_tree/level_iterator_test.cpp
        test/test_tree/tree_fixtures.cpp

        src/io/readimage.h
        src/io/writeimage.h
        src/io/parameters.cpp

        src/data_structures/meshclass.h
        src/data_structures/structure_parts.cpp

        src/algorithm/pipeline.cpp

        )

set(SOURCE_FILES_COMPILED
        src/data_structures/structure_parts.cpp
        src/io/parameters.cpp
        src/io/hdf5functions.cpp
        src/io/hdf5functions_blosc.cpp
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -ffast-math -I /usr/include/hdf5/serial/ -Bdynamic -ldl")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Bdynamic -ldl ")
endif(CMAKE_COMPILER_IS_GNUCC)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -Bdynamic")
    if (NOT NO_OPENMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
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
            )

    add_executable(testTreeNeighboursFace
            test/test_tree/tree_fixtures.cpp
            test/test_tree/face_neighbours_test.cpp
            )

    add_executable(testSparsePartStruc
            test/test_tree/tree_fixtures.cpp
	     test/test_tree/create_part_cell_structure.cpp
            test/test_tree/sparse_rep_test.cpp
            ${SOURCE_FILES_COMPILED}
            test/utils.cpp
	     src/io/hdf5functions_blosc.cpp
            )

    target_link_libraries(testBspline ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc)
    target_link_libraries(testVariance ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc)
    target_link_libraries(testLevel ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc)
    target_link_libraries(testResult ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc)
    target_link_libraries(testLevelIterator ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc)
    target_link_libraries(testTreeNeighbours ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc)
    target_link_libraries(testTreeNeighboursFace ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc)
    target_link_libraries(testSparsePartStruc ${HDF5_LIBRARIES} ${GTEST_LIBRARIES} pthread tiff blosc)

endif(TESTS)

add_executable(Example_get_apr
        test/Examples/get_apr.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
	 test/utils.cpp
        )

add_executable(Example_compute_cells
        test/Examples/compute_cells.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_compute_parts
        test/Examples/compute_parts.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_convert_full_format
        test/Examples/write_full_apr_format.cpp
        src/io/write_parts.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(Example_ray_cast
        test/Examples/ray_cast.cpp
        test/utils.cpp
        ${SOURCE_FILES_COMPILED}
        )

add_executable(pipeline
    src/algorithm/pipeline.cpp
    src/io/write_parts.cpp
    ${SOURCE_FILES_COMPILED}
)

target_link_libraries(pipeline ${HDF5_LIBRARIES} pthread tiff blosc)
target_link_libraries(Example_get_apr ${HDF5_LIBRARIES} pthread tiff blosc)
target_link_libraries(Example_compute_cells ${HDF5_LIBRARIES} pthread tiff blosc)
target_link_libraries(Example_compute_parts ${HDF5_LIBRARIES} pthread tiff blosc)
target_link_libraries(Example_convert_full_format ${HDF5_LIBRARIES} pthread tiff blosc)
target_link_libraries(Example_ray_cast ${HDF5_LIBRARIES} pthread tiff blosc)
