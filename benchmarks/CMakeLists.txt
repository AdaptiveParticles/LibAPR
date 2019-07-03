macro(buildTarget TARGET SRC)
    add_executable(${TARGET} ${SRC})
    target_link_libraries(${TARGET} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${GTEST_LIBRARIES} ${APR_BUILD_LIBRARY})
    add_test( ${TARGET} ${TARGET} )
endmacro(buildTarget)


## Performance Benchmark Tests

buildTarget(BenchIteration BenchIteration.cpp)
buildTarget(BenchPartData BenchPartData.cpp)

