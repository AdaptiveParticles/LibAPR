macro(buildTarget TARGET SRC)
    add_executable(${TARGET} ${SRC})
    target_link_libraries(${TARGET} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${GTEST_LIBRARIES} ${APR_BUILD_LIBRARY})
    add_test( ${TARGET} ${TARGET} )
endmacro(buildTarget)

buildTarget(testMeshData MeshDataTest.cpp)
buildTarget(testTiff TiffTest.cpp)
buildTarget(testAPR APRTest.cpp)
buildTarget(testComputeGradient ComputeGradientTest.cpp)
buildTarget(testLocalIntensityScale LocalIntensityScaleTest.cpp)
buildTarget(testPullingScheme PullingSchemeTest.cpp)

#APR GPU Tests
#if(APR_USE_CUDA)
   buildTarget(testAPRCuda APRTestCuda.cpp)
#endif()

## Performance Benchmark Tests

#buildTarget(BenchmarkTest BenchmarkTest.cpp)

# Naive python test
# Working directory is changed to build since python module is availabe there
#if(APR_BUILD_PYTHON_WRAPPERS)
#    add_test(NAME pythondWrapperTest WORKING_DIRECTORY ${PROJECT_BINARY_DIR} COMMAND python ${CMAKE_CURRENT_LIST_DIR}/pythonTest.py)
#endif()
