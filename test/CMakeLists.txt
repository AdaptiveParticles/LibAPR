macro(buildTarget TARGET SRC)
    add_executable(${TARGET} ${SRC})
    target_link_libraries(${TARGET} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${GTEST_LIBRARIES} ${APR_BUILD_LIBRARY} Threads::Threads)
    add_test( ${TARGET} ${TARGET} )
endmacro(buildTarget)

buildTarget(testMeshData MeshDataTest.cpp)
buildTarget(testTiff TiffTest.cpp)
buildTarget(testAPR APRTest.cpp)
buildTarget(testComputeGradient ComputeGradientTest.cpp)
buildTarget(testLocalIntensityScale LocalIntensityScaleTest.cpp)
buildTarget(testPullingScheme PullingSchemeTest.cpp)

#APR GPU Tests
if(APR_USE_CUDA)
   buildTarget(testAPRCuda APRTestCuda.cpp)
endif()


if(APR_BUILD_EXAMPLES)
    buildTarget(testExamples ExamplesTest.cpp)
endif()

if(APR_DENOISE)
    buildTarget(testDenoise DenoiseTest.cpp)
endif()


## Performance Benchmark Tests

#buildTarget(BenchmarkTest BenchmarkTest.cpp)


