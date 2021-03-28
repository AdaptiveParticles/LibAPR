macro(buildTarget TARGET SRC)
    add_executable(${TARGET} ${SRC})
    target_link_libraries(${TARGET} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${GTEST_LIBRARIES} ${APR_BUILD_LIBRARY} Threads::Threads ${OPENMP_LINK})
endmacro(buildTarget)

## Performance Benchmark Tests
buildTarget(BenchIteration BenchIteration.cpp)
buildTarget(BenchPartData BenchPartData.cpp)
buildTarget(BenchFilter BenchFilter.cpp)
buildTarget(BenchIO BenchIO.cpp)
buildTarget(BenchPipeline BenchPipeline.cpp)

if(APR_USE_CUDA)

buildTarget(BenchCudaAccessInit BenchCudaAccessInit.cpp)
buildTarget(BenchFilterCuda BenchFilterCuda.cpp)

endif()


