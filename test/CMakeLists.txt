macro(buildTarget TARGET SRC)
    add_executable(${TARGET} ${SRC})
    target_link_libraries(${TARGET} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${GTEST_LIBRARIES} ${APR_BUILD_LIBRARY})
endmacro(buildTarget)

buildTarget(testMeshData MeshDataTest.cpp)
buildTarget(testTiff TiffTest.cpp)
buildTarget(testAPR APRTest.cpp)
buildTarget(testComputeGradient ComputeGradientTest.cpp)
