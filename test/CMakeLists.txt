macro(buildTarget TARGET SRC)
    add_executable(${TARGET} ${SRC})
    target_link_libraries(${TARGET} ${GTEST_LIBRARIES} libapr_dynamic)
endmacro(buildTarget)

buildTarget(testMeshData MeshDataTest.cpp)
buildTarget(testTiff TiffTest.cpp)
buildTarget(testAPR APRTest.cpp)
buildTarget(testComputeGradient ComputeGradientTest.cpp)
