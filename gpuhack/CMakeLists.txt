macro(buildTarget TARGET)
    add_executable(${TARGET} ${TARGET}.cpp)
    target_link_libraries(${TARGET} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${APR_BUILD_LIBRARY})
endmacro(buildTarget)

buildTarget(Cpu_example)
buildTarget(Cpu_dense_neighbour)
buildTarget(Cpu_dense_identity_check)
buildTarget(Cpu_dense_mesh)
buildTarget(Cpu_dense_simplified_access)
buildTarget(Cpu_dense_tree)


