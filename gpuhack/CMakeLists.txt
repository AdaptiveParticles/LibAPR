macro(buildTarget TARGET)
    add_executable(${TARGET} ${TARGET}.cpp)
    target_link_libraries(${TARGET} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${APR_BUILD_LIBRARY})
endmacro(buildTarget)

macro(buildGPUTarget TARGET)
  if(APR_USE_CUDA)

    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
    list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

    add_executable(${TARGET} ${TARGET}.cu)
    target_link_libraries(${TARGET} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${APR_BUILD_LIBRARY})
    set_target_properties(${TARGET} PROPERTIES
      CUDA_STANDARD 14

      POSITION_INDEPENDENT_CODE True
      COMPILE_FLAGS "-DAPR_USE_CUDA --expt-extended-lambda"
      )

  else()
    message(WARNING "skipping building ${TARGET} as libAPR was not configured for CUDA support")
  endif()
endmacro(buildGPUTarget)

buildTarget(Cpu_example)
buildTarget(Cpu_dense_neighbour)
buildTarget(Cpu_dense_identity_check)
buildTarget(Cpu_dense_mesh)
buildTarget(Cpu_dense_simplified_access)
buildTarget(Cpu_dense_tree)
buildTarget(CPU_full_storage)
buildTarget(Cpu_neighbour_access_Tree)
buildTarget(Cpu_neighbour_access_Tree_stencil)


buildGPUTarget(GPU_full_storage)
buildGPUTarget(GPU_insert_dense)
