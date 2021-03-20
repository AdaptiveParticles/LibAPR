macro(buildTarget TARGET)
    add_executable(${TARGET} ${TARGET}.cpp)
    target_link_libraries(${TARGET} ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${APR_BUILD_LIBRARY} Threads::Threads ${OPENMP_LINK})
endmacro(buildTarget)

buildTarget(Example_get_apr)
buildTarget(Example_get_apr_by_block)
buildTarget(Example_apr_iterate)
buildTarget(Example_reconstruct_image)
buildTarget(Example_compress_apr)
buildTarget(Example_apr_neighbour_access)
buildTarget(Example_compute_gradient)
buildTarget(Example_apr_filter)
buildTarget(Example_apr_deconvolution)
buildTarget(Example_random_access)

#buildTarget(Example_reconstruct_patch)  #The way this is working is going to be re-designed.
buildTarget(Example_apr_tree)

buildTarget(Example_ray_cast)

if(APR_DENOISE)
    #requires eigen
    buildTarget(Example_denoise)
endif()


