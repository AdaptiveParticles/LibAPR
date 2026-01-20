#ifndef GEN_INFO_GPU_ACCESS_CUH
#define GEN_INFO_GPU_ACCESS_CUH

// CUDA version of GenInfo structure
typedef struct GenInfoCuda_t {
    int l_min;
    int l_max;

    int *org_dims; // fixed size: [3]

    uint8_t number_dimensions;

    int *x_num;
    int *y_num;
    int *z_num;

    // this differs from original GenInfo structure
    // since we need to be able to send data back from GPU to CPU
    uint64_t *total_number_particles;

    int *level_size;

    uint64_t get_total_number_particles() const { return *total_number_particles; }

    __device__ int level_max() const { return l_max; }
    __device__ int level_min() const { return l_min; }

} GenInfoCuda;

// -----------------------------

/*
 * Class for easy transfering to/from GPU of GenInfo structure.
 */
class GenInfoGpuAccess {
    GenInfo &gi;

    cudaStream_t iStream;

    ScopedCudaMemHandler<int*, H2D | D2H> org_dims;
    ScopedCudaMemHandler<int*, H2D | D2H> x_num;
    ScopedCudaMemHandler<int*, H2D | D2H> y_num;
    ScopedCudaMemHandler<int*, H2D | D2H> z_num;

    // total_number_particles_pinned is used as "middle" variable for transfering value to/from GPU
    // The reason behind this is that original GenInfo structure has total_number_particles as unpinned memory.
    // When transfering data is causes to all streams synchronize. To avoid that we do the following:
    // GPU.total_number_particles -> total_number_particles_pinned -> CPU.total_number_particles
    // and opposite direction when copying into GPU.
    // In that way we do not 'break' streams.
    VectorData<uint64_t> total_number_particles_pinned;
    ScopedCudaMemHandler<uint64_t*, H2D | D2H> total_number_particles;

    ScopedCudaMemHandler<int*, H2D | D2H> level_size;


public:
    GenInfoGpuAccess(GenInfo &genInfo, cudaStream_t cudaStream) :
        gi(genInfo),
        iStream(cudaStream),
        org_dims(gi.org_dims, 3, iStream),
        x_num(gi.x_num.data(), gi.x_num.size(), iStream),
        y_num(gi.y_num.data(), gi.y_num.size(), iStream),
        z_num(gi.z_num.data(), gi.z_num.size(), iStream),
        total_number_particles_pinned(true),
        level_size(gi.level_size.data(), gi.level_size.size(), iStream)
    {
        total_number_particles_pinned.resize(1);
        total_number_particles_pinned[0] = gi.total_number_particles;
        total_number_particles.initialize(total_number_particles_pinned.data(), 1, iStream);
    }

    GenInfoCuda getGenInfoCuda() {
        GenInfoCuda gic;

        gic.l_min = gi.l_min;
        gic.l_max = gi.l_max;
        gic.org_dims = org_dims.get();
        gic.number_dimensions = gi.number_dimensions;
        gic.x_num = x_num.get();
        gic.y_num = y_num.get();
        gic.z_num = z_num.get();
        gic.total_number_particles = total_number_particles.get();
        gic.level_size = level_size.get();

        return gic;
    }

    ~GenInfoGpuAccess() {
        // TODO: When freeing stream in GpuProcessingTaskImpl is fixed this should be uncommented
        // copyDtoH();
    }

    void copyHtoD() {
        // The only data that can change between CPU & GPU (the rest values are fixed based on input image dimension)

        // Check description of 'total_number_particles_pinned' for explanation
        total_number_particles_pinned[0] = gi.total_number_particles;
        total_number_particles.copyH2D();
    }

    void copyDtoH() {
        // The only data that can change between CPU & GPU (the rest values are fixed based on input image dimension)

        // Check description of 'total_number_particles_pinned' for explanation
        total_number_particles.copyD2H();
        checkCuda(cudaStreamSynchronize(iStream));
        gi.total_number_particles = total_number_particles_pinned[0];
    }
};

#endif // GEN_INFO_GPU_ACCESS_CUH