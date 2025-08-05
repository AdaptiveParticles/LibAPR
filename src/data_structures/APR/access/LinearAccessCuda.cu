#include "LinearAccessCuda.hpp"

#include "misc/CudaTools.cuh"
#include "algorithm/ParticleCellTreeCuda.cuh"


// *********************************************************************************************************************
//                       FULL RESOLUTION
// *********************************************************************************************************************
/**
 * Handle edge case for #levels <= 2
 * For performance reasons and clarity of the code,
 * it doesn't make sense here to handle these cases.
 * Below assumes there is at least levels <=2;
 * @param level_xz
 * @param xz_end
 * @param y
 * @param gic - cuda version of GenInfo
 */
__global__ void fullResolution(const uint64_t *level_xz, uint64_t *xz_end, uint16_t *y, GenInfoCuda gic) {

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    const unsigned levelMax = gic.level_max();
    const uint64_t xMax = gic.x_num[levelMax];
    const uint64_t yMax = gic.y_num[levelMax];
    const uint64_t zMax = gic.z_num[levelMax];


    if (x < xMax && z < zMax) {
        const uint64_t levelStart = level_xz[levelMax];
        uint64_t offset_pc_data = z * xMax + x;
        uint64_t particleCounter = (1 + x + z * xMax) * yMax;

        xz_end[levelStart + offset_pc_data] = particleCounter;

        for (int i = 0; i < yMax; ++i) {
            uint64_t idx = (xMax * z + x) * yMax + i;
            y[idx] = i;
        }
    }

    if (x == 0 && z == 0) {
        *gic.total_number_particles = xMax * yMax * zMax;
    }
}

void runFullResolution(const uint64_t *level_xz, uint64_t *xz_end, uint16_t *y, const GenInfo &gi, GenInfoGpuAccess &giga, cudaStream_t aStream) {
    dim3 threadsPerBlock(32, 1, 1);

    dim3 numBlocks( (gi.x_num[gi.l_max] + threadsPerBlock.x - 1)/threadsPerBlock.x,
                    1,
                    (gi.z_num[gi.l_max] + threadsPerBlock.z - 1)/threadsPerBlock.z);
    fullResolution<<<numBlocks, threadsPerBlock, 0, aStream>>>(level_xz, xz_end, y, giga.getGenInfoCuda());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("----------------------------------Error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("runFullResolution failed");
    }
}


// *********************************************************************************************************************
//                       FIRST STEP
// *********************************************************************************************************************

static constexpr uint8_t seed_us = UPSAMPLING_SEED_TYPE; //deal with the equivalence optimization


__global__ void firstStepLAC(const uint8_t *prevLevel, uint8_t *currLevel, int level, uint8_t min_type, GenInfoCuda gic) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    const uint64_t xLen = gic.x_num[level];
    const uint64_t yLen = gic.y_num[level];
    const uint64_t zLen = gic.z_num[level];
    const uint64_t xLenDS = gic.x_num[level - 1];
    const uint64_t yLenDS = gic.y_num[level - 1];

    if (x < xLen && z < zLen) {
        const size_t offset_part_map_ds = (x / 2) * yLenDS + (z / 2) * yLenDS * xLenDS;
        const size_t offset_part_map = x * yLen + z * yLen * xLen;

        for (size_t y = 0; y < yLenDS; ++y) {
            uint8_t  status = prevLevel[offset_part_map_ds + y];
            if (status > 0 && status <= min_type) {
                currLevel[offset_part_map + 2 * y] = seed_us;                    // 2 * y
                currLevel[offset_part_map + min(2 * y + 1, yLen - 1)] = seed_us; // 2 * y + 1
            }
        }
    }
}

void runFirstStep(const GenInfo &gi, GenInfoGpuAccess &giga, ParticleCellTreeCuda &p_map, uint8_t min_type, cudaStream_t aStream) {
    dim3 threadsPerBlock(128, 1, 1);
    auto genInfoCuda = giga.getGenInfoCuda();
    for (int level = gi.l_min + 1; level < gi.l_max; ++level) {
        dim3 numBlocks( (gi.x_num[level] + threadsPerBlock.x - 1)/threadsPerBlock.x,
                        1,
                        (gi.z_num[level] + threadsPerBlock.z - 1)/threadsPerBlock.z);
        auto *p_mapPrev = p_map[level - 1];
        auto *p_mapCurr = p_map[level];
        firstStepLAC<<<numBlocks, threadsPerBlock, 0, aStream>>>(p_mapPrev, p_mapCurr, level, min_type, genInfoCuda);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("----------------------------------Error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("runFirstStep failed");
    }
}


// *********************************************************************************************************************
//                       SECOND STEP
// *********************************************************************************************************************


__global__ void secondStep(const uint8_t *currLevel, int level, uint8_t min_type, GenInfoCuda gic, const uint64_t *level_xz, uint64_t *xz_end) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    const uint64_t xLen = gic.x_num[level];
    const uint64_t yLen = gic.y_num[level];
    const uint64_t zLen = gic.z_num[level];

    const uint64_t level_start = level_xz[level];

    if (x < xLen && z < zLen) {
        const size_t offset_pc_data = z * xLen + x;
        const size_t offset_part_map = yLen * offset_pc_data;

        uint64_t counter = 0;

        for (size_t y = 0; y < yLen; ++y) {
            uint8_t  status = currLevel[offset_part_map + y];
            if (status > min_type && status <= UPSAMPLING_SEED_TYPE) {
                counter++;
            }
        }

        xz_end[level_start + offset_pc_data] = counter;
    }
}

void runSecondStep(const GenInfo &gi, GenInfoGpuAccess &giga, ParticleCellTreeCuda &p_map, uint8_t min_type, const uint64_t *level_xz, uint64_t *xz_end, cudaStream_t aStream) {
    dim3 threadsPerBlock(32, 1, 1);

    for (int level = gi.l_min; level < gi.l_max - 1; ++level) {
        dim3 numBlocks( (gi.x_num[level] + threadsPerBlock.x - 1)/threadsPerBlock.x,
                        1,
                        (gi.z_num[level] + threadsPerBlock.z - 1)/threadsPerBlock.z);
        auto *p_mapCurr = p_map[level];
        secondStep<<<numBlocks, threadsPerBlock, 0, aStream>>>(p_mapCurr, level, min_type, giga.getGenInfoCuda(), level_xz, xz_end);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("----------------------------------Error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("runSecondStep failed");
    }
}


// *********************************************************************************************************************
//                       SECOND STEP LAST LEVEL
//
//    l_max - 1 is special as it also has the l_max information that then needs to be upsampled.
// *********************************************************************************************************************


__global__ void secondStepLastLevel(const uint8_t *currLevel, int level_minus_1, uint8_t min_type, GenInfoCuda gic, const uint64_t *level_xz, uint64_t *xz_end) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    const uint64_t xLen = gic.x_num[level_minus_1];
    const uint64_t yLen = gic.y_num[level_minus_1];
    const uint64_t zLen = gic.z_num[level_minus_1];

    const uint64_t  xLen_m = gic.x_num[level_minus_1 + 1]; // level max
    const uint64_t  yLen_m = gic.y_num[level_minus_1 + 1]; // level max
    const uint64_t  zLen_m = gic.z_num[level_minus_1 + 1]; // level max

    const uint64_t level_start = level_xz[level_minus_1];
    const uint64_t level_start_m = level_xz[level_minus_1 + 1]; // level max


    if (x < xLen && z < zLen) {
        const size_t offset_pc_data = z * xLen + x;
        const size_t offset_part_map = yLen * offset_pc_data;

        uint64_t counter = 0;
        uint64_t counter_l = 0;

        for (size_t y = 0; y < yLen; ++y) {
            uint8_t  status = currLevel[offset_part_map + y];
            if (status > min_type && status <= UPSAMPLING_SEED_TYPE) {
                counter++;
            }
            else if (status > 0 && status <= min_type) {
                counter_l++;

                if ((2 * y) < (yLen_m - 1)) {
                    counter_l++;
                }
            }
        }

        xz_end[level_start + offset_pc_data] = counter;

        // In original CPU code value of counter_l is remembered in temporary buffer and later
        // write down to xz_end vector. Here is the solution without need of temp. buffer.
        for (size_t dz = 0; dz <= 1; dz++) {
            for (size_t dx = 0; dx <= 1; dx++) {
                size_t uz = 2 * z + dz; // upsampled z
                size_t ux = 2 * x + dx; // upsampled x
                if (uz < zLen_m && ux < xLen_m) {
                    const size_t offset_pc_data_m = uz * xLen_m + ux;
                    xz_end[level_start_m + offset_pc_data_m] = counter_l;
                }
            }
        }

    }
}


__global__ void secondStepCountParticles(GenInfoCuda gic, uint64_t *xz_end, uint64_t counter_total) {
    extern __shared__ uint64_t b[];
    const int idx = threadIdx.x;
    // std::partial_sum on one CUDA core implementation
    //
    // How it works:
    // 1. Each thread reads a value from xz_end into shared memory as coalesced memory access to speed up process
    // 2. The first thread (idx == 0) computes the partial sum for the block (this is fast enough)
    // 3. The partial sums are written back to xz_end in coalesced memory manner.
    // In general sice this is multi-warp code __syncthreads() are used to be sure that all threads are at same
    // place in code.

    uint64_t sum = 0;
    for (uint64_t i = 0; i < counter_total; i+= blockDim.x) {
        if ((i + idx) < counter_total) b[idx] = xz_end[i + idx];
        __syncthreads();

        if (idx == 0) {
            size_t endIdx = blockDim.x;
            if ((i + blockDim.x - 1) >= counter_total) {
                endIdx = counter_total - i;
            }
            for (size_t n = 0; n < endIdx; n++) {
                sum += b[n];
                b[n] = sum;
            }
        }
        __syncthreads();

        if ((i + idx) < counter_total) xz_end[i + idx] = b[idx];
        __syncthreads();
    }

    if (idx == 0) *gic.total_number_particles = xz_end[counter_total - 1];
}

void runSecondStepLastLevel(const GenInfo &gi, GenInfoGpuAccess &giga, ParticleCellTreeCuda &p_map, uint8_t min_type, const uint64_t *level_xz, uint64_t *xz_end, uint64_t counter_total, cudaStream_t aStream) {
    dim3 threadsPerBlock(32, 1, 1);
    dim3 numBlocks( (gi.x_num[gi.l_max - 1] + threadsPerBlock.x - 1)/threadsPerBlock.x,
                    1,
                    (gi.z_num[gi.l_max - 1] + threadsPerBlock.z - 1)/threadsPerBlock.z);

    int level = gi.l_max - 1;
    auto *p_mapCurr = p_map[level];
    secondStepLastLevel<<<numBlocks, threadsPerBlock, 0, aStream>>>(p_mapCurr, level, min_type, giga.getGenInfoCuda(), level_xz, xz_end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("----------------------------------Error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("runSecondStepLastLevel #1 failed");
    }

    int numOfWorkers = 128;
    secondStepCountParticles<<<1, numOfWorkers, numOfWorkers * sizeof(uint64_t), aStream>>>(giga.getGenInfoCuda(), xz_end, counter_total);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("----------------------------------Error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("runSecondStepLastLevel #2 failed");
    }
}


// *********************************************************************************************************************
//                       THIRD STEP - Get Y values
// *********************************************************************************************************************


__global__ void getYvalues(const uint8_t *currLevel, int level, uint8_t min_type, GenInfoCuda gic, const uint64_t *level_xz, uint64_t *xz_end, uint16_t *y_vec) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    const uint64_t xLen = gic.x_num[level];
    const uint64_t yLen = gic.y_num[level];
    const uint64_t zLen = gic.z_num[level];

    const uint64_t level_start = level_xz[level];

    if (x < xLen && z < zLen) {
        const size_t offset_pc_data = z * xLen + x;
        const size_t offset_part_map = yLen * offset_pc_data;

        uint64_t counter = 0;

        uint64_t offset_y  = xz_end[level_start + offset_pc_data - 1];

        for (size_t y = 0; y < yLen; ++y) {
            uint8_t  status = currLevel[offset_part_map + y];
            if (status > min_type && status <= UPSAMPLING_SEED_TYPE) {
                y_vec[counter + offset_y] = y;
                counter++;
            }
        }
    }
}

void runGetYvalues(const GenInfo &gi, GenInfoGpuAccess &giga, ParticleCellTreeCuda &p_map, uint8_t min_type, const uint64_t *level_xz, uint64_t *xz_end, uint16_t *y_vec, cudaStream_t aStream) {
    dim3 threadsPerBlock(32, 1, 1);

    for (int level = gi.l_min; level < gi.l_max - 1; ++level) {
        dim3 numBlocks( (gi.x_num[level] + threadsPerBlock.x - 1)/threadsPerBlock.x,
                        1,
                        (gi.z_num[level] + threadsPerBlock.z - 1)/threadsPerBlock.z);
        auto *p_mapCurr = p_map[level];
        getYvalues<<<numBlocks, threadsPerBlock, 0, aStream>>>(p_mapCurr, level, min_type, giga.getGenInfoCuda(), level_xz, xz_end, y_vec);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("----------------------------------Error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("runGetYvalues failed");
    }
}


// *********************************************************************************************************************
//                       4th STEP LAST LEVEL
//
//    l_max - 1 is special as it also has the l_max information that then needs to be upsampled.
// *********************************************************************************************************************


__global__ void fourthStep(const uint8_t *currLevel, int level_minus_1, uint8_t min_type, GenInfoCuda gic, const uint64_t *level_xz, uint64_t *xz_end, uint16_t *y_vec) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    const uint64_t xLen = gic.x_num[level_minus_1];
    const uint64_t yLen = gic.y_num[level_minus_1];
    const uint64_t zLen = gic.z_num[level_minus_1];

    const uint64_t  xLen_m = gic.x_num[level_minus_1 + 1]; // level max
    const uint64_t  yLen_m = gic.y_num[level_minus_1 + 1]; // level max

    const uint64_t level_start_minus_1 = level_xz[level_minus_1];
    const uint64_t level_start_m = level_xz[level_minus_1 + 1]; // level max


    if (x < xLen && z < zLen) {
        const size_t offset_pc_data = z * xLen + x;

        const size_t offset_pc_data_m = (z*2) * xLen_m + x * 2;
        const size_t offset_part_map = yLen * offset_pc_data; // current level

        uint64_t counter = 0;
        uint64_t counter_l = 0;

        uint64_t offset_y = xz_end[level_start_minus_1 + offset_pc_data - 1];
        uint64_t offset_y_m = xz_end[level_start_m + offset_pc_data_m -1];

        for (size_t y = 0; y < yLen; ++y) {
            uint8_t  status = currLevel[offset_part_map + y];
            if (status > min_type && status <= UPSAMPLING_SEED_TYPE) {
                y_vec[counter + offset_y] = y;
                counter++;
            }
            else if (status > 0 && status <= min_type) {
                y_vec[counter_l + offset_y_m] = 2*y;
                counter_l++;

                if ((2 * y) < (yLen_m - 1)) {
                    y_vec[counter_l + offset_y_m] = 2*y + 1;
                    counter_l++;
                }
            }
        }
    }
}

__global__ void fourthStepLastLevel(GenInfoCuda gic, const uint64_t *level_xz, uint64_t *xz_end, uint16_t *y_vec) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    int maxLevel = gic.level_max();
    const uint64_t  xLen_m = gic.x_num[maxLevel]; // level max
    const uint64_t  zLen_m = gic.z_num[maxLevel]; // level max

    const uint64_t level_start_m = level_xz[maxLevel];


    if (x < xLen_m && z < zLen_m) {

        // first check if it's not already there
        if ( ((z % 2) != 0) || ((x % 2) != 0) ) {
            const size_t offset_pc_data_m = z * xLen_m + x;
            const size_t offset_pc_data_m_f = (z/2) * 2 * xLen_m + (x/2) * 2;

            uint64_t offset_y_b_f = xz_end[level_start_m + offset_pc_data_m_f - 1];
            uint64_t offset_y_e_f = xz_end[level_start_m + offset_pc_data_m_f];
            uint64_t offset_y_b   = xz_end[level_start_m + offset_pc_data_m - 1];

            for (uint64_t idx = offset_y_b_f; idx < offset_y_e_f; ++idx) {
                y_vec[offset_y_b++] = y_vec[idx];
            }
        }

    }
}

void runFourthStep(const GenInfo &gi, GenInfoGpuAccess &giga, ParticleCellTreeCuda &p_map, uint8_t min_type, const uint64_t *level_xz, uint64_t *xz_end, uint16_t *y_vec, uint64_t counter_total, cudaStream_t aStream) {
    dim3 threadsPerBlock(32, 1, 1);
    dim3 numBlocks( (gi.x_num[gi.l_max] + threadsPerBlock.x - 1)/threadsPerBlock.x,
                    1,
                    (gi.z_num[gi.l_max] + threadsPerBlock.z - 1)/threadsPerBlock.z);

    int level = gi.l_max - 1;
    auto *p_mapCurr = p_map[level];
    fourthStep<<<numBlocks, threadsPerBlock, 0, aStream>>>(p_mapCurr, level, min_type, giga.getGenInfoCuda(), level_xz, xz_end, y_vec);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("----------------------------------Error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("runFourthStep #1 failed");
    }

    fourthStepLastLevel<<<numBlocks, threadsPerBlock, 0, aStream>>>(giga.getGenInfoCuda(), level_xz, xz_end, y_vec);

    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("----------------------------------Error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("runFourthStep #2 failed");
    }
}


// *********************************************************************************************************************
//   MAIN FUNC TO CALL - implements logic of  LinearAccess::initialize_linear_structure CPU func.
// *********************************************************************************************************************


/*
 * This function is for testing purposes only.
 *
 * This function does everything:
 * - creates CPU structures
 * - copies everything to GPU
 * - run computation of all linear-structures
 * - copy it back to CPU
 * - returns all the structure
 */
LinearAccessCudaStructs initializeLinearStructureCuda(GenInfo &gi, const APRParameters &apr_parameters, std::vector<PixelData<uint8_t>> &pct) {

    cudaStream_t aStream = nullptr;

    // Copy input to CUDA mem and prepare CUDA representation of particle cell tree which will be filled after computing
    // all steps
    ParticleCellTreeCuda p_map (gi, aStream);
    p_map.uploadPCT2GPU(pct);

    uint8_t min_type = apr_parameters.neighborhood_optimization ? 1 : 2;

    VectorData<uint16_t> y_vec(true);
    VectorData<uint64_t> xz_end_vec(true);
    VectorData<uint64_t> level_xz_vec(true);

    // initialize_xz_linear() - CPU impl.
    uint64_t counter_total = 1; //the buffer val to allow -1 calls without checking.
    level_xz_vec.resize(gi.l_max + 2, 0); //includes a buffer for -1 calls, and therefore needs to be called with level + 1;
    level_xz_vec[0] = 1; //allowing for the offset.
    for (int i = 0; i <= gi.l_max; ++i) {
        counter_total += gi.x_num[i] * gi.z_num[i];
        level_xz_vec[i + 1] = counter_total;
    }
    xz_end_vec.resize(counter_total, 0);

    // auto prt = [&](const auto& v){ std::cout << "size=" << v.size() << " data="; for (size_t i = 0; i < v.size(); i++) std::cout << v[i] << ", "; std::cout << std::endl; };
    // prt(y_vec);
    // prt(xz_end_vec);
    // prt(level_xz_vec);

    // TODO: This is temporary solution.
    //       Since in CPU code size of y_vec is calculated 'on the fly' and in CUDA code it would be much better
    //       to have pre-allocated memory for that - currently y_vec is pre-allocated to have maximum size. This is not
    //       optimal but always working solution. If any better idea pop up - it will be changed.
    size_t maxYvecSize = gi.x_num[gi.l_max] * gi.y_num[gi.l_max] * gi.z_num[gi.l_max];
    y_vec.resize(maxYvecSize);


    {
        ScopedCudaMemHandler<uint16_t *, D2H> y_vec_cuda(y_vec.data(), y_vec.size(), aStream);
        ScopedCudaMemHandler<uint64_t *, D2H> xz_end_vec_cuda(xz_end_vec.data(), xz_end_vec.size(), aStream);
        ScopedCudaMemHandler<uint64_t *, H2D | D2H> level_xz_vec_cuda(level_xz_vec.data(), level_xz_vec.size(), aStream);
        GenInfoGpuAccess giga(gi, aStream);
        if (gi.l_max <= 2) {
            runFullResolution(level_xz_vec_cuda.get(), xz_end_vec_cuda.get(), y_vec_cuda.get(), gi, giga, aStream);
        }
        else {
            runFirstStep(gi, giga, p_map, min_type, aStream);
            runSecondStep(gi, giga, p_map, min_type, level_xz_vec_cuda.get(), xz_end_vec_cuda.get(), aStream);
            runSecondStepLastLevel(gi, giga, p_map, min_type, level_xz_vec_cuda.get(), xz_end_vec_cuda.get(), counter_total, aStream);
            runGetYvalues(gi, giga, p_map, min_type, level_xz_vec_cuda.get(), xz_end_vec_cuda.get(), y_vec_cuda.get(), aStream);
            runFourthStep(gi, giga, p_map, min_type, level_xz_vec_cuda.get(), xz_end_vec_cuda.get(), y_vec_cuda.get(), counter_total, aStream);
        }
        // TODO: This is temporary solution. Fix GIGA destructor and stream handling!
        giga.copyDtoH();
    }

    // TODO: Resized back to correct size, should it be initialized to this size in the first place or pre-allocation for
    //       full size is more than enough? (for example in case of computing particles for multiple frames with same resolution
    //       we can get different size of particles for each frame - with preallocated buffer we can do all of them on it).
    y_vec.resize(gi.total_number_particles);

    // Transfer changes to PCT from GPU to CPU (this is needed only for tests)
    p_map.downloadPCTfromGPU(pct);


    LinearAccessCudaStructs lac;
    lac.y_vec.swap(y_vec);
    lac.xz_end_vec.swap(xz_end_vec);
    lac.level_xz_vec.swap(level_xz_vec);

    return lac;
}

void computeLinearStructureCuda(uint16_t *y_vec_cuda, uint64_t *xz_end_vec_cuda, const uint64_t *level_xz_vec_cuda, ParticleCellTreeCuda &p_map, GenInfo &gi, GenInfoGpuAccess &giga, const APRParameters &apr_parameters, uint64_t counter_total, cudaStream_t aStream) {

    const uint8_t min_type = apr_parameters.neighborhood_optimization ? 1 : 2;

    if (gi.l_max <= 2) {
        runFullResolution(level_xz_vec_cuda, xz_end_vec_cuda, y_vec_cuda, gi, giga, aStream);
    }
    else {
        runFirstStep(gi, giga, p_map, min_type, aStream);
        runSecondStep(gi, giga, p_map, min_type, level_xz_vec_cuda, xz_end_vec_cuda, aStream);
        runSecondStepLastLevel(gi, giga, p_map, min_type, level_xz_vec_cuda, xz_end_vec_cuda, counter_total, aStream);
        runGetYvalues(gi, giga, p_map, min_type, level_xz_vec_cuda, xz_end_vec_cuda, y_vec_cuda, aStream);
        runFourthStep(gi, giga, p_map, min_type, level_xz_vec_cuda, xz_end_vec_cuda, y_vec_cuda, counter_total, aStream);
    }
}
