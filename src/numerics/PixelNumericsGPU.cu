//
// Created by joel on 07.04.20.
//
#include "PixelNumericsGPU.hpp"

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_333_chunked(const inputType* input_image,
                                       outputType* output_image,
                                       const stencilType* stencil,
                                       const int z_num,
                                       const int x_num,
                                       const int y_num) {

    const int block_dim = 4;

    const int x_index = blockIdx.x * 2 + threadIdx.y - 1;
    const int z_index = blockIdx.z * 2 + threadIdx.z - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    __syncthreads();

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {

        // out of bounds --> return
        return;
    }

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < (block_dim - 1)) &&
                           (threadIdx.z > 0) && (threadIdx.z < (block_dim - 1));
    //(threadIdx.y > 0) && (threadIdx.y < (blockDim.y - 1));


    const size_t row_begin = x_index * y_num + z_index * x_num * y_num;

    __syncthreads();

    int y_0;

    // overlapping y chunks

    const int chunkSizeInternal = chunkSize-2;
    const int number_y_chunks = (y_num + chunkSizeInternal - 1) / chunkSizeInternal;

    __syncthreads();

    for(int y_chunk = 0; y_chunk < number_y_chunks; ++y_chunk) {

        __syncthreads();

        y_0 = y_chunk * chunkSizeInternal - 1 + threadIdx.x;

        if( (y_0 >= 0) && (y_0 < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % N] = input_image[row_begin + y_0];
        }

        __syncthreads();

        if( (y_0 >= (y_chunk*(chunkSizeInternal))) && (y_0 < min(((y_chunk+1)*(chunkSizeInternal)), y_num)) ) {

            float neighbour_sum = 0;
            LOCALPATCHCONV333_N(output_image, (row_begin+y_0), threadIdx.z, threadIdx.y, y_0 + 1, neighbour_sum)

        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void
__launch_bounds__(1024, 2)
conv_pixel_555_chunked(const inputType* input_image,
                       outputType* output_image,
                       const stencilType* stencil,
                       const int z_num,
                       const int x_num,
                       const int y_num) {

    //const int block_dim = 4;

    const int x_index = blockIdx.x * (blockSize-4) + threadIdx.y - 2;
    const int z_index = blockIdx.z * (blockSize-4) + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[5][5][5];

    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }

    __syncthreads();

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {

        // out of bounds --> return
        return;
    }

    const bool not_ghost = (threadIdx.y > 1) && (threadIdx.y < (blockSize - 2)) &&
                           (threadIdx.z > 1) && (threadIdx.z < (blockSize - 2));

    const size_t row_begin = x_index * y_num + z_index * x_num * y_num;

    __syncthreads();

    int y_0;

    // overlapping y chunks

    const int chunkSizeInternal = chunkSize-4;
    const int number_y_chunks = (y_num + chunkSizeInternal - 1) / chunkSizeInternal;

    __syncthreads();

    for(int y_chunk = 0; y_chunk < number_y_chunks; ++y_chunk) {

        __syncthreads();

        y_0 = y_chunk * chunkSizeInternal - 2 + threadIdx.x;

        if( (y_0 >= 0) && (y_0 < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+2) % N] = input_image[row_begin + y_0];
        }

        __syncthreads();

        if( (y_0 >= (y_chunk*(chunkSizeInternal))) && (y_0 < min(((y_chunk+1)*(chunkSizeInternal)), y_num)) ) {

            float neighbour_sum = 0;
            LOCALPATCHCONV555_N(output_image, (row_begin+y_0), threadIdx.z, threadIdx.y, (y_0+2), neighbour_sum)

        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_333_reflective(const inputType* input_image,
                                          outputType* output_image,
                                          const stencilType* stencil,
                                          const int z_num,
                                          const int x_num,
                                          const int y_num) {

    const int block_dim = 4;

    int x_index = blockIdx.x * 2 + threadIdx.y - 1;
    int z_index = blockIdx.z * 2 + threadIdx.z - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    __syncthreads();

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < (block_dim - 1)) &&
                     (threadIdx.z > 0) && (threadIdx.z < (block_dim - 1));

    if(x_index < 0) {
        x_index = -x_index;
        not_ghost = false;
    }

    if(x_index >= x_num) {
        x_index = x_num - 1 - (x_index - x_num + 1);
        not_ghost = false;
    }

    if(z_index < 0) {
        z_index = -z_index;
        not_ghost = false;
    }

    if(z_index >= z_num) {
        z_index = z_num - 1 - (z_index - z_num + 1);
        not_ghost = false;
    }

    const size_t row_begin = x_index * y_num + z_index * x_num * y_num;

    __syncthreads();

    int y_0;

    // overlapping y chunks

    const int chunkSizeInternal = chunkSize-2;
    const int number_y_chunks = (y_num + chunkSizeInternal - 1) / chunkSizeInternal;

    __syncthreads();

    for(int y_chunk = 0; y_chunk < number_y_chunks; ++y_chunk) {

        __syncthreads();

        y_0 = y_chunk * chunkSizeInternal - 1 + threadIdx.x;

        if( (y_0 >= 0) && (y_0 < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % N] = input_image[row_begin + y_0];
        }

        __syncthreads();

        if(y_chunk == 0) {
            if(threadIdx.x == 0) {
                local_patch[threadIdx.z][threadIdx.y][0] = local_patch[threadIdx.z][threadIdx.x][2];
            }
        }

        if( (y_chunk+1)*chunkSizeInternal > y_num-1 ) {
            if(threadIdx.x == 0) {
                local_patch[threadIdx.z][threadIdx.y][(y_num+1) % N] = local_patch[threadIdx.z][threadIdx.y][(y_num-1) % N];
            }
        }

        __syncthreads();

        if( (y_0 >= (y_chunk*(chunkSizeInternal))) && (y_0 < min(((y_chunk+1)*(chunkSizeInternal)), y_num)) ) {

            float neighbour_sum = 0;
            LOCALPATCHCONV333_N(output_image, (row_begin+y_0), threadIdx.z, threadIdx.y, y_0 + 1, neighbour_sum)

        }
    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void
__launch_bounds__(1024, 2)
conv_pixel_555_reflective(const inputType* input_image,
                          outputType* output_image,
                          const stencilType* stencil,
                          const int z_num,
                          const int x_num,
                          const int y_num) {

    //const int block_dim = 4;

    int x_index = blockIdx.x * (blockSize-4) + threadIdx.y - 2;
    int z_index = blockIdx.z * (blockSize-4) + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[5][5][5];

    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }

    __syncthreads();

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    bool not_ghost = (threadIdx.y > 1) && (threadIdx.y < (blockSize - 2)) &&
                     (threadIdx.z > 1) && (threadIdx.z < (blockSize - 2));

    if(x_index < 0) {
        x_index = -x_index;
        not_ghost = false;
    }

    if(x_index >= x_num) {
        x_index = x_num - 1 - (x_index - x_num + 1);
        not_ghost = false;
    }

    if(z_index < 0) {
        z_index = -z_index;
        not_ghost = false;
    }

    if(z_index >= z_num) {
        z_index = z_num - 1 - (z_index - z_num + 1);
        not_ghost = false;
    }

    const size_t row_begin = x_index * y_num + z_index * x_num * y_num;

    __syncthreads();

    int y_0;

    // overlapping y chunks

    const int chunkSizeInternal = chunkSize-4;
    const int number_y_chunks = (y_num + chunkSizeInternal - 1) / chunkSizeInternal;

    __syncthreads();

    for(int y_chunk = 0; y_chunk < number_y_chunks; ++y_chunk) {

        __syncthreads();

        y_0 = y_chunk * chunkSizeInternal - 2 + threadIdx.x;

        if( (y_0 >= 0) && (y_0 < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+2) % N] = input_image[row_begin + y_0];
        }

        __syncthreads();

        if(y_chunk == 0) {
            if(threadIdx.x < 2) {
                local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = local_patch[threadIdx.z][threadIdx.x][4-threadIdx.x];
            }
        }

        if( (y_chunk+1)*chunkSizeInternal > y_num-2 ) {
            int limit = max( (y_chunk+1)*chunkSizeInternal - y_num + 2, 2 );
            if(threadIdx.x < limit) {
                local_patch[threadIdx.z][threadIdx.y][(y_num+2+threadIdx.x) % N] = local_patch[threadIdx.z][threadIdx.y][(y_num-threadIdx.x) % N];
            }
        }

        __syncthreads();

        if( (y_0 >= (y_chunk*(chunkSizeInternal))) && (y_0 < min(((y_chunk+1)*(chunkSizeInternal)), y_num)) ) {

            float neighbour_sum = 0;
            LOCALPATCHCONV555_N(output_image, (row_begin+y_0), threadIdx.z, threadIdx.y, (y_0+2), neighbour_sum)

        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    } // end for y_chunk
}



template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_333(inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, int y_num, int x_num, int z_num) {

    const int chunkSize = 32;
    const int blockSize = 4;

    int x_blocks = (x_num + blockSize - 3) / (blockSize - 2);
    int z_blocks = (z_num + blockSize - 3) / (blockSize - 2);

    dim3 blocks_l(x_blocks, 1, z_blocks);
    dim3 threads_l(chunkSize, blockSize, blockSize);

    conv_pixel_333_chunked<chunkSize, blockSize>
                                << < blocks_l, threads_l >> >
                                    (input_gpu, output_gpu, stencil_gpu, z_num, x_num, y_num);
}


template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_333_reflective(inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, int y_num, int x_num, int z_num) {

    const int chunkSize = 32;
    const int blockSize = 4;

    int x_blocks = (x_num + blockSize - 3) / (blockSize - 2);
    int z_blocks = (z_num + blockSize - 3) / (blockSize - 2);

    dim3 blocks_l(x_blocks, 1, z_blocks);
    dim3 threads_l(chunkSize, blockSize, blockSize);

    conv_pixel_333_reflective<chunkSize, blockSize>
                                 << < blocks_l, threads_l >> >
                                    (input_gpu, output_gpu, stencil_gpu, z_num, x_num, y_num);
}


template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_555(inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, int y_num, int x_num, int z_num) {

    const int chunkSize = 16;
    const int blockSize = 8;

    int x_blocks = (x_num + blockSize - 5) / (blockSize-4);
    int z_blocks = (z_num + blockSize - 5) / (blockSize-4);

    dim3 blocks_l(x_blocks, 1, z_blocks);
    dim3 threads_l(chunkSize, blockSize, blockSize);

    conv_pixel_555_chunked<chunkSize, blockSize>
                            << < blocks_l, threads_l >> >
                                (input_gpu, output_gpu, stencil_gpu, z_num, x_num, y_num);
}


template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_555_reflective(inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, int y_num, int x_num, int z_num) {

    const int chunkSize = 16;
    const int blockSize = 8;

    int x_blocks = (x_num + blockSize - 5) / (blockSize-4);
    int z_blocks = (z_num + blockSize - 5) / (blockSize-4);

    dim3 blocks_l(x_blocks, 1, z_blocks);
    dim3 threads_l(chunkSize, blockSize, blockSize);

    conv_pixel_555_reflective<chunkSize, blockSize>
                                << < blocks_l, threads_l >> >
                                    (input_gpu, output_gpu, stencil_gpu, z_num, x_num, y_num);
}


template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_333(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil, bool reflective_bc) {

    assert(stencil.mesh.size() == 27);

    if(output.size() < input.size()) {
        output.init(input);
    }

    /// allocate GPU memory
    ScopedCudaMemHandler<PixelData<inputType>, JUST_ALLOC> input_gpu(input);
    ScopedCudaMemHandler<PixelData<outputType>, JUST_ALLOC> output_gpu(output);
    ScopedCudaMemHandler<PixelData<stencilType>, JUST_ALLOC> stencil_gpu(stencil);

    /// copy input and stencil to the GPU
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    /// launch the kernel
    if(reflective_bc) {
        convolve_pixel_333_reflective(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
    } else {
        convolve_pixel_333(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
    }
    error_check( cudaDeviceSynchronize() )

    /// transfer the results back to the host
    output_gpu.copyD2H();
}


template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_555(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil, bool reflective_bc) {

    assert(stencil.mesh.size() == 125);

    if(output.size() < input.size()) {
        output.init(input);
    }

    /// allocate GPU memory
    ScopedCudaMemHandler<PixelData<inputType>, JUST_ALLOC> input_gpu(input);
    ScopedCudaMemHandler<PixelData<outputType>, JUST_ALLOC> output_gpu(output);
    ScopedCudaMemHandler<PixelData<stencilType>, JUST_ALLOC> stencil_gpu(stencil);

    /// copy input and stencil to the GPU
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    if(reflective_bc) {
        convolve_pixel_555_reflective(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
    } else {
        convolve_pixel_555(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
    }

    error_check( cudaDeviceSynchronize() )

    /// transfer the results back to the host
    output_gpu.copyD2H();
}



template<typename inputType, typename stencilType>
void richardson_lucy_pixel(inputType* input, stencilType* output, stencilType* psf, stencilType* psf_flipped, int kernel_size, int npixels, int niter, std::vector<int>& dims) {

    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> relative_blur(NULL, npixels);
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> error_est(NULL, npixels);

    dim3 grid_dim(8, 1, 1);
    dim3 block_dim(256, 1, 1);

    const int chunkSize = (kernel_size == 3) ? 32 : 16;
    const int convBlockSize = (kernel_size == 3) ? 4 : 8;

    const int x_blocks = (kernel_size == 3) ? (dims[1] + convBlockSize - 3) / (convBlockSize-2) : (dims[1] + convBlockSize - 5) / (convBlockSize-4);
    const int z_blocks = (kernel_size == 3) ? (dims[0] + convBlockSize - 3) / (convBlockSize-2) : (dims[0] + convBlockSize - 5) / (convBlockSize-4);

    dim3 conv_grid_dim(x_blocks, 1, z_blocks);
    dim3 conv_block_dim(chunkSize, convBlockSize, convBlockSize);

    /// initialize output with 1s
    fillWithValue<<< grid_dim, block_dim >>> (output, (stencilType)1, npixels);
    error_check( cudaDeviceSynchronize() )

    for(int i = 0; i < niter; ++i) {

        if(kernel_size == 5) {
            conv_pixel_555_reflective<16, 8><<< conv_grid_dim, conv_block_dim >>> (output, relative_blur.get(), psf_flipped, dims[0], dims[1], dims[2]);
        } else {
            conv_pixel_333_reflective<32, 4><<< conv_grid_dim, conv_block_dim >>> (output, relative_blur.get(), psf_flipped, dims[0], dims[1], dims[2]);
        }
        error_check( cudaDeviceSynchronize() )

        elementWiseDiv<< < grid_dim, block_dim >> > (input, relative_blur.get(), relative_blur.get(), npixels);
        error_check( cudaDeviceSynchronize() )

        if(kernel_size == 5) {
            conv_pixel_555_reflective<16, 8><<< conv_grid_dim, conv_block_dim >>> (relative_blur.get(), error_est.get(), psf, dims[0], dims[1], dims[2]);
        } else {
            conv_pixel_333_reflective<32, 4><<< conv_grid_dim, conv_block_dim >>> (relative_blur.get(), error_est.get(), psf, dims[0], dims[1], dims[2]);
        }
        error_check( cudaDeviceSynchronize() )

        elementWiseMult<<< grid_dim, block_dim >>> (output, error_est.get(), npixels);
        error_check( cudaDeviceSynchronize() )
    }
}


template<typename inputType, typename stencilType>
void richardson_lucy_pixel(PixelData<inputType>& input, PixelData<stencilType>& output, PixelData<stencilType>& psf, int niter) {

    assert(psf.z_num == psf.x_num && psf.x_num == psf.y_num);
    int kernel_size = psf.y_num;
    assert(kernel_size == 3 || kernel_size == 5);

    if(output.mesh.size() != input.mesh.size()) {
        output.init(input);
    }

    std::vector<int> dims = {static_cast<int>(input.z_num), static_cast<int>(input.x_num), static_cast<int>(input.y_num)};

    PixelData<stencilType> psf_flipped(psf, false);
    for(int i = 0; i < psf.mesh.size(); ++i) {
        psf_flipped.mesh[i] = psf.mesh[psf.mesh.size() - 1 - i];
    }

    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.mesh.get(), input.mesh.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> output_gpu(output.mesh.get(), output.mesh.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> psf_gpu(psf.mesh.get(), psf.mesh.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> psf_flipped_gpu(psf.mesh.get(), psf.mesh.size());

    input_gpu.copyH2D();
    psf_gpu.copyH2D();
    psf_flipped_gpu.copyH2D();

    richardson_lucy_pixel(input_gpu.get(), output_gpu.get(), psf_gpu.get(), psf_flipped_gpu.get(), kernel_size, input.mesh.size(), niter, dims);

    output_gpu.copyD2H();
}


/// force template instantiation for some different type combinations
//pixels 333
template void convolve_pixel_333(PixelData<uint16_t>&, PixelData<float>&, PixelData<float>&, bool);
template void convolve_pixel_333(PixelData<uint16_t>&, PixelData<double>&, PixelData<double>&, bool);
template void convolve_pixel_333(PixelData<float>&, PixelData<float>&, PixelData<float>&, bool);

template void convolve_pixel_333(uint16_t* input_gpu, float* output_gpu, float* stencil_gpu, int y_num, int x_num, int z_num);
template void convolve_pixel_333(float* input_gpu, float* output_gpu, float* stencil_gpu, int y_num, int x_num, int z_num);

template void convolve_pixel_333_reflective(uint16_t* input_gpu, float* output_gpu, float* stencil_gpu, int y_num, int x_num, int z_num);
template void convolve_pixel_333_reflective(float* input_gpu, float* output_gpu, float* stencil_gpu, int y_num, int x_num, int z_num);

//pixels 555
template void convolve_pixel_555(PixelData<uint16_t>&, PixelData<float>&, PixelData<float>&, bool);
template void convolve_pixel_555(PixelData<uint16_t>&, PixelData<double>&, PixelData<double>&, bool);
template void convolve_pixel_555(PixelData<float>&, PixelData<float>&, PixelData<float>&, bool);

template void convolve_pixel_555(uint16_t* input_gpu, float* output_gpu, float* stencil_gpu, int y_num, int x_num, int z_num);
template void convolve_pixel_555(float* input_gpu, float* output_gpu, float* stencil_gpu, int y_num, int x_num, int z_num);

template void convolve_pixel_555_reflective(uint16_t* input_gpu, float* output_gpu, float* stencil_gpu, int y_num, int x_num, int z_num);
template void convolve_pixel_555_reflective(float* input_gpu, float* output_gpu, float* stencil_gpu, int y_num, int x_num, int z_num);

// richardson lucy
template void richardson_lucy_pixel(float*, float*, float*, float*, int, int, int, std::vector<int>&);
template void richardson_lucy_pixel(uint16_t*, float*, float*, float*, int, int, int, std::vector<int>&);

template void richardson_lucy_pixel(PixelData<float>&, PixelData<float>&, PixelData<float>&, int);
template void richardson_lucy_pixel(PixelData<uint16_t>&, PixelData<float>&, PixelData<float>&, int);

