//
// Created by joel on 08.04.20.
//

#include "APRNumericsGPU.hpp"


template<typename InputType>
void APRNumericsGPU::gradient_magnitude(GPUAccessHelper &access, GPUAccessHelper &tree_access,
                                        VectorData<InputType> &inputParticles, VectorData<float> &outputParticles,
                                        VectorData<float> &stencil_vec_y, VectorData<float> &stencil_vec_x,
                                        VectorData<float> &stencil_vec_z) {

    // initialize GPU access data
    tree_access.init_gpu();
    access.init_gpu(tree_access);

    // check stencils
    assert(stencil_vec_y.size() >= 27);
    assert(stencil_vec_x.size() >= 27);
    assert(stencil_vec_z.size() >= 27);
    const bool downsampled_y = (stencil_vec_y.size() >= 27 * (access.level_max() - access.level_min()));
    const bool downsampled_x = (stencil_vec_x.size() >= 27 * (access.level_max() - access.level_min()));
    const bool downsampled_z = (stencil_vec_z.size() >= 27 * (access.level_max() - access.level_min()));

    // initialize output
    outputParticles.resize(access.total_number_particles());

    // find non-empty rows
    VectorData<int> ne_counter_ds;
    VectorData<int> ne_counter_333;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_333_gpu;
    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
    compute_ne_rows_cuda<16, 32>(access, ne_counter_333, ne_rows_333_gpu, 2);

    // allocate GPU memory
    ScopedCudaMemHandler<InputType*, H2D> input_gpu(inputParticles.data(), inputParticles.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> output_gpu(outputParticles.data(), outputParticles.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> tmp_output(NULL, access.total_number_particles());
    ScopedCudaMemHandler<float*, JUST_ALLOC> tree_gpu(NULL, tree_access.total_number_particles());
    ScopedCudaMemHandler<float*, H2D> stencil_y_gpu(stencil_vec_y.data(), stencil_vec_y.size());
    ScopedCudaMemHandler<float*, H2D> stencil_x_gpu(stencil_vec_x.data(), stencil_vec_x.size());
    ScopedCudaMemHandler<float*, H2D> stencil_z_gpu(stencil_vec_z.data(), stencil_vec_z.size());

    // compute block and grid size for elementwise operations
    const size_t N = access.total_number_particles();
    int blockSize, minGridSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, addSquare, 0, 0);
    gridSize = (N + blockSize - 1) / blockSize;

    // fill tree by average downsampling
    downsample_avg(access, tree_access, input_gpu.get(), tree_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    error_check( cudaDeviceSynchronize() )

    // compute y gradient
    isotropic_convolve_333_reflective(access, tree_access, input_gpu.get(), output_gpu.get(),
                                      stencil_y_gpu.get(), tree_gpu.get(), ne_rows_333_gpu.get(),
                                      ne_counter_333, downsampled_y);
    error_check( cudaDeviceSynchronize() )

    // square y gradient
    elementWiseMult<<<blockSize, gridSize>>>(output_gpu.get(), output_gpu.get(), N);

    // compute x gradient
    isotropic_convolve_333_reflective(access, tree_access, input_gpu.get(), tmp_output.get(),
                                      stencil_x_gpu.get(), tree_gpu.get(), ne_rows_333_gpu.get(),
                                      ne_counter_333, downsampled_x);

    error_check( cudaDeviceSynchronize() )

    // add square of x gradient to output
    addSquare<<<blockSize, gridSize>>>(output_gpu.get(), tmp_output.get(), N);

    error_check( cudaDeviceSynchronize() )

    // compute z gradient
    isotropic_convolve_333_reflective(access, tree_access, input_gpu.get(), tmp_output.get(),
                                      stencil_z_gpu.get(), tree_gpu.get(), ne_rows_333_gpu.get(),
                                      ne_counter_333, downsampled_z);

    error_check( cudaDeviceSynchronize() )

    // add square of x gradient to output
    addSquare<<<blockSize, gridSize>>>(output_gpu.get(), tmp_output.get(), N);

    error_check( cudaDeviceSynchronize() )

    elementWiseSqrt<<<blockSize, gridSize>>>(output_gpu.get(), N);

    error_check( cudaDeviceSynchronize() )

    output_gpu.copyD2H();
}


template<typename inputType, typename stencilType>
void APRNumericsGPU::richardson_lucy(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input,
                                     stencilType* output, stencilType* psf, stencilType* psf_flipped, int kernel_size,
                                     int niter, bool use_stencil_downsample, bool resume) {

    VectorData<int> ne_counter_ds;
    VectorData<int> ne_counter_333;
    VectorData<int> ne_counter_555;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_333_gpu;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_555_gpu;

    /// non-empty rows precalculation (can be reused in all iterations)
    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
    if(kernel_size == 3 || use_stencil_downsample) {
        compute_ne_rows_cuda<16, 32>(access, ne_counter_333, ne_rows_333_gpu, 2);
    }
    if(kernel_size == 5) {
        compute_ne_rows_cuda<16, 32>(access, ne_counter_555, ne_rows_555_gpu, 4);
    }

    /// allocate GPU memory
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> relative_blur(NULL, access.total_number_particles());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> error_est(NULL, access.total_number_particles());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> tree_data_gpu(NULL, tree_access.total_number_particles());

    /// set block/grid size for cuda kernels
    const size_t numParts = access.total_number_particles();

    dim3 grid_dim(8, 1, 1);
    dim3 block_dim(256, 1, 1);

    error_check( cudaDeviceSynchronize() )

    /// if not resuming from previous iterations, initialize estimate with 1s
    if(!resume) {
        fillWithValue<<< grid_dim, block_dim >>>(output, (stencilType) 1, numParts);
    }

    error_check( cudaDeviceSynchronize() )

    for(int i = 0; i < niter; ++i) {

        downsample_avg(access, tree_access, output, tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);

        error_check( cudaDeviceSynchronize() )

        if(kernel_size == 5) {
            if(use_stencil_downsample) {
                isotropic_convolve_555_ds_reflective(access, tree_access, output, relative_blur.get(), psf_flipped,
                                                     tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555, ne_rows_333_gpu.get(), ne_counter_333);
            } else {
                isotropic_convolve_555_reflective(access, tree_access, output, relative_blur.get(), psf_flipped,
                                                  tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555);
            }
        } else {
            isotropic_convolve_333_reflective(access, tree_access, output, relative_blur.get(), psf_flipped,
                                              tree_data_gpu.get(), ne_rows_333_gpu.get(), ne_counter_333, use_stencil_downsample);
        }

        error_check( cudaDeviceSynchronize() )

        elementWiseDiv<<< grid_dim, block_dim >>> (input, relative_blur.get(), relative_blur.get(), numParts);

        error_check( cudaDeviceSynchronize() )

        downsample_avg(access, tree_access, relative_blur.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);

        error_check( cudaDeviceSynchronize() )

        if(kernel_size == 5) {
            if(use_stencil_downsample) {
                isotropic_convolve_555_ds_reflective(access, tree_access, relative_blur.get(), error_est.get(), psf,
                                                     tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555, ne_rows_333_gpu.get(), ne_counter_333);
            } else {
                isotropic_convolve_555_reflective(access, tree_access, relative_blur.get(), error_est.get(), psf,
                                                  tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555);
            }
        } else {
            isotropic_convolve_333_reflective(access, tree_access, relative_blur.get(), error_est.get(), psf,
                                              tree_data_gpu.get(), ne_rows_333_gpu.get(), ne_counter_333, use_stencil_downsample);
        }
        error_check( cudaDeviceSynchronize() )

        elementWiseMult<<< grid_dim, block_dim >>> (output, error_est.get(), numParts);

        error_check( cudaDeviceSynchronize() )
    }
}


template<typename inputType, typename stencilType>
void APRNumericsGPU::richardson_lucy(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                                     VectorData<stencilType>& output, PixelData<stencilType>& psf, int niter,
                                     bool use_stencil_downsample, bool normalize_stencil, bool resume) {

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    int kernel_size;
    if(psf.mesh.size() == 27) {
        kernel_size = 3;
    } else if(psf.mesh.size() == 125) {
        kernel_size = 5;
    } else {
        throw std::runtime_error("richardson_lucy is currently only implemented for 3x3x3 and 5x5x5 kernels");
    }

    PixelData<stencilType> psf_flipped(psf, false);
    for(int i = 0; i < psf.mesh.size(); ++i) {
        psf_flipped.mesh[i] = psf.mesh[psf.mesh.size()-1-i];
    }

    VectorData<stencilType> psf_vec;
    VectorData<stencilType> psf_flipped_vec;

    if(use_stencil_downsample) {
        APRStencil::get_downsampled_stencils(psf, psf_vec, access.level_max() - access.level_min(), normalize_stencil);
        APRStencil::get_downsampled_stencils(psf_flipped, psf_flipped_vec, access.level_max() - access.level_min(), normalize_stencil);
    }

    output.resize(input.size());

    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> psf_gpu;
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> psf_flipped_gpu;

    if(use_stencil_downsample) {
        psf_gpu.initialize(psf_vec.data(), psf_vec.size());
        psf_flipped_gpu.initialize(psf_flipped_vec.data(), psf_flipped_vec.size());
    } else {
        psf_gpu.initialize(psf.mesh.get(), psf.mesh.size());
        psf_flipped_gpu.initialize(psf_flipped.mesh.get(), psf_flipped.mesh.size());
    }

    /// copy input and psf to the device
    input_gpu.copyH2D();
    psf_gpu.copyH2D();
    psf_flipped_gpu.copyH2D();

    if(resume) {
        output_gpu.copyH2D();
    }

    APRNumericsGPU::richardson_lucy(access, tree_access, input_gpu.get(), output_gpu.get(), psf_gpu.get(),
                                    psf_flipped_gpu.get(), kernel_size, niter, use_stencil_downsample, resume);
    error_check( cudaDeviceSynchronize() )

    /// copy result back to host
    output_gpu.copyD2H();
}


template void APRNumericsGPU::gradient_magnitude(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint8_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template void APRNumericsGPU::gradient_magnitude(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template void APRNumericsGPU::gradient_magnitude(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint64_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template void APRNumericsGPU::gradient_magnitude(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);

template void APRNumericsGPU::richardson_lucy(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, float*, float*, float*, int, int, bool, bool);
template void APRNumericsGPU::richardson_lucy(GPUAccessHelper&, GPUAccessHelper&, float*, float*, float*, float*, int, int, bool, bool);

template void APRNumericsGPU::richardson_lucy(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, PixelData<float>&, int, bool, bool, bool);
template void APRNumericsGPU::richardson_lucy(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, PixelData<float>&, int, bool, bool, bool);