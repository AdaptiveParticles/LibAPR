////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// APR Converter class handles the methods and functions for creating an APR from an input image
///
////////////////////////////////

#ifndef __APR_CONVERTER_HPP__
#define __APR_CONVERTER_HPP__

#include <list>

#include "AutoParameters.hpp"
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "io/TiffUtils.hpp"
#include "numerics/APRReconstruction.hpp"

#include "PullingScheme.hpp"

#include "PullingSchemeSparse.hpp"

#include "LocalParticleCellSet.hpp"
#include "LocalIntensityScale.hpp"
#include "ComputeGradient.hpp"
#include <iterator>

#ifdef APR_USE_CUDA
#include "algorithm/ComputeGradientCuda.hpp"
#endif

template<typename ImageType>
class APRConverter {

    template<typename T>
    friend class APRConverterBatch;

protected:
    PullingScheme iPullingScheme;
    LocalParticleCellSet iLocalParticleSet;
    LocalIntensityScale iLocalIntensityScale;
    ComputeGradient iComputeGradient;

    PullingSchemeSparse iPullingSchemeSparse;

    bool generate_linear = true; //default is now the new structures
    bool sparse_pulling_scheme = false;

public:

    void set_generate_linear(bool flag){
        generate_linear = flag;
    }

    void set_sparse_pulling_scheme(bool flag){
        sparse_pulling_scheme = flag;
    }

    APRTimer fine_grained_timer;
    APRTimer method_timer;
    APRTimer total_timer;
    APRTimer allocation_timer;
    APRTimer computation_timer;
    APRParameters par;

    template <typename T>
    bool get_apr(APR &aAPR, PixelData<T> &input_image);

    template <typename T>
    bool get_apr_cpu(APR &aAPR, PixelData<T> &input_image);

#ifdef APR_USE_CUDA
    template <typename T>
    bool get_apr_cuda(APR &aAPR, PixelData<T> &input_image);
#endif

    bool verbose = true;

    void get_apr_custom_grad_scale(APR& aAPR,PixelData<ImageType>& grad,PixelData<float>& lis,bool down_sampled = true);

    template <typename T>
    bool initPipelineAPR(APR &aAPR, PixelData<T> &input_image) {

        if (par.check_input) {
            if (!check_input_dimensions(input_image)) {
                std::cout << "Input dimension check failed. Make sure the input image is filled in order x -> y -> z, or try using the option -swap_dimension" << std::endl;
                return false;
            }
        }

        //  Initializes the APR datastructures for the given image.
        aAPR.parameters = par;
        aAPR.aprInfo.init(input_image.y_num,input_image.x_num,input_image.z_num);
        aAPR.linearAccess.genInfo = &aAPR.aprInfo;
        aAPR.apr_access.genInfo = &aAPR.aprInfo;

        return true;
    }

    float bspline_offset = 0;

protected:

    template<typename T>
    bool get_lrf(APR &aAPR, PixelData<T> &input_image);

    bool get_ds(APR &aAPR);

    //get apr without setting parameters, and with an already loaded image.

    //DATA (so it can be re-used)

    PixelData<ImageType> grad_temp; // should be a down-sampled image
    PixelData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
    PixelData<float> local_scale_temp2;

    void applyParameters(APR& aAPR,APRParameters& aprParameters);

    template<typename T>
    void computeL(APR& aAPR,PixelData<T>& input_image);

    void solveForAPR(APR& aAPR);

    void generateDatastructures(APR& aAPR);

    template<typename T>
    bool check_input_dimensions(PixelData<T> &input_image);

    void initPipelineMemory(int y_num, int x_num = 1, int z_num = 1);

};


template<typename ImageType>
void APRConverter<ImageType>::initPipelineMemory(int y_num,int x_num,int z_num){
    //initializes the internal memory to be used in the pipeline.
    allocation_timer.start_timer("init_ds_images");

    const int z_num_ds = ceil(1.0*z_num/2.0);
    const int x_num_ds = ceil(1.0*x_num/2.0);
    const int y_num_ds = ceil(1.0*y_num/2.0);

    grad_temp.initWithResize(y_num_ds, x_num_ds, z_num_ds); //this needs to be initialized to zero
    grad_temp.fill(0);

    float not_needed;
    std::vector<int> var_win;
    iLocalIntensityScale.get_window_alt(not_needed, var_win, par, grad_temp);

    int padding_y = 2*std::max(var_win[0],var_win[3]);
    int padding_x = 2*std::max(var_win[1],var_win[4]);
    int padding_z = 2*std::max(var_win[2],var_win[5]);

    //Compute dimensions

    //This ensures enough memory is allocated for the padding.
    local_scale_temp.initWithResize(y_num_ds+padding_y, x_num_ds+padding_x, z_num_ds+padding_z);
    local_scale_temp.initWithResize(y_num_ds, x_num_ds, z_num_ds);

    local_scale_temp2.initWithResize(y_num_ds+padding_y, x_num_ds+padding_x, z_num_ds+padding_z);
    local_scale_temp2.initWithResize(y_num_ds, x_num_ds, z_num_ds);

    allocation_timer.stop_timer();
}

template<typename ImageType>
void APRConverter<ImageType>::get_apr_custom_grad_scale(APR& aAPR,PixelData<ImageType>& grad,PixelData<float>& lis,bool down_sampled){

    //APR must already be initialized.

    if(down_sampled){

        //need to check that they are initialized.
        grad_temp.swap(grad);
        lis.swap(local_scale_temp);

    } else {
        // To be done. The L(y) needs to be computed then max downsampled.
        std::cerr << "Not implimented" << std::endl;

    }

    aAPR.parameters = par;
    applyParameters(aAPR,par);
    solveForAPR(aAPR);
    generateDatastructures(aAPR);

}


template<typename ImageType> template<typename T>
void APRConverter<ImageType>::computeL(APR& aAPR,PixelData<T>& input_image){
    //
    //  Computes the local resolution estimate L(y), the input for the Pulling Scheme and setting the resolution everywhere.
    //


    ////////////////////////////////////////
    /// Memory allocation of variables
    ////////////////////////////////////////

    //assuming uint16, the total memory cost shoudl be approximately (1 + 1 + 1/8 + 2/8 + 2/8) = 2 5/8 original image size in u16bit
    //storage of the particle cell tree for computing the pulling scheme
    allocation_timer.start_timer("init and copy image");
    PixelData<ImageType> image_temp(input_image, false /* don't copy */, false /* pinned memory */); // global image variable useful for passing between methods, or re-using memory (should be the only full sized copy of the image)

    allocation_timer.stop_timer();

    /////////////////////////////////
    /// Pipeline
    ////////////////////////

    fine_grained_timer.start_timer("offset image");

    if (std::is_floating_point<ImageType>::value) {
        image_temp.copyFromMesh(input_image);
    } else {
        bspline_offset = compute_bspline_offset<ImageType>(input_image, par.lambda);
        image_temp.copyFromMeshWithUnaryOp(input_image, [=](const auto &a) { return (a + bspline_offset); });
    }

    fine_grained_timer.stop_timer();

    //method_timer.verbose_flag = true;
    method_timer.start_timer("compute_gradient_magnitude_using_bsplines");
    iComputeGradient.get_gradient(image_temp, grad_temp, local_scale_temp, par);
    method_timer.stop_timer();
#ifdef HAVE_LIBTIFF
    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "gradient_step.tif", grad_temp);
    }
#endif
    method_timer.start_timer("compute_local_intensity_scale");
    iLocalIntensityScale.get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
    method_timer.stop_timer();

#ifdef HAVE_LIBTIFF
    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "local_intensity_scale_step.tif", local_scale_temp);
    }
#endif


}

template<typename ImageType>
void APRConverter<ImageType>::applyParameters(APR& aAPR,APRParameters& aprParameters) {
    //
    //  Apply the main parameters
    //

    fine_grained_timer.start_timer("load_and_apply_mask");
    // Apply mask if given
    if(par.mask_file != ""){
        iComputeGradient.mask_gradient(grad_temp, aprParameters);
    }
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("threshold");
    iComputeGradient.threshold_gradient(grad_temp,local_scale_temp2,aprParameters.Ip_th + bspline_offset);
    fine_grained_timer.stop_timer();

    float max_th = 60000;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
    for (size_t i = 0; i < grad_temp.mesh.size(); ++i) {

        float rescaled = local_scale_temp.mesh[i];
        if (rescaled < aprParameters.sigma_th) {
            rescaled = (rescaled < aprParameters.sigma_th_max) ? max_th : par.sigma_th;
            local_scale_temp.mesh[i] = rescaled;
        }
    }

#ifdef HAVE_LIBTIFF
    if(par.output_steps) {
        TiffUtils::saveMeshAsTiff(par.output_dir + "local_intensity_scale_rescaled.tif", local_scale_temp);
    }
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
    for (size_t i = 0; i < grad_temp.mesh.size(); ++i) {

        if(grad_temp.mesh[i] < aprParameters.grad_th){
            grad_temp.mesh[i] = 0;
        }
    }
}


template<typename ImageType>
void APRConverter<ImageType>::solveForAPR(APR& aAPR){

    method_timer.start_timer("compute_levels");
    iLocalParticleSet.computeLevels(grad_temp, local_scale_temp, aAPR.level_max(), par.rel_error, par.dx, par.dy, par.dz);
    method_timer.stop_timer();

    if(!sparse_pulling_scheme) {

        method_timer.start_timer("initialize_particle_cell_tree");
        iPullingScheme.initialize_particle_cell_tree(aAPR.aprInfo);
        method_timer.stop_timer();

        method_timer.start_timer("compute_local_particle_set");
        iLocalParticleSet.get_local_particle_cell_set(iPullingScheme,local_scale_temp, local_scale_temp2,par);
        method_timer.stop_timer();

        method_timer.start_timer("compute_pulling_scheme");
        iPullingScheme.pulling_scheme_main();
        method_timer.stop_timer();
    } else {

        method_timer.start_timer("initialize_particle_cell_tree");
        iPullingSchemeSparse.initialize_particle_cell_tree(aAPR.aprInfo);
        method_timer.stop_timer();

        method_timer.start_timer("compute_local_particle_set");
        iLocalParticleSet.get_local_particle_cell_set_sparse(iPullingSchemeSparse,local_scale_temp, local_scale_temp2,par);
        method_timer.stop_timer();

        method_timer.start_timer("compute_pulling_scheme");
        iPullingSchemeSparse.pulling_scheme_main();
        method_timer.stop_timer();

    }

}

template<typename ImageType>
void APRConverter<ImageType>::generateDatastructures(APR& aAPR){

    method_timer.start_timer("compute_apr_datastructure");
    if(!generate_linear) {
        if(!sparse_pulling_scheme){
            aAPR.apr_access.initialize_structure_from_particle_cell_tree(aAPR.parameters,
                                                                         iPullingScheme.getParticleCellTree());
        } else{
            aAPR.apr_access.initialize_structure_from_particle_cell_tree_sparse(aAPR.parameters,
                                                                                iPullingSchemeSparse.particle_cell_tree);
        }
        aAPR.apr_initialized_random = true;

    } else {
        if(!sparse_pulling_scheme) {
            aAPR.linearAccess.initialize_linear_structure(aAPR.parameters,
                                                          iPullingScheme.getParticleCellTree());
        } else {
            aAPR.linearAccess.initialize_linear_structure_sparse(aAPR.parameters,
                                                                 iPullingSchemeSparse.particle_cell_tree);
        }
        aAPR.apr_initialized = true;
    }
    method_timer.stop_timer();
}

/**
 * Main method for constructing the input steps to the computation to the APR before parameters are applied.
 *
 * Note: currently only used by the python wrappers for interactive parameter selection
 */
template<typename ImageType> template<typename T>
inline bool APRConverter<ImageType>::get_lrf(APR &aAPR, PixelData<T>& input_image) {

    computation_timer.verbose_flag = false;

    aAPR.parameters = par;

    initPipelineAPR(aAPR, input_image);

    computation_timer.start_timer("init_mem");

    initPipelineMemory(input_image.y_num, input_image.x_num, input_image.z_num);

    computation_timer.stop_timer();

    computation_timer.start_timer("compute_L");

    //Compute the local resolution estimate
    computeL(aAPR,input_image);

    computation_timer.stop_timer();

    return true;

}

/**
 * Main method for constructing the input steps to the computation to the APR before parameters are applied.
 *
 * Note: currently only used by the python wrappers for interactive parameter selection
 */
template<typename ImageType>
inline bool APRConverter<ImageType>::get_ds(APR &aAPR) {

    applyParameters(aAPR,par);
    aAPR.parameters = par;

    solveForAPR(aAPR);

    generateDatastructures(aAPR);

    return true;

}


#ifdef APR_USE_CUDA
/**
 * Implementation of pipeline for GPU/CUDA
 *
 * @param aAPR - the APR datastructure
 * @param input_image - input image
 */
template<typename ImageType> template<typename T>
inline bool APRConverter<ImageType>::get_apr_cuda(APR &aAPR, PixelData<T>& input_image) {
    if (!initPipelineAPR(aAPR, input_image)) return false;


    initPipelineMemory(input_image.y_num, input_image.x_num, input_image.z_num);

    method_timer.start_timer("compute_gradient_magnitude_using_bsplines and local instensity scale CUDA");
    APRTimer t(true);
    APRTimer d(true);
    t.start_timer(" =========== ALL");
    {

        computation_timer.start_timer("init_mem");
        PixelData<ImageType> image_temp(input_image, false /* don't copy */, true /* pinned memory */); // global image variable useful for passing between methods, or re-using memory (should be the only full sized copy of the image)

        /////////////////////////////////
        /// Pipeline
        ////////////////////////
        //offset image by factor (this is required if there are zero areas in the background with uint16_t and uint8_t images, as the Bspline co-efficients otherwise may be negative!)
        // Warning both of these could result in over-flow (if your image is non zero, with a 'buffer' and has intensities up to uint16_t maximum value then set image_type = "", i.e. uncomment the following line)

        if (std::is_same<uint16_t, ImageType>::value) {
            bspline_offset = 100;
            image_temp.copyFromMeshWithUnaryOp(input_image, [=](const auto &a) { return (a + bspline_offset); });
        } else if (std::is_same<uint8_t, ImageType>::value) {
            bspline_offset = 5;
            image_temp.copyFromMeshWithUnaryOp(input_image, [=](const auto &a) { return (a + bspline_offset); });
        } else {
            image_temp.copyFromMesh(input_image);
        }

        computation_timer.stop_timer();

        std::vector<GpuProcessingTask<ImageType>> gpts;

        int numOfStreams = 1;
        int repetitionsPerStream = 1;

        computation_timer.start_timer("compute_L");
        // Create streams and send initial task to do
        for (int i = 0; i < numOfStreams; ++i) {
            gpts.emplace_back(GpuProcessingTask<ImageType>(image_temp, local_scale_temp, par, bspline_offset, aAPR.level_max()));
            gpts.back().sendDataToGpu();
            gpts.back().processOnGpu();
        }
        computation_timer.stop_timer();


        for (int i = 0; i < numOfStreams * repetitionsPerStream; ++i) {
            int c = i % numOfStreams;

            computation_timer.start_timer("apply_parameters");
            // get data from previous task
            gpts[c].getDataFromGpu();

            computation_timer.stop_timer();

            // in theory we get new data and send them to task
            if (i  < numOfStreams * (repetitionsPerStream - 1)) {
                gpts[c].sendDataToGpu();
                gpts[c].processOnGpu();
            }

            // Postprocess on CPU
            std::cout << "--------- start CPU processing ---------- " << i << std::endl;

            computation_timer.start_timer("solve_for_apr");
            iPullingScheme.initialize_particle_cell_tree(aAPR.aprInfo);

            PixelData<float> lst(local_scale_temp, true);

#ifdef HAVE_LIBTIFF
            if (par.output_steps){
                TiffUtils::saveMeshAsTiff(par.output_dir + "local_intensity_scale_step.tif", lst);
            }
#endif

#ifdef HAVE_LIBTIFF
            if (par.output_steps){
                TiffUtils::saveMeshAsTiff(par.output_dir + "gradient_step.tif", grad_temp);
            }
#endif

            iLocalParticleSet.get_local_particle_cell_set(iPullingScheme,lst, local_scale_temp2,par);

            iPullingScheme.pulling_scheme_main();

            computation_timer.stop_timer();

            computation_timer.start_timer("generate_data_structures");
            generateDatastructures(aAPR);
            computation_timer.stop_timer();
        }
        std::cout << "Total n ENDED" << std::endl;

    }
    t.stop_timer();
    method_timer.stop_timer();

    return true;
}
#endif


/**
 * Implementation of pipeline for CPU
 *
 * @param aAPR - the APR datastructure
 * @param input_image - input image
 */
template<typename ImageType> template<typename T>
inline bool APRConverter<ImageType>::get_apr_cpu(APR &aAPR, PixelData<T> &input_image) {

    if (!initPipelineAPR(aAPR, input_image)) return false;

    total_timer.start_timer("full_pipeline");

    computation_timer.start_timer("init_mem");

    initPipelineMemory(input_image.y_num, input_image.x_num, input_image.z_num);

    computation_timer.stop_timer();

    computation_timer.start_timer("compute_L");

    //Compute the local resolution estimate
    computeL(aAPR,input_image);

    computation_timer.stop_timer();

    computation_timer.start_timer("apply_parameters");

    if (par.auto_parameters) {
        method_timer.start_timer("autoParameters");
        autoParametersLiEntropy(par, local_scale_temp2, local_scale_temp, grad_temp, bspline_offset, verbose);
        aAPR.parameters = par;
        method_timer.stop_timer();
    }

    applyParameters(aAPR,par);

    computation_timer.stop_timer();

    computation_timer.start_timer("solve_for_apr");

    solveForAPR(aAPR);

    computation_timer.stop_timer();

    computation_timer.start_timer("generate_data_structures");

    generateDatastructures(aAPR);

    computation_timer.stop_timer();

    total_timer.stop_timer();

    return true;
}


/**
 * Main method for constructing the APR from an input image
 *
 * @param aAPR - the APR data structure
 * @param input_image - input image
 */
template<typename ImageType> template<typename T>
inline bool APRConverter<ImageType>::get_apr(APR &aAPR, PixelData<T> &input_image) {
// TODO: CUDA pipeline is temporarily turned off and CPU version is always chosen.
//       After revising a CUDA pipeline remove "#if true // " part.
#if true // #ifndef APR_USE_CUDA
    return get_apr_cpu(aAPR, input_image);
#else
    return get_apr_cuda(aAPR, input_image);
#endif
}


/**
 * Checks if the memory dimension (y) is filled
 */
template<typename ImageType> template<typename T>
bool APRConverter<ImageType>::check_input_dimensions(PixelData<T> &input_image) {
    bool x_present = input_image.x_num>1;
    bool y_present = input_image.y_num>1;
    bool z_present = input_image.z_num>1;

    uint8_t number_dims = x_present + y_present + z_present;

    if(number_dims == 0) { return false; }
    if(number_dims == 3) { return true; }

    if(verbose) {
        std::cout << "Used parameters: " << std::endl;
        std::cout << "I_th: " << par.Ip_th << std::endl;
        std::cout << "sigma_th: " << par.sigma_th << std::endl;
        std::cout << "sigma_th_max: " << par.sigma_th_max << std::endl;
        std::cout << "relative error (E): " << par.rel_error << std::endl;
        std::cout << "lambda: " << par.lambda << std::endl;
    }

    // number_dims equals 1 or 2
    if(y_present) {
        return true;
    } else if(par.swap_dimensions){
        if(x_present) {
            std::swap(input_image.x_num, input_image.y_num);
        } else {
            std::swap(input_image.z_num, input_image.y_num);
        }
        return true;
    }

    return false;

}



#endif // __APR_CONVERTER_HPP__
