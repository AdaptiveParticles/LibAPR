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
    friend class PyAPRConverter;

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

    template<typename T>
    bool get_apr(APR &aAPR, PixelData<T> &input_image);

    bool verbose = true;

    void get_apr_custom_grad_scale(APR& aAPR,PixelData<ImageType>& grad,PixelData<float>& lis,bool down_sampled = true);

    void initPipelineAPR(APR &aAPR, int y_num, int x_num = 1, int z_num = 1){
        //
        //  Initializes the APR datastructures for the given image.
        //

        aAPR.aprInfo.init(y_num,x_num,z_num);
        aAPR.linearAccess.genInfo = &aAPR.aprInfo;
        aAPR.apr_access.genInfo = &aAPR.aprInfo;
    }

protected:

    template<typename T>
    bool get_lrf(APR &aAPR, PixelData<T> &input_image);

    bool get_ds(APR &aAPR);

    //get apr without setting parameters, and with an already loaded image.

    float bspline_offset = 0;

    //DATA (so it can be re-used)

    PixelData<ImageType> grad_temp; // should be a down-sampled image
    PixelData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
    PixelData<float> local_scale_temp2;

    void applyParameters(APR& aAPR,APRParameters& aprParameters);

    template<typename T>
    void computeL(APR& aAPR,PixelData<T>& input_image);

    void solveForAPR(APR& aAPR);

    void generateDatastructures(APR& aAPR);

    template<typename T,typename S>
    void autoParameters(const PixelData<T> &localIntensityScale,const PixelData<S> &grad);

    template<typename T,typename S>
    void autoParametersLiEntropy(const PixelData<T> &image, const PixelData<T> &localIntensityScale, const PixelData<S> &grad);

    template<typename T>
    bool check_input_dimensions(PixelData<T> &input_image);

    void initPipelineMemory(int y_num,int x_num = 1,int z_num = 1);


};


template <typename T>
struct MinMax{T min; T max; };

template <typename T>
static MinMax<T> getMinMax(const PixelData<T>& input_image) {
    T minVal = std::numeric_limits<T>::max();
    T maxVal = std::numeric_limits<T>::min();

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) reduction(max:maxVal) reduction(min:minVal)
#endif
    for (size_t i = 0; i < input_image.mesh.size(); ++i) {
        T val = input_image.mesh[i];
        if (val > maxVal) maxVal = val;
        if (val < minVal) minVal = val;
    }

    return MinMax<T>{minVal, maxVal};
}

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
    PixelData<ImageType> image_temp(input_image, false /* don't copy */, true /* pinned memory */); // global image variable useful for passing between methods, or re-using memory (should be the only full sized copy of the image)

    allocation_timer.stop_timer();

    /////////////////////////////////
    /// Pipeline
    ////////////////////////

    fine_grained_timer.start_timer("offset image");
    //offset image by factor (this is required if there are zero areas in the background with uint16_t and uint8_t images, as the Bspline co-efficients otherwise may be negative!)
    // Warning both of these could result in over-flow (if your image is non zero, with a 'buffer' and has intensities up to uint16_t maximum value then set image_type = "", i.e. uncomment the following line)

    if (std::is_same<uint16_t, ImageType>::value) {
        bspline_offset = 100;
        image_temp.copyFromMeshWithUnaryOp(input_image, [=](const auto &a) { return (a + bspline_offset); });
    } else if (std::is_same<uint8_t, ImageType>::value){
        bspline_offset = 5;
        image_temp.copyFromMeshWithUnaryOp(input_image, [=](const auto &a) { return (a + bspline_offset); });
    } else {
        image_temp.copyFromMesh(input_image);
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
 */
template<typename ImageType> template<typename T>
inline bool APRConverter<ImageType>::get_lrf(APR &aAPR, PixelData<T>& input_image) {

    computation_timer.verbose_flag = false;

    aAPR.parameters = par;

    initPipelineAPR(aAPR, input_image.y_num, input_image.x_num, input_image.z_num);

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
 */
template<typename ImageType>
inline bool APRConverter<ImageType>::get_ds(APR &aAPR) {

    applyParameters(aAPR,par);
    aAPR.parameters = par;

    solveForAPR(aAPR);

    generateDatastructures(aAPR);

    return true;

}


/**
 * Main method for constructing the APR from an input image
 */
template<typename ImageType> template<typename T>
inline bool APRConverter<ImageType>::get_apr(APR &aAPR, PixelData<T>& input_image) {

    aAPR.parameters = par;

    if(par.check_input) {
        if(!check_input_dimensions(input_image)) {
            std::cout << "Input dimension check failed. Make sure the input image is filled in order x -> y -> z, or try using the option -swap_dimension" << std::endl;
            return false;
        }
    }


    initPipelineAPR(aAPR, input_image.y_num, input_image.x_num, input_image.z_num);

#ifndef APR_USE_CUDA

    total_timer.start_timer("full_pipeline");

    computation_timer.start_timer("init_mem");

    initPipelineMemory(input_image.y_num, input_image.x_num, input_image.z_num);

    computation_timer.stop_timer();


    computation_timer.start_timer("compute_L");

    //Compute the local resolution estimate
    computeL(aAPR,input_image);

    computation_timer.stop_timer();

    computation_timer.start_timer("apply_parameters");

    if( par.auto_parameters ) {
        method_timer.start_timer("autoParameters");
//        autoParameters(local_scale_temp,grad_temp);
        autoParametersLiEntropy(local_scale_temp2, local_scale_temp, grad_temp);
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

#else


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
        } else if (std::is_same<uint8_t, ImageType>::value){
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
            if(par.output_steps){
                TiffUtils::saveMeshAsTiff(par.output_dir + "local_intensity_scale_step.tif", lst);
            }
#endif

#ifdef HAVE_LIBTIFF
            if(par.output_steps){
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
#endif

    return true;
}

template<typename T>
void compute_means(const std::vector<T>& data, float threshold, float& mean_back, float& mean_fore) {
    float sum_fore=0.f, sum_back=0.f;
    size_t count_fore=0, count_back=0;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) shared(data, threshold) reduction(+:sum_fore, sum_back, count_fore, count_back)
#endif
    for(size_t idx = 0; idx < data.size(); ++idx) {
        if(data[idx] > threshold) {
            sum_fore += data[idx];
            count_fore++;
        } else {
            sum_back += data[idx];
            count_back++;
        }
    }
    mean_fore = sum_fore / count_fore;
    mean_back = sum_back / count_back;
}


/**
 * Compute threshold value by Li's iterative Minimum Cross Entropy method [1]
 *
 * Note: it is assumed that the input elements are non-negative (as is the case for the gradient and local intensity
 * scale). To apply the method to data that may be negative, subtract the minimum value from the input, and then add
 * that value to the computed threshold.
 *
 * [1] Li, C. H., & Tam, P. K. S. (1998). "An iterative algorithm for minimum cross entropy thresholding."
 *     Pattern recognition letters, 19(8), 771-776.
 */
template<typename T>
float threshold_li(const std::vector<T> &input) {

    if(input.empty()) { return 0.f; }     // return 0 if input is empty

    const T image_min = *std::min_element(input.begin(), input.end());
    const T image_max = *std::max_element(input.begin(), input.end());

    if(image_min == image_max) { return image_min; }  // if all inputs are equal, return that value

    float tolerance = 0.5f;   // tolerance 0.5 should suffice for integer inputs

    // For floating point inputs we set the tolerance to the lesser of 0.01 and a fraction of the data range
    // This could be improved, by instead taking e.g. half the smallest difference between any two non-equal elements
    if(std::is_floating_point<T>::value) {
        float range = image_max - image_min;
        tolerance = std::min(0.01f, range*1e-4f);
    }

    // Take the mean of the input as initial value
    float t_next = std::accumulate(input.begin(), input.end(), 0.f) / (float) input.size();
    float t_curr = -2.f * tolerance; //this ensures that the first iteration is performed, since t_next is non-negative

    // For integer inputs, we have to ensure a large enough initial value, such that mean_back > 0
    if(!std::is_floating_point<T>::value) {
        // if initial value is <1, try to increase it to 1.5 unless the range is too narrow
        if(t_next < 1.f) {
            t_next = std::min(1.5f, (image_min+image_max)/2.f);
        }
    }

    // Perform Li iterations until convergence
    while(std::abs(t_next - t_curr) > tolerance) {
        t_curr = t_next;

        // Compute averages of values above and below the current threshold
        float mean_back, mean_fore;
        compute_means(input, t_curr, mean_back, mean_fore);

        // Handle the edge case where all values < t_curr are 0
        if(mean_back == 0) {
            std::wcout << "log(0) encountered in APRConverter::threshold_li, returning current threshold" << std::endl;
            return t_curr;
        }

        // Update the threshold (one-point iteration)
        t_next = (mean_fore - mean_back) / (std::log(mean_fore) - std::log(mean_back));
    }
    return t_next;
}


template<typename ImageType>
template<typename T,typename S>
void APRConverter<ImageType>::autoParametersLiEntropy(const PixelData<T> &image,
                                                      const PixelData<T> &localIntensityScale,
                                                      const PixelData<S> &grad) {

    fine_grained_timer.start_timer("autoparameters: subsample buffers");

    std::vector<S> grad_subsampled;
    std::vector<T> lis_subsampled;

    {   // intentional scope
        /// First we extract the gradient and local intensity scale values at all locations where the image
        /// intensity exceeds the intensity threshold. This allows better adaptation in certain cases, e.g.
        /// when there is significant background AND signal noise/autofluorescence (provided that par.Ip_th
        /// is set appropriately).
        std::vector<S> grad_foreground(grad.mesh.size());
        std::vector<T> lis_foreground(localIntensityScale.mesh.size());

        const auto threshold = par.Ip_th + bspline_offset;
        size_t counter = 0;
        for(size_t idx = 0; idx < grad.mesh.size(); ++idx) {
            if(image.mesh[idx] > threshold) {
                grad_foreground[counter] = grad.mesh[idx];
                lis_foreground[counter] = localIntensityScale.mesh[idx];
                counter++;
            }
        }

        const size_t num_foreground_pixels = counter;

        grad_foreground.resize(num_foreground_pixels); //setting size to non-zero elements.
        lis_foreground.resize(num_foreground_pixels);

        /// Then we uniformly subsample these signals, as we typically don't need all elements to compute the thresholds
        const size_t num_elements = std::min((size_t)32*512*512, num_foreground_pixels); //arbitrary number.
        const size_t delta = num_foreground_pixels / num_elements;
        grad_subsampled.resize(num_elements);
        lis_subsampled.resize(num_elements);

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for(size_t idx = 0; idx < num_elements; ++idx) {
            grad_subsampled[idx] = grad_foreground[idx*delta];
            lis_subsampled[idx] = lis_foreground[idx*delta];
        }
    }
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("autoparameters: compute gradient threshold");
    par.grad_th = threshold_li(grad_subsampled);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("autoparameters: compute sigma threshold");
    par.sigma_th = threshold_li(lis_subsampled);
    fine_grained_timer.stop_timer();

    if(verbose) {
        std::cout << "Automatic parameter tuning found sigma_th = " << par.sigma_th <<
                     " and grad_th = " << par.grad_th << std::endl;
    }
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
