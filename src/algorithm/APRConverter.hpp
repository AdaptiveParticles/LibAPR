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

#ifdef APR_USE_CUDA
#include "algorithm/ComputeGradientCuda.hpp"
#endif

template<typename ImageType>
class APRConverter {

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

    void get_apr_custom_grad_scale(APR& aAPR,PixelData<ImageType>& grad,PixelData<float>& lis,bool down_sampled = true){

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

    void initPipelineAPR(APR &aAPR, int y_num, int x_num = 1, int z_num = 1){
        //
        //  Initializes the APR datastructures for the given image.
        //

        aAPR.aprInfo.init(y_num,x_num,z_num);
        aAPR.linearAccess.genInfo = &aAPR.aprInfo;
        aAPR.apr_access.genInfo = &aAPR.aprInfo;
    }

protected:

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

    template<typename T>
    void auto_parameters(const PixelData<T> &input_img);

    template<typename T>
    bool check_input_dimensions(PixelData<T> &input_image);



    void initPipelineMemory(int y_num,int x_num = 1,int z_num = 1){
        //initializes the internal memory to be used in the pipeline.
        allocation_timer.start_timer("init ds images");


        grad_temp.initDownsampled(y_num, x_num, z_num, 0, false); //#TODO: why are these differnet?


        local_scale_temp.initDownsampled(y_num, x_num, z_num, true);


        local_scale_temp2.initDownsampled(y_num, x_num, z_num, false);

        allocation_timer.stop_timer();
    }





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
template<typename ImageType> template<typename T>
void APRConverter<ImageType>::computeL(APR& aAPR,PixelData<T>& input_image){
    //
    //  Computes the local resolution estimate L(y), the input for the Pulling Scheme and setting the resolution everywhere.
    //


    total_timer.start_timer("Total_pipeline_excluding_IO");

    initPipelineMemory(input_image.y_num, input_image.x_num, input_image.z_num);

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

    computation_timer.start_timer("Calculations");

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

#ifndef APR_USE_CUDA
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
    //method_timer.verbose_flag = false;
#ifdef HAVE_LIBTIFF
    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "local_intensity_scale_step.tif", local_scale_temp);
    }

#else
    method_timer.start_timer("compute_gradient_magnitude_using_bsplines and local instensity scale CUDA");
    //getFullPipeline(image_temp, grad_temp, local_scale_temp, local_scale_temp2,bspline_offset, par);
    method_timer.stop_timer();

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
    iComputeGradient.threshold_gradient(grad_temp,local_scale_temp,aprParameters.Ip_th + bspline_offset);
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

    if(!generate_linear) {

        if(!sparse_pulling_scheme){
            method_timer.start_timer("compute_apr_datastructure");
            aAPR.apr_access.initialize_structure_from_particle_cell_tree(aAPR.parameters,
                                                                         iPullingScheme.getParticleCellTree());
            method_timer.stop_timer();
        } else{
            method_timer.start_timer("compute_apr_datastructure");
            aAPR.apr_access.initialize_structure_from_particle_cell_tree_sparse(aAPR.parameters,
                                                                                iPullingSchemeSparse.particle_cell_tree);
            method_timer.stop_timer();

        }

        aAPR.apr_initialized_random = true;

    } else {
        method_timer.start_timer("compute_apr_datastructure");

        if(!sparse_pulling_scheme) {

            aAPR.linearAccess.initialize_linear_structure(aAPR.parameters,
                                                          iPullingScheme.getParticleCellTree());
        } else {

            aAPR.linearAccess.initialize_linear_structure_sparse(aAPR.parameters,
                                                                 iPullingSchemeSparse.particle_cell_tree);
        }
        aAPR.apr_initialized = true;

        method_timer.stop_timer();
    }

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

    if( par.auto_parameters ) {
        auto_parameters(input_image);
    }

    initPipelineAPR(aAPR, input_image.y_num, input_image.x_num, input_image.z_num);

    //Compute the local resolution estimate
    computeL(aAPR,input_image);

    applyParameters(aAPR,par);

    solveForAPR(aAPR);

    generateDatastructures(aAPR);

#else
    method_timer.start_timer("compute_gradient_magnitude_using_bsplines and local instensity scale CUDA");
    APRTimer t(true);
    APRTimer d(true);
    t.start_timer(" =========== ALL");
    {


        std::vector<GpuProcessingTask<ImageType>> gpts;

        int numOfStreams = 1;
        int repetitionsPerStream = 1;

        // Create streams and send initial task to do
        for (int i = 0; i < numOfStreams; ++i) {
            gpts.emplace_back(GpuProcessingTask<ImageType>(image_temp, local_scale_temp, par, bspline_offset, (*apr).level_max()));
            gpts.back().sendDataToGpu();
            gpts.back().processOnGpu();
        }

        for (int i = 0; i < numOfStreams * repetitionsPerStream; ++i) {
            int c = i % numOfStreams;

            // get data from previous task
            gpts[c].getDataFromGpu();

            // in theory we get new data and send them to task
            if (i  < numOfStreams * (repetitionsPerStream - 1)) {
                gpts[c].sendDataToGpu();
                gpts[c].processOnGpu();
            }

            // Postprocess on CPU
            std::cout << "--------- start CPU processing ---------- " << i << std::endl;
            init_apr(aAPR, input_image);
            d.start_timer("1");
            iPullingScheme.initialize_particle_cell_tree(aAPR.apr_access);
            d.stop_timer();
            d.start_timer("2");
            PixelData<float> lst(local_scale_temp, true);
            d.stop_timer();
            d.start_timer("3");
            get_local_particle_cell_set(lst, local_scale_temp2);
            d.stop_timer();
            d.start_timer("4");
            iPullingScheme.pulling_scheme_main();
            d.stop_timer();
            d.start_timer("5");
            PixelData<T> inImg(input_image, true);
            d.stop_timer();
            d.start_timer("6");
            std::vector<PixelData<T>> downsampled_img;
            downsamplePyrmaid(inImg, downsampled_img, aAPR.level_max(), aAPR.level_min());
            d.stop_timer();
            d.start_timer("7");
            aAPR.apr_access.initialize_structure_from_particle_cell_tree(aAPR.parameters, iPullingScheme.getParticleCellTree());
            d.stop_timer();
            d.start_timer("8");
            aAPR.get_parts_from_img(downsampled_img, aAPR.particles_intensities);
            d.stop_timer();

        }
        std::cout << "Total n ENDED" << std::endl;

    }
    t.stop_timer();
    method_timer.stop_timer();
#endif

    computation_timer.stop_timer();

    total_timer.stop_timer();

    return true;
}




template<typename ImageType> template<typename T>
inline void APRConverter<ImageType>::auto_parameters(const PixelData<T>& input_img){
    //
    //  Simple automatic parameter selection for 3D APR Flouresence Images
    //

    // TODO: fix auto params for 2D
    if(input_img.y_num > 1 && input_img.x_num > 1 && input_img.z_num > 1) {
        //take the current input parameters
        float lambda_input = par.lambda;
        float rel_error_input = par.rel_error;
        float ip_th_input = par.Ip_th;
        float min_signal_input = par.min_signal;

        APRTimer par_timer;
        par_timer.verbose_flag = false;

        //
        //  Do not compute the statistics over the whole image, but only a smaller sub-set.
        //
        const double total_required_pixel = 10*1000*1000;
        size_t num_slices = std::min((unsigned int)ceil(total_required_pixel/(1.0*input_img.y_num*input_img.x_num)),(unsigned int)input_img.z_num);
        size_t delta = std::max((unsigned int)1,(unsigned int)(input_img.z_num/num_slices));
        std::vector<size_t> selectedSlicesOffsets;
        selectedSlicesOffsets.reserve(num_slices);
        //evenly space the slices across the image
        for (size_t i1 = 0; i1 < num_slices; ++i1) {
            selectedSlicesOffsets.push_back(delta*i1);
        }

        // Get min value
        par_timer.start_timer("get_min");
        float min_val = 99999999;
        for (size_t k1 = 0; k1 < selectedSlicesOffsets.size(); ++k1) {
            min_val = std::min((float)*std::min_element(input_img.mesh.begin() + selectedSlicesOffsets[k1]*(input_img.y_num*input_img.x_num),input_img.mesh.begin()  + (selectedSlicesOffsets[k1]+1)*(input_img.y_num*input_img.x_num)),min_val);
        }
        par_timer.stop_timer();

        // will need to deal with grouped constant or zero sections in the image somewhere.... but lets keep it simple for now.
        const size_t num_bins = 10000;
        std::vector<uint64_t> freq(num_bins);
        uint64_t counter = 0;
        double total=0;

        size_t xnumynum = input_img.x_num*input_img.y_num;
        par_timer.start_timer("get_histogram");
        for (size_t s = 0; s < selectedSlicesOffsets.size(); ++s) {
            for (size_t q= selectedSlicesOffsets[s]*xnumynum; q < (selectedSlicesOffsets[s]+1)*xnumynum; ++q) {
                if(input_img.mesh[q] < (min_val + num_bins-1)){
                    freq[input_img.mesh[q]-min_val]++;
                    if(input_img.mesh[q] > 0) {
                        counter++;
                        total += input_img.mesh[q];
                    }
                }
            }

        }
        par_timer.stop_timer();

        float img_mean = counter > 0 ? total/(counter*1.0) : 1;
        float prop_total_th = 0.05; //assume there is atleast 5% background in the image
        float prop_total = 0;
        uint64_t min_j = 0;

        // set to start at one to ignore potential constant regions thresholded out. (Common in some images)
        for (unsigned int j = 1; j < num_bins; ++j) {
            prop_total += freq[j]/(counter*1.0);

            if(prop_total > prop_total_th){
                min_j = j;
                break;
            }

        }

        float proportion_flat = freq[0]/(counter*1.0f);
        float proportion_next = freq[1]/(counter*1.0f);


        PixelData<T> histogram;
        histogram.init(num_bins, 1, 1);
        std::copy(freq.begin(),freq.end(),histogram.mesh.begin());

        float tol = 0.0001;
        float lambda = 3;

        //Y direction bspline

        ///
        /// Smooth the histogram results using Bsplines
        ///
        iComputeGradient.bspline_filt_rec_y(histogram,lambda,tol);

        iComputeGradient.calc_inv_bspline_y(histogram);

        ///
        /// Calculate the local maximum after 5%  of the background on the smoothed histogram
        ///

        unsigned int local_max_j = 0;
        uint64_t local_max = 0;

        for (size_t k = min_j; k < num_bins; ++k) {

            if(histogram.mesh[k] >= ((histogram.mesh[k-1] + histogram.mesh[k-2])/2.0)){
            } else {
                local_max = histogram.mesh[k];
                local_max_j = k;
                break;
            }
        }

        //possible due to quantization your histogrtam is actually quite sparse, this corrects for the case where the smoothed intensity doesn't exist in the original image.
        if(freq[local_max_j]==0){
            while(freq[local_max_j]==0){
                local_max_j++;
                local_max=freq[local_max_j];
            }
        }

        T estimated_first_mode = local_max_j + min_val;

        std::vector<std::vector<T>> patches;

        patches.resize(std::min(local_max,(uint64_t)10000));

        for (unsigned int l = 0; l < patches.size(); ++l) {
            patches[l].resize(27, 0);
        }


        int64_t z_num = input_img.z_num;
        int64_t x_num = input_img.x_num;
        int64_t y_num = input_img.y_num;

        par_timer.start_timer("get_patches");

        uint64_t counter_p = 0;
        if (patches.size() > 0) {
            for (size_t s = 0; s < selectedSlicesOffsets.size(); ++s) {
                // limit slice to range [1, z_num-2]
                int64_t z = std::min((int) z_num - 2, std::max((int) selectedSlicesOffsets[s], (int) 1));
                for (int64_t x = 1; x < (x_num - 1); ++x) {
                    for (int64_t y = 1; y < (y_num - 1); ++y) {
                        float val = input_img.mesh[z * x_num * y_num + x * y_num + y];
                        if (val == estimated_first_mode) {
                            uint64_t counter_n = 0;
                            for (int64_t sz = -1; sz <= 1; ++sz) {
                                for (int64_t sx = -1; sx <= 1; ++sx) {
                                    for (int64_t sy = -1; sy <= 1; ++sy) {
                                        size_t idx = (z + sz) * x_num * y_num + (x + sx) * y_num + (y + sy);
                                        const auto &val = input_img.mesh[idx];
                                        patches[counter_p][counter_n] = val;
                                        counter_n++;
                                    }
                                }
                            }
                            counter_p++;
                            if (counter_p >= patches.size()) {
                                goto finish;
                            }
                        }
                    }
                }
            }
        }

        finish:
        par_timer.stop_timer();

        //first compute the mean over all the patches.
        double total_p=0;
        counter = 0;

        for (size_t i = 0; i < patches.size(); ++i) {
            for (size_t j = 0; j < patches[i].size(); ++j) {

                if(patches[i][j] > 0){
                    total_p += patches[i][j];
                    counter++;
                }
            }
        }

        float mean = counter > 0 ? total_p/(counter*1.0) : 1;

        //now compute the standard deviation (sd) of the patches

        double var=0;

        for (size_t i = 0; i < patches.size(); ++i) {
            for (size_t j = 0; j < patches[i].size(); ++j) {

                if(patches[i][j] > 0){
                    var += pow(patches[i][j]-mean,2);
                }
            }
        }

        var = (counter > 0) ? var/(counter*1) : 1;

        float sd = sqrt(var);

        par.noise_sd_estimate = sd;

        for (size_t l1 = 1; l1 < histogram.mesh.size(); ++l1) {
            if(histogram.mesh[l1] > 0){
                par.background_intensity_estimate = l1 + min_val;
            }
        }





        float min_snr = 6;


    if(par.SNR_min > 0){
        min_snr = par.SNR_min;
    } else {
        if(verbose) {
            std::cout << "**Assuming a minimum SNR of 6" << std::endl;
        }
    }


        float Ip_th = mean + sd;
        float var_th = (img_mean/(mean*1.0f))*sd*min_snr;

        float var_th_max = sd*min_snr*.5f;

        par.background_intensity_estimate = estimated_first_mode;



    //
    //  Detecting background subtracted images, or no-noise, in these cases the above estimates do not work
    //
    if((proportion_flat > 1.0f) && (proportion_next > 0.00001f)){
        if(verbose) {
            std::cout
                    << "AUTOPARAMTERS:**Warning** Detected that there is likely noisy background, instead assuming background subtracted and minimum signal of 5 (absolute), if this is not the case please set parameters manually"
                    << std::endl;
        }
        Ip_th = 1;
        var_th = 5;
        lambda = 0.5;
        var_th_max = 2;
    } else {
       // std::cout << "AUTOPARAMTERS: **Assuming image has atleast 5% dark background" << std::endl;
    }


        /*
         *  Input parameter over-ride.
         *
         */

        if(min_signal_input < 0) {
            par.sigma_th = var_th;
            par.sigma_th_max = var_th_max;
        } else {
            par.sigma_th_max = par.min_signal*0.5f;
            par.sigma_th = par.min_signal;
        }


        if(lambda_input != -1) {
            par.lambda = lambda_input;
        }else{
            par.lambda = lambda;
        }

        if(par.lambda < 0.05){
            par.lambda = 0;
            std::cout << "setting lambda to zero, bsplines algorithm cannot work with such small lambda" << std::endl;
        }

        if(ip_th_input != -1){
            par.Ip_th = ip_th_input;
        } else {
            par.Ip_th = Ip_th;
        }

        if(rel_error_input != 0.1){
            par.rel_error = rel_error_input;
        }

        std::cout << "Used parameters: " << std::endl;
        std::cout << "I_th: " << par.Ip_th << std::endl;
        std::cout << "sigma_th: " << par.sigma_th << std::endl;
        std::cout << "sigma_th_max: " << par.sigma_th_max << std::endl;
        std::cout << "relative error (E): " << par.rel_error << std::endl;
        std::cout << "lambda: " << par.lambda << std::endl;
    } else {
        par.Ip_th = 1000;
        par.sigma_th = 100;
        par.sigma_th_max = 10;
        par.rel_error = 0.1;
        par.lambda = 1;

        std::cout << "Used parameters: " << std::endl;
        std::cout << "I_th: " << par.Ip_th << std::endl;
        std::cout << "sigma_th: " << par.sigma_th << std::endl;
        std::cout << "sigma_th_max: " << par.sigma_th_max << std::endl;
        std::cout << "relative error (E): " << par.rel_error << std::endl;
        std::cout << "lambda: " << par.lambda << std::endl;
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
