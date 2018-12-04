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
#include <data_structures/APR/APR.hpp>

#include "data_structures/APR/APR.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "io/TiffUtils.hpp"

#include "PullingScheme.hpp"
#include "LocalParticleCellSet.hpp"
#include "LocalIntensityScale.hpp"
#include "ComputeGradient.hpp"
#include "algorithm/OVPC.h"

#ifdef APR_USE_CUDA
#include "algorithm/ComputeGradientCuda.hpp"
#endif


template<typename ImageType>
class APRConverter {

    PullingScheme iPullingScheme;
    LocalParticleCellSet iLocalParticleSet;
    LocalIntensityScale iLocalIntensityScale;
    ComputeGradient iComputeGradient;


public:

    APRTimer fine_grained_timer;
    APRTimer method_timer;
    APRTimer total_timer;
    APRTimer allocation_timer;
    APRTimer computation_timer;
    APRParameters par;

    bool get_apr(APR<ImageType> &aAPR);

    //get apr without setting parameters, and with an already loaded image.
    template<typename T>
    bool get_apr_method(APR<ImageType> &aAPR, PixelData<T> &input_image);

    template<typename T>
    void auto_parameters(const PixelData<T> &input_img);


private:

    //pointer to the APR structure so member functions can have access if they need
    const APR<ImageType> *apr;

    template<typename T>
    void init_apr(APR<ImageType>& aAPR, PixelData<T>& input_image);

    template<typename T>
    bool get_apr_method_from_file(APR<ImageType> &aAPR, PixelData<T> input_image);

public:
    void get_gradient(PixelData<ImageType> &image_temp, PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par);
    void get_local_intensity_scale(PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, const APRParameters &par);
    void computeLevels(const PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp, int maxLevel, float relError, float dx = 1, float dy = 1, float dz = 1);
    void get_local_particle_cell_set(PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2);
};

template<typename ImageType>
inline bool APRConverter<ImageType>::get_apr(APR<ImageType> &aAPR) {
    apr = &aAPR;
#ifdef HAVE_LIBTIFF
    TiffUtils::TiffInfo inputTiff(par.input_dir + par.input_image_name);
    if (!inputTiff.isFileOpened()) return false;


    if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT8) {
        return get_apr_method_from_file<uint8_t>(aAPR, TiffUtils::getMesh<uint8_t>(inputTiff));
    } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_FLOAT) {
        return get_apr_method_from_file<float>(aAPR, TiffUtils::getMesh<float>(inputTiff));
    } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT16) {
        return get_apr_method_from_file<uint16_t>(aAPR, TiffUtils::getMesh<uint16_t>(inputTiff));
    } else {
        std::cerr << "Wrong file type" << std::endl;
        return false;
    }
#else
    return false;
#endif 
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

/**
 * Main method for constructing the APR from an input image
 */
template<typename ImageType> template<typename T>
inline bool APRConverter<ImageType>::get_apr_method_from_file(APR<ImageType> &aAPR, PixelData<T> inputImage) {
//    allocation_timer.start_timer("read tif input image");
//    PixelData<T> inputImage = TiffUtils::getMesh<T>(aTiffFile);
//    allocation_timer.stop_timer();

    method_timer.start_timer("calculate automatic parameters");

    if(par.normalized_input) {
        MinMax<T> mm;
        T maxValue;
        if ((std::is_same<uint16_t, ImageType>::value) || (std::is_same<uint8_t, ImageType>::value)) {
            mm = getMinMax(inputImage);
            maxValue = static_cast<T>((float) std::numeric_limits<ImageType>::max() * 0.8);
            std::cout << "Normalizing image with min: " << mm.min << " max: " << mm.max << " to a dynamic range of: " << maxValue << std::endl;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
            for (size_t i = 0; i < inputImage.mesh.size(); ++i) {
                inputImage.mesh[i] = (inputImage.mesh[i] - mm.min) * maxValue / (mm.max - mm.min);
            }

            //normalize the input parameters if required
            if(par.Ip_th!=-1){
                std::cout << "Scaled input intensity threshold" << std::endl;
                par.Ip_th = (par.Ip_th - mm.min)* maxValue / (mm.max - mm.min);
            }

            if(par.min_signal!=-1){
                std::cout << "Scaled input min signal threshold" << std::endl;
                par.min_signal = (par.min_signal)* maxValue / (mm.max - mm.min);
            }
        }
    }


    //auto_parameters(inputImage);
    method_timer.stop_timer();

    return get_apr_method(aAPR, inputImage);
}

/**
 * Main method for constructing the APR from an input image
 */
template<typename ImageType> template<typename T>
inline bool APRConverter<ImageType>::get_apr_method(APR<ImageType> &aAPR, PixelData<T>& input_image) {

    apr = &aAPR; // in case it was called directly

    if( par.auto_parameters ) {
        auto_parameters(input_image);
    }

    total_timer.start_timer("Total_pipeline_excluding_IO");

    init_apr(aAPR, input_image);

    ////////////////////////////////////////
    /// Memory allocation of variables
    ////////////////////////////////////////

    //assuming uint16, the total memory cost shoudl be approximately (1 + 1 + 1/8 + 2/8 + 2/8) = 2 5/8 original image size in u16bit
    //storage of the particle cell tree for computing the pulling scheme
    allocation_timer.start_timer("init and copy image");
    PixelData<ImageType> image_temp(input_image, false /* don't copy */, true /* pinned memory */); // global image variable useful for passing between methods, or re-using memory (should be the only full sized copy of the image)
    PixelData<ImageType> grad_temp; // should be a down-sampled image
    grad_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0, false);
    PixelData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
    local_scale_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, true);
    PixelData<float> local_scale_temp2;
    local_scale_temp2.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, false);
    allocation_timer.stop_timer();

    /////////////////////////////////
    /// Pipeline
    ////////////////////////

    computation_timer.start_timer("Calculations");

    fine_grained_timer.start_timer("offset image");
    //offset image by factor (this is required if there are zero areas in the background with uint16_t and uint8_t images, as the Bspline co-efficients otherwise may be negative!)
    // Warning both of these could result in over-flow (if your image is non zero, with a 'buffer' and has intensities up to uint16_t maximum value then set image_type = "", i.e. uncomment the following line)
    float bspline_offset = 0;
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
    method_timer.verbose_flag = true;
    method_timer.start_timer("compute_gradient_magnitude_using_bsplines");
    get_gradient(image_temp, grad_temp, local_scale_temp, local_scale_temp2, bspline_offset, par);
    method_timer.stop_timer();
#ifdef HAVE_LIBTIFF
    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "gradient_step.tif", grad_temp);
    }
#endif
    method_timer.start_timer("compute_local_intensity_scale");
    get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
    method_timer.stop_timer();
    //method_timer.verbose_flag = false;
#ifdef HAVE_LIBTIFF
    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "local_intensity_scale_step.tif", local_scale_temp);
    }
#endif
    method_timer.start_timer("compute_levels");
    computeLevels(grad_temp, local_scale_temp, (*apr).level_max(), par.rel_error, par.dx, par.dy, par.dz);
    method_timer.stop_timer();

    method_timer.start_timer("initialize_particle_cell_tree");
    iPullingScheme.initialize_particle_cell_tree(aAPR.apr_access);
    method_timer.stop_timer();

    method_timer.start_timer("compute_local_particle_set");
    get_local_particle_cell_set(local_scale_temp, local_scale_temp2);
    method_timer.stop_timer();

    method_timer.start_timer("compute_pulling_scheme");
    iPullingScheme.pulling_scheme_main();
    method_timer.stop_timer();

    method_timer.start_timer("downsample_pyramid");
    std::vector<PixelData<T>> downsampled_img;
    //Down-sample the image for particle intensity estimation
    downsamplePyrmaid(input_image, downsampled_img, aAPR.level_max(), aAPR.level_min());
    method_timer.stop_timer();

    method_timer.start_timer("compute_apr_datastructure");
    aAPR.apr_access.initialize_structure_from_particle_cell_tree(aAPR.parameters.neighborhood_optimization, iPullingScheme.getParticleCellTree());
    method_timer.stop_timer();

    method_timer.start_timer("sample_particles");
    aAPR.get_parts_from_img(downsampled_img,aAPR.particles_intensities);
    method_timer.stop_timer();
#else
    method_timer.start_timer("compute_gradient_magnitude_using_bsplines and local instensity scale CUDA");
    APRTimer t(true);
    APRTimer d(true);
    t.start_timer(" =========== ALL");

        int numOfStreams = 1;
        int repetitionsPerStream = 1;

   
 {


        std::vector<GpuProcessingTask<ImageType>> gpts;

//        int numOfStreams = 4;
  //      int repetitionsPerStream = 10;

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

            d.start_timer("1 - initialize particle cell tree");
            iPullingScheme.initialize_particle_cell_tree(aAPR.apr_access);
            d.stop_timer();

            d.start_timer("2 - copy LST");
        PixelData<float> &lst = local_scale_temp;   
 //	 PixelData<float> lst(local_scale_temp, true);
    //        PixelData<float> lst2(local_scale_temp, true);
            d.stop_timer();

            d.start_timer("3 - get local particle cell set");
            get_local_particle_cell_set(lst, local_scale_temp2);
            d.stop_timer();

            ////////////////////////////////
            d.start_timer("4 - pulling main");
            iPullingScheme.pulling_scheme_main();
            d.stop_timer();

            //d.start_timer("4 - new pulling main");
            //OVPC nps(aAPR.apr_access, lst2);
            //nps.generateTree();
            //d.stop_timer();


            PixelData<T> &inImg = input_image; // user redy data and later...

            d.start_timer("5 - init struct from particle cell tree");
            aAPR.apr_access.initialize_structure_from_particle_cell_tree(aAPR.parameters.neighborhood_optimization, iPullingScheme.getParticleCellTree());
            d.stop_timer();

            d.start_timer("6 - downsample pyramid");
            // - downsample (mean) pyramid from input image
            std::vector<PixelData<T>> downsampled_img;
            downsamplePyrmaid(inImg, downsampled_img, aAPR.level_max(), aAPR.level_min());
            d.stop_timer();
            ////////////////////////////////


            d.start_timer("7 - sample particles");
            // - sample particle values on all levels
            aAPR.get_parts_from_img(downsampled_img, aAPR.particles_intensities);
            input_image.swap(downsampled_img.back()); // ... revert
            d.stop_timer();

        }
        std::cout << "Total n ENDED" << std::endl;

    }
    double allT = t.stop_timer();
    std::cout << "BW=" << (numOfStreams * repetitionsPerStream * input_image.size()) / allT / 1000000000.0 << "GB/s" << std::endl;
    method_timer.stop_timer();
#endif

    computation_timer.stop_timer();

    double totT = total_timer.stop_timer();
std::cout << "BW=" << (numOfStreams * repetitionsPerStream * input_image.size()) / totT / 1000000000.0 << "GB/s" << std::endl;
    return true;
}

template<typename ImageType>
inline void APRConverter<ImageType>::computeLevels(const PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp, int maxLevel, float relError, float dx, float dy, float dz) {
    //divide gradient magnitude by Local Intensity Scale (first step in calculating the Local Resolution Estimate L(y), minus constants)
    fine_grained_timer.start_timer("compute_level_first");
    #ifdef HAVE_OPENMP
    #pragma omp parallel for default(shared)
    #endif
    for (size_t i = 0; i < grad_temp.mesh.size(); ++i) {
        local_scale_temp.mesh[i] = grad_temp.mesh[i] / local_scale_temp.mesh[i];
    }
    fine_grained_timer.stop_timer();

    float min_dim = std::min(dy, std::min(dx, dz));
    float level_factor = pow(2, maxLevel) * min_dim;

    //incorporate other factors and compute the level of the Particle Cell, effectively construct LPC L_n
    fine_grained_timer.start_timer("compute_level_second");
    iLocalParticleSet.compute_level_for_array(local_scale_temp, level_factor, relError);
    fine_grained_timer.stop_timer();
}

template<typename ImageType>
inline void APRConverter<ImageType>::get_local_particle_cell_set(PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2) {
    //  Computes the Local Particle Cell Set from a down-sampled local intensity scale (\sigma) and gradient magnitude
    //  Down-sampled due to the Equivalence Optimization

#ifdef HAVE_LIBTIFF
    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "local_particle_set_level_step.tif", local_scale_temp);
    }
#endif

    int l_max = (*apr).level_max() - 1;
    int l_min = (*apr).level_min();

    fine_grained_timer.start_timer("pulling_scheme_fill_max_level");
    iPullingScheme.fill(l_max,local_scale_temp);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("level_loop_initialize_tree");
    for(int l_ = l_max - 1; l_ >= l_min; l_--){

        //down sample the resolution level k, using a max reduction
        downsample(local_scale_temp, local_scale_temp2,
                   [](const float &x, const float &y) -> float { return std::max(x, y); },
                   [](const float &x) -> float { return x; }, true);
        //for those value of level k, add to the hash table
        iPullingScheme.fill(l_,local_scale_temp2);
        //assign the previous mesh to now be resampled.
        local_scale_temp.swap(local_scale_temp2);
    }
    fine_grained_timer.stop_timer();
}

template<typename ImageType>
inline void APRConverter<ImageType>::get_gradient(PixelData<ImageType> &image_temp, PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par) {
    //  Bevan Cheeseman 2018
    //  Calculate the gradient from the input image. (You could replace this method with your own)
    //  Input: full sized image.
    //  Output: down-sampled by 2 gradient magnitude (Note, the gradient is calculated at pixel level then maximum down sampled within the loops below)

    //fine_grained_timer.verbose_flag = true;

    fine_grained_timer.start_timer("threshold");

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < image_temp.mesh.size(); ++i) {
        if (image_temp.mesh[i] <= (par.Ip_th + bspline_offset)) { image_temp.mesh[i] = par.Ip_th + bspline_offset; }
    }
    fine_grained_timer.stop_timer();


    fine_grained_timer.start_timer("smooth_bspline");
    if(par.lambda > 0) {
        iComputeGradient.get_smooth_bspline_3D(image_temp, par.lambda);
    }
    fine_grained_timer.stop_timer();


    fine_grained_timer.start_timer("calc_bspline_fd_mag_ds");
    iComputeGradient.calc_bspline_fd_ds_mag(image_temp,grad_temp,par.dx,par.dy,par.dz);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("down-sample_b-spline");
    downsample(image_temp, local_scale_temp,
               [](const float &x, const float &y) -> float { return x + y; },
               [](const float &x) -> float { return x / 8.0; });
    fine_grained_timer.stop_timer();

    if(par.lambda > 0){
        fine_grained_timer.start_timer("calc_inv_bspline_y");
        iComputeGradient.calc_inv_bspline_y(local_scale_temp);
        fine_grained_timer.stop_timer();
        fine_grained_timer.start_timer("calc_inv_bspline_x");
        iComputeGradient.calc_inv_bspline_x(local_scale_temp);
        fine_grained_timer.stop_timer();
        fine_grained_timer.start_timer("calc_inv_bspline_z");
        iComputeGradient.calc_inv_bspline_z(local_scale_temp);
        fine_grained_timer.stop_timer();
    }

    fine_grained_timer.start_timer("load_and_apply_mask");
    // Apply mask if given
    if(par.mask_file != ""){
        iComputeGradient.mask_gradient(grad_temp,local_scale_temp2,image_temp, par);
    }
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("threshold");
    iComputeGradient.threshold_gradient(grad_temp,local_scale_temp,par.Ip_th + bspline_offset);
    fine_grained_timer.stop_timer();
}

template<typename ImageType>
void APRConverter<ImageType>::get_local_intensity_scale(PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, const APRParameters &par) {
    //
    //  Calculate the Local Intensity Scale (You could replace this method with your own)
    //
    //  Input: full sized image.
    //
    //  Output: down-sampled Local Intensity Scale (h) (Due to the Equivalence Optimization we only need down-sampled values)
    //

    fine_grained_timer.start_timer("copy_intensities_from_bsplines");
    //copy across the intensities
    local_scale_temp2.copyFromMesh(local_scale_temp);
    fine_grained_timer.stop_timer();

    float var_rescale;
    std::vector<int> var_win;
    iLocalIntensityScale.get_window(var_rescale,var_win,par);
    size_t win_y = var_win[0];
    size_t win_x = var_win[1];
    size_t win_z = var_win[2];
    size_t win_y2 = var_win[3];
    size_t win_x2 = var_win[4];
    size_t win_z2 = var_win[5];

    std::cout << "CPU WINDOWS: " << win_y << " " << win_x << " " << win_z << " " << win_y2 << " " << win_x2 << " " << win_z2 << std::endl;
    fine_grained_timer.start_timer("calc_sat_mean_y");
    iLocalIntensityScale.calc_sat_mean_y(local_scale_temp,win_y);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("calc_sat_mean_x");
    iLocalIntensityScale.calc_sat_mean_x(local_scale_temp,win_x);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("calc_sat_mean_z");
    iLocalIntensityScale.calc_sat_mean_z(local_scale_temp,win_z);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("second_pass_and_rescale");
    //calculate abs and subtract from original
    iLocalIntensityScale.calc_abs_diff(local_scale_temp2,local_scale_temp);
    //Second spatial average
    iLocalIntensityScale.calc_sat_mean_y(local_scale_temp,win_y2);
    iLocalIntensityScale.calc_sat_mean_x(local_scale_temp,win_x2);
    iLocalIntensityScale.calc_sat_mean_z(local_scale_temp,win_z2);
    iLocalIntensityScale.rescale_var_and_threshold( local_scale_temp, var_rescale,par);
    fine_grained_timer.stop_timer();
}


template<typename ImageType> template<typename T>
inline void APRConverter<ImageType>::init_apr(APR<ImageType>& aAPR,PixelData<T>& input_image){
    //
    //  Initializing the size of the APR, min and maximum level (in the data structures it is called depth)
    //

    aAPR.apr_access.org_dims[0] = input_image.y_num;
    aAPR.apr_access.org_dims[1] = input_image.x_num;
    aAPR.apr_access.org_dims[2] = input_image.z_num;

    int max_dim = std::max(std::max(aAPR.apr_access.org_dims[1], aAPR.apr_access.org_dims[0]), aAPR.apr_access.org_dims[2]);
    int min_dim = std::min(std::min(aAPR.apr_access.org_dims[1], aAPR.apr_access.org_dims[0]), aAPR.apr_access.org_dims[2]);

    int levelMax = ceil(std::log2(max_dim));
    // TODO: why minimum level is forced here to be 2?
    int levelMin = std::max( (int)(levelMax - floor(std::log2(min_dim))), 2);

    aAPR.apr_access.l_min = levelMin;
    aAPR.apr_access.l_max = levelMax;

    aAPR.parameters = par;
}

template<typename ImageType> template<typename T>
inline void APRConverter<ImageType>::auto_parameters(const PixelData<T>& input_img){
    //
    //  Simple automatic parameter selection for 3D APR Flouresence Images
    //

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
        std::cout << "**Assuming a minimum SNR of 6" << std::endl;
    }

    float Ip_th = mean + sd;
    float var_th = (img_mean/(mean*1.0f))*sd*min_snr;

    float var_th_max = sd*min_snr*.5f;

    par.background_intensity_estimate = estimated_first_mode;


    //
    //  Detecting background subtracted images, or no-noise, in these cases the above estimates do not work
    //
    if((proportion_flat > 1.0f) && (proportion_next > 0.00001f)){
        std::cout << "AUTOPARAMTERS:**Warning** Detected that there is likely noisy background, instead assuming background subtracted and minimum signal of 5 (absolute), if this is not the case please set parameters manually" << std::endl;
        Ip_th = 1;
        var_th = 5;
        lambda = 0.5;
        var_th_max = 2;
    } else {
        std::cout << "AUTOPARAMTERS: **Assuming image has atleast 5% dark background" << std::endl;
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
}


#endif // __APR_CONVERTER_HPP__
