////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// APR Converter class handles the methods and functions for creating an APR from an input image
///
////////////////////////////////

#ifndef PARTPLAY_APR_CONVERTER_HPP
#define PARTPLAY_APR_CONVERTER_HPP

#include "../data_structures/Mesh/MeshData.hpp"
#include "../io/TiffUtils.hpp"
#include "../data_structures/APR/APR.hpp"

#include "ComputeGradient.hpp"
#include "LocalIntensityScale.hpp"
#include "LocalParticleCellSet.hpp"
#include "PullingScheme.hpp"


template<typename ImageType>
class APRConverter: public LocalIntensityScale, public ComputeGradient, public LocalParticleCellSet, public PullingScheme {

public:
    APRParameters par;
    APRTimer fine_grained_timer;
    APRTimer method_timer;
    APRTimer total_timer;
    APRTimer allocation_timer;
    APRTimer computation_timer;

    bool get_apr(APR<ImageType> &aAPR) {
        apr = &aAPR;

        TiffUtils::TiffInfo inputTiff(par.input_dir + par.input_image_name);
        if (!inputTiff.isFileOpened()) return false;

        if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT8) {
            return get_apr_method_from_file<uint8_t>(aAPR, inputTiff);
        } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_FLOAT) {
            return get_apr_method_from_file<float>(aAPR, inputTiff);
        } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT16) {
            return get_apr_method_from_file<uint16_t>(aAPR, inputTiff);
        } else {
            std::cerr << "Wrong file type" << std::endl;
            return false;
        }
    };

private:
    //get apr without setting parameters, and with an already loaded image.
    template<typename T>
    bool get_apr_method(APR<ImageType> &aAPR, MeshData<T> &input_image);

    //pointer to the APR structure so member functions can have access if they need
    const APR<ImageType> *apr;

    template<typename T>
    void init_apr(APR<ImageType>& aAPR, MeshData<T>& input_image);

    template<typename T>
    void auto_parameters(const MeshData<T> &input_img);

    template<typename T>
    bool get_apr_method_from_file(APR<ImageType> &aAPR, const TiffUtils::TiffInfo &aTiffFile);

    void get_gradient(MeshData<ImageType> &image_temp, MeshData<ImageType> &grad_temp, MeshData<float> &local_scale_temp, MeshData<float> &local_scale_temp2, float bspline_offset);
    void get_local_intensity_scale(MeshData<float> &local_scale_temp, MeshData<float> &local_scale_temp2);
    void get_local_particle_cell_set(MeshData<ImageType> &grad_temp, MeshData<float> &local_scale_temp, MeshData<float> &local_scale_temp2);
};


/**
 * Main method for constructing the APR from an input image
 */
template<typename ImageType> template<typename T>
bool APRConverter<ImageType>::get_apr_method_from_file(APR<ImageType> &aAPR, const TiffUtils::TiffInfo &aTiffFile) {
    allocation_timer.start_timer("read tif input image");
    MeshData<T> inputImage = TiffUtils::getMesh<T>(aTiffFile);
    allocation_timer.stop_timer();

    method_timer.start_timer("calculate automatic parameters");
    auto_parameters(inputImage);
    method_timer.stop_timer();

    return get_apr_method(aAPR, inputImage);
}

/**
 * Main method for constructing the APR from an input image
 */
template<typename ImageType> template<typename T>
bool APRConverter<ImageType>::get_apr_method(APR<ImageType> &aAPR, MeshData<T>& input_image) {
    apr = &aAPR; // in case it was called directly

    total_timer.start_timer("Total_pipeline_excluding_IO");

    init_apr(aAPR, input_image);

    ////////////////////////////////////////
    /// Memory allocation of variables
    ////////////////////////////////////////

    //assuming uint16, the total memory cost shoudl be approximately (1 + 1 + 1/8 + 2/8 + 2/8) = 2 5/8 original image size in u16bit
    //storage of the particle cell tree for computing the pulling scheme
    allocation_timer.start_timer("init and copy image");
    MeshData<ImageType> image_temp(input_image, false /* don't copy */); // global image variable useful for passing between methods, or re-using memory (should be the only full sized copy of the image)
    MeshData<ImageType> grad_temp; // should be a down-sampled image
    grad_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0);
    MeshData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
    local_scale_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
    MeshData<float> local_scale_temp2;
    local_scale_temp2.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
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

    method_timer.start_timer("compute_gradient_magnitude_using_bsplines");
    get_gradient(image_temp, grad_temp, local_scale_temp, local_scale_temp2, bspline_offset);
    method_timer.stop_timer();

    method_timer.start_timer("compute_local_intensity_scale");
    get_local_intensity_scale(local_scale_temp, local_scale_temp2);
    method_timer.stop_timer();

    method_timer.start_timer("initialize_particle_cell_tree");
    initialize_particle_cell_tree(aAPR);
    method_timer.stop_timer();

    method_timer.start_timer("compute_local_particle_set");
    get_local_particle_cell_set(grad_temp, local_scale_temp, local_scale_temp2);
    method_timer.stop_timer();

    method_timer.start_timer("compute_pulling_scheme");
    PullingScheme::pulling_scheme_main();
    method_timer.stop_timer();

    method_timer.start_timer("downsample_pyramid");
    std::vector<MeshData<T>> downsampled_img;
    //Down-sample the image for particle intensity estimation
    downsamplePyrmaid(input_image, downsampled_img, aAPR.level_max(), aAPR.level_min());
    method_timer.stop_timer();

    method_timer.start_timer("compute_apr_datastructure");
    aAPR.apr_access.initialize_structure_from_particle_cell_tree(aAPR,particle_cell_tree);
    method_timer.stop_timer();

    method_timer.start_timer("sample_particles");
    aAPR.get_parts_from_img(downsampled_img,aAPR.particles_intensities);
    method_timer.stop_timer();

    computation_timer.stop_timer();

    aAPR.parameters = par;

    total_timer.stop_timer();

    return true;
}

template<typename ImageType>
void APRConverter<ImageType>::get_local_particle_cell_set(MeshData<ImageType> &grad_temp, MeshData<float> &local_scale_temp, MeshData<float> &local_scale_temp2) {
    //
    //  Computes the Local Particle Cell Set from a down-sampled local intensity scale (\sigma) and gradient magnitude
    //
    //  Down-sampled due to the Equivalence Optimization
    //

    fine_grained_timer.start_timer("compute_level_first");
    //divide gradient magnitude by Local Intensity Scale (first step in calculating the Local Resolution Estimate L(y), minus constants)
    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared)
    #endif
    for(size_t i = 0; i < grad_temp.mesh.size(); ++i) {
        local_scale_temp.mesh[i] = (1.0*grad_temp.mesh[i])/(local_scale_temp.mesh[i]*1.0);
    }
    fine_grained_timer.stop_timer();

    float min_dim = std::min(par.dy,std::min(par.dx,par.dz));
    float level_factor = pow(2,(*apr).level_max())*min_dim;

    int l_max = (*apr).level_max() - 1;
    int l_min = (*apr).level_min();

    fine_grained_timer.start_timer("compute_level_second");
    //incorporate other factors and compute the level of the Particle Cell, effectively construct LPC L_n
    compute_level_for_array(local_scale_temp,level_factor,par.rel_error);
    fill(l_max,local_scale_temp);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("level_loop_initialize_tree");
    for(int l_ = l_max - 1; l_ >= l_min; l_--){

        //down sample the resolution level k, using a max reduction
        downsample(local_scale_temp, local_scale_temp2,
                   [](const float &x, const float &y) -> float { return std::max(x, y); },
                   [](const float &x) -> float { return x; }, true);
        //for those value of level k, add to the hash table
        fill(l_,local_scale_temp2);
        //assign the previous mesh to now be resampled.
        local_scale_temp.swap(local_scale_temp2);
    }
    fine_grained_timer.stop_timer();
}

template<typename ImageType>
void APRConverter<ImageType>::get_gradient(MeshData<ImageType> &image_temp, MeshData<ImageType> &grad_temp, MeshData<float> &local_scale_temp, MeshData<float> &local_scale_temp2, float bspline_offset) {
    //  Bevan Cheeseman 2018
    //  Calculate the gradient from the input image. (You could replace this method with your own)
    //  Input: full sized image.
    //  Output: down-sampled by 2 gradient magnitude (Note, the gradient is calculated at pixel level then maximum down sampled within the loops below)

    fine_grained_timer.verbose_flag = false;

    fine_grained_timer.start_timer("smooth_bspline");
    if(par.lambda > 0) {
        get_smooth_bspline_3D(image_temp, par.lambda);
    }
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("calc_bspline_fd_mag_ds");
    calc_bspline_fd_ds_mag(image_temp,grad_temp,par.dx,par.dy,par.dz);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("down-sample_b-spline");
    downsample(image_temp, local_scale_temp,
               [](const float &x, const float &y) -> float { return x + y; },
               [](const float &x) -> float { return x / 8.0; });
    fine_grained_timer.stop_timer();

    if(par.lambda > 0){
        fine_grained_timer.start_timer("calc_inv_bspline_y");
        calc_inv_bspline_y(local_scale_temp);
        fine_grained_timer.stop_timer();
        fine_grained_timer.start_timer("calc_inv_bspline_x");
        calc_inv_bspline_x(local_scale_temp);
        fine_grained_timer.stop_timer();
        fine_grained_timer.start_timer("calc_inv_bspline_z");
        calc_inv_bspline_z(local_scale_temp);
        fine_grained_timer.stop_timer();
    }

    fine_grained_timer.start_timer("load_and_apply_mask");
    // Apply mask if given
    if(par.mask_file != ""){
        mask_gradient(grad_temp,local_scale_temp2,image_temp, par);
    }
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("threshold");
    threshold_gradient(grad_temp,local_scale_temp,par.Ip_th + bspline_offset);
    fine_grained_timer.stop_timer();
}

template<typename ImageType>
void APRConverter<ImageType>::get_local_intensity_scale(MeshData<float> &local_scale_temp, MeshData<float> &local_scale_temp2) {
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
    get_window(var_rescale,var_win,par);

    size_t win_y = var_win[0];
    size_t win_x = var_win[1];
    size_t win_z = var_win[2];
    size_t win_y2 = var_win[3];
    size_t win_x2 = var_win[4];
    size_t win_z2 = var_win[5];

    fine_grained_timer.start_timer("calc_sat_mean_y");
    calc_sat_mean_y(local_scale_temp,win_y);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("calc_sat_mean_x");
    calc_sat_mean_x(local_scale_temp,win_x);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("calc_sat_mean_z");
    calc_sat_mean_z(local_scale_temp,win_z);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("second_pass_and_rescale");
    //calculate abs and subtract from original
    calc_abs_diff(local_scale_temp2,local_scale_temp);
    //Second spatial average
    calc_sat_mean_y(local_scale_temp,win_y2);
    calc_sat_mean_x(local_scale_temp,win_x2);
    calc_sat_mean_z(local_scale_temp,win_z2);
    rescale_var_and_threshold( local_scale_temp, var_rescale,par);
    fine_grained_timer.stop_timer();
}


template<typename ImageType> template<typename T>
void APRConverter<ImageType>::init_apr(APR<ImageType>& aAPR,MeshData<T>& input_image){
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

    aAPR.apr_access.level_min = levelMin;
    aAPR.apr_access.level_max = levelMax;
}

template<typename ImageType> template<typename T>
void APRConverter<ImageType>::auto_parameters(const MeshData<T>& input_img){
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

    MeshData<T> histogram;
    histogram.init(num_bins, 1, 1);
    std::copy(freq.begin(),freq.end(),histogram.mesh.begin());

    float tol = 0.0001;
    float lambda = 3;

    //Y direction bspline

    ///
    /// Smooth the histogram results using Bsplines
    ///
    bspline_filt_rec_y(histogram,lambda,tol);

    calc_inv_bspline_y(histogram);

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


#endif //PARTPLAY_APR_CONVERTER_HPP
