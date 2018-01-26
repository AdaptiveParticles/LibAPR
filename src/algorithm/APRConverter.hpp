////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// APR Converter class handles the methods and functions for creating an APR from an input image
///
////////////////////////////////

#ifndef PARTPLAY_APR_CONVERTER_HPP
#define PARTPLAY_APR_CONVERTER_HPP

#include "src/data_structures/Mesh/MeshData.hpp"
#include "src/io/TiffUtils.hpp"
#include "src/data_structures/APR/APR.hpp"

#include "src/algorithm/ComputeGradient.hpp"
#include "src/algorithm/LocalIntensityScale.hpp"
#include "src/algorithm/LocalParticleCellSet.hpp"
#include "src/algorithm/PullingScheme.hpp"


template<typename ImageType>
class APRConverter: public LocalIntensityScale, public ComputeGradient, public LocalParticleCellSet, public PullingScheme {

public:

    APRConverter():image_type("uint16"){
    }

    APRParameters par;

    APRTimer total_timer;
    APRTimer allocation_timer;

    APRTimer computation_timer;

    APRTimer method_timer;

    APRTimer fine_grained_timer;

    std::string image_type; //default uint16

    /*
     * Declerations
     */

    bool get_apr(APR<ImageType>& apr){
        //
        //  Different input image types
        //

        //set the pointer ot the data-structure
        apr_ = &apr;

        TiffUtils::TiffInfo inputTiff(par.input_dir + par.input_image_name);
        if (!inputTiff.isFileOpened()) return false;

        if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT8) {
            image_type = "uint8";
            return get_apr_method_from_file<uint8_t>(apr, inputTiff);
        } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_FLOAT) {
            image_type = "float";
            return get_apr_method_from_file<float>(apr, inputTiff);
        } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT16) {
            image_type = "uint16";
            return get_apr_method_from_file<uint16_t>(apr, inputTiff);
        } else {
            std::cerr << "Wrong file type" << std::endl;
            return false;
        }

    };

    //get apr without setting parameters, and with an already loaded image.
    template<typename T>
    bool get_apr_method(APR<ImageType>& apr, MeshData<T>& input_image);

private:

    /*
     * Private member variables
     */

    //pointer to the APR structure so member functions can have access if they need
    APR<ImageType>* apr_;

    MeshData<ImageType> image_temp; // global image variable useful for passing between methods, or re-using memory (should be the only full sized copy of the image)

    MeshData<ImageType> grad_temp; // should be a down-sampled image

    MeshData<float> local_scale_temp; //   Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors

    MeshData<float> local_scale_temp2;  //   Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors

    //assuming uint16, the total memory cost shoudl be approximately (1 + 1 + 1/8 + 2/8 + 2/8) = 2 5/8 original image size in u16bit

    //storage of the particle cell tree for computing the pulling scheme


    float bspline_offset=0;

    /*
     * Private member functions
     */

    template<typename T>
    void init_apr(APR<ImageType>& apr,MeshData<T>& input_image);

    template<typename T>
    void auto_parameters(const MeshData<T>& input_img);

    template<typename T>
    bool get_apr_method_from_file(APR<ImageType>& apr, const TiffUtils::TiffInfo &tiffFile);

    template<typename T,typename S>
    void get_gradient(MeshData<T>& input_img,MeshData<S>& gradient);

    template<typename T,typename S>
    void get_local_intensity_scale(MeshData<T>& input_img,MeshData<S>& local_intensity_scale);

    template<typename T,typename S>
    void get_local_particle_cell_set(MeshData<T>& grad_image_ds,MeshData<S>& local_intensity_scale_ds);



};

/*
 * Implimentations
 */

template<typename ImageType> template<typename T>
bool APRConverter<ImageType>::get_apr_method_from_file(APR<ImageType>& apr, const TiffUtils::TiffInfo &tiffFile) {
    //
    //  Main method for constructing the APR from an input image
    //

    allocation_timer.start_timer("read tif input image");

    std::cout << image_type << std::endl;

    //input type
    MeshData<T> input_image = TiffUtils::getMesh<T>(tiffFile);

    allocation_timer.stop_timer();

    //    was there an image found
    if (input_image.mesh.size() == 0) {
        std::cout << "Image Not Found" << std::endl;
        return false;
    }

    computation_timer.start_timer("calculate automatic parameters");
    auto_parameters(input_image);
    computation_timer.stop_timer();

    return get_apr_method(apr,input_image);

}


template<typename ImageType> template<typename T>
bool APRConverter<ImageType>::get_apr_method(APR<ImageType>& apr, MeshData<T>& input_image) {
    //
    //  Main method for constructing the APR from an input image
    //

    apr_ = &apr; // in case it was called directly


    //Initialize the apr size parameters from the image
    init_apr(apr,input_image);

    ////////////////////////////////////////
    ///
    /// Memory allocation of variables
    ///
    ////////////////////////////////////////


    allocation_timer.start_timer("init and copy image");

    //initialize the storage of the B-spline co-efficients
    image_temp.initialize(input_image);



    //allocate require memory for the down-sampled structures

    //compute the gradient
    grad_temp.preallocate(input_image.y_num,input_image.x_num,input_image.z_num,0);

    local_scale_temp.preallocate(input_image.y_num,input_image.x_num,input_image.z_num,0);

    local_scale_temp2.preallocate(input_image.y_num,input_image.x_num,input_image.z_num,0);

    allocation_timer.stop_timer();

    /////////////////////////////////
    ///
    /// Pipeline
    ///
    ////////////////////////

    computation_timer.start_timer("Calculations");

    fine_grained_timer.start_timer("offset image");

    //offset image by factor (this is required if there are zero areas in the background with uint16_t and uint8_t images, as the Bspline co-efficients otherwise may be negative!)
    // Warning both of these could result in over-flow
    if(this->image_type == "uint16"){
        //
        block_offset_by_100(input_image,image_temp);

        bspline_offset = 100;
    } else if (this->image_type == "uint8"){
        block_offset_by_5(input_image,image_temp);
        bspline_offset = 5;
    } else {
        image_temp.block_copy_data(input_image);
    }

    fine_grained_timer.stop_timer();


    method_timer.start_timer("compute_gradient_magnitude_using_bsplines");
    MeshData<T> gradient;
    this->get_gradient(input_image,gradient); //note in the current pipeline don't actually use these variables, but here for interface (Use shared member allocated above variables)
    method_timer.stop_timer();

    MeshData<T> local_scale;
    method_timer.start_timer("compute_local_intensity_scale");
    this->get_local_intensity_scale(input_image,local_scale);  //note in the current pipeline don't actually use these variables, but here for interface (Use shared member allocated above variables)
    method_timer.stop_timer();

    method_timer.start_timer("initialize_particle_cell_tree");
    initialize_particle_cell_tree(apr);
    method_timer.stop_timer();

    method_timer.start_timer("compute_local_particle_set");
    this->get_local_particle_cell_set(local_scale,gradient); //note in the current pipeline don't actually use these variables, but here for interface (Use shared member allocated above variables)
    method_timer.stop_timer();

    method_timer.start_timer("compute_pulling_scheme");
    PullingScheme::pulling_scheme_main();
    method_timer.stop_timer();

    method_timer.start_timer("downsample_pyramid");
    std::vector<MeshData<T>> downsampled_img;
    //Down-sample the image for particle intensity estimation
    downsample_pyrmaid(input_image,downsampled_img,apr.level_max()-1,apr.level_min());
    method_timer.stop_timer();

    method_timer.start_timer("compute_apr_datastructure");
    apr.apr_access.initialize_structure_from_particle_cell_tree(apr,particle_cell_tree);
    method_timer.stop_timer();

    method_timer.start_timer("sample_particles");
    apr.get_parts_from_img(downsampled_img,apr.particles_intensities);
    method_timer.stop_timer();

    computation_timer.stop_timer();

    apr.parameters = par;

    return true;
}



template<typename ImageType> template<typename T,typename S>
void APRConverter<ImageType>::get_local_particle_cell_set(MeshData<T>& grad_image_ds,MeshData<S>& local_intensity_scale_ds) {
    //
    //  Computes the Local Particle Cell Set from a down-sampled local intensity scale (\sigma) and gradient magnitude
    //
    //  Down-sampled due to the Equivalence Optimization
    //

    fine_grained_timer.start_timer("compute_level_first");

    int i = 0;
    //divide gradient magnitude by Local Intensity Scale (first step in calculating the Local Resolution Estimate L(y), minus constants)
#ifdef HAVE_OPENMP
	#pragma omp parallel for private(i) default(shared)
#endif
    for(i = 0; i < grad_temp.mesh.size(); i++)
    {
        local_scale_temp.mesh[i] = (1.0*grad_temp.mesh[i])/(local_scale_temp.mesh[i]*1.0);
    }

    fine_grained_timer.stop_timer();

    float level_factor;

    float min_dim = std::min(this->par.dy,std::min(this->par.dx,this->par.dz));

    level_factor = pow(2,(*apr_).level_max())*min_dim;

    unsigned int l_max = (*apr_).level_max() - 1;
    unsigned int l_min = (*apr_).level_min();

    fine_grained_timer.start_timer("compute_level_second");

    //incorporate other factors and compute the level of the Particle Cell, effectively construct LPC L_n
    compute_level_for_array(local_scale_temp,level_factor,this->par.rel_error);

    fill(l_max,local_scale_temp);

    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("level_loop_initialize_tree");

    for(int l_ = l_max - 1; l_ >= l_min; l_--){

        //down sample the resolution level k, using a max reduction
        down_sample(local_scale_temp,local_scale_temp2,
                    [](float x, float y) { return std::max(x,y); },
                    [](float x) { return x; }, true);
        //for those value of level k, add to the hash table
        fill(l_,local_scale_temp2);
        //assign the previous mesh to now be resampled.
        local_scale_temp.swap(local_scale_temp2);
    }

    fine_grained_timer.stop_timer();

}


template<typename ImageType> template<typename T,typename S>
void APRConverter<ImageType>::get_gradient(MeshData<T>& input_img,MeshData<S>& gradient){
    //
    //  Bevan Cheeseman 2018
    //
    //  Calculate the gradient from the input image. (You could replace this method with your own)
    //
    //  Input: full sized image.
    //
    //  Output: down-sampled by 2 gradient magnitude (Note, the gradient is calculated at pixel level then maximum down sampled within the loops below)
    //


    fine_grained_timer.start_timer("smooth_bspline");

    if(par.lambda > 0) {

        get_smooth_bspline_3D(image_temp, this->par);

    }
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("calc_bspline_fd_mag_ds");
    calc_bspline_fd_ds_mag(image_temp,grad_temp,par.dx,par.dy,par.dz);
    fine_grained_timer.stop_timer();


    fine_grained_timer.start_timer("down-sample_b-spline");
    down_sample(image_temp,local_scale_temp,
                [](const T x,const  T y) { return (x*8.0+1.0*y)/8.0; },
                [](const T x) { return x ; });
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
    if(this->par.mask_file != ""){
        mask_gradient(grad_temp,local_scale_temp2,image_temp, this->par);
    }
    fine_grained_timer.stop_timer();

//    std::vector<ImageType>().swap(image_temp.mesh);

    fine_grained_timer.start_timer("threshold");
    threshold_gradient(grad_temp,local_scale_temp,par.Ip_th + bspline_offset);
    fine_grained_timer.stop_timer();

}

template<typename ImageType> template<typename T,typename S>
void APRConverter<ImageType>::get_local_intensity_scale(MeshData<T>& input_img,MeshData<S>& local_intensity_scale){
    //
    //  Calculate the Local Intensity Scale (You could replace this method with your own)
    //
    //  Input: full sized image.
    //
    //  Output: down-sampled Local Intensity Scale (h) (Due to the Equivalence Optimization we only need down-sampled values)
    //
    fine_grained_timer.start_timer("copy_intensities_from_bsplines");
    //copy across the intensities
    std::copy(local_scale_temp.mesh.begin(),local_scale_temp.mesh.end(),local_scale_temp2.mesh.begin());
    fine_grained_timer.stop_timer();

    float var_rescale;
    std::vector<int> var_win;

    get_window(var_rescale,var_win,this->par);

    int win_x,win_y,win_z,win_y2,win_x2,win_z2;

    win_y = var_win[0];
    win_x = var_win[1];
    win_z = var_win[2];

    win_y2 = var_win[3];
    win_x2 = var_win[4];
    win_z2 = var_win[5];

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


    //free up the memory not needed anymore
//    std::vector<float>().swap(local_scale_temp2.mesh);

    //Second spatial average
    calc_sat_mean_y(local_scale_temp,win_y2);
    calc_sat_mean_x(local_scale_temp,win_x2);
    calc_sat_mean_z(local_scale_temp,win_z2);

    rescale_var_and_threshold( local_scale_temp,var_rescale,this->par);

    fine_grained_timer.stop_timer();

}


template<typename ImageType> template<typename T>
void APRConverter<ImageType>::init_apr(APR<ImageType>& apr,MeshData<T>& input_image){
    //
    //  Initializing the size of the APR, min and maximum level (in the data structures it is called depth)
    //
    //

    apr.apr_access.org_dims[0] = input_image.y_num;
    apr.apr_access.org_dims[1] = input_image.x_num;
    apr.apr_access.org_dims[2] = input_image.z_num;

    int max_dim;
    int min_dim;

    if(input_image.z_num == 1) {
        max_dim = (std::max(apr.apr_access.org_dims[1], apr.apr_access.org_dims[0]));
        min_dim = (std::min(apr.apr_access.org_dims[1], apr.apr_access.org_dims[0]));
    }
    else{
        max_dim = std::max(std::max(apr.apr_access.org_dims[1], apr.apr_access.org_dims[0]), apr.apr_access.org_dims[2]);
        min_dim = std::min(std::min(apr.apr_access.org_dims[1], apr.apr_access.org_dims[0]), apr.apr_access.org_dims[2]);
    }

    int k_max_ = ceil(M_LOG2E*log(max_dim)) - 1;
    int k_min_ = std::max( (int)(k_max_ - floor(M_LOG2E*log(min_dim)) + 1),2);

    apr.apr_access.level_min = k_min_;
    apr.apr_access.level_max = k_max_ + 1;

}




template<typename ImageType> template<typename T>
void APRConverter<ImageType>::auto_parameters(const MeshData<T>& input_img){
    //
    //  Simple automatic parameter selection for 3D APR Flouresence Images
    //

    APRTimer par_timer;
    par_timer.verbose_flag = true;

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
    uint64_t q =0;

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

//    for (q = 0; q < input_img.mesh.size(); ++q) {
//
//        if(input_img.mesh[q] < (min_val + num_bins-1)){
//            freq[input_img.mesh[q]-min_val]++;
//            if(input_img.mesh[q] > 0) {
//                counter++;
//                total += input_img.mesh[q];
//            }
//        }
//    }

    std::cout << "img_mean: " << total << " counter: " << counter << std::endl;
    float img_mean = counter > 0 ? total/(counter*1.0) : 1;
    float prop_total_th = 0.05; //assume there is atleast 5% background in the image
    float prop_total = 0;
    uint64_t min_j = 0;

    // set to start at one to ignore potential constant regions thresholded out. (Common in some images)
    for (int j = 1; j < num_bins; ++j) {
        prop_total += freq[j]/(counter*1.0);

        if(prop_total > prop_total_th){
            min_j = j;
            break;
        }

    }


    MeshData<T> histogram;
    histogram.initialize(num_bins,1,1);

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

    int stop = 1;

    std::vector<std::vector<T>> patches;

    patches.resize(std::min(local_max,(uint64_t)10000));

    for (int l = 0; l < patches.size(); ++l) {
        patches[l].resize(27,0);
    }


    int64_t z_num = input_img.z_num;
    int64_t x_num = input_img.x_num;
    int64_t y_num = input_img.y_num;

    int64_t j = 0;
    int64_t k = 0;
    int64_t i = 0;

    int j_n = 0;
    int k_n = 0;
    int i_n = 0;

    par_timer.start_timer("get_patches");

    uint64_t counter_p = 0;

    for (size_t s = 0; s < selectedSlicesOffsets.size(); ++s) {
        j = std::min((int)z_num - 2, std::max((int)selectedSlicesOffsets[s],(int)1));
        for(i = 1; i < (x_num-1);i++){
            for(k = 1;k < (y_num-1);k++){
                float val = input_img.mesh[j*x_num*y_num + i*y_num + k];
                if (val == estimated_first_mode) {
                    uint64_t counter_n = 0;
                    for (int64_t l = -1; l < 2; ++l) {
                        for (int64_t m = -1; m < 2; ++m) {
                            for (int64_t n = -1; n < 2; ++n) {
                                size_t idx = (size_t)(j+l)*x_num*y_num + (i+m)*y_num + (k+n);
                                const auto &val = input_img.mesh[idx];
                                patches[counter_p][counter_n] = val;
                                counter_n++;
                            }
                        }
                    }
                    counter_p++;
                }
                if (counter_p > (patches.size()-1)){
                    goto finish;
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

    T mean = counter > 0 ? total_p/(counter*1.0) : 1;

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

    if(this->par.SNR_min > 0){
        min_snr = this->par.SNR_min;
    } else {
        std::cout << "**Assuming a minimum SNR of 6" << std::endl;
    }

    std::cout << "**Assuming image has atleast 5% dark background" << std::endl;

    float Ip_th = mean + sd;
    std::cout << "mean: " << mean << " counter: " << counter << std::endl;
    float var_th = (img_mean/mean)*sd*min_snr;

    float var_th_max = sd*min_snr*.5;

    if(this->par.Ip_th < 0 ){
        this->par.Ip_th = Ip_th;
    }

    if(this->par.lambda < 0){
        this->par.lambda = 3.0;
    }

    this->par.background_intensity_estimate = estimated_first_mode;

    if(this->par.min_signal < 0) {
        this->par.sigma_th = var_th;
        this->par.sigma_th_max = var_th_max;
        std::cout << "1: " << var_th << std::endl;
    } else if (this->par.sigma_th > 0){
        //keep the defaults

        std::cout << "defaults!" << std::endl;
    } else{
        this->par.sigma_th_max = this->par.min_signal*0.5;
        this->par.sigma_th = this->par.min_signal;
        std::cout << "3: " << var_th << std::endl;
    }

    std::cout << "I_th: " << this->par.Ip_th << std::endl;
    std::cout << "sigma_th: " << this->par.sigma_th << std::endl;
    std::cout << "sigma_th_max: " << this->par.sigma_th_max << std::endl;
    std::cout << "relative error (E): " << this->par.rel_error << std::endl;
    std::cout << "lambda: " << this->par.lambda << std::endl;
}


#endif //PARTPLAY_APR_CONVERTER_HPP
