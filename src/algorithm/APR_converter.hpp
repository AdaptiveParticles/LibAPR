////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// APR Converter class handles the methods and functions for creating an APR from an input image
///
////////////////////////////////

#ifndef PARTPLAY_APR_CONVERTER_HPP
#define PARTPLAY_APR_CONVERTER_HPP

#include "src/data_structures/Mesh/meshclass.h"
#include "src/data_structures/APR/APR.hpp"

#include "src/algorithm/ComputeGradient.hpp"
#include "src/algorithm/LocalIntensityScale.hpp"
#include "src/algorithm/LocalParticleCellSet.hpp"
#include "src/algorithm/PullingScheme.hpp"


template<typename ImageType>
class APR_converter: public LocalIntensityScale, public ComputeGradient, public LocalParticleCellSet, public PullingScheme {

public:

    APR_converter():image_type("uint16"){

    }

    APR_parameters par;

    APR_timer total_timer;
    APR_timer allocation_timer;

    APR_timer computation_timer;

    APR_timer misc_timer;

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


        if(image_type == "uint8"){
            return get_apr_method<uint8_t>(apr);
        } else if (image_type == "float"){
            return get_apr_method<float>(apr);

        } else {
            return get_apr_method<uint16_t>(apr);
        }



    };

private:

    /*
     * Private member variables
     */

    //pointer to the APR structure so member functions can have access if they need
    APR<ImageType>* apr_;

    Mesh_data<ImageType> image_temp; // global image variable useful for passing between methods, or re-using memory (should be the only full sized copy of the image)

    Mesh_data<ImageType> grad_temp; // should be a down-sampled image

    Mesh_data<float> local_scale_temp; //   Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors

    Mesh_data<float> local_scale_temp2;  //   Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors

    //assuming uint16, the total memory cost shoudl be approximately (1 + 1 + 1/8 + 2/8 + 2/8) = 2 5/8 original image size in u16bit

    //storage of the particle cell tree for computing the pulling scheme


    float bspline_offset=0;

    /*
     * Private member functions
     */

    template<typename T>
    void init_apr(APR<ImageType>& apr,Mesh_data<T>& input_image);

    template<typename T>
    void auto_parameters(Mesh_data<T>& input_img);

    template<typename T>
    bool get_apr_method(APR<ImageType>& apr);

    template<typename T,typename S>
    void get_gradient(Mesh_data<T>& input_img,Mesh_data<S>& gradient);

    template<typename T,typename S>
    void get_local_intensity_scale(Mesh_data<T>& input_img,Mesh_data<S>& local_intensity_scale);

    template<typename T,typename S>
    void get_local_particle_cell_set(Mesh_data<T>& grad_image_ds,Mesh_data<S>& local_intensity_scale_ds);

};

/*
 * Implimentations
 */
template<typename ImageType> template<typename T>
bool APR_converter<ImageType>::get_apr_method(APR<ImageType>& apr) {
    //
    //  Main method for constructing the APR from an input image
    //
    
    APR_timer full;
    full.verbose_flag = true;

    full.start_timer("GET APR");

    APR_timer timer;
    timer.verbose_flag = true;

    timer.start_timer("read tif input image");

    //input type
    Mesh_data<T> input_image;

    input_image.load_image_tiff(par.input_dir + par.input_image_name);

    timer.stop_timer();

    //    was there an image found
    if(input_image.mesh.size() == 0){
        std::cout << "Image Not Found" << std::endl;
        return false;
    }

    init_apr(apr,input_image);

    timer.start_timer("calculate automatic parameters");
    auto_parameters(input_image);
    timer.stop_timer();


    timer.start_timer("init and copy image");

    //initialize the storage of the B-spline co-efficients
    image_temp.initialize(input_image);

    std::copy(input_image.mesh.begin(),input_image.mesh.end(),image_temp.mesh.begin());

    //allocate require memory for the down-sampled structures

    ////////////////////////////////////////
    ///
    /// Memory allocation of variables
    ///
    ////////////////////////////////////////

    //compute the gradient
    grad_temp.preallocate(input_image.y_num,input_image.x_num,input_image.z_num,0);

    local_scale_temp.preallocate(input_image.y_num,input_image.x_num,input_image.z_num,0);

    local_scale_temp2.preallocate(input_image.y_num,input_image.x_num,input_image.z_num,0);

    timer.stop_timer();

    computation_timer.start_timer("Calculations");

    computation_timer.verbose_flag = true;

    APR_timer st;
    st.verbose_flag = true;

    st.start_timer("grad");

    Mesh_data<T> gradient;

    this->get_gradient(input_image,gradient); //note in the current pipeline don't actually use these variables, but here for interface (Use shared member allocated above variables)

    st.stop_timer();

    Mesh_data<T> local_scale;

    this->get_local_intensity_scale(input_image,local_scale);  //note in the current pipeline don't actually use these variables, but here for interface (Use shared member allocated above variables)

    st.start_timer("init Particle Cell Image Pyramid structure");
    initialize_particle_cell_tree(apr);
    st.stop_timer();

    st.start_timer("Compute LPC");
    this->get_local_particle_cell_set(local_scale,gradient); //note in the current pipeline don't actually use these variables, but here for interface (Use shared member allocated above variables)
    st.stop_timer();


    st.start_timer("Pulling Scheme: Compute OVPC V_n from LPC");
    PullingScheme::pulling_scheme_main();
    st.stop_timer();


    st.start_timer("Down sample image");
    std::vector<Mesh_data<T>> downsampled_img;
    //Down-sample the image for particle intensity estimation
    downsample_pyrmaid(input_image,downsampled_img,apr.depth_max()-1,apr.depth_min());
    st.stop_timer();


    st.start_timer("Init data structure");
    apr.init_from_pulling_scheme(particle_cell_tree);
    st.stop_timer();


    st.start_timer("sample particles");
    apr.get_parts_from_img(downsampled_img,apr.particles_int);
    st.stop_timer();

    full.stop_timer();

    computation_timer.stop_timer();

    return true;
}



template<typename ImageType> template<typename T,typename S>
void APR_converter<ImageType>::get_local_particle_cell_set(Mesh_data<T>& grad_image_ds,Mesh_data<S>& local_intensity_scale_ds) {
    //
    //  Computes the Local Particle Cell Set from a down-sampled local intensity scale (\sigma) and gradient magnitude
    //
    //  Down-sampled due to the Equivalence Optimization
    //

    APR_timer timer;

    //divide gradient magnitude by Local Intensity Scale (first step in calculating the Local Resolution Estimate L(y), minus constants)
#pragma omp parallel for default(shared)
    for(int i = 0; i < grad_temp.mesh.size(); i++)
    {
        local_scale_temp.mesh[i] = (1.0*grad_temp.mesh[i])/(local_scale_temp.mesh[i]*1.0);
    }

    float level_factor;

    float min_dim = std::min(this->par.dy,std::min(this->par.dx,this->par.dz));

    level_factor = pow(2,(*apr_).depth_max())*min_dim;

    unsigned int l_max = (*apr_).depth_max() - 1;
    unsigned int l_min = (*apr_).depth_min();

    //incorporate other factors and compute the level of the Particle Cell, effectively construct LPC L_n
    compute_level_for_array(local_scale_temp,level_factor,this->par.rel_error);

    fill(l_max,local_scale_temp);

    timer.start_timer("level_loop");

    for(int l_ = l_max - 1; l_ >= l_min; l_--){

        //down sample the resolution level k, using a max reduction
        down_sample(local_scale_temp,local_scale_temp2,
                    [](float x, float y) { return std::max(x,y); },
                    [](float x) { return x; }, true);
        //for those value of level k, add to the hash table
        fill(l_,local_scale_temp2);
        //assign the previous mesh to now be resampled.
        std::swap(local_scale_temp, local_scale_temp2);

    }

    timer.stop_timer();

}


template<typename ImageType> template<typename T,typename S>
void APR_converter<ImageType>::get_gradient(Mesh_data<T>& input_img,Mesh_data<S>& gradient){
    //
    //  Bevan Cheeseman 2018
    //
    //  Calculate the gradient from the input image. (You could replace this method with your own)
    //
    //  Input: full sized image.
    //
    //  Output: down-sampled by 2 gradient magnitude (Note, the gradient is calculated at pixel level then maximum down sampled within the loops below)
    //

    APR_timer timer;

    timer.verbose_flag = false;


    timer.start_timer("offset image");

    //offset image by factor (this is required if there are zero areas in the background with uint16_t and uint8_t images, as the Bspline co-efficients otherwise may be negative!)
    // Warning both of these could result in over-flow
    if(this->image_type == "uint16"){
        //
        std::transform(image_temp.mesh.begin(),image_temp.mesh.end(),image_temp.mesh.begin(),[](const float &a) { return a + 100; });
        bspline_offset = 100;
    } else if (this->image_type == "uint8"){
        std::transform(image_temp.mesh.begin(),image_temp.mesh.end(),image_temp.mesh.begin(),[](const float &a) { return a + 5; });
        bspline_offset = 5;
    } else {
        bspline_offset = 0;
    }

    timer.stop_timer();

    timer.start_timer("smooth bspline");

    if(par.lambda > 0) {

        get_smooth_bspline_3D(image_temp, this->par);

    }

    timer.stop_timer();


    timer.start_timer("calc_bspline_fd_x_y_ds");
    calc_bspline_fd_ds_mag(image_temp,grad_temp,par.dx,par.dy,par.dz);
    timer.stop_timer();



    timer.start_timer("down-sample b-spline");
    down_sample(image_temp,local_scale_temp,
                [](T x, T y) { return (x*8.0+1.0*y)/8.0; },
                [](T x) { return x ; });
    timer.stop_timer();


    timer.start_timer("compute smoothed function for local intenisty scale");
    if(par.lambda > 0){
        timer.start_timer("calc_inv_bspline_y");
        calc_inv_bspline_y(local_scale_temp);
        timer.stop_timer();
        timer.start_timer("calc_inv_bspline_x");
        calc_inv_bspline_x(local_scale_temp);
        timer.stop_timer();
        timer.start_timer("calc_inv_bspline_z");
        calc_inv_bspline_z(local_scale_temp);
        timer.stop_timer();
    }
    timer.stop_timer();



    timer.start_timer("load and apply mask");
    // Apply mask if given
    if(this->par.mask_file != ""){
        mask_gradient(grad_temp,local_scale_temp2,image_temp, this->par);
    }
    timer.stop_timer();

    std::vector<ImageType>().swap(image_temp.mesh);

    timer.start_timer("Threshold ");
    threshold_gradient(grad_temp,local_scale_temp,par.Ip_th + bspline_offset);
    timer.stop_timer();

}

template<typename ImageType> template<typename T,typename S>
void APR_converter<ImageType>::get_local_intensity_scale(Mesh_data<T>& input_img,Mesh_data<S>& local_intensity_scale){
    //
    //  Calculate the Local Intensity Scale (You could replace this method with your own)
    //
    //  Input: full sized image.
    //
    //  Output: down-sampled Local Intensity Scale (h) (Due to the Equivalence Optimization we only need down-sampled values)
    //

    APR_timer timer;

    APR_timer var_timer;
    var_timer.verbose_flag = true;

    var_timer.start_timer("compute local intensity scale");


    //copy across the intensities
    std::copy(local_scale_temp.mesh.begin(),local_scale_temp.mesh.end(),local_scale_temp2.mesh.begin());

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

    timer.start_timer("calc_sat_mean_y");

    calc_sat_mean_y(local_scale_temp,win_y);

    timer.stop_timer();

    timer.start_timer("calc_sat_mean_x");

    calc_sat_mean_x(local_scale_temp,win_x);

    timer.stop_timer();

    timer.start_timer("calc_sat_mean_z");

    calc_sat_mean_z(local_scale_temp,win_z);

    timer.stop_timer();


    timer.start_timer("second pass and rescale");

    //calculate abs and subtract from original
    calc_abs_diff(local_scale_temp2,local_scale_temp);


    //free up the memory not needed anymore
    std::vector<float>().swap(local_scale_temp2.mesh);

    //Second spatial average
    calc_sat_mean_y(local_scale_temp,win_y2);
    calc_sat_mean_x(local_scale_temp,win_x2);
    calc_sat_mean_z(local_scale_temp,win_z2);


    rescale_var_and_threshold( local_scale_temp,var_rescale,this->par);

    timer.stop_timer();

    var_timer.stop_timer();




}


template<typename ImageType> template<typename T>
void APR_converter<ImageType>::init_apr(APR<ImageType>& apr,Mesh_data<T>& input_image){
    //
    //  Initializing the size of the APR, min and maximum level (in the data structures it is called depth)
    //
    //

    apr.pc_data.org_dims.resize(3,0);

    apr.pc_data.org_dims[0] = input_image.y_num;
    apr.pc_data.org_dims[1] = input_image.x_num;
    apr.pc_data.org_dims[2] = input_image.z_num;

    int max_dim;
    int min_dim;

    if(input_image.z_num == 1) {
        max_dim = (std::max(apr.pc_data.org_dims[1], apr.pc_data.org_dims[0]));
        min_dim = (std::min(apr.pc_data.org_dims[1], apr.pc_data.org_dims[0]));
    }
    else{
        max_dim = std::max(std::max(apr.pc_data.org_dims[1], apr.pc_data.org_dims[0]), apr.pc_data.org_dims[2]);
        min_dim = std::min(std::min(apr.pc_data.org_dims[1], apr.pc_data.org_dims[0]), apr.pc_data.org_dims[2]);
    }

    int k_max_ = ceil(M_LOG2E*log(max_dim)) - 1;
    int k_min_ = std::max( (int)(k_max_ - floor(M_LOG2E*log(min_dim)) + 1),2);

    apr.pc_data.depth_min = k_min_;
    apr.pc_data.depth_max = k_max_ + 1;

}




template<typename ImageType> template<typename T>
void APR_converter<ImageType>::auto_parameters(Mesh_data<T>& input_img){
    //
    //  Simple automatic parameter selection for 3D APR Flouresence Images
    //



    APR_timer par_timer;

    par_timer.verbose_flag = true;

    //
    //  Do not compute the statistics over the whole image, but only a smaller sub-set.
    //

    double total_required_pixel = 10*1000*1000;

    std::vector<unsigned int> selected_slices;

    unsigned int num_slices = std::min((unsigned int)ceil(total_required_pixel/(1.0*input_img.y_num*input_img.x_num)),(unsigned int)input_img.z_num);

    unsigned int delta = std::max((unsigned int)1,(unsigned int)(input_img.z_num/num_slices));

    //evenly space the slices across the image
    for (int i1 = 0; i1 < num_slices; ++i1) {
        selected_slices.push_back(delta*i1);
    }

    float min_val = 99999999;

    par_timer.start_timer("get_min");

    for (int k1 = 0; k1 < selected_slices.size(); ++k1) {
        min_val = std::min((float)*std::min_element(input_img.mesh.begin() + selected_slices[k1]*(input_img.y_num*input_img.x_num),input_img.mesh.begin()  + (selected_slices[k1]+1)*(input_img.y_num*input_img.x_num)),min_val);
    }

    par_timer.stop_timer();


    //minimum element
    //T min_val = *std::min_element(input_img.mesh.begin(),input_img.mesh.end());



    // will need to deal with grouped constant or zero sections in the image somewhere.... but lets keep it simple for now.

    std::vector<uint64_t> freq;
    unsigned int num_bins = 10000;
    freq.resize(num_bins);

    uint64_t counter = 0;
    double total=0;

    uint64_t q =0;
//#pragma omp parallel for default(shared) private(q)

    unsigned int xnumynum = input_img.x_num*input_img.y_num;

    par_timer.start_timer("get_histogram");

    for (int s = 0; s < selected_slices.size(); ++s) {

        for (int q= selected_slices[s]*xnumynum; q < (selected_slices[s]+1)*xnumynum; ++q) {
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

    float img_mean = total/(counter*1.0);

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


    Mesh_data<T> histogram;
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

    for (int k = min_j; k < num_bins; ++k) {

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


    unsigned int z_num = input_img.z_num;
    unsigned int x_num = input_img.x_num;
    unsigned int y_num = input_img.y_num;

    int j = 0;
    int k = 0;
    int i = 0;

    int j_n = 0;
    int k_n = 0;
    int i_n = 0;

    par_timer.start_timer("get_patches");

    uint64_t counter_p = 0;

    for (int s = 0; s < selected_slices.size(); ++s) {
        j = std::max((int)selected_slices[s],(int)1);
        for(i = 1; i < (x_num-1);i++){
            for(k = 1;k < (y_num-1);k++){

                float val = input_img.mesh[j*x_num*y_num + i*y_num + k];

                if(val == estimated_first_mode) {

                    uint64_t counter_n = 0;

                    for (int l = -1; l < 2; ++l) {
                        for (int m = -1; m < 2; ++m) {
                            for (int n = -1; n < 2; ++n) {
                                patches[counter_p][counter_n] = input_img.mesh[(j+l)*x_num*y_num + (i+m)*y_num + (k+n)];
                                counter_n++;
                            }
                        }
                    }

                    counter_p++;

                }
                if(counter_p > (patches.size()-1)){
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

    for (int i = 0; i < patches.size(); ++i) {
        for (int j = 0; j < patches[i].size(); ++j) {

            if(patches[i][j] > 0){
                total_p += patches[i][j];
                counter++;
            }
        }
    }

    T mean = total_p/(counter*1.0);

    //now compute the standard deviation (sd) of the patches

    double var=0;

    for (int i = 0; i < patches.size(); ++i) {
        for (int j = 0; j < patches[i].size(); ++j) {

            if(patches[i][j] > 0){
                var += pow(patches[i][j]-mean,2);
            }
        }
    }

    var = var/(counter*1);

    float sd = sqrt(var);

    float min_snr = 6;

    if(this->par.SNR_min > 0){
        min_snr = this->par.SNR_min;
    } else {
        std::cout << "**Assuming a minimum SNR of 6" << std::endl;
    }

    std::cout << "**Assuming image has atleast 5% dark background" << std::endl;

    float Ip_th = mean + sd;

    float var_th = (img_mean/mean)*sd*min_snr;

    float var_th_max = sd*min_snr*.5;

    if(this->par.Ip_th < 0 ){
        this->par.Ip_th = Ip_th;
    }

    if(this->par.lambda < 0){
        this->par.lambda = 3.0;
    }

    if(this->par.min_signal < 0){
        this->par.sigma_th = var_th;
        this->par.sigma_th_max = var_th_max;
    } else{
        this->par.sigma_th_max = this->par.min_signal*0.5;
        this->par.sigma_th = this->par.min_signal;
    }

    std::cout << "I_th: " << this->par.Ip_th << std::endl;
    std::cout << "sigma_th: " << this->par.sigma_th << std::endl;
    std::cout << "sigma_th_max: " << this->par.sigma_th_max << std::endl;
    std::cout << "relative error (E): " << this->par.rel_error << std::endl;
    std::cout << "Lambda: " << this->par.lambda << std::endl;

}


#endif //PARTPLAY_APR_CONVERTER_HPP
