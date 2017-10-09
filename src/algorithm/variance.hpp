#ifndef PARTPLAY_VARIANCE_H
#define PARTPLAY_VARIANCE_H

#include <algorithm>
#include <functional>

#include "../data_structures/meshclass.h"
#include "../data_structures/structure_parts.h"

class Map_calc_cpu{
    //
    //
    //  This is a class containing different processing methods used to estimate the part map on the GPU
    //
    //

public:
    unsigned int dir;  //direction of the filter


    //image parameters
    std::vector<float> real_sigmas;
    std::vector<float> real_window_size;
    float cut_th;

    float var_rescale;
    float window_ref;
    float var_scale;

    float max_filter_size;

    std::vector<float> sampling_delta;


    unsigned int max_filter_len1 ;
    unsigned int max_filter_len2 ;
    unsigned int max_filter_len3 ;

    unsigned int var_window_size;

    unsigned int var_window_size1;
    unsigned int var_window_size2;
    unsigned int var_window_size3;

    std::vector<int> var_win;


    Proc_par par;

    Map_calc_cpu(unsigned int dir_,Proc_par par_){
        dir = dir_;
        par = par_;
        max_filter_len2 = 0;
        max_filter_len1 = 0;
        max_filter_len3 = 0;


        max_filter_size = round(par_.xdim/(par_.dx*2.0)-1);

        cut_th = 0.01;

        var_scale = par_.var_scale;
        var_rescale = 10;

        real_sigmas.resize(3);
        real_sigmas[0] = par_.psfy;
        real_sigmas[1] = par_.psfx;
        real_sigmas[2] = par_.psfz;

        sampling_delta.resize(3);

        sampling_delta[0] = par_.dy;
        sampling_delta[1] = par_.dx;
        sampling_delta[2] = par_.dz;

        real_window_size.resize(3);
        //calc the parameters
        for (int i = 0;  i < real_window_size.size(); i++) {
            real_window_size[i] = sqrt(log(1/cut_th)*2*pow(real_sigmas[i],2));
        }


    }


    void set_up_var_filters_3D(){


        var_window_size1 = std::min((float)(1.0+2*round(var_scale*real_window_size[0]/sampling_delta[0])),max_filter_size);

        var_window_size1 = std::max((float)3.0,(float)var_window_size1);

        var_window_size2 = std::min((float)(1.0+2*round(var_scale*real_window_size[1]/sampling_delta[1])),max_filter_size);

        var_window_size2 = std::max((float)3.0,(float)var_window_size2);

        var_window_size3 = std::min((float)(1.0+2*round(var_scale*real_window_size[2]/sampling_delta[2])),max_filter_size);

        var_window_size3 = std::max((float)3.0,(float)var_window_size3);


        max_filter_len1 = std::max(max_filter_len1,(unsigned int)((var_window_size1-1)/2.0));
        max_filter_len2 = std::max(max_filter_len2,(unsigned int)((var_window_size2-1)/2.0));
        max_filter_len3 = std::max(max_filter_len3,(unsigned int)((var_window_size3-1)/2.0));


        window_ref = std::max((float)3.0,(float)(1+2*round(real_window_size[0]/sampling_delta[0])));


        float window_rel1 = 1.0*var_window_size1/(std::max((float)3.0,(float)(1+2*round(real_window_size[0]/sampling_delta[0]))));

        float window_rel2 = 1.0*var_window_size2/(std::max((float)3.0,(float)(1+2*round(real_window_size[1]/sampling_delta[1]))));

        float window_rel3 = 1.0*var_window_size3/(std::max((float)3.0,(float)(1+2*round(real_window_size[2]/sampling_delta[2]))));


        float rescale_par = pow(window_rel1*window_rel2*window_rel3,1.0/3.0);

        window_ref = rescale_par;


        float sig = pow(real_sigmas[0]*real_sigmas[1]*real_sigmas[2],1.0/3.0);


        var_rescale = 1.0/(0.02201*pow(rescale_par,3.0) - 0.146*pow(rescale_par,2.0) + 0.3521*rescale_par - 0.09969);

        if(sig < 0.73){

            var_rescale = 1.0/(0.79*pow(sig,3.0) - 1.7*pow(sig,2.0) + .87*pow(sig,1.0)-.012);


        } else {
            var_rescale = 1.0/(.015 + .046*sig);
        }


        std::vector<std::vector<std::vector<float>>> rescale_store  ={{{{7.1893},{9.4188},{8.4697}},{{8.7616},{7.6561},{9.4535}},{{8.2584},{10.3079},{8.1669}},{{8.7885},{9.0379},{9.2111}},{{8.8016},{8.5888},{9.3444}},{{9.0287},{9.1967},{8.396}},{{9.5409},{8.4426},{8.2812}},{{8.4263},{8.0129},{7.383}}},{{{4.5055},{4.3203},{5.3158}},{{4.635},{3.9027},{4.8302}},{{4.2916},{4.3896},{4.2315}},{{4.6119},{4.22},{4.9655}},{{4.7524},{4.557},{5.1137}},{{4.5965},{5.1327},{4.3396}},{{3.5912},{4.6375},{4.9569}},{{4.3082},{5.5035},{4.1922}}},{{{4.8563},{5.1238},{5.0085}},{{4.5107},{4.2069},{4.2616}},{{3.8026},{4.0777},{4.0023}},{{3.8633},{4.7123},{4.0766}},{{3.3432},{4.6071},{4.1871}},{{4.591},{4.0171},{4.2956}},{{4.0567},{4.0784},{4.3953}},{{4.3465},{4.3179},{4.1182}}},{{{4.8079},{4.901},{4.7586}},{{5.5722},{4.1117},{4.8205}},{{5.4289},{3.8836},{4.802}},{{3.6394},{4.5734},{4.3615}},{{4.6738},{4.6656},{5.162}},{{3.9805},{4.5518},{4.9345}},{{4.1835},{5.4492},{4.9338}},{{4.2355},{3.8741},{4.1255}}},{{{4.5989},{5.5532},{5.0333}},{{4.1989},{4.4257},{4.8997}},{{4.4176},{3.8287},{3.899}},{{4.935},{4.5538},{3.8128}},{{5.3005},{4.493},{5.0622}},{{3.7672},{4.7798},{4.2321}},{{3.5788},{5.8633},{4.3694}},{{4.7605},{4.4981},{5.2156}}},{{{4.9932},{5.5232},{4.7861}},{{4.5736},{4.484},{4.4487}},{{5.3733},{4.7776},{4.7036}},{{4.9047},{5.2998},{4.0408}},{{5.4856},{5.0934},{4.7911}},{{4.3178},{4.6986},{5.4765}},{{4.5732},{5.9212},{4.8737}},{{4.896},{4.9144},{4.3474}}}};

        var_rescale = par.var_scale;

        std::vector<int> windows_1 = {1,2,3};
        std::vector<int> windows_2 = {1,2,3,4,5,6,7,8};

        int psf_ind = std::max(((float) (round(par.psfx/par.dx) - 1)),((float)0.0f));

        psf_ind = std::min(psf_ind,5);


        std::vector<int> win_1 = {1,1,1,1,1,1};
        std::vector<int> win_2 = {2,5,3,4,5,6};

        var_win.resize(6);

//        if(par.dx == par.dz) {
//
//            var_win[0] = win_1[psf_ind];
//            var_win[1] = win_1[psf_ind];
//            var_win[2] = win_1[psf_ind];
//            var_win[3] = win_2[psf_ind];
//            var_win[4] = win_2[psf_ind];
//            var_win[5] = win_2[psf_ind];
//
//            int window_ind_1 = 0;
//            int window_ind_2 = 0;
//
//            int curr_dist_1 = 99;
//            int curr_dist_2 = 99;
//
//            for (int i = 0; i < windows.size(); ++i) {
//                if (abs(windows[i] - var_win[0]) < curr_dist_1) {
//                    window_ind_1 = i;
//                    curr_dist_1 = abs(windows[i] - var_win[0]);
//                }
//
//                if (abs(windows[i] - var_win[3]) < curr_dist_2) {
//                    window_ind_2 = i;
//                    curr_dist_2 = abs(windows[i] - var_win[3]);
//                }
//
//            }
//
//            //var_rescale = rescale_store[psf_ind][window_ind_2][window_ind_1];
//
//        } else {
//
//
//            int psf_indz = std::max(((float) (round(par.psfz/par.dz) - 1)),((float)0.0f));
//
//            psf_indz = std::min(psf_indz,5);
//
//           // psf_indz = psf_ind;
//
//            var_win[0] = win_1[psf_ind];
//            var_win[1] = win_1[psf_ind];
//            var_win[2] = win_1[psf_indz];
//            var_win[3] = win_2[psf_ind];
//            var_win[4] = win_2[psf_ind];
//            var_win[5] = win_2[psf_indz];
//
//            int window_ind_1 = 0;
//            int window_ind_2 = 0;
//
//            int window_ind_1_z = 0;
//            int window_ind_2_z = 0;
//
//            int curr_dist_1 = 99;
//            int curr_dist_2 = 99;
//
//            for (int i = 0; i < windows.size(); ++i) {
//                if (abs(windows[i] - var_win[0]) < curr_dist_1) {
//                    window_ind_1 = i;
//                    curr_dist_1 = abs(windows[i] - var_win[0]);
//                }
//
//                if (abs(windows[i] - var_win[3]) < curr_dist_2) {
//                    window_ind_2 = i;
//                    curr_dist_2 = abs(windows[i] - var_win[3]);
//                }
//
//
//            }
//
//           // var_rescale = rescale_store[psf_ind][window_ind_2][window_ind_1];
//
//        }


        var_win = par.padd_dims;


        var_rescale = 1;

    }

    void set_up_var_filters_2D(){


        var_window_size1 = std::min((float)(1.0+2*round(var_scale*real_window_size[0]/sampling_delta[0])),max_filter_size);

        var_window_size1 = std::max((float)3.0,(float)var_window_size1);

        var_window_size2 = std::min((float)(1.0+2*round(var_scale*real_window_size[1]/sampling_delta[1])),max_filter_size);

        var_window_size2 = std::max((float)3.0,(float)var_window_size2);

        var_window_size3 = std::min((float)(1.0+2*round(var_scale*real_window_size[2]/sampling_delta[2])),max_filter_size);

        var_window_size3 = std::max((float)3.0,(float)var_window_size3);

        var_window_size3 = 1;



        max_filter_len1 = std::max(max_filter_len1,(unsigned int)((var_window_size1-1)/2.0));
        max_filter_len2 = std::max(max_filter_len2,(unsigned int)((var_window_size2-1)/2.0));
        max_filter_len3 = std::max(max_filter_len3,(unsigned int)((var_window_size3-1)/2.0));


        window_ref =std::max((float)3.0,(float)(1+2*round(real_window_size[0]/sampling_delta[0])));


        float window_rel1 = 1.0*var_window_size1/(std::max((float)3.0,(float)(1+2*round(real_window_size[0]/sampling_delta[0]))));

        float window_rel2 = 1.0*var_window_size2/(std::max((float)3.0,(float)(1+2*round(real_window_size[1]/sampling_delta[1]))));

        float window_rel3 = 1.0*var_window_size3/(std::max((float)3.0,(float)(1+2*round(real_window_size[2]/sampling_delta[2]))));
        window_rel3 = 1;


        float rescale_par = pow(window_rel1*window_rel2*window_rel3,1.0/3.0);

        window_ref = rescale_par;

        var_rescale = 1.0/(0.02201*pow(rescale_par,3.0) - 0.146*pow(rescale_par,2.0) + 0.3521*rescale_par - 0.09969);



        //var_rescale = 6.9541;
        //var_rescale = 7.1748;

    }




};

template<typename T>
void calc_sat_mean_y(Mesh_data<T>& input,const int offset){
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculates a O(1) recursive mean using SAT.
    //


    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    std::vector<T> temp_vec;
    temp_vec.resize(y_num,0);


    const int offset_n = offset;
    int i, k, index;
    float counter, temp, divisor = 2*offset_n + 1;

#pragma omp parallel for default(shared) private(i,k,counter,temp,index) firstprivate(temp_vec)
    for(int j = 0;j < z_num;j++){
        for(i = 0;i < x_num;i++){

            index = j*x_num*y_num + i*y_num;

            //first pass over and calculate cumsum
            temp = 0;
            for (k = 0; k < y_num;k++){
                temp += input.mesh[index + k];
                temp_vec[k] = temp;
            }

            input.mesh[index] = 0;
            //handling boundary conditions (LHS)
            for (k = 1; k <= (offset+1);k++){
                input.mesh[index + k] = -temp_vec[0]/divisor;
            }

            //second pass calculate mean
            for (k = offset + 1; k < y_num;k++){
                input.mesh[index + k] = -temp_vec[k - offset - 1]/divisor;
            }


            //second pass calculate mean
            for (k = 0; k < (y_num-offset);k++){
                input.mesh[index + k] += temp_vec[k + offset]/divisor;
            }


            counter = 0;
            //handling boundary conditions (RHS)
            for (k = ( y_num - offset); k < (y_num);k++){
                counter++;
                input.mesh[index + k]*= divisor;
                input.mesh[index + k]+= temp_vec[y_num-1];
                input.mesh[index + k]*= 1.0/(divisor - counter);
            }

            //handling boundary conditions (LHS), need to rehandle the boundary
            for (k = 1; k < (offset + 1);k++){
                input.mesh[index + k] *= divisor/(1.0*k + offset_n);
            }
            //end point boundary condition
            input.mesh[index] *= divisor/(offset_n+1);
        }
    }



}

template<typename T>
void calc_sat_mean_x(Mesh_data<T>& input,const int offset){
    // The same, but in place

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    std::vector<T> temp_vec;
    temp_vec.resize(y_num*(2*offset + 1),0);

    int i,k;
    float temp;
    int index_modulo, previous_modulo, current_index, jxnumynum;

#pragma omp parallel for default(shared) private(i,k,temp,index_modulo, previous_modulo, current_index, jxnumynum) \
        firstprivate(temp_vec)
    for(int j = 0; j < z_num; j++) {

        jxnumynum = j * x_num * y_num;

        //prefetching

        for(k = 0; k < y_num ; k++){
            // std::copy ?
            temp_vec[k] = input.mesh[jxnumynum + k];
        }

        for(i = 1; i < 2 * offset + 1; i++) {
            for(k = 0; k < y_num; k++) {
                temp_vec[i*y_num + k] = input.mesh[jxnumynum + i*y_num + k] + temp_vec[(i-1)*y_num + k];
            }
        }

        // LHS boundary

        for(i = 0; i < offset + 1; i++){
            for(k = 0; k < y_num; k++) {
                input.mesh[jxnumynum + i * y_num + k] = (temp_vec[(i + offset) * y_num + k]) / (i + offset + 1);
            }
        }

        // middle

        current_index = offset + 1;

        for(i = offset + 1; i < x_num - offset; i++){
            // the current cumsum
            index_modulo = (current_index + offset) % (2*offset + 1); // current_index - offset - 1
            previous_modulo = (current_index + offset - 1) % (2*offset + 1); // the index of previous cumsum

            for(k = 0; k < y_num; k++) {
                temp = input.mesh[jxnumynum + (i + offset)*y_num + k] + temp_vec[previous_modulo*y_num + k];
                input.mesh[jxnumynum + i*y_num + k] = (temp - temp_vec[index_modulo*y_num + k]) /
                        (2*offset + 1);
                temp_vec[index_modulo*y_num + k] = temp;
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }

        // RHS boundary
        current_index = (current_index + offset) % (2*offset + 1);

        for(i = x_num - offset; i < x_num; i++){
            for(k = 0; k < y_num; k++){
                input.mesh[jxnumynum + i*y_num + k] = (temp_vec[index_modulo*y_num + k] -
                        temp_vec[current_index*y_num + k]) / (x_num - i + offset);
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }
    }


}


template<typename T>
void calc_sat_mean_z(Mesh_data<T>& input,const int offset) {

    // The same, but in place

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    std::vector<T> temp_vec;
    temp_vec.resize(y_num*(2*offset + 1),0);

    int j,k;
    float temp;
    int index_modulo, previous_modulo, current_index, iynum;
    int xnumynum = x_num * y_num;

#pragma omp parallel for default(shared) private(j,k,temp,index_modulo, previous_modulo, current_index, iynum) \
        firstprivate(temp_vec)
    for(int i = 0; i < x_num; i++) {

        iynum = i * y_num;

        //prefetching

        for(k = 0; k < y_num ; k++){
            // std::copy ?
            temp_vec[k] = input.mesh[iynum + k];
        }

        for(j = 1; j < 2 * offset + 1; j++) {
            for(k = 0; k < y_num; k++) {
                temp_vec[j*y_num + k] = input.mesh[j * xnumynum + iynum + k] + temp_vec[(j-1)*y_num + k];
            }
        }

        // LHS boundary

        for(j = 0; j < offset + 1; j++){
            for(k = 0; k < y_num; k++) {
                input.mesh[j * xnumynum + iynum + k] = (temp_vec[(j + offset)*y_num + k]) / (j + offset + 1);
            }
        }

        // middle

        current_index = offset + 1;

        for(j = offset + 1; j < z_num - offset; j++){

            index_modulo = (current_index + offset) % (2*offset + 1); // current_index - offset - 1
            previous_modulo = (current_index + offset - 1) % (2*offset + 1); // the index of previous cumsum

            for(k = 0; k < y_num; k++) {
                // the current cumsum
                temp = input.mesh[(j + offset) * xnumynum + iynum + k] + temp_vec[previous_modulo*y_num + k];
                input.mesh[j * xnumynum + iynum + k] = (temp - temp_vec[index_modulo*y_num + k]) /
                        (2*offset + 1);
                temp_vec[index_modulo*y_num + k] = temp;
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }

        // RHS boundary
        current_index = (current_index + offset) % (2*offset + 1);

        for(j = z_num - offset; j < z_num; j++){
            for(k = 0; k < y_num; k++){
                input.mesh[j * xnumynum + iynum + k] = (temp_vec[index_modulo*y_num + k] -
                                                        temp_vec[current_index*y_num + k]) / (z_num - j + offset);
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }
    }

}


template<typename T>
void calc_abs_diff(Mesh_data<T>& input_image,Mesh_data<T>& var){
    //
    //  Bevan Cheeseman 2016
    //
    //

    const int z_num = input_image.z_num;
    const int x_num = input_image.x_num;
    const int y_num = input_image.y_num;

    int i,k;

#pragma omp parallel for default(shared) private(i,k)
    for(int j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){

            for (k = 0; k < (y_num);k++){
                var.mesh[j*x_num*y_num + i*y_num + k] = std::abs(var.mesh[j*x_num*y_num + i*y_num + k] - input_image.mesh[j*x_num*y_num + i*y_num + k]);
            }

        }
    }


}
template<typename T>
void intensity_th(Mesh_data<T>& input_image,Mesh_data<T>& var,const float threshold,float max_th_input = 60000){
    //
    //  Bevan Cheeseman 2016
    //
    //

    const int z_num = var.z_num;
    const int x_num = var.x_num;
    const int y_num = var.y_num;

    const int z_num_i = input_image.z_num;
    const int x_num_i = input_image.x_num;
    const int y_num_i = input_image.y_num;

    const float max_th = max_th_input;

    int i,k;

#pragma omp parallel for default(shared) private(i,k)
    for(int j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){

            for (k = 0; k < (y_num);k++){

                if(input_image.mesh[j*x_num_i*y_num_i + i*y_num_i + k] < threshold){
                    var.mesh[j*x_num*y_num + i*y_num + k] = max_th;
                }

            }

        }
    }


}

template<typename T>
void rescale_var_and_threshold(Mesh_data<T>& var,const float var_rescale,Part_rep& p_rep){
    //
    //  Bevan Cheeseman 2016
    //
    //

    const int z_num = var.z_num;
    const int x_num = var.x_num;
    const int y_num = var.y_num;
    const float max_th = 60000.0;

    int i,k;
    float rescaled;

#pragma omp parallel for default(shared) private(i,k,rescaled)
    for(int j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){

            for (k = 0; k < (y_num);k++){

                float rescaled = var.mesh[j*x_num*y_num + i*y_num + k] * var_rescale;
                if(rescaled < p_rep.pars.var_th_max){
                    rescaled = max_th;
                }
                if(rescaled < p_rep.pars.var_th){
                    rescaled = p_rep.pars.var_th;
                }
                var.mesh[j*x_num*y_num + i*y_num + k] = rescaled;
            }

        }
    }

}

template<typename T>
void get_variance_2D(Part_rep &p_rep, Mesh_data<T> &input_image, Mesh_data<T> &var){
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculates the local variance using recursive SAT
    //

    Part_timer timer;


    // first down sample the image by 2, then calculate...
    down_sample(input_image,var,
                [](T x, T y) { return x+y; },
                [](T x) { return x * (1.0/8.0); });

    // copy constructor
    Mesh_data<T> temp = var;

    Map_calc_cpu calc_map(0,p_rep.pars);

    calc_map.set_up_var_filters_2D();

    int win_y = ceil((calc_map.var_window_size1 - 1)/4.0);
    int win_x = ceil((calc_map.var_window_size2 - 1)/4.0);
    int win_z = ceil((calc_map.var_window_size3 - 1)/4.0);

    //Perform first spatial average output to var

    debug_write(temp,"temp");

    timer.start_timer("calc_sat_mean_y");

    calc_sat_mean_y(var,win_y);

    timer.stop_timer();


    timer.start_timer("calc_sat_mean_x");


    calc_sat_mean_x(var,win_x);

    timer.stop_timer();




    timer.start_timer("calc_abs_diff");


    //calculate abs and subtract from original
    calc_abs_diff(temp,var);

    timer.stop_timer();
    //Second spatial average
    calc_sat_mean_y(var,win_y);
    calc_sat_mean_x(var,win_x);



    //if needed threshold the results
    if(p_rep.pars.I_th > 0) {
        intensity_th(temp, var,
                     p_rep.pars.I_th);
    }

    timer.start_timer("rescale_var_and_threshold");

    //rescaling the variance estimate
    rescale_var_and_threshold( var,calc_map.var_rescale,p_rep);

    timer.stop_timer();


}

template<typename T>
void get_variance_3D(Part_rep &p_rep, Mesh_data<T> &input_image, Mesh_data<T> &var){
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculates the local variance using recursive SAT
    //

    Part_timer timer;


    // first down sample the image by 2, then calculate...
    down_sample(input_image,var,
                [](T x, T y) { return x+y; },
                [](T x) { return x * (1.0/8.0); });

    // copy constructor
    Mesh_data<T> temp = var;

    Map_calc_cpu calc_map(0,p_rep.pars);

    calc_map.set_up_var_filters_3D();



    int win_y = ceil((calc_map.var_window_size1 - 1)/4.0);
    int win_x = ceil((calc_map.var_window_size2 - 1)/4.0);
    int win_z = ceil((calc_map.var_window_size3 - 1)/4.0);

    int win_y2 = ceil((calc_map.var_window_size1 - 1)/4.0);
    int win_x2 = ceil((calc_map.var_window_size2 - 1)/4.0);
    int win_z2 = ceil((calc_map.var_window_size3 - 1)/4.0);



        win_y = calc_map.var_win[0];
        win_x = calc_map.var_win[1];
        win_z = calc_map.var_win[2];

        win_y2 = calc_map.var_win[3];
        win_x2 = calc_map.var_win[4];
        win_z2 = calc_map.var_win[5];



    //Perform first spatial average output to var

    timer.start_timer("calc_sat_mean_y");

    calc_sat_mean_y(var,win_y);

    timer.stop_timer();


    timer.start_timer("calc_sat_mean_x");


    calc_sat_mean_x(var,win_x);

    timer.stop_timer();

    timer.start_timer("calc_sat_mean_z");

    calc_sat_mean_z(var,win_z);

    timer.stop_timer();



    timer.start_timer("calc_abs_diff");


    //calculate abs and subtract from original
    calc_abs_diff(temp,var);

    timer.stop_timer();
    //Second spatial average
    calc_sat_mean_y(var,win_y2);
    calc_sat_mean_x(var,win_x2);
    calc_sat_mean_z(var,win_z2);

    //if needed threshold the results
    if(p_rep.pars.I_th > 0) {
        intensity_th(temp, var,
        p_rep.pars.I_th);
    }

    timer.start_timer("rescale_var_and_threshold");

    //rescaling the variance estimate
    rescale_var_and_threshold( var,calc_map.var_rescale,p_rep);

    timer.stop_timer();


}

#endif