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

        var_rescale = par.var_scale;

        std::vector<std::vector<std::vector<float>>> rescale_store ={{{{6.9005},{3.0015},{1.5595},{2.9951},{2.7954},{1.5053},{1.959},{1.9693},{0.94017},{3.1533}},{{6.9585},{4.5416},{2.9428},{5.0018},{1.7866},{2.6356},{1.857},{0.22761},{0.72876},{1.2207}},{{6.4195},{6.2791},{0.77803},{2.0814},{1.0562},{1.1767},{0.31025},{0.193},{1.6856},{6.3676}},{{11.5081},{4.4167},{4.8724},{3.3629},{2.3503},{5.9221},{3.8502},{2.8936},{2.2604},{4.4294}},{{15.5754},{5.4097},{5.1949},{4.6281},{3.3316},{8.6997},{6.9946},{10.0258},{2.9945},{6.7075}},{{16.4329},{7.9348},{4.5819},{5.1859},{7.5101},{6.8357},{3.5212},{6.5328},{3.4555},{5.2596}},{{12.8818},{9.4402},{13.5714},{7.4091},{7.5802},{4.9943},{5.1142},{7.3422},{6.9476},{6.4024}},{{11.8528},{14.4033},{10.8678},{8.8582},{11.0998},{8.2039},{8.2856},{11.1421},{7.3569},{10.8666}},{{14.9621},{10.9815},{12.664},{11.5217},{8.7983},{10.899},{13.3429},{9.3479},{9.7507},{8.8858}},{{15.9225},{17.5693},{15.8052},{14.0109},{9.87},{8.2022},{8.415},{10.5893},{8.4724},{9.9762}}},{{{18.1995},{10.6846},{6.1133},{5.6048},{7.0095},{10.4368},{10.9563},{9.7628},{17.7793},{9.98}},{{18.0142},{10.05},{7.6529},{8.3367},{8.7006},{8.6996},{5.9568},{5.5279},{4.8856},{10.9143}},{{17.6632},{6.266},{8.2092},{5.6068},{5.1155},{4.7086},{6.7828},{9.3271},{4.6387},{10.7991}},{{19.4002},{11.9303},{7.265},{7.5419},{9.3333},{9.1014},{9.5534},{13.9675},{6.2189},{12.3625}},{{19.9439},{16.3377},{19.8603},{8.1125},{10.6712},{7.5874},{5.6108},{11.4737},{12.4551},{8.8511}},{{18.436},{11.8581},{12.3504},{6.4589},{8.587},{9.2417},{7.8166},{8.0202},{10.0012},{11.8252}},{{17.1532},{16.8461},{13.4653},{12.836},{11.2832},{9.0022},{13.2367},{6.5446},{11.3811},{9.7403}},{{13.8384},{15.772},{18.1144},{18.5878},{13.9045},{9.3872},{13.7415},{17.0552},{20.6475},{11.3636}},{{17.2368},{17.9718},{14.2753},{14.1354},{14.2511},{9.4021},{12.5552},{9.9775},{8.7759},{16.7331}},{{20.1458},{24.9106},{11.959},{16.3301},{15.8906},{17.4266},{17.3624},{12.1818},{12.0454},{20.5409}}},{{{21.0239},{14.905},{13.2771},{12.7423},{12.9801},{16.9697},{11.7524},{12.9517},{12.4513},{12.8633}},{{20.8622},{14.1937},{17.4366},{10.2144},{13.1675},{8.3207},{17.5321},{11.9755},{11.2743},{9.0227}},{{18.9805},{13.4195},{15.1406},{12.5176},{14.1265},{12.8629},{10.5821},{9.1722},{8.3375},{15.0463}},{{14.6695},{19.3547},{9.4346},{8.2773},{10.4729},{11.4219},{6.8861},{9.1746},{12.5293},{11.7671}},{{19.9519},{19.9777},{14.354},{9.1452},{14.9452},{9.6351},{7.7188},{9.3573},{6.2866},{5.9898}},{{22.3896},{15.7745},{15.0568},{15.3291},{10.0262},{14.1917},{14.6996},{15.4821},{5.7422},{10.8119}},{{13.9642},{17.4047},{18.3078},{14.2676},{9.7538},{16.5824},{7.2839},{9.7706},{21.2349},{10.3308}},{{19.8133},{20.4587},{14.7187},{17.3917},{16.5444},{15.6303},{14.8159},{14.1005},{17.7932},{15.749}},{{22.4706},{17.3228},{21.4321},{15.5333},{21.335},{14.3796},{13.0556},{16.6923},{13.6033},{13.1584}},{{17.3054},{21.589},{25.6187},{12.8294},{14.0788},{18.0558},{17.9466},{14.5477},{16.2883},{17.2984}}},{{{17.5045},{18.0076},{13.3874},{17.465},{13.5223},{11.3675},{11.1137},{13.8726},{15.3783},{12.5056}},{{17.6451},{20.3019},{17.7219},{20.326},{15.557},{11.9901},{18.093},{9.0467},{14.5785},{11.2668}},{{21.2771},{17.7472},{13.6287},{11.2761},{13.0062},{15.6102},{13.9448},{10.7312},{10.2994},{14.3447}},{{15.6681},{21.7888},{15.4115},{21.3667},{15.3697},{17.3483},{12.3834},{15.7095},{12.9071},{15.3094}},{{18.0197},{16.7813},{17.5096},{14.8957},{10.8196},{10.9031},{12.385},{8.9942},{13.4166},{10.9977}},{{18.9135},{20.6092},{16.2611},{18.353},{14.8385},{15.0185},{9.0612},{16.6081},{13.8304},{17.6951}},{{15.3446},{15.9367},{14.0045},{18.9721},{14.6801},{14.256},{13.8505},{12.7035},{14.3475},{17.4769}},{{17.2851},{19.107},{14.047},{15.5935},{18.2465},{13.3975},{16.5503},{13.5881},{15.3646},{10.7214}},{{19.8053},{15.3045},{22.5614},{17.5283},{14.7689},{18.13},{15.6822},{16.7456},{12.1229},{23.2503}},{{17.7875},{18.8442},{18.3366},{16.8903},{19.483},{18.3018},{19.1622},{12.6454},{12.9021},{17.4428}}},{{{13.4685},{15.2219},{14.0731},{13.1983},{15.3928},{16.6228},{12.5885},{15.2535},{13.4833},{17.0484}},{{17.4924},{14.5395},{19.5577},{12.8522},{12.0348},{13.6049},{14.6738},{19.6508},{19.845},{15.4846}},{{22.6794},{16.7043},{17.4535},{19.2934},{20.0959},{10.9205},{16.2226},{12.8742},{22.5305},{15.5704}},{{16.4777},{16.0454},{17.8349},{15.1879},{19.0365},{19.4915},{18.3484},{13.6509},{24.2359},{10.5372}},{{15.7332},{14.0167},{14.4875},{16.5083},{12.8817},{15.5988},{13.1469},{14.9859},{13.579},{12.9107}},{{18.8724},{13.1602},{17.4165},{14.1934},{13.9666},{14.5261},{15.3258},{18.1127},{13.8503},{15.3016}},{{20.9572},{18.814},{18.1938},{23.8946},{17.0175},{15.7097},{18.5029},{14.9155},{19.5808},{12.2099}},{{16.9476},{20.2683},{19.3154},{13.5934},{11.1492},{14.9726},{19.0322},{16.7739},{13.4148},{16.2037}},{{17.4194},{15.0752},{15.3107},{17.9344},{15.0174},{19.3117},{20.7877},{15.5961},{15.2122},{16.866}},{{19.3997},{16.1417},{13.4364},{17.9089},{15.0435},{18.7739},{16.4096},{18.8551},{14.6208},{18.5913}}}};

        //(PSF,window2,window1) with 0 indexing

        int window_ind_1 = std::min(par.padd_dims[0] - 1,9);
        int window_ind_2 = std::min(par.padd_dims[3] - 1,9);
        int psf_ind = std::max(((float) (round(par.psfx/par.dx) - 1)),((float)0.0f));

       // var_rescale = 1.0/rescale_store[psf_ind][window_ind_2][window_ind_1];

        var_rescale = par.var_scale;

        std::cout << "**scale: " << var_rescale << std::endl;

        //int stop = 1;

        //var_rescale = 6.9541;
        //var_rescale = 7.1748;

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

    if(p_rep.pars.padd_dims.size() == 6) {

        win_y = p_rep.pars.padd_dims[0];
        win_x = p_rep.pars.padd_dims[1];
        win_z = p_rep.pars.padd_dims[2];

        win_y2 = p_rep.pars.padd_dims[3];
        win_x2 = p_rep.pars.padd_dims[4];
        win_z2 = p_rep.pars.padd_dims[5];

    }

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