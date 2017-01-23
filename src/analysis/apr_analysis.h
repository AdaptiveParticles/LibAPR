//
// Created by cheesema on 23/01/17.
//

#ifndef PARTPLAY_APR_ANALYSIS_H
#define PARTPLAY_APR_ANALYSIS_H

#include "../algorithm/apr_pipeline.hpp"

template<typename S,typename U>
void compare_reconstruction_to_original(Mesh_data<S>& org_img,PartCellStructure<float,U>& pc_struct,cmdLineOptions& options){
    //
    //  Bevan Cheeseman 2017
    //
    //  Compares an APR pc reconstruction with original image.
    //
    //

    Mesh_data<S> rec_img;
    pc_struct.interp_parts_to_pc(rec_img,pc_struct.part_data.particle_data);

    //get the MSE
    double MSE = calc_mse(org_img,rec_img);

    compare_E(org_img,rec_img,options);

}
template<typename S>
void compare_E(Mesh_data<S>& org_img,Mesh_data<S>& rec_img,cmdLineOptions& options){

    Mesh_data<float> variance;

    get_variance(variance,options);

    uint64_t z_num_o = org_img.z_num;
    uint64_t x_num_o = org_img.x_num;
    uint64_t y_num_o = org_img.y_num;

    uint64_t z_num_r = rec_img.z_num;
    uint64_t x_num_r = rec_img.x_num;
    uint64_t y_num_r = rec_img.y_num;

    variance.x_num = x_num_r;
    variance.y_num = y_num_r;
    variance.z_num = z_num_r;

    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;

    Mesh_data<float> SE;
    SE.initialize(y_num_o,x_num_o,z_num_o,0);

    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){

                SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] += 1000.0*abs(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k])/(1.0*variance.mesh[j*x_num_r*y_num_r + i*y_num_r + k]);

            }
        }
    }


    debug_write(SE,"E_diff");
    debug_write(variance,"var");
    debug_write(rec_img,"rec_img");


}


template<typename S>
double calc_mse(Mesh_data<S>& org_img,Mesh_data<S>& rec_img){
    //
    //  Bevan Cheeseman 2017
    //
    //  Calculates the mean squared error
    //

    uint64_t z_num_o = org_img.z_num;
    uint64_t x_num_o = org_img.x_num;
    uint64_t y_num_o = org_img.y_num;

    uint64_t z_num_r = rec_img.z_num;
    uint64_t x_num_r = rec_img.x_num;
    uint64_t y_num_r = rec_img.y_num;

    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;

    double MSE = 0;

//#pragma omp parallel for default(shared) private(j,i,k)
    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){

                MSE += pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2);

            }
        }
    }

    MSE = MSE/(z_num_o*x_num_o*y_num_o*1.0);


    Mesh_data<S> SE;
    SE.initialize(y_num_o,x_num_o,z_num_o,0);

    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){

                SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] += pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2);

            }
        }
    }


    debug_write(SE,"squared_error");

    return MSE;


}







#endif //PARTPLAY_ANALYZE_APR_H
