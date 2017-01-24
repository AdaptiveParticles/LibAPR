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

    std::string name = "input";
    compare_E(org_img,rec_img,options,name);

}
template<typename S>
void compare_E(Mesh_data<S>& org_img,Mesh_data<S>& rec_img,cmdLineOptions& options,std::string name){

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

    double mean = 0;
    double inf_norm = 0;
    uint64_t counter = 0;

    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){
                double val = abs(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k])/(1.0*variance.mesh[j*x_num_r*y_num_r + i*y_num_r + k]);
                SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] += val;

                if(variance.mesh[j*x_num_r*y_num_r + i*y_num_r + k] < 50000) {
                    mean += val;
                    inf_norm = std::max(inf_norm, val);
                    counter++;
                }
            }
        }
    }

    mean = mean/(1.0*counter);

    debug_write(SE,name + "E_diff");
    debug_write(variance,name + "var");
    debug_write(rec_img,name +"rec_img");

    //calculate the variance
    double var = 0;
    counter = 0;

    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){
                double val = pow(SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - mean,2.0);

                if(variance.mesh[j*x_num_r*y_num_r + i*y_num_r + k] < 50000) {
                    var+=val;
                    counter++;
                }
            }
        }
    }

    //get variance
    var = var/(1.0*counter);
    double se = 1.96*sqrt(var);

    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Var: " << var << std::endl;
    std::cout << "SE: " << se << std::endl;
    std::cout << "inf: " << inf_norm << std::endl;


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

    double var = 0;
    double counter = 0;

    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){

                SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] += pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2);
                var += pow(rec_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - MSE,2);
                counter++;
            }
        }
    }

    var = var/(counter*1.0);

    debug_write(SE,"squared_error");

    double se = sqrt(var)*1.96;

    double PSNR = 10*log10(64000.0/MSE);

    std::cout << "MSE: " << MSE << std::endl;
    std::cout << "PSNR: " << PSNR << std::endl;
    std::cout << "se: " << se << std::endl;


    return MSE;


}





#endif //PARTPLAY_ANALYZE_APR_H
