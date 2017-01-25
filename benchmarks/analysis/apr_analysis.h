//
// Created by cheesema on 23/01/17.
//

#ifndef PARTPLAY_APR_ANALYSIS_H
#define PARTPLAY_APR_ANALYSIS_H

#include "../../src/algorithm/apr_pipeline.hpp"
#include "AnalysisData.hpp"


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
    std::cout << "SE(1.96*sd): " << se << std::endl;

}
template<typename T>
void produce_apr_analysis(AnalysisData& analysis_data) {

    Part_rep p_rep;

    Proc_par pars;

    analysis_data.push_proc_par(pars);

    Mesh_data<T> input_image;

//////////////////////////////////////
//
//  Initialize - data anslysis fields
//
/////////////////////////////////////////

//first check if its first run and the variables need to be set up
    Part_data<int> *check_ref = analysis_data.get_data_ref<int>("num_parts");

    if (check_ref == nullptr) {
// First experiment need to set up the variables

//pipeline parameters
        analysis_data.create_int_dataset("num_parts", 0);
        analysis_data.create_int_dataset("num_cells", 0);
        analysis_data.create_int_dataset("num_seed_cells", 0);
        analysis_data.create_int_dataset("num_boundary_cells", 0);
        analysis_data.create_int_dataset("num_filler_cells", 0);
        analysis_data.create_int_dataset("num_ghost_cells", 0);

        analysis_data.create_int_dataset("num_pixels", 0);

        analysis_data.create_float_dataset("apr_comp_size", 0);
        analysis_data.create_float_dataset("apr_full_size", 0);
        analysis_data.create_float_dataset("image_size", 0);
        analysis_data.create_float_dataset("comp_image_size", 0);

        analysis_data.create_float_dataset("information_content", 0);
        analysis_data.create_float_dataset("rel_error", 0);

//set up timing variables

        for (int i = 0; i < p_rep.timer.timings.size(); i++) {
            analysis_data.create_float_dataset(p_rep.timer.timing_names[i], 0);
        }


    }

/////////////////////////////////////////////
//
//  APR Size Info
//
//////////////////////////////////////////////


    analysis_data.get_data_ref<int>("num_parts")->data.push_back(p_rep.get_part_num());
    analysis_data.part_data_list["num_parts"].print_flag = true;

//calc number of active cells
    int num_seed_cells = 0;
    int num_boundary_cells = 0;
    int num_filler_cells = 0;
    int num_ghost_cells = 0;
    int num_cells = 0;

    for (int i = 0; i < p_rep.pl_map.cells.size(); i++) {

        if (p_rep.status.data[i] == 2) {
            num_seed_cells++;
            num_cells++;
        } else if (p_rep.status.data[i] == 4) {
            num_boundary_cells++;
            num_cells++;
        } else if (p_rep.status.data[i] == 5) {
            num_filler_cells++;
            num_cells++;
        } else {
            num_ghost_cells++;
            num_cells++;
        }

    }


    analysis_data.get_data_ref<int>("num_cells")->data.push_back(num_cells);
    analysis_data.part_data_list["num_cells"].print_flag = true;

    analysis_data.get_data_ref<int>("num_boundary_cells")->data.push_back(num_boundary_cells);
    analysis_data.part_data_list["num_boundary_cells"].print_flag = true;

    analysis_data.get_data_ref<int>("num_filler_cells")->data.push_back(num_filler_cells);
    analysis_data.part_data_list["num_filler_cells"].print_flag = true;


    analysis_data.get_data_ref<int>("num_seed_cells")->data.push_back(num_seed_cells);
    analysis_data.part_data_list["num_seed_cells"].print_flag = true;

    analysis_data.get_data_ref<int>("num_ghost_cells")->data.push_back(num_ghost_cells);
    analysis_data.part_data_list["num_ghost_cells"].print_flag = true;

    analysis_data.get_data_ref<int>("num_pixels")->data.push_back(
            input_image.y_num * input_image.x_num * input_image.z_num);
    analysis_data.part_data_list["num_pixels"].print_flag = true;

////////////////////////////////////////////////////////////////////
//
//  Timing Information
//
////////////////////////////////////////////////////////////////////

    for (int i = 0; i < p_rep.timer.timings.size(); i++) {
        analysis_data.get_data_ref<float>(p_rep.timer.timing_names[i])->data.push_back(p_rep.timer.timings[i]);
        analysis_data.part_data_list[p_rep.timer.timing_names[i]].print_flag = true;
    }


////////////////////////////////////////////////////////////////////
//
//  File Size Information
//
/////////////////////////////////////////////////////////////////////

//write apr and image to file

    std::string file_name = p_rep.pars.output_path + p_rep.pars.name;

//for debug
// write_apr_to_hdf5(p_rep, p_rep.pars.output_path,  p_rep.pars.name);


// write_qcompress_to_hdf5(p_rep, p_rep.pars.output_path,  p_rep.pars.name);
// write_qcompress_to_hdf5(p_rep, p_rep.pars.output_path,  p_rep.pars.name);

//  write_nocompress_to_hdf5(p_rep, p_rep.pars.output_path,  p_rep.pars.name);

    //write_lossy_wavelet_apr_to_hdf5(p_rep, p_rep.pars.output_path, p_rep.pars.name);

    write_apr_to_hdf5(p_rep, p_rep.pars.output_path, p_rep.pars.name);

    write_image_tiff(input_image, file_name + ".tif");

    analysis_data.get_data_ref<float>("apr_comp_size")->data.push_back(GetFileSize(file_name + "_wavelet_lossy.h5"));
    analysis_data.part_data_list["apr_comp_size"].print_flag = true;

    analysis_data.get_data_ref<float>("apr_full_size")->data.push_back(GetFileSize(file_name + "_nc.h5"));
    analysis_data.part_data_list["apr_full_size"].print_flag = true;

    analysis_data.get_data_ref<float>("image_size")->data.push_back(GetFileSize(file_name + ".tif"));
    analysis_data.part_data_list["image_size"].print_flag = true;

    analysis_data.get_data_ref<float>("rel_error")->data.push_back(p_rep.pars.rel_error);
    analysis_data.part_data_list["rel_error"].print_flag = true;

//produce the compressed image file to get baseline
    std::string compress_file_name = file_name + ".bz2";
    std::string system_string = "pbzip2 -c -9 <" + file_name + ".tif" + ">" + compress_file_name;
    system(system_string.c_str());

    analysis_data.get_data_ref<float>("comp_image_size")->data.push_back(GetFileSize(compress_file_name));
    analysis_data.part_data_list["comp_image_size"].print_flag = true;


}






#endif //PARTPLAY_ANALYZE_APR_H
