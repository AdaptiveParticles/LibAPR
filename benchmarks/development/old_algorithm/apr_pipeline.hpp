//
// Created by cheesema on 23/01/17.
//

#ifndef PARTPLAY_APR_PIPELINE_HPP
#define PARTPLAY_APR_PIPELINE_HPP

#include "benchmarks/development/old_structures/structure_parts.h"

#include "src/data_structures/Mesh/meshclass.h"
#include "src/io/readimage.h"

#include "gradient.hpp"
#include "benchmarks/development/old_structures/particle_map.hpp"
#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "level.hpp"
#include "src/io/writeimage.h"
#include "src/io/write_parts.h"
#include "src/io/partcell_io.h"
#include "benchmarks/development/Tree/PartCellParent.hpp"
#include "benchmarks/analysis/AnalysisData.hpp"
#include "src/data_structures/APR/APR.hpp"

struct cmdLineOptions{
    std::string gt_input = "";
    std::string output_dir = "";
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    std::string mask_file = "";
    bool stats_file = false;

    float Ip_th = -1;
    float SNR_min = 0;
    float lambda = -1;
    float min_signal = -1;
    float rel_error = 0.1;

};

bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv, Proc_par& pars){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_get_apr -i inputfile [-t] [-s statsfile -d directory] [-o outputfile]\"" << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    if(command_option_exists(argv, argv + argc, "-s"))
    {
        result.stats = std::string(get_command_option(argv, argv + argc, "-s"));
        result.stats_file = get_image_stats(pars, result.directory, result.stats);

    }

    if(command_option_exists(argv, argv + argc, "-od"))
    {
        result.output_dir = std::string(get_command_option(argv, argv + argc, "-od"));
    } else {
        result.output_dir = result.directory;
    }

    if(command_option_exists(argv, argv + argc, "-gt"))
    {
        result.gt_input = std::string(get_command_option(argv, argv + argc, "-gt"));
    } else {
        result.gt_input = "";
    }

    if(command_option_exists(argv, argv + argc, "-lambda"))
    {
        result.lambda = std::stof(std::string(get_command_option(argv, argv + argc, "-lambda")));
    }

    if(command_option_exists(argv, argv + argc, "-I_th"))
    {
        result.Ip_th = std::stof(std::string(get_command_option(argv, argv + argc, "-I_th")));
    }

    if(command_option_exists(argv, argv + argc, "-SNR_min"))
    {
        result.SNR_min = std::stof(std::string(get_command_option(argv, argv + argc, "-SNR_min")));
    }

    if(command_option_exists(argv, argv + argc, "-min_signal"))
    {
        result.min_signal = std::stof(std::string(get_command_option(argv, argv + argc, "-min_signal")));
    }

    if(command_option_exists(argv, argv + argc, "-rel_error"))
    {
        result.rel_error = std::stof(std::string(get_command_option(argv, argv + argc, "-rel_error")));
    }

    if(command_option_exists(argv, argv + argc, "-mask_file"))
    {
        result.mask_file = std::string(get_command_option(argv, argv + argc, "-mask_file"));
    }


    return result;

}

template<typename U>
U median_3(U& a,U& b,U& c) {
    // 3 value median
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

template<typename U>
void calc_median_filter(Mesh_data<U>& input_img){


    uint64_t offset_min;
    uint64_t offset_max;

    const int x_num_m = input_img.x_num;
    const int y_num_m = input_img.y_num;
    const int z_num_m = input_img.z_num;

    Part_timer timer;

    timer.start_timer("compute gradient y");

    std::vector<U> temp_vec;
    temp_vec.resize(y_num_m);


    uint64_t x_,z_;

#pragma omp parallel for default(shared) private(z_,x_,offset_min,offset_max) firstprivate(temp_vec)
    for(z_ = 0;z_ < z_num_m;z_++){
        //both z and x are explicitly accessed in the structure

        for(x_ = 0;x_ < x_num_m;x_++){

            for (int k = 0; k < y_num_m; ++k) {
                temp_vec[k] = input_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k];
            }

            for (int k = 0; k < y_num_m; ++k) {
                offset_max = std::min((int)(k + 1),(int)(y_num_m-1));
                offset_min = std::max((int)(k - 1),(int)0);

                input_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k] = median_3(temp_vec[offset_min],temp_vec[k],temp_vec[offset_max]);
            }


        }
    }

    timer.stop_timer();

    float time = (timer.t2 - timer.t1);

    std::vector<U> temp_vecp;
    temp_vecp.resize(y_num_m);

    std::vector<U> temp_vecm;
    temp_vecm.resize(y_num_m);


#pragma omp parallel for default(shared) private(z_,x_,offset_min,offset_max) firstprivate(temp_vec,temp_vecm,temp_vecp)
    for(z_ = 0;z_ < z_num_m;z_++){
        //both z and x are explicitly accessed in the structure

        for (int k = 0; k < y_num_m; ++k) {
            temp_vecp[k] = input_img.mesh[z_*x_num_m*y_num_m + k];
            temp_vec[k] = input_img.mesh[z_*x_num_m*y_num_m + k];
        }

        for(x_ = 0;x_ < x_num_m;x_++){

            std::swap(temp_vec,temp_vecm);
            std::swap(temp_vecp,temp_vec);

            offset_max = std::min((int)(x_ + 1),(int)(x_num_m-1));
            //offset_min = std::max((int)(x_ - 1),(int)0);

            //get plus value
            for (int k = 0; k < y_num_m; ++k) {
                temp_vecp[k] = input_img.mesh[z_*x_num_m*y_num_m + offset_max*y_num_m + k];
            }


            for (int k = 0; k < y_num_m; ++k) {

                input_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k] = median_3(temp_vec[k],temp_vecp[k],temp_vecm[k]);
            }


        }
    }

    timer.stop_timer();

    time = (timer.t2 - timer.t1);




#pragma omp parallel for default(shared) private(z_,x_,offset_min,offset_max) firstprivate(temp_vec,temp_vecm,temp_vecp)
    for(x_ = 0;x_ < x_num_m;x_++){
        //both z and x are explicitly accessed in the structure

        for (int k = 0; k < y_num_m; ++k) {
            temp_vecp[k] = input_img.mesh[ x_*y_num_m + k];
            temp_vec[k] = input_img.mesh[ x_*y_num_m + k];
        }

        for(z_ = 0;z_ < z_num_m;z_++){

            std::swap(temp_vec,temp_vecm);
            std::swap(temp_vecp,temp_vec);

            offset_max = std::min((int)(z_ + 1),(int)(z_num_m-1));
            //offset_min = std::max((int)(z_ - 1),(int)0);

            //get plus value
            for (int k = 0; k < y_num_m; ++k) {
                temp_vecp[k] = input_img.mesh[offset_max*x_num_m*y_num_m + x_*y_num_m + k];
            }


            for (int k = 0; k < y_num_m; ++k) {

                input_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k] = median_3(temp_vec[k],temp_vecp[k],temp_vecm[k]);
            }


        }
    }

    timer.stop_timer();

    time = (timer.t2 - timer.t1);




}
template<typename U>
void calc_median_filter_n(Mesh_data<U>& output_img,Mesh_data<U>& input_img){


    uint64_t offset_min_y;
    uint64_t offset_max_y;

    uint64_t offset_min_x;
    uint64_t offset_max_x;

    uint64_t offset_min_z;
    uint64_t offset_max_z;

    const int x_num_m = input_img.x_num;
    const int y_num_m = input_img.y_num;
    const int z_num_m = input_img.z_num;

    Part_timer timer;

    timer.start_timer("compute gradient y");


    uint64_t x_,z_;

#pragma omp parallel for default(shared) private(z_,x_,offset_min_x,offset_max_x,offset_min_y,offset_max_y,offset_min_z,offset_max_z)
    for(z_ = 0;z_ < z_num_m;z_++){
        //both z and x are explicitly accessed in the structure

        offset_max_z = std::min((int)(z_ + 1),(int)(z_num_m-1));
        offset_min_z = std::max((int)(z_ - 1),(int)0);

        for(x_ = 0;x_ < x_num_m;x_++){

            offset_max_x = std::min((int)(x_ + 1),(int)(x_num_m-1));
            offset_min_x = std::max((int)(x_ - 1),(int)0);


            for (int k = 0; k < y_num_m; ++k) {
                offset_max_y = std::min((int)(k + 1),(int)(y_num_m-1));
                offset_min_y = std::max((int)(k - 1),(int)0);

                float curr = input_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k];

                float y_d_p = input_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + offset_max_y];
                float y_d_m =  input_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + offset_min_y];

                float x_d_p = input_img.mesh[z_*x_num_m*y_num_m + offset_max_x*y_num_m + k] ;
                float x_d_m = input_img.mesh[z_*x_num_m*y_num_m + offset_min_x*y_num_m + k] ;

                float z_d_p = input_img.mesh[offset_max_z*x_num_m*y_num_m + x_*y_num_m + k];
                float z_d_m = input_img.mesh[offset_min_z*x_num_m*y_num_m + x_*y_num_m + k];


                x_d_m = median_3(x_d_m,x_d_p,curr);

                        //Z
                y_d_m = median_3(z_d_m,z_d_p,curr);

                        //y
                z_d_m = median_3(y_d_m,y_d_p,curr);


                        //z
                output_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k] = median_3(z_d_m,x_d_m,y_d_m);






            }


        }
    }

    timer.stop_timer();

    float time = (timer.t2 - timer.t1);

    std::swap(input_img.mesh,output_img.mesh);


}
template<typename U>
void calc_median_filter_n2(Mesh_data<U>& output_img,Mesh_data<U>& input_img){


    uint64_t offset_min_y;
    uint64_t offset_max_y;

    uint64_t offset_min_x;
    uint64_t offset_max_x;

    uint64_t offset_min_z;
    uint64_t offset_max_z;

    const int x_num_m = input_img.x_num;
    const int y_num_m = input_img.y_num;
    const int z_num_m = input_img.z_num;

    Part_timer timer;

    timer.start_timer("compute gradient y");


    uint64_t x_,z_;

#pragma omp parallel for default(shared) private(z_,x_,offset_min_x,offset_max_x,offset_min_y,offset_max_y,offset_min_z,offset_max_z)
    for(z_ = 0;z_ < z_num_m;z_++){
        //both z and x are explicitly accessed in the structure

        offset_max_z = std::min((int)(z_ + 1),(int)(z_num_m-1));
        offset_min_z = std::max((int)(z_ - 1),(int)0);

        for(x_ = 0;x_ < x_num_m;x_++){

            offset_max_x = std::min((int)(x_ + 1),(int)(x_num_m-1));
            offset_min_x = std::max((int)(x_ - 1),(int)0);


            for (int k = 0; k < y_num_m; ++k) {
                offset_max_y = std::min((int)(k + 1),(int)(y_num_m-1));
                offset_min_y = std::max((int)(k - 1),(int)0);

                float curr = input_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k];

                float y_d_p = input_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + offset_max_y];
                float y_d_m =  input_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + offset_min_y];

                float x_d_p = input_img.mesh[z_*x_num_m*y_num_m + offset_max_x*y_num_m + k] ;
                float x_d_m = input_img.mesh[z_*x_num_m*y_num_m + offset_min_x*y_num_m + k] ;

                float z_d_p = input_img.mesh[offset_max_z*x_num_m*y_num_m + x_*y_num_m + k];
                float z_d_m = input_img.mesh[offset_min_z*x_num_m*y_num_m + x_*y_num_m + k];

                if (std::abs(y_d_p - y_d_m) <  std::abs(x_d_p - x_d_m)){
                    if (std::abs(z_d_p - z_d_m) <  std::abs(x_d_p - x_d_m)){
                        //X
                        output_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k] = (x_d_m + x_d_p + curr)/3.0;

                    } else {
                        //Z
                        output_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k] = (z_d_m + z_d_p + curr)/3.0;

                    }


                } else {
                    if (std::abs(z_d_p - z_d_m) < std::abs(y_d_p - y_d_m)){
                        //y
                        output_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k] = (y_d_m + y_d_p + curr)/3.0;

                    } else {
                        //z
                        output_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + k] = (z_d_m + z_d_p + curr)/3.0;

                    }


                }

            }


        }
    }

    timer.stop_timer();

    float time = (timer.t2 - timer.t1);

    std::swap(input_img.mesh,output_img.mesh);


}
template<typename S>
void get_variance(Mesh_data<S>& input_image,Mesh_data<float>& variance_u,Proc_par& pars) {

    Part_rep part_rep;


    Mesh_data<float> gradient,variance;
    Mesh_data<float> interp_img;


    Mesh_data<float> input_image_float;

    input_image_float.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);

    std::copy(input_image.mesh.begin(), input_image.mesh.end(), input_image_float.mesh.begin());


    {
        gradient.initialize(input_image_float.y_num, input_image_float.x_num, input_image_float.z_num, 0);
        part_rep.initialize(input_image_float.y_num, input_image_float.x_num, input_image_float.z_num);

        // After this block, input_image will be freed.
    }

    part_rep.pars = pars;

    Part_timer t;
    t.verbose_flag = false;

    // preallocate_memory
    Particle_map<float> part_map(part_rep);
    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);
    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    Mesh_data<float> temp;
    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    t.start_timer("whole");

    part_rep.timer.start_timer("get_gradient_3D");
    get_gradient_3D(part_rep, input_image_float, gradient);
    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("get_variance_3D");
    get_variance_3D(part_rep, input_image_float, variance);
    part_rep.timer.stop_timer();


    int x_dim = ceil(gradient.x_num/2.0)*2;
    int z_dim = ceil(gradient.z_num/2.0)*2;
    int y_dim = ceil(gradient.y_num/2.0)*2;

    variance_u.mesh.resize(x_dim*z_dim*y_dim,0);

    std::vector<unsigned int> dims = {(unsigned int)gradient.y_num,(unsigned int)gradient.x_num,(unsigned int)gradient.z_num};

    const_upsample_img(variance_u,variance,dims);

    t.stop_timer();

}

template<typename S>
void spread_values(Mesh_data<S>& input_image){
    // spread the value if zero to the neighbours.


    int x_num = input_image.x_num;
    int y_num = input_image.y_num;
    int z_num = input_image.z_num;

    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

    Mesh_data<S> output_data;

    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);

    int j = 0;
    int k = 0;
    int i = 0;

    int j_n = 0;
    int k_n = 0;
    int i_n = 0;

    //float neigh_sum = 0;


#pragma omp parallel for default(shared) private(j,i,k,i_n,k_n,j_n)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                for(k = 0;k < y_num;k++){
                    float neigh_sum = 0;
                    int count = 0;


                    for(int  d  = 0;d < 6;d++){

                        i_n = i + dir_x[d];
                        k_n = k + dir_y[d];
                        j_n = j + dir_z[d];

                        //check boundary conditions
                        if((i_n >=0) & (i_n < x_num) ){
                            if((j_n >=0) & (j_n < z_num) ){
                                if((k_n >=0) & (k_n < y_num) ){
                                    if(input_image.mesh[j_n * x_num * y_num + i_n * y_num + k_n] > 0) {
                                        neigh_sum += input_image.mesh[j_n * x_num * y_num + i_n * y_num + k_n];
                                        count++;
                                    }
                                }
                            }
                        }

                    }
                    if(count > 0) {
                        output_data.mesh[j * x_num * y_num + i * y_num + k] = neigh_sum/count;
                    } else{
                        output_data.mesh[j * x_num * y_num + i * y_num + k] = 0;
                    }


                }
            }
        }

    std::swap(input_image,output_data);

}

template<typename T,typename S>
ExtraPartCellData<T> get_scale_parts_guided(APR<T>& apr,Mesh_data<S>& input_image,Proc_par& pars,Part_rep& part_rep){
    //
    //  Produces a new scale estimate that uses the previous time steps APR to estimate.
    //

    Mesh_data<float> variance_u;

    get_variance(input_image,variance_u,pars);


    // now need the pyramid data-structure
    Particle_map<float> part_map(part_rep);

    std::vector<Mesh_data<float>> layers;

    int k_max = apr.y_vec.depth_max;
    int k_min = part_map.k_min;

    std::vector<unsigned int> dims = apr.y_vec.org_dims;

    layers.resize(k_max + 1);

    for(int k_ = k_min; k_ < (k_max + 1) ;k_ ++){
        layers[k_].initialize(ceil(1.0*dims[0]/pow(2.0,1.0*k_max - k_)),
                              ceil(1.0*dims[1]/pow(2.0,1.0*k_max - k_)),
                              ceil(1.0*dims[2]/pow(2.0,1.0*k_max - k_)),
                              EMPTY);
    }

    int z_, x_, j_, y_, i, k;
    //loop over particle locations and set equal to the var value

    uint64_t depth = apr.y_vec.depth_max;
    //loop over the resolutions of the structure
    const unsigned int x_num_ = apr.y_vec.x_num[depth];
    const unsigned int z_num_ = apr.y_vec.z_num[depth];

    const float step_size_x = pow(2, apr.y_vec.depth_max - depth);
    const float step_size_y = pow(2, apr.y_vec.depth_max - depth);
    const float step_size_z = pow(2, apr.y_vec.depth_max - depth);

#pragma omp parallel for default(shared) private(z_,x_,j_,i,k) schedule(guided) if(z_num_*x_num_ > 1000)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_ * z_ + x_;

                for (j_ = 0; j_ < apr.y_vec.data[depth][pc_offset].size(); j_++) {


                    const int y = apr.y_vec.data[depth][pc_offset][j_];

                    const float y_actual = floor((y+0.5) * step_size_y);
                    const float x_actual = floor((x_+0.5) * step_size_x);
                    const float z_actual = floor((z_+0.5) * step_size_z);

                    layers[depth](y_actual,x_actual,z_actual)=variance_u(y_actual,x_actual,z_actual);

                }
            }
        }



    for (uint64_t depth = apr.y_vec.depth_max; depth > apr.y_vec.depth_min; depth--) {

        spread_values(layers[depth]);

        //loop over the levels

        //then create a spread function
        down_sample(layers[depth], layers[depth - 1],
                    [](T x, T y) { return std::max(x, y); },
                    [](T x) { return x; }, false);

        //then down-sample
    }

    //then sample from either the new one if non-zero, else the old one.


    ExtraPartCellData<T> scale_parts;

    scale_parts.initialize_structure_parts(apr.particles_int);




    for (uint64_t depth = (apr.y_vec.depth_min); depth <= apr.y_vec.depth_max; depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = apr.y_vec.x_num[depth];
        const unsigned int z_num_ = apr.y_vec.z_num[depth];

        const float step_size_x = pow(2, apr.y_vec.depth_max - depth);
        const float step_size_y = pow(2, apr.y_vec.depth_max - depth);
        const float step_size_z = pow(2, apr.y_vec.depth_max - depth);


#pragma omp parallel for default(shared) private(z_,x_,j_,i,k) schedule(guided) if(z_num_*x_num_ > 1000)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_ * z_ + x_;

                for (j_ = 0; j_ < apr.y_vec.data[depth][pc_offset].size(); j_++) {


                    const int y = apr.y_vec.data[depth][pc_offset][j_];

                    const float y_actual = floor((y+0.5) * step_size_y);
                    const float x_actual = floor((x_+0.5) * step_size_x);
                    const float z_actual = floor((z_+0.5) * step_size_z);

                    float var = layers[depth](y,x_,z_);

                    if(var > 0){
                        scale_parts.data[depth][pc_offset][j_] = var;
                    } else {

                        scale_parts.data[depth][pc_offset][j_] = variance_u(y_actual, x_actual, z_actual);

                    }


                }
            }
        }
    }

    debug_write(variance_u,"var");

    return scale_parts;

}


template<typename T,typename S>
ExtraPartCellData<T> get_scale_parts(APR<T>& apr,Mesh_data<S>& input_image,Proc_par& pars){

    Mesh_data<float> variance_u;

    get_variance(input_image,variance_u,pars);

    ExtraPartCellData<T> scale_parts;

    scale_parts.initialize_structure_parts(apr.particles_int);

    int z_, x_, j_, y_, i, k;

    for (uint64_t depth = (apr.y_vec.depth_min); depth <= apr.y_vec.depth_max; depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = apr.y_vec.x_num[depth];
        const unsigned int z_num_ = apr.y_vec.z_num[depth];

        const float step_size_x = pow(2, apr.y_vec.depth_max - depth);
        const float step_size_y = pow(2, apr.y_vec.depth_max - depth);
        const float step_size_z = pow(2, apr.y_vec.depth_max - depth);


#pragma omp parallel for default(shared) private(z_,x_,j_,i,k) schedule(guided) if(z_num_*x_num_ > 1000)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_ * z_ + x_;

                for (j_ = 0; j_ < apr.y_vec.data[depth][pc_offset].size(); j_++) {


                    const int y = apr.y_vec.data[depth][pc_offset][j_];

                    const float y_actual = floor((y+0.5) * step_size_y);
                    const float x_actual = floor((x_+0.5) * step_size_x);
                    const float z_actual = floor((z_+0.5) * step_size_z);

                    scale_parts.data[depth][pc_offset][j_]=variance_u(y_actual,x_actual,z_actual);


                }
            }
        }
    }



    return scale_parts;

}


template<typename T>
void auto_parameters(Mesh_data<T>& input_img,Proc_par& pars){
    //
    //  Simple automatic parameter selection for 3D APR Flouresence Images
    //


    //minimum element
    T min_val = *std::min_element(input_img.mesh.begin(),input_img.mesh.end());

    // will need to deal with grouped constant or zero sections in the image somewhere.... but lets keep it simple for now.

    std::vector<uint64_t> freq;
    unsigned int num_bins = 10000;
    freq.resize(num_bins);

    uint64_t counter = 0;
    double total=0;

    uint64_t q =0;
//#pragma omp parallel for default(shared) private(q)
    for (q = 0; q < input_img.mesh.size(); ++q) {

        if(input_img.mesh[q] < (min_val + num_bins-1)){
            freq[input_img.mesh[q]-min_val]++;
            if(input_img.mesh[q] > 0) {
                counter++;
                total += input_img.mesh[q];
            }
        }
    }

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
    bspline_filt_rec_y(histogram,lambda,tol);

    calc_inv_bspline_y(histogram);

    //calc_bspline_fd_y(Mesh_data<T>& input)

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

    uint64_t counter_p = 0;

    for(j = 1; j < (z_num-1);j++){
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

    if(pars.noise_scale > 0){
        min_snr = pars.noise_scale;
    } else {
        std::cout << "**Assuming a minimum SNR of 6" << std::endl;
    }

    std::cout << "**Assuming image has atleast 5% dark background" << std::endl;

    float Ip_th = mean + sd;

    float var_th = (img_mean/mean)*sd*min_snr;

    float var_th_max = sd*min_snr*.5;

    if(pars.I_th < 0 ){
        pars.I_th = Ip_th;
    }

    if(pars.lambda < 0){
        pars.lambda = 3.0;
    }

    if(pars.var_th < 0){
        pars.var_th = var_th;
        pars.var_th_max = var_th_max;
    } else{
        pars.var_th_max = pars.var_th*0.5;
    }


    std::cout << "I_th: " << pars.I_th << std::endl;
    std::cout << "sigma_th: " << pars.var_th << std::endl;
    std::cout << "sigma_th_max: " << pars.var_th_max << std::endl;
    std::cout << "relative error (E): " << pars.rel_error << std::endl;
    std::cout << "Lambda: " << pars.lambda << std::endl;

}



void get_variance(Mesh_data<float>& variance_u,cmdLineOptions& options){

    Proc_par pars;

    Mesh_data<uint16_t> input_image;

    load_image_tiff(input_image, options.directory + options.input);

    get_image_stats(pars, options.directory, options.stats);


    get_variance(input_image,variance_u,pars);

}


void get_apr_part_timing(Mesh_data<uint16_t >& input_image,Part_rep& part_rep,PartCellStructure<float,uint64_t>& pc_struct,AnalysisData& analysis_data){

    int interp_type = part_rep.pars.interp_type;

    // COMPUTATIONS

    Mesh_data<float> input_image_float;
    Mesh_data<float> gradient, variance;
    Mesh_data<float> interp_img;

    gradient.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);
    part_rep.initialize(input_image.y_num, input_image.x_num, input_image.z_num);

    input_image_float = input_image.to_type<float>();
    interp_img = input_image.to_type<float>();
    // After this block, input_image will be freed.

    Part_timer t;
    t.verbose_flag = false;

    // preallocate_memory
    Particle_map<float> part_map(part_rep);
    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);
    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);
    std::vector<Mesh_data<float>> down_sampled_images;

    Mesh_data<float> temp;
    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    t.start_timer("whole");

    //    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);

    part_rep.timer.start_timer("get_gradient_3D");
    get_gradient_3D(part_rep, input_image_float, gradient);
    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("get_variance_3D");
    get_variance_3D(part_rep, input_image_float, variance);
    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("get_level_3D");
    get_level_3D(variance, gradient, part_rep, part_map, temp);
    part_rep.timer.stop_timer();

    // free memory (not used anymore)
    std::vector<float>().swap(gradient.mesh);
    std::vector<float>().swap(variance.mesh);


    part_rep.timer.start_timer("sample");

    if (interp_type == 0) {
        part_map.downsample(interp_img);
        calc_median_filter(part_map.downsampled[part_map.k_max+1]);
    }

    if (interp_type == 1) {
        part_map.downsample(input_image_float);
    } else if (interp_type == 2) {
        part_map.downsample(interp_img);
    } else if (interp_type ==3){
        part_map.downsample(interp_img);
        calc_median_filter_n(input_image_float,part_map.downsampled[part_map.k_max+1]);
    }

    part_rep.timer.stop_timer();

    float num_reps = 20;


    part_rep.timer.start_timer("pushing_scheme");
    part_map.pushing_scheme(part_rep);
    part_rep.timer.stop_timer();

    Part_timer timer;

    float total_time = 0;

    for (int i = 0; i < num_reps; ++i) {
        Particle_map<float> part_map_t(part_rep);
        preallocate(part_map_t.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);

        timer.start_timer("pushing_scheme");

        part_map.pushing_scheme(part_rep);

        timer.stop_timer();

        total_time += (timer.t2 - timer.t1);
    }


    analysis_data.add_float_data("pushing_scheme_avg",total_time/num_reps);

    part_rep.timer.start_timer("construct_pcstruct");

    pc_struct.initialize_structure(part_map);

    part_rep.timer.stop_timer();

    total_time = 0;

    for (int i = 0; i < num_reps; ++i) {
        PartCellStructure<float,uint64_t> pc_struct_t;

        timer.start_timer("construct_pc");

        pc_struct_t.initialize_structure(part_map);

        timer.stop_timer();

        total_time += (timer.t2 - timer.t1);
    }

    analysis_data.add_float_data("construct_pc_avg",total_time/num_reps);


    t.stop_timer();

    //add the timer data
    //analysis_data.add_timer(part_rep.timer);
    //analysis_data.add_timer(t);


}


void get_apr_2D(Mesh_data<uint16_t >& input_image,Part_rep& part_rep,PartCellStructure<float,uint64_t>& pc_struct,AnalysisData& analysis_data){

    int interp_type = part_rep.pars.interp_type;

    // COMPUTATIONS

    Mesh_data<float> input_image_float;
    Mesh_data<float> gradient, variance;
    Mesh_data<float> interp_img;

    gradient.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);
    part_rep.initialize(input_image.y_num, input_image.x_num, input_image.z_num);

    input_image_float = input_image.to_type<float>();
    interp_img = input_image.to_type<float>();
    // After this block, input_image will be freed.

    Part_timer t;
    t.verbose_flag = false;

    // preallocate_memory
    Particle_map<float> part_map(part_rep);
    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);
    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);
    std::vector<Mesh_data<float>> down_sampled_images;

    Mesh_data<float> temp;
    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    t.start_timer("whole");

    //    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);

    part_rep.timer.start_timer("get_gradient_2D");
    get_gradient_2D(part_rep, input_image_float, gradient);
    part_rep.timer.stop_timer();

    debug_write(gradient,"gradient");

    part_rep.timer.start_timer("get_variance_2D");
    get_variance_2D(part_rep, input_image_float, variance);
    part_rep.timer.stop_timer();

    debug_write(variance,"variance");

    part_rep.timer.start_timer("get_level_2D");
    get_level_2D(variance, gradient, part_rep, part_map, temp);
    part_rep.timer.stop_timer();



    // free memory (not used anymore)
    std::vector<float>().swap(gradient.mesh);
    std::vector<float>().swap(variance.mesh);

    part_rep.timer.start_timer("pushing_scheme");
    part_map.pushing_scheme(part_rep);
    part_rep.timer.stop_timer();


    part_rep.timer.start_timer("sample");

    if (interp_type == 0) {
        part_map.downsample(interp_img);
        calc_median_filter(part_map.downsampled[part_map.k_max+1]);
    }

    if (interp_type == 1) {
        part_map.downsample(input_image_float);
    } else if (interp_type == 2) {
        part_map.downsample(interp_img);
    } else if (interp_type ==3){
        part_map.downsample(interp_img);
        calc_median_filter_n(part_map.downsampled[part_map.k_max+1],input_image_float);
    }

    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("construct_pcstruct");

    pc_struct.initialize_structure(part_map);

    part_rep.timer.stop_timer();

    t.stop_timer();

    //add the timer data
    analysis_data.add_timer(part_rep.timer);
    analysis_data.add_timer(t);


}
void mask_variance(Mesh_data<float>& variance,Mesh_data<float>& temp,APR<float>& apr){
    //
    //  Bevan Cheeseman 2018
    //
    //  Loads in a tiff image file and masks the variance according to it
    //
    //

    Mesh_data<uint16_t> mask;

    load_image_tiff(mask, apr.pars.mask_file);

    down_sample(mask,temp,
                [](float x, float y) { return std::max(x,y); },
                [](float x) { return x; });
    uint64_t i = 0;
#pragma omp parallel for default(shared) private(i)
    for ( i = 0; i < variance.mesh.size(); ++i) {

        if(temp.mesh[i]==0){
            variance.mesh[i] = 64000;
        }

    }

}


void get_apr(Mesh_data<uint16_t >& input_image,APR<float>& apr){
    //
    //  NEEDS TO BE CLEANED UP *quick hacks to get it workgin directly with the new datastructures
    //

    Part_timer t;
    t.verbose_flag = false;

    Part_timer timer;
    timer.verbose_flag = true;

    int interp_type = apr.pars.interp_type;

    // COMPUTATIONS
    Part_rep part_rep;
    part_rep.pars = apr.pars;

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

    //the part_rep needs to be removed and replaced, but leaving in for backward combatability at the moment

    part_rep.pl_map.k_max = k_max_;
    part_rep.pl_map.k_min = k_min_;

    timer.start_timer("initializing memory");

    Mesh_data<float> input_image_float;
    Mesh_data<float> gradient, variance;
    Mesh_data<float> interp_img;

    part_rep.org_dims = apr.pc_data.org_dims;

    gradient.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);

    input_image_float = input_image.to_type<float>();
    interp_img = input_image.to_type<float>();
    // After this block, input_image will be freed.

    // preallocate_memory
    Particle_map<float> part_map(apr.pc_data.org_dims,apr.pc_data.depth_min,apr.pc_data.depth_max);
    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, apr.pc_data.depth_max - 1,apr.pc_data.depth_min);


    //allocating to be half size
    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    Mesh_data<float> temp;
    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);


    timer.stop_timer();

    t.start_timer("whole");


    timer.start_timer("compute gradient magnitude");
    get_gradient_3D(part_rep, input_image_float, gradient);
    timer.stop_timer();

    timer.start_timer("calculate Local Intensity Scale");
    get_variance_3D(part_rep, input_image_float, variance);
    timer.stop_timer();

    if((interp_type == 0) | (interp_type == 2)){
        //frees the memory of the smooth image
        std::vector<float>().swap(input_image_float.mesh);
    }

    if(apr.pars.mask_file!= ""){
        //
        //  Now threshold the variance according to the mask (think this should be changed to the gradient actually)
        //
        timer.start_timer("masking variance");
        mask_variance(variance,temp,apr);
        timer.stop_timer();
    }


    timer.start_timer("construct Local Particle Set (L_n)");
    get_level_3D(variance, gradient, part_rep, part_map, temp);
    timer.stop_timer();

    // free memory (not used anymore)
    std::vector<float>().swap(gradient.mesh);
    std::vector<float>().swap(variance.mesh);

    timer.start_timer("pulling scheme (compute OVPC V_n)");
    part_map.pushing_scheme(part_rep);
    timer.stop_timer();

    timer.start_timer("sample particle intensities");

    if (interp_type == 2) {
        part_map.downsample(interp_img);
    }


    if (interp_type == 0) {
        part_map.downsample(interp_img);
        calc_median_filter(part_map.downsampled[part_map.k_max+1]);
    }

    if (interp_type == 1) {
        part_map.downsample(input_image_float);
    } else if (interp_type ==3){
        part_map.downsample(interp_img);
        //calc_median_filter_n(part_map.downsampled[part_map.k_max+1],input_image_float);
        part_map.downsampled[part_map.k_max+1]=input_image_float;
    } else if (interp_type ==4){
        part_map.closest_pixel(interp_img);
    }

    timer.stop_timer();

    timer.start_timer("construct the APR data-structure");

    //construct the pc-data
    apr.init_from_pulling_scheme(part_map.layers);

    apr.get_parts_from_img(part_map.downsampled,apr.particles_int);

    //then get the particle data

    timer.stop_timer();

    t.stop_timer();

//    //add the timer data
//    analysis_data.add_timer(timer);
//    analysis_data.add_timer(t);


}







void get_apr(Mesh_data<uint16_t >& input_image,Part_rep& part_rep,PartCellStructure<float,uint64_t>& pc_struct,AnalysisData& analysis_data){

    int interp_type = part_rep.pars.interp_type;

    // COMPUTATIONS

    Mesh_data<float> input_image_float;
    Mesh_data<float> gradient, variance;
    Mesh_data<float> interp_img;

    gradient.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);
    part_rep.initialize(input_image.y_num, input_image.x_num, input_image.z_num);

    input_image_float = input_image.to_type<float>();
    interp_img = input_image.to_type<float>();
    // After this block, input_image will be freed.

    Part_timer t;
    t.verbose_flag = false;

    // preallocate_memory
    Particle_map<float> part_map(part_rep);
    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);
    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);
    std::vector<Mesh_data<float>> down_sampled_images;

    Mesh_data<float> temp;
    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    t.start_timer("whole");

    //    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);

    part_rep.timer.start_timer("get_gradient_3D");
    get_gradient_3D(part_rep, input_image_float, gradient);
    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("get_variance_3D");
    get_variance_3D(part_rep, input_image_float, variance);
    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("get_level_3D");
    get_level_3D(variance, gradient, part_rep, part_map, temp);
    part_rep.timer.stop_timer();

    // free memory (not used anymore)
    std::vector<float>().swap(gradient.mesh);
    std::vector<float>().swap(variance.mesh);

    part_rep.timer.start_timer("pushing_scheme");
    part_map.pushing_scheme(part_rep);
    part_rep.timer.stop_timer();


    part_rep.timer.start_timer("sample");

    if (interp_type == 0) {
        part_map.downsample(interp_img);
        calc_median_filter(part_map.downsampled[part_map.k_max+1]);
    }

    if (interp_type == 1) {
        part_map.downsample(input_image_float);
    } else if (interp_type == 2) {
        part_map.downsample(interp_img);
    } else if (interp_type ==3){
        part_map.downsample(interp_img);
        //calc_median_filter_n(part_map.downsampled[part_map.k_max+1],input_image_float);
        part_map.downsampled[part_map.k_max+1]=input_image_float;
    } else if (interp_type ==4){
        part_map.closest_pixel(interp_img);
    }

    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("construct_pcstruct");

    pc_struct.initialize_structure(part_map);

    part_rep.timer.stop_timer();

    t.stop_timer();

    //add the timer data
    analysis_data.add_timer(part_rep.timer);
    analysis_data.add_timer(t);


}
void get_apr_perfect(Mesh_data<uint16_t >& input_image,Mesh_data<float>& grad_gt,Mesh_data<float>& var_gt,Part_rep& part_rep,PartCellStructure<float,uint64_t>& pc_struct,AnalysisData& analysis_data){

    int interp_type = part_rep.pars.interp_type;

    // COMPUTATIONS

    Mesh_data<float> input_image_float;
    Mesh_data<float> gradient, variance;
    Mesh_data<float> interp_img;

    gradient.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);
    part_rep.initialize(input_image.y_num, input_image.x_num, input_image.z_num);

    input_image_float = input_image.to_type<float>();
    interp_img = input_image.to_type<float>();
    // After this block, input_image will be freed.

    Part_timer t;
    t.verbose_flag = false;

    // preallocate_memory
    Particle_map<float> part_map(part_rep);
    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);
    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);
    std::vector<Mesh_data<float>> down_sampled_images;

    Mesh_data<float> temp;
    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    t.start_timer("whole");

    //    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);

    //part_rep.timer.start_timer("get_gradient_3D");
    //get_gradient_3D(part_rep, input_image_float, gradient);
    //part_rep.timer.stop_timer();

    //part_rep.timer.start_timer("get_variance_3D");
   // get_variance_3D(part_rep, input_image_float, variance);
    //part_rep.timer.stop_timer();


    down_sample(var_gt,variance,
                [](float x, float y) { return std::max(x,y); },
                [](float x) { return x; });


    part_rep.timer.start_timer("get_level_3D");
    get_level_3D(variance, grad_gt, part_rep, part_map, temp);
    part_rep.timer.stop_timer();

    // free memory (not used anymore)
    std::vector<float>().swap(gradient.mesh);
    std::vector<float>().swap(variance.mesh);

    part_rep.timer.start_timer("pushing_scheme");
    part_map.pushing_scheme(part_rep);
    part_rep.timer.stop_timer();


    part_rep.timer.start_timer("sample");

    if (interp_type == 0) {
        part_map.downsample(interp_img);
        calc_median_filter(part_map.downsampled[part_map.k_max+1]);
    }

    if (interp_type == 1) {
        part_map.downsample(input_image_float);
    } else if (interp_type == 2) {
        part_map.downsample(interp_img);
    } else if (interp_type ==3){
        part_map.downsample(interp_img);
        calc_median_filter_n(part_map.downsampled[part_map.k_max+1],input_image_float);
    } else if (interp_type ==4){
        part_map.closest_pixel(interp_img);
    }

    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("construct_pcstruct");

    pc_struct.initialize_structure(part_map);

    part_rep.timer.stop_timer();

    t.stop_timer();

    //add the timer data
    analysis_data.add_timer(part_rep.timer);
    analysis_data.add_timer(t);


}



template<typename S,typename U>
void get_apr_t(Mesh_data<S >& input_image,Part_rep& part_rep,PartCellStructure<U,uint64_t>& pc_struct,AnalysisData& analysis_data){

    int interp_type = part_rep.pars.interp_type;

    // COMPUTATIONS

    Mesh_data<S> input_image_float;
    Mesh_data<S> gradient, variance;
    Mesh_data<S> interp_img;

    gradient.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);
    part_rep.initialize(input_image.y_num, input_image.x_num, input_image.z_num);

    //input_image_float = input_image.to_type<float>();
    interp_img = input_image;
    // After this block, input_image will be freed.

    Part_timer t;
    t.verbose_flag = false;

    // preallocate_memory
    Particle_map<S> part_map(part_rep);
    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);
    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);
    std::vector<Mesh_data<S>> down_sampled_images;

    Mesh_data<S> temp;
    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    t.start_timer("whole");

    //    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);

    part_rep.timer.start_timer("get_gradient_3D");
    get_gradient_3D(part_rep, input_image, gradient);
    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("get_variance_3D");
    get_variance_3D(part_rep, input_image, variance);
    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("get_level_3D");
    get_level_3D(variance, gradient, part_rep, part_map, temp);
    part_rep.timer.stop_timer();

    // free memory (not used anymore)
    std::vector<S>().swap(gradient.mesh);
    std::vector<S>().swap(variance.mesh);

    part_rep.timer.start_timer("pushing_scheme");
    part_map.pushing_scheme(part_rep);
    part_rep.timer.stop_timer();


    part_rep.timer.start_timer("sample");

    if (interp_type == 0) {
        part_map.downsample(interp_img);
        calc_median_filter(part_map.downsampled[part_map.k_max+1]);
    }

    if (interp_type == 1) {
        part_map.downsample(input_image);
    } else if (interp_type == 2) {
        part_map.downsample(interp_img);
    } else if (interp_type ==3){
        part_map.downsample(interp_img);
        calc_median_filter_n(part_map.downsampled[part_map.k_max+1],input_image_float);
    }

    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("construct_pcstruct");

    pc_struct.initialize_structure(part_map);

    part_rep.timer.stop_timer();

    t.stop_timer();

    //add the timer data
    analysis_data.add_timer(part_rep.timer);
    analysis_data.add_timer(t);


}

void get_apr(Mesh_data<uint16_t >& input_image,Part_rep& part_rep,PartCellStructure<float,uint64_t>& pc_struct){
    //interface without analysis_data
    AnalysisData analysis_data;

    if(input_image.z_num == 1){
        if(input_image.y_num == 1){
            //1D no code yet

        } else{
            //2D
            get_apr_2D(input_image,part_rep,pc_struct,analysis_data);

        }

    } else {
        get_apr(input_image,part_rep,pc_struct,analysis_data);
    }


}

void get_apr(int argc, char **argv,PartCellStructure<float,uint64_t>& pc_struct,cmdLineOptions& options) {

    Part_rep part_rep;

    // INPUT PARSING

    options = read_command_line_options(argc, argv, part_rep.pars);

    Mesh_data<uint16_t> input_image;

    load_image_tiff(input_image, options.directory + options.input);

    if (!options.stats_file) {
        // defaults

        part_rep.pars.dy = part_rep.pars.dx = part_rep.pars.dz = 1;
        part_rep.pars.psfx = part_rep.pars.psfy = part_rep.pars.psfz = 1;
        part_rep.pars.rel_error = 0.1;
        part_rep.pars.var_th = 0;
        part_rep.pars.var_th_max = 0;

        std::cout << "Need status file" << std::endl;

        return;
    }

    get_apr(input_image,part_rep,pc_struct);

    pc_struct.pars = part_rep.pars;

}


bool get_apr(int argc, char **argv,APR<float>& apr,cmdLineOptions& options) {

    // INPUT PARSING

    options = read_command_line_options(argc, argv, apr.pars);

    Part_timer timer;
    timer.verbose_flag = true;

    timer.start_timer("read tif input image");

    Mesh_data<uint16_t> input_image;

    load_image_tiff(input_image, options.directory + options.input);

    timer.stop_timer();

    if (!options.stats_file) {
        // defaults

        apr.pars.dy = apr.pars.dx = apr.pars.dz = 1;
        apr.pars.psfx = apr.pars.psfy = apr.pars.psfz = 2;
        //apr.pars.rel_error = 0.1;
        apr.pars.var_th = 0;
        apr.pars.var_th_max = 0;

        // setting the command line options
        apr.pars.I_th = options.Ip_th;
        apr.pars.noise_scale = options.SNR_min;
        apr.pars.lambda = options.lambda;
        apr.pars.var_th = options.min_signal;

        if(options.mask_file != "") {
            apr.pars.mask_file = options.directory + options.mask_file;
        }

        if(input_image.mesh.size() == 0){
            std::cout << "Image Not Found" << std::endl;

            return false;
        }

        timer.start_timer("calculate automatic parameters");
        auto_parameters(input_image,apr.pars);
        timer.stop_timer();

        apr.pars.name = options.output;
        apr.name = options.output;

        //return false;
    }

    get_apr(input_image,apr);

    return true;
}



#endif //PARTPLAY_APR_PIPELINE_HPP
