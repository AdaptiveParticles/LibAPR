//
// Created by cheesema on 23/01/17.
//

#ifndef PARTPLAY_APR_PIPELINE_HPP
#define PARTPLAY_APR_PIPELINE_HPP

#include "../../src/data_structures/structure_parts.h"

#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"
#include "../../src/data_structures/Tree/PartCellParent.hpp"
#include "../../benchmarks/analysis/AnalysisData.hpp"
#include "../../src/data_structures/APR/APR.hpp"

struct cmdLineOptions{
    std::string gt_input = "";
    std::string output_dir = "";
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    bool stats_file = false;
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

cmdLineOptions read_command_line_options(int argc, char **argv, Part_rep& part_rep){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"pipeline -i inputfile [-t] [-s statsfile -d directory] [-o outputfile]\"" << std::endl;
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
        get_image_stats(part_rep.pars, result.directory, result.stats);
        result.stats_file = true;
    }
    if(command_option_exists(argv, argv + argc, "-t"))
    {
        part_rep.timer.verbose_flag = true;
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

    options = read_command_line_options(argc, argv, part_rep);

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

}






#endif //PARTPLAY_APR_PIPELINE_HPP
