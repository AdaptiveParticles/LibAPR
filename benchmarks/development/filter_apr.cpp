#include <algorithm>
#include <iostream>

#include "filter_apr.h"
#include "src/data_structures/Mesh/meshclass.h"
#include "benchmarks/development/old_io/readimage.h"

#include "benchmarks/development/old_algorithm/gradient.hpp"
#include "benchmarks/development/old_structures/particle_map.hpp"
#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "benchmarks/development/Tree/ParticleDataNew.hpp"
#include "benchmarks/development/old_algorithm/level.hpp"
#include "benchmarks/development/old_io/writeimage.h"
#include "benchmarks/development/old_io/write_parts.h"
#include "benchmarks/development/old_io/partcell_io.h"

#include "test/utils.h"

#include "benchmarks/development/old_numerics/misc_numerics.hpp"
#include "benchmarks/development/old_numerics/filter_numerics.hpp"
#include "benchmarks/development/old_numerics/enhance_parts.hpp"



bool command_option_exists_filter(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option_filter(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions_filter read_command_line_options_filter(int argc, char **argv, Part_rep& part_rep){
    
    cmdLineOptions_filter result;
    
    if(argc == 1) {
        std::cerr << "Usage: \"pipeline -i inputfile -d directory [-t] [-o outputfile]\"" << std::endl;
        exit(1);
    }
    
    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-org_image"))
    {
        result.original_file = std::string(get_command_option(argv, argv + argc, "-org_image"));
    }
    
    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    
    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if(command_option_exists(argv, argv + argc, "-gt"))
    {
        result.gt = std::string(get_command_option(argv, argv + argc, "-gt"));
    }
    
    if(command_option_exists(argv, argv + argc, "-t"))
    {
        part_rep.timer.verbose_flag = true;
    }
    
    return result;
    
}

int main(int argc, char **argv) {
    
    Part_rep part_rep;
    
    // INPUT PARSING
    
    cmdLineOptions_filter options = read_command_line_options_filter(argc, argv, part_rep);
    
    // APR data structure
    PartCellStructure<float,uint64_t> pc_struct;
    
    // Filename
    std::string file_name = options.directory + options.input;
    
    // Read the apr file into the part cell structure
    read_apr_pc_struct(pc_struct,file_name);
    
    
    //////////////////////////////////
    //
    //  Different access and filter test examples
    //
    //////////////////////////////////

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    ExtraPartCellData<float> particle_data;

    part_new.create_particles_at_cell_structure(particle_data);

    //set up some new structures used in this test
    AnalysisData analysis_data;

    float num_repeats = 1;

    std::vector<float> filter = {.025f,.95f,.025f};
    std::vector<float> delta = {1.0,1.0,4.0};

    int num_tap = 1;

    ExtraPartCellData<float> smoothed_parts = adaptive_smooth(pc_data,particle_data,num_tap,filter);

    ExtraPartCellData<float> gradient_mag = adaptive_grad(pc_data,particle_data,3,delta);

    ExtraPartCellData<float> smoothed_gradient_mag = adaptive_grad(pc_data,smoothed_parts,3,delta);

    MeshData<float> output_img;
    interp_img(output_img, pc_data, part_new, smoothed_parts,true);

    debug_write(output_img,"adapt_smooth");

    interp_img(output_img, pc_data, part_new, gradient_mag,true);

    debug_write(output_img,"grad_mag");

    interp_img(output_img, pc_data, part_new, smoothed_gradient_mag,true);

    debug_write(output_img,"grad_mag_smooth");

    ExtraPartCellData<float> diff_diff = adaptive_grad(pc_data,gradient_mag,3,delta);

    interp_img(output_img, pc_data, part_new, diff_diff,true);

    debug_write(output_img,"second_dir");



    if(options.original_file != ""){
        //
        //  If there is input for the original image, perform the gradient on it
        //

        MeshData<uint16_t> input_image;

        load_image_tiff(input_image, options.original_file);

        MeshData<float> input_image_float;

        input_image_float = input_image.to_type<float>();

        // Grad magniute FD

        MeshData<float> grad;

        grad = compute_grad(input_image_float,delta);

        debug_write(grad,"input_grad");

        MeshData<float> grad_bspline;

        grad_bspline.initialize(grad.y_num,grad.x_num,grad.z_num,0);

        //Grad magnitude Smoothing bsplines

        //fit the splines using recursive filters
        float tol = 0.0001;
        float lambda = .5;

        //Y direction bspline
        bspline_filt_rec_y(input_image_float,lambda,tol);

        //Z direction bspline
        bspline_filt_rec_z(input_image_float,lambda,tol);

        //X direction bspline
        bspline_filt_rec_x(input_image_float,lambda,tol);


        calc_bspline_fd_x_y_alt(input_image_float,grad_bspline,delta[0],delta[1]);

        calc_bspline_fd_z_alt(input_image_float,grad_bspline,delta[2]);

        debug_write(grad_bspline,"input_grad_bspline");

    }


    //Get neighbours (linear)

    //particles
    //
   // particle_linear_neigh_access(pc_struct,num_repeats,analysis_data);

//    particle_linear_neigh_access(pc_struct,num_repeats,analysis_data);



//    lin_access_parts(pc_struct);
//
//    ParticleDataNew<float, uint64_t> part_new;
//
//    part_new.initialize_from_structure(pc_struct);
//
//    PartCellData<uint64_t> pc_data;
//    part_new.create_pc_data_new(pc_data);
//
//    uint64_t counter = 0;
//
//
//    for(uint64_t depth = (pc_data.depth_min);depth <= pc_data.depth_max;depth++) {
//        //loop over the resolutions of the structure
//        for(int i = 0;i < pc_data.data[depth].size();i++){
//
//            counter += pc_data.data[depth][i].size();
//        }
//
//    }
//
//    std::cout << counter << std::endl;
//    std::cout << pc_struct.get_number_parts() << std::endl;

    //particle_linear_neigh_access_alt_1(pc_struct);

    //pixels
   // pixels_linear_neigh_access(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],num_repeats,analysis_data);


    //Get neighbours (random access)

    //particle_random_access(pc_struct,analysis_data);

   // pixel_neigh_random(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],analysis_data);


    // Filtering

    uint64_t filter_offset = 10;

    //apr_filter_full(pc_struct,filter_offset,num_repeats,analysis_data);






    //new_filter_part(pc_struct,filter_offset,num_repeats,analysis_data);

    //interp_slice<float,float>(pc_struct,pc_struct.part_data.particle_data,dir,num);

    //get_slices<float>(pc_struct);

//    MeshData<uint16_t> output;
//
//    Part_timer timer;
//
//    timer.verbose_flag = true;
//
//
//    std::vector<float> filter;
//
//    filter = create_dog_filter<float>(filter_offset,1.5,3);
//
//    //filter = {-1,0,1};
//
//    ExtraPartCellData<float> filter_output;
//
//    filter_output = filter_apr_by_slice<float>(pc_struct,filter,analysis_data,num_repeats,true);
//
//    MeshData<float> input_image;
//
//    pc_struct.interp_parts_to_pc(input_image,pc_struct.part_data.particle_data);
//
//    MeshData<float> output_image;
//
//    output_image =  pixel_filter_full(input_image,filter,num_repeats,analysis_data);
//
//    for (int k = 0; k < output_image.mesh.size(); ++k) {
//        output_image.mesh[k] = 10 * fabs(output_image.mesh[k]);
//    }
//    debug_write(output_image,"img_filter_full");
//
//    MeshData<uint16_t> input_image_;
//
//    load_image_tiff(input_image_,options.gt);
//
//    input_image = input_image_.to_type<float>();
//
//    output_image =  pixel_filter_full(input_image,filter,num_repeats,analysis_data);
//
//    for (int k = 0; k < output_image.mesh.size(); ++k) {
//        output_image.mesh[k] = 10 * fabs(output_image.mesh[k]);
//    }
//    debug_write(output_image,"img_filter_org");
//
//    ExtraPartCellData<float> filter_output_mesh;
//
//    filter_output_mesh = filter_apr_input_img<float>(input_image,pc_struct,filter,analysis_data,num_repeats,true);

}


