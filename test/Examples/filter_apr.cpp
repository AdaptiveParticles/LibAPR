#include <algorithm>
#include <iostream>

#include "filter_apr.h"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellBase.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/data_structures/Tree/ParticleDataNew.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"

#include "../../test/utils.h"

#include "../../src/numerics/misc_numerics.hpp"
#include "../../src/numerics/filter_numerics.hpp"



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
    
    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    
    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
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
    
    cmdLineOptions options = read_command_line_options(argc, argv, part_rep);
    
    // APR data structure
    PartCellStructure<float,uint64_t> pc_struct;
    
    // Filename
    std::string file_name = options.directory + options.input;
    
    // Read the apr file into the part cell structure
    read_apr_pc_struct(pc_struct,file_name);
    
    ExtraPartCellData<float> filter_output;
    
    filter_output.initialize_structure_parts(pc_struct.part_data.particle_data);
    
    
  //  threshold_speed(pc_struct);
    
   // threshold_pixels(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2]);
    
    //filter y
    //convolution_filter_y(pc_struct,filter_output);
    
    //convolution_filter_y_new(pc_struct,filter_output);
    //convolution_filter_y_new(pc_struct,filter_output);
    
   // convolution_filter_pixels(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2]);
    
    convolution_filter_pixels_temp(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2]);
    //convolution_filter_pixels_off(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2]);
    //convolution_filter_pixels_random(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2]);
//    
//    
    uint64_t num_parts = pc_struct.get_number_parts();
    uint64_t dim = ceil(pow(num_parts,1.0/3.0));
//    
    //convolution_filter_pixels(pc_struct,dim,dim,dim);
    //convolution_filter_pixels_temp(pc_struct,dim,dim,dim);
    //convolution_filter_pixels_off(pc_struct,dim,dim,dim);
    convolution_filter_pixels_temp(pc_struct,dim,dim,dim);
    
  
    //convolution_filter_pixels_random(pc_struct,dim,dim,dim);
    
    //part_new.utest_structure(pc_struct,link_array);
    
    //compute_gradient(pc_struct,filter_output);
    
  //  compute_gradient(pc_struct,filter_output);
    
    //compute_gradient_new(pc_struct,filter_output);
    
    //compute_gradient(pc_struct,filter_output);
    
    //neigh_cells(pc_struct.pc_data);
    
    Mesh_data<uint16_t> filter_img;
    
    std::cout << pc_struct.get_number_parts() << std::endl;
    std::cout << pc_struct.get_number_cells() << std::endl;
    
    
    ParticleDataNew<float, uint64_t> part_new;
    
    part_new.initialize_from_structure(pc_struct);
    
    
    PartCellData<uint64_t> pc_data_new;
    part_new.create_pc_data_new(pc_data_new);
    
    //neigh_cells(pc_data_new);
    //neigh_cells_new(pc_data_new,part_new);
    
    //neigh_cells_new_random(pc_data_new,part_new,pc_struct.get_number_parts());
    
    Part_timer timer;
    timer.verbose_flag = true;
    
    timer.start_timer("interp");
    
    pc_struct.interp_parts_to_pc(filter_img,pc_struct.part_data.particle_data);
    
    timer.stop_timer();
    
    apr_filter_full(pc_struct);
    
    
   // debug_write(filter_img,"test");
   // debug_write(int_array[3],"int_array_3");
   // debug_write(int_array[5],"int_array_5");
   // debug_write(int_array[6],"int_array_6");
    
    
    //get_neigh_check(pc_struct,link_array,int_array);
    
    //utest_neigh_cells(pc_struct);
    
    //utest_neigh_parts(pc_struct);
    
    //utest_alt_part_struct(pc_struct);
    
    //utest_neigh_parts(pc_struct);
    
   // get_neigh_check2(pc_struct,link_array);
    
    Mesh_data<uint8_t> seg_img;
    
    
    //interp_depth_to_mesh(seg_img,pc_struct);
    
    //debug_write(seg_img,"k_mask");
    
}


