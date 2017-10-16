#include <algorithm>
#include <iostream>

#include "resample_img.h"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"
#include "../utils.h"
#include "../../src/numerics/misc_numerics.hpp"
#include "../../src/algorithm/apr_pipeline.hpp"
#include "../../src/numerics/apr_segment.hpp"

int main(int argc, char **argv) {
    
    Part_rep part_rep;
    
    // INPUT PARSING
    
    cmdLineOptions options = read_command_line_options(argc, argv, part_rep);
    
    // COMPUTATIONS
    PartCellStructure<float,uint64_t> pc_struct;
    
    //output
    std::string file_name = options.directory + options.input;
    
    read_apr_pc_struct(pc_struct,file_name);
    
    int num_cells = pc_struct.get_number_cells();
    int num_parts = pc_struct.get_number_parts();
    
    std::cout << "Number cells: " << num_cells << std::endl;
    std::cout << "Number parts: " << num_parts << std::endl;
    
    Mesh_data<uint16_t> interp;
    
    Part_timer timer;
    
    timer.verbose_flag = true;

    pc_struct.name = options.input;

    write_apr_pc_struct(pc_struct,options.directory,pc_struct.name + "_comp");

    timer.start_timer("interp to pc");
    //creates pc interpolation mesh from the apr
    //pc_struct.interp_parts_to_pc(interp,pc_struct.part_data.particle_data);

    timer.stop_timer();

   // debug_write(interp,"interp_pc");

    std::vector<float> scale = {1,1,2};

    Mesh_data<float> smooth_img;
    timer.start_timer("smooth recon");
    //interp_parts_to_smooth(smooth_img,pc_struct.part_data.particle_data,pc_struct,scale);
    timer.stop_timer();

    //debug_write(smooth_img,"interp_smooth");

    //Mesh_data<uint16_t> comp_label;

    //Mesh_data<uint8_t> k_img;
    //interp_adapt_to_mesh(k_img,pc_struct);
    //debug_write(k_img,"depth_debug");

    //calc_cc_mesh(k_img,(uint8_t) 28,comp_label);

    //debug_write(comp_label,"k_cc");

//    Mesh_data<uint8_t> k_img;
//    //creates a depth interpoaltion from the apr
//    interp_depth_to_mesh(k_img,pc_struct);
//
//    debug_write(k_img,"k_img");
//
//    Mesh_data<uint8_t> status_img;
//    //creates a depth interpoaltion from the apr
//    interp_status_to_mesh(status_img,pc_struct );
//    debug_write(status_img,"status_img");

//
//    ParticleDataNew<float, uint64_t> part_new;
//    //flattens format to particle = cell, this is in the classic access/part paradigm
//    part_new.initialize_from_structure(pc_struct);
//
//    //generates the nieghbour structure
//    PartCellData<uint64_t> pc_data;
//    part_new.create_pc_data_new(pc_data);
//
//    pc_data.org_dims = pc_struct.org_dims;
//    part_new.access_data.org_dims = pc_struct.org_dims;
//
//    part_new.particle_data.org_dims = pc_struct.org_dims;
//
//    Mesh_data<float> interp_out;
//
//    interp_img(interp_out, pc_data, part_new, part_new.particle_data,false);
//
//    debug_write(interp_out,"interp_out_n");
//
//    Mesh_data<float> w_interp_out;
//
//    weigted_interp_img(w_interp_out, pc_data, part_new, part_new.particle_data,false);
//
//    debug_write(w_interp_out,"weighted_interp_out_n");

}


