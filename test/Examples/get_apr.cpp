
#include <algorithm>
#include <iostream>

#include "get_apr.h"
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

int main(int argc, char **argv) {

    //input parsing
    cmdLineOptions options;

    //init structure
    PartCellStructure<float,uint64_t> pc_struct;

    get_apr(argc,argv,pc_struct,options);
    
    //output
    std::string save_loc = options.output_dir;
    std::string file_name = options.output;

    Part_timer timer;

    timer.verbose_flag = true;

    timer.start_timer("writing output");
  
    write_apr_pc_struct(pc_struct,save_loc,file_name);

    timer.stop_timer();

    Mesh_data<uint8_t> k_img;
    interp_depth_to_mesh(k_img,pc_struct,0);
    debug_write(k_img,"k_debug");

    Mesh_data<uint16_t> interp_img;

    pc_struct.interp_parts_to_pc(interp_img,pc_struct.part_data.particle_data);
    debug_write(interp_img,"interp_img");

    write_apr_full_format(pc_struct,options.output_dir + "Full/",options.output);

    std::vector<float> scale = {1,1,4};

    Mesh_data<float> smooth_img;
    timer.start_timer("smooth recon");
    interp_parts_to_smooth(smooth_img,pc_struct.part_data.particle_data,pc_struct,scale);
    timer.stop_timer();

    debug_write(smooth_img,"smooth_test");
    
}


