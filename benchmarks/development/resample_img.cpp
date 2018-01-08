#include <algorithm>
#include <iostream>

#include "resample_img.h"
#include "src/data_structures/Mesh/meshclass.h"
#include "benchmarks/development/old_io/readimage.h"

#include "benchmarks/development/old_algorithm/gradient.hpp"
#include "benchmarks/development/old_structures/particle_map.hpp"
#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "benchmarks/development/old_algorithm/level.hpp"
#include "benchmarks/development/old_io/writeimage.h"
#include "benchmarks/development/old_io/write_parts.h"
#include "benchmarks/development/old_io/partcell_io.h"
#include "test/utils.h"
#include "benchmarks/development/old_numerics/misc_numerics.hpp"
#include "benchmarks/development/old_algorithm/apr_pipeline.hpp"
#include "benchmarks/development/old_numerics/apr_segment.hpp"

#include "external/blitzwave/src/Wavelet.h"
#include "external/blitzwave/src/WaveletDecomp.h"
#include "external/blitzwave/src/arrayTools.h"

#include "benchmarks/development/old_io/wavelet_comp.hpp"

using namespace std;
using namespace blitz;
using namespace bwave;

int main(int argc, char **argv) {
    
    Part_rep part_rep;
    
    // INPUT PARSING
    
    cmdLineOptions options = read_command_line_options(argc, argv, part_rep);
    
    // COMPUTATIONS
    PartCellStructure<float,uint64_t> pc_struct;
    
    //output
    std::string file_name = options.directory + options.input;

    read_apr_pc_struct(pc_struct,file_name);

    Mesh_data<uint16_t> interp;

    //creates pc interpolation mesh from the apr
    pc_struct.interp_parts_to_pc(interp,pc_struct.part_data.particle_data);

    debug_write(interp,"interp_pc_org");

    PartCellStructure<float,uint64_t> pc_struct_t = pc_struct;

    float Th = 400;
    int level = 2;
    Wavelet wl = WL_CDF_97;

    //test_wavelet(pc_struct_t,Th,level,wl);;

    //creates pc interpolation mesh from the apr
    //pc_struct_t.interp_parts_to_pc(interp,pc_struct_t.part_data.particle_data);

    //debug_write(interp,"interp_pc_wavelet");

    //

    //write_apr_pc_struct(pc_struct,options.directory,pc_struct.name  + "org");

    write_apr_wavelet(pc_struct,options.directory,pc_struct.name + "wavelet",Th,level,wl);

    write_apr_wavelet_partnew(pc_struct,options.directory,pc_struct.name + "wavelet_part",Th,wl);

    PartCellStructure<float,uint64_t> pc_struct_w;

    read_apr_wavelet(pc_struct_w,options.directory + pc_struct.name + "wavelet_pcstruct_part.h5",wl);

    pc_struct_w.interp_parts_to_pc(interp,pc_struct_w.part_data.particle_data);

    debug_write(interp,"interp_pc_wavelet_2");

//    int num_cells = pc_struct.get_number_cells();
//    int num_parts = pc_struct.get_number_parts();
//
//    std::cout << "Number cells: " << num_cells << std::endl;
//    std::cout << "Number parts: " << num_parts << std::endl;
//
//    Mesh_data<uint16_t> interp;
//
//    Part_timer timer;
//
//    timer.verbose_flag = true;
//
//    pc_struct.name = options.input;
//
   // write_apr_pc_struct(pc_struct,options.directory,pc_struct.name + "_comp");
//
    //write_apr_pc_struct_hilbert(pc_struct,options.directory,pc_struct.name + "_comp_hilbert");
//

//
    //std::vector<float> scale = {1,1,2};
//
    //Mesh_data<float> smooth_img;
//    timer.start_timer("smooth recon");
    //interp_parts_to_smooth(smooth_img,pc_struct_t.part_data.particle_data,pc_struct_t,scale);
//    timer.stop_timer();
   // debug_write(smooth_img,"smooth_recon");

//    unsigned nBits = 6;
//    unsigned nDims = 3;
//    std::vector<bitmask_t> coord = {2,5,6};
//
//    hilbert_incr_v(nBits,nDims , coord);
//
//    std::cout << coord[0] << coord[1] << coord[2] << std::endl;
//
//    hilbert_incr_v(nBits,nDims , coord);
//
//    std::cout << coord[0] << coord[1] << coord[2] << std::endl;
//
//    hilbert_incr_v(nBits,nDims , coord);
//
//    std::cout << coord[0] << coord[1] << coord[2] << std::endl;
//
//    for (int i = 0; i < 20; ++i) {
//        bitmask_t index = i;
//        bitmask_t coord[3] = {0,0,0};
//
//        hilbert_i2c(nDims,nBits, index, coord);
//
//        std::cout << coord[0] << coord[1] << coord[2] << std::endl;
//    }


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


