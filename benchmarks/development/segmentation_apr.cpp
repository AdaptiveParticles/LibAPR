#include <algorithm>
#include <iostream>

#include "segmentation_apr.h"
#include "src/data_structures/Mesh/MeshData.hpp"
#include "benchmarks/development/old_io/readimage.h"

#include "benchmarks/development/old_algorithm/gradient.hpp"
#include "benchmarks/development/old_structures/particle_map.hpp"
#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "benchmarks/development/old_algorithm/level.hpp"
#include "benchmarks/development/old_io/writeimage.h"
#include "benchmarks/development/old_io/write_parts.h"
#include "benchmarks/development/old_io/partcell_io.h"
#include "benchmarks/development/old_numerics/parent_numerics.hpp"
#include "benchmarks/development/old_numerics/misc_numerics.hpp"
#include "benchmarks/development/old_numerics/graph_cut_seg.hpp"
#include "benchmarks/development/old_numerics/apr_segment.hpp"
#include "src/numerics/ray_cast.hpp"
#include "src/data_structures/APR/APR.hpp"

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
    
    //Part Segmentation
   // MeshData<uint8_t> k_img;
    //interp_depth_to_mesh(k_img,pc_struct);
    //debug_write(k_img,"k_debug_old");

    
    AnalysisData analysis_data;
    
    ExtraPartCellData<uint16_t> seg_parts;
    
    //nuclei
    std::array<uint64_t,10> parameters_nuc = {100,2000,1,1,2,2,2,3,0,0};
    
    //nuclei
    std::array<uint64_t,10> parameters_mem = {100,2000,2,2,2,2,2,3,0,0};
    
    //calc_graph_cuts_segmentation(pc_struct, seg_parts,parameters_nuc,analysis_data);
    
    
    std::cout << "Num_parts: " << pc_struct.get_number_parts() << std::endl;

    MeshData<uint16_t> seg_mesh;

    //calc_graph_cuts_segmentation_mesh(pc_struct,seg_mesh,parameters_nuc,analysis_data);
    
    std::cout << "Num_pixels: " <<seg_mesh.mesh.size() << std::endl;
    
    float Ip_threshold = pc_struct.pars.I_th;
    float Ip_max = 10000;
    float beta = 100;
    float var_th = 0;

    std::array<float,13> parameters_new = {Ip_threshold,1,2,3,1,2,3,1,1,1.5,Ip_max,beta,var_th};

    Part_timer timer;
    timer.verbose_flag = true;
    timer.start_timer("full seg");

    calc_graph_cuts_segmentation_new(pc_struct, seg_parts,analysis_data,parameters_new);

    timer.stop_timer();

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    interp_img(seg_mesh, pc_data, part_new, seg_parts,true);

    debug_write(seg_mesh,pc_struct.name + "_gc_seg");


    proj_par proj_pars;

    proj_pars.theta_0 = -3.14;
    proj_pars.theta_final = 3.14;
    proj_pars.radius_factor = 1.1;
    proj_pars.theta_delta = 0.025;
    proj_pars.scale_z = pc_struct.pars.aniso;

    //APR<float> curr_apr(pc_struct);

    //shift_particles_from_cells(curr_apr.part_new,seg_parts);

    //ExtraPartCellData<uint16_t> seg_parts_depth = multiply_by_depth(seg_parts);

    proj_pars.name = pc_struct.name + "depth";

    //apr_perspective_raycast_depth(curr_apr.y_vec,seg_parts,seg_parts_depth,proj_pars,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);},true);

    ////////////////////////////////////
    //
    //
    //  Connected Component
    //
    //
    /////////////////////////////////////

    bool connected_comp = false;

    if(connected_comp) {

        //ExtraPartCellData<uint16_t> component_label;

        //calculate the connected component

        //calc_connected_component(pc_struct, seg_parts, component_label);

        //Now we will view the output by creating the binary image implied by the segmentation

        std::cout << "boom" << std::endl;

        MeshData<uint16_t> comp_label;

        calc_cc_mesh(seg_mesh,(uint16_t) 255,comp_label);


        //pc_struct.interp_parts_to_pc(comp_img, component_label);

        //interp_img(comp_img, pc_data, part_new, component_label,true);

        debug_write(comp_label,pc_struct.name + "_cc");


    }
    

}


