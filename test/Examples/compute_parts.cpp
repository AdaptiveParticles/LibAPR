#include <algorithm>
#include <iostream>

#include "compute_parts.h"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"
#include "../../src/data_structures/Tree/ExtraPartCellData.hpp"

#include "../../src/data_structures/APR/APR.hpp"

#include <map>
#include <unordered_map>

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
    
    part_rep.timer.start_timer("Read Parts");
    // Read the apr file into the part cell structure
    read_apr_pc_struct(pc_struct,file_name);

    APR<float> apr;
    apr.init_cells(pc_struct);

    apr.init_random_access();

    bool test = apr.random_access_pc(4,3,3,3);

    Part_timer timer;

    timer.verbose_flag = true;


    ExtraPartCellData<float> neigh;
    neigh.initialize_structure_cells(apr.pc_data);

    timer.start_timer("random access");

    std::vector<float> neigh_val;

    for (apr.begin(); apr.end() == true ; apr.it_forward()) {

        float temp = 0;
        float counter = 0;

        for (int face = 0; face < 6; ++face) {
            apr.get_neigh_random(face,apr.particles_int,neigh_val);

            for (int i = 0; i < neigh_val.size(); ++i) {
                temp +=neigh_val[i];
                counter++;
            }

        }

        apr(neigh) = temp/counter;

    }

    timer.stop_timer();

    Mesh_data<float> output;
    apr.interp_img(output,neigh);
    debug_write(output,"test_random");

    timer.start_timer("normal access");


    for (apr.begin(); apr.end() == true ; apr.it_forward()) {
        //
        //  Demo APR iterator
        //

        apr(neigh) = apr.type();

    }

    timer.stop_timer();

    APR_iterator<float> apr_it;
    uint64_t part;
    apr.init_by_part_iteration(apr_it);

    std::vector<std::vector<float>> neigh_vals;

    ExtraPartCellData<float> neigh_avg;
    neigh_avg.initialize_structure_cells(apr.pc_data);

    timer.start_timer("Parrelell neighbour iterator loop");

#pragma omp parallel for schedule(static) private(part) firstprivate(apr_it,neigh_vals)
    for (part = 0; part < apr.num_parts_total; ++part) {
        apr_it.set_part(part);

        apr_it.get_neigh_all(apr.particles_int,neigh_vals);

        float counter = 0;
        float temp = 0;

        for (int i = 0; i < neigh_vals.size(); ++i) {

            for (int j = 0; j < neigh_vals[i].size(); ++j) {
                temp += neigh_vals[i][j];
                counter++;
            }
        }

        apr_it(neigh_avg) = temp/(counter);

    }

    timer.stop_timer();

    std::cout << "end" << std::endl;

    apr.interp_img(output,neigh_avg);
    debug_write(output,"test_par");

//    part_rep.timer.stop_timer();
//
//    //initialize variables required
//    uint64_t node_val_pc; // node variable encoding neighbour and cell information
//    uint64_t node_val_part; // node variable encoding part offset status information
//    int x_; // iteration variables
//    int z_; // iteration variables
//    uint64_t j_; // index variable
//    uint64_t curr_key = 0; // key used for accessing and particles and cells
//    PartCellNeigh<uint64_t> neigh_part_keys; // data structure for holding particle or cell neighbours
//    PartCellNeigh<uint64_t> neigh_cell_keys;
//    //
//    // Extra variables required
//    //
//
//    uint64_t status=0;
//    uint64_t part_offset=0;
//    uint64_t p;
//
//
//    int num_cells = pc_struct.get_number_cells();
//    int num_parts = pc_struct.get_number_parts();
//
//    //
//    //
//    //  Example 1:
//    //
//    //
//    //  Get particle neighbour loop
//    //
//    //
//
//    part_rep.timer.start_timer("Loop over parts and get neighbours");
//
//    for(int i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
//        //loop over the resolutions of the structure
//        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
//        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
//
//
//#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset) firstprivate(neigh_part_keys,neigh_cell_keys) if(z_num_*x_num_ > 100)
//        for(z_ = 0;z_ < z_num_;z_++){
//            //both z and x are explicitly accessed in the structure
//            curr_key = 0;
//
//            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
//            pc_struct.pc_data.pc_key_set_depth(curr_key,i);
//
//
//            for(x_ = 0;x_ < x_num_;x_++){
//
//                pc_struct.pc_data.pc_key_set_x(curr_key,x_);
//
//                const size_t offset_pc_data = x_num_*z_ + x_;
//
//                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
//
//                //the y direction loop however is sparse, and must be accessed accordinagly
//                for(j_ = 0;j_ < j_num;j_++){
//
//                    //particle cell node value, used here as it is requried for getting the particle neighbours
//                    node_val_pc = pc_struct.pc_data.data[i][offset_pc_data][j_];
//
//                    if (!(node_val_pc&1)){
//                        //Indicates this is a particle cell node
//                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
//
//                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
//
//                        status = pc_struct.part_data.access_node_get_status(node_val_part);
//                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
//
//                        //loop over the particles
//                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
//                            //first set the particle index value in the particle_data array (stores the intensities)
//                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
//                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
//                            pc_struct.part_data.get_part_neighs_all(p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
//
//                        }
//
//                    } else {
//                        // Inidicates this is not a particle cell node, and is a gap node
//                    }
//
//                }
//
//            }
//
//        }
//    }
//
//
//    part_rep.timer.stop_timer();
//
//    uint64_t y_coord; // y coordinate needs to be tracked and is not explicitly stored in the structure
//
//    //
//    //
//    //  Example 2:
//    //
//    //
//    //  Compute Filter
//    //
//    //
//
//
//    ExtraPartCellData<float> filter_output(pc_struct.part_data.particle_data);
//
//    part_rep.timer.start_timer("Loop over parts and get -z neighbour and its intensity, and get there coordinates");
//
//    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
//        //loop over the resolutions of the structure
//        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
//        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
//
//
//#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset) firstprivate(neigh_part_keys,neigh_cell_keys) if(z_num_*x_num_ > 100)
//        for(z_ = 0;z_ < z_num_;z_++){
//            //both z and x are explicitly accessed in the structure
//            curr_key = 0;
//
//            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
//            pc_struct.pc_data.pc_key_set_depth(curr_key,i);
//
//
//            for(x_ = 0;x_ < x_num_;x_++){
//
//                pc_struct.pc_data.pc_key_set_x(curr_key,x_);
//
//                const size_t offset_pc_data = x_num_*z_ + x_;
//
//                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
//
//                uint64_t status_current;
//                uint64_t x_current;
//                uint64_t y_current;
//                uint64_t z_current;
//                uint64_t depth_current;
//
//                uint64_t x_neigh;
//                uint64_t y_neigh;
//                uint64_t z_neigh;
//                uint64_t depth_neigh;
//                uint64_t status_neigh;
//
//
//                //the y direction loop however is sparse, and must be accessed accordinagly
//                for(j_ = 0;j_ < j_num;j_++){
//
//                    //particle cell node value, used here as it is requried for getting the particle neighbours
//                    node_val_pc = pc_struct.pc_data.data[i][offset_pc_data][j_];
//
//                    if (!(node_val_pc&1)){
//                        //Indicates this is a particle cell node
//                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
//
//                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
//
//                        status = pc_struct.part_data.access_node_get_status(node_val_part);
//                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
//
//                        //loop over the particles
//                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
//                            //first set the particle index value in the particle_data array (stores the intensities)
//                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
//
//                            // First get some details about the current part
//                            pc_struct.part_data.access_data.get_coordinates_part(y_coord,curr_key,x_current,z_current,y_current,depth_current,status_current);
//
//                            // Get the intensity of the particle
//                            uint16_t curr_intensity = pc_struct.part_data.get_part(curr_key);
//                            (void) curr_intensity; //force execution
//
//                            //get all the neighbour particles in -z direction
//                            uint64_t face = 5;
//                            pc_struct.part_data.get_part_neighs_face(face,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
//
//                            //loop over the nieghbours
//                            for(int n = 0; n < neigh_part_keys.neigh_face[face].size();n++){
//                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
//                                uint64_t neigh_part_key = neigh_part_keys.neigh_face[face][n];
//
//                                if(neigh_part_key > 0){
//                                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
//                                    pc_struct.pc_data.get_neigh_coordinates_part(neigh_part_keys,face,n,y_coord,y_neigh,x_neigh,z_neigh,depth_neigh);
//
//                                    //then we can get the status from this
//                                    status_neigh = pc_struct.pc_data.pc_key_get_status(neigh_part_key);
//
//                                    filter_output.get_part(curr_key) = 0.5*pc_struct.part_data.get_part(neigh_part_key);
//
//                                }
//
//                            }
//
//                        }
//
//                    } else {
//                        // Inidicates this is not a particle cell node, and is a gap node
//                    }
//
//                }
//
//            }
//
//        }
//    }
//
//
//    part_rep.timer.stop_timer();
    
    

    
}


