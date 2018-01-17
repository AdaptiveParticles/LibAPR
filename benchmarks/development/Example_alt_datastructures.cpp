#include <algorithm>
#include <iostream>

#include "Example_alt_datastructures.h"
#include "src/data_structures/Mesh/MeshData.hpp"
#include "benchmarks/development/old_io/readimage.h"

#include "benchmarks/development/old_algorithm/gradient.hpp"
#include "benchmarks/development/old_structures/particle_map.hpp"
#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "benchmarks/development/old_algorithm/level.hpp"
#include "benchmarks/development/old_io/writeimage.h"
#include "benchmarks/development/old_io/write_parts.h"
#include "benchmarks/development/old_io/partcell_io.h"

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
    
    // COMPUTATIONS
    PartCellStructure<float,uint64_t> pc_struct;
    
    //output
    std::string file_name = options.directory + options.input;
    
    read_apr_pc_struct(pc_struct,file_name);
    
    //initialize
    uint64_t node_val;
    
    int x_;
    int z_;
    uint64_t j_;
    uint64_t curr_key = 0;
    PartCellNeigh<uint64_t> neigh_keys;
    PartCellNeigh<uint64_t> neigh_cell_keys;
    
    int num_cells = pc_struct.get_number_cells();
    int num_parts = pc_struct.get_number_parts();
    
    std::cout << "Number cells: " << num_cells << std::endl;
    std::cout << "Number parts: " << num_parts << std::endl;
    
    Part_timer timer;

    ////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///  Original SARI (Sparse APR Random Iteration) data-structure. The Particle Cells structure stores the location information using V_n, and the particles are stored in seperate arrays and must be accessed through
    ///  the correct index. Filler and Boundary Cells have 1 particle and Seed have 8 particles
    ///
    //////////////////////////////////////////////////////////////////////////////////

    //Example 1. Looping of Particle Cells and getting neighbour, co-ordinate and status information.


    uint64_t y_coord; //keeps track of y coordinate

    timer.start_timer("Loop over cells and compare +x neighbour status and get coordinates");

    for(int i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){

        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];


#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,y_coord) firstprivate(neigh_keys) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){

            curr_key = 0;

            //set the key values
            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);


            for(x_ = 0;x_ < x_num_;x_++){

                pc_struct.pc_data.pc_key_set_x(curr_key,x_);

                const size_t offset_pc_data = x_num_*z_ + x_;

                //number of nodes on the level
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();

                uint64_t status_current;
                uint64_t x_current;
                uint64_t y_current;
                uint64_t z_current;
                uint64_t depth_current;

                uint64_t x_neigh;
                uint64_t y_neigh;
                uint64_t z_neigh;
                uint64_t depth_neigh;
                uint64_t status_neigh;

                for(j_ = 0;j_ < j_num;j_++){

                    //this value encodes the state and neighbour locations of the particle cell
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];

                    if (!(node_val&1)){
                        y_coord++; //iterate y

                        //This node represents a particle cell

                        //set the key index
                        pc_struct.pc_data.pc_key_set_j(curr_key,j_);
                        //add the status to the key (required for the next line to have the status encoded in the key)
                        pc_struct.pc_data.pc_key_set_status(curr_key,pc_struct.pc_data.get_status(node_val));

                        //get some information about the current cell
                        pc_struct.pc_data.get_coordinates_cell(y_coord,curr_key,x_current,z_current,y_current,depth_current,status_current);

                        //get all the neighbours
                        // (Neighbour directions are (+y,-y,+x,-x,+z,-z)
                        uint64_t face = 2; // +x direction

                        pc_struct.pc_data.get_neighs_face(curr_key,node_val,face,neigh_keys);

                        //loop over the nieghbours
                        for(int n = 0; n < neigh_keys.neigh_face[face].size();n++){
                            // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                            uint64_t neigh_key = neigh_keys.neigh_face[face][n];

                            if(neigh_key > 0){
                                //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                pc_struct.pc_data.get_neigh_coordinates_cell(neigh_keys,face,n,y_coord,y_neigh,x_neigh,z_neigh,depth_neigh);

                                //Get the neighbour status, we need to access the cell
                                uint64_t neigh_node = pc_struct.pc_data.get_val(neigh_key);
                                //then we can get the status from this
                                status_neigh = pc_struct.pc_data.get_status(neigh_node);
                            }

                        }


                    } else {
                        //This is a gap node

                        //Gap nodes store the next and previous coodinate
                        y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                    }

                }

            }

        }
    }

    timer.stop_timer();


    //////////////////////////
    //
    //  PARTICLES (SARI)
    //
    ////////////////////////////


    // Example 2. Particle Iteration using SARI structure accessing co-ordinates status and intensity information

    //initializations
    unsigned int p ;
    uint64_t node_val_pc,node_val_part,status,part_offset;

    timer.start_timer("Loop over particles with status intensity and co-ordinates");

    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];


#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;

            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);

            for(x_ = 0;x_ < x_num_;x_++){

                pc_struct.pc_data.pc_key_set_x(curr_key,x_);

                const size_t offset_pc_data = x_num_*z_ + x_;

                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();

                uint64_t status_current;
                uint64_t x_current;
                uint64_t y_current;
                uint64_t z_current;
                uint64_t depth_current;

                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){

                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_struct.pc_data.data[i][offset_pc_data][j_];

                    if (!(node_val_pc&1)){

                        y_coord++; //iterate y

                        //Indicates this is a particle cell node
                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];

                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);

                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);

                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);

                            // First get some details about the current part
                            pc_struct.part_data.access_data.get_coordinates_part(y_coord,curr_key,x_current,z_current,y_current,depth_current,status_current);

                            // Get the intensity of the particle
                            uint16_t curr_intensity = pc_struct.part_data.get_part(curr_key);
                            (void) curr_intensity; //force execution

                        }

                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                        y_coord = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                    }

                }

            }

        }
    }


    timer.stop_timer();


    //Example 2: Loop over particles with and access z neighbour and store in APR structure

    PartCellNeigh<uint64_t> neigh_part_keys; //datastructures for storing the neighbour information during iteration

    ExtraPartCellData<float> filter_output(pc_struct.part_data.particle_data); //structure with same layout as particle intensities and therefore can be accessed using the same keys

    timer.start_timer("Loop over parts and get -z neighbour and its intensity, and get there coordinates");

    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];


#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset) firstprivate(neigh_part_keys,neigh_cell_keys) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;

            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);


            for(x_ = 0;x_ < x_num_;x_++){

                pc_struct.pc_data.pc_key_set_x(curr_key,x_);

                const size_t offset_pc_data = x_num_*z_ + x_;

                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();

                uint64_t status_current;
                uint64_t x_current;
                uint64_t y_current;
                uint64_t z_current;
                uint64_t depth_current;

                uint64_t x_neigh;
                uint64_t y_neigh;
                uint64_t z_neigh;
                uint64_t depth_neigh;
                uint64_t status_neigh;


                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){

                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_struct.pc_data.data[i][offset_pc_data][j_];

                    if (!(node_val_pc&1)){

                        y_coord++; //iterate y

                        //Indicates this is a particle cell node
                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];

                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);

                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);

                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);

                            // First get some details about the current part (this is spatial index and depth)
                            pc_struct.part_data.access_data.get_coordinates_part(y_coord,curr_key,x_current,z_current,y_current,depth_current,status_current);

                            // Get the intensity of the particle
                            uint16_t curr_intensity = pc_struct.part_data.get_part(curr_key);
                            (void) curr_intensity; //force execution

                            //get all the neighbour particles in -z direction
                            uint64_t face = 5;
                            pc_struct.part_data.get_part_neighs_face(face,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);

                            //loop over the nieghbours
                            for(int n = 0; n < neigh_part_keys.neigh_face[face].size();n++){
                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                uint64_t neigh_part_key = neigh_part_keys.neigh_face[face][n];

                                if(neigh_part_key > 0){
                                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                    pc_struct.pc_data.get_neigh_coordinates_part(neigh_part_keys,face,n,y_coord,y_neigh,x_neigh,z_neigh,depth_neigh);

                                    //then we can get the status from this
                                    status_neigh = pc_struct.pc_data.pc_key_get_status(neigh_part_key);

                                    filter_output.get_part(curr_key) = 0.5*pc_struct.part_data.get_part(neigh_part_key);

                                }

                            }

                        }

                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                        y_coord = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                    }

                }

            }

        }
    }


    timer.stop_timer();

    
}


