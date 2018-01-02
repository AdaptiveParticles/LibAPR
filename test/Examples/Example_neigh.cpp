//
// Created by cheesema on 14/03/17.
//

#include <algorithm>
#include <iostream>

#include "Example_neigh.hpp"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"
#include "../../src/numerics/parent_numerics.hpp"
#include "../../src/numerics/misc_numerics.hpp"

#include "../../test/utils.h"

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
        std::cerr << "Usage: \"Example_neigh -i input_apr_file -d directory [-t] [-o outputfile]\"" << std::endl;
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


    Part_timer timer;

    timer.verbose_flag = true;

    ////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///  Original SARI (Sparse APR Random Iteration) data-structure. The Particle Cells structure stores the location information using V_n, and the particles are stored in seperate arrays and must be accessed through
    ///  the correct index. Filler and Boundary Cells have 1 particle and Seed have 8 particles
    ///
    //////////////////////////////////////////////////////////////////////////////////

    //Example 1. Looping of Particle Cells and getting neighbour, co-ordinate and status information.

    //initialize
    uint64_t node_val;

    int x_;
    int z_;
    uint64_t j_;
    uint64_t curr_key = 0;
    PartCellNeigh<uint64_t> neigh_keys;

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
    PartCellNeigh<uint64_t> neigh_cell_keys;

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


    //////////////////////////////////////////////
    //
    //  New Data-structures (SARI) (Particles at cells in datastructure) Storing V (not V_n) (Higher memory over-head, less complex access)
    //
    ////////////////////////////////////////////////

    APR<float> apr; //new datastructure
    apr.init_cells(pc_struct);

    // Example 4: SARI Particle Neighbour Access

    timer.start_timer("loop over particles and access neighbours");


    for(int i = apr.pc_data.depth_min;i <= apr.pc_data.depth_max;i++){

        const unsigned int x_num_ = apr.pc_data.x_num[i];
        const unsigned int z_num_ = apr.pc_data.z_num[i];


#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,y_coord) firstprivate(neigh_keys) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){

            curr_key = 0;

            //set the key values
            apr.pc_data.pc_key_set_z(curr_key,z_);
            apr.pc_data.pc_key_set_depth(curr_key,i);


            for(x_ = 0;x_ < x_num_;x_++){

                apr.pc_data.pc_key_set_x(curr_key,x_);

                const size_t offset_pc_data = x_num_*z_ + x_;

                //number of nodes on the level
                const size_t j_num = apr.pc_data.data[i][offset_pc_data].size();


                for(j_ = 0;j_ < j_num;j_++){

                    //this value encodes the state and neighbour locations of the particle cell
                    node_val = apr.pc_data.data[i][offset_pc_data][j_];

                    if (!(node_val&1)){
                        y_coord++; //iterate y

                        //This node represents a particle cell

                        //set the key index
                        apr.pc_data.pc_key_set_j(curr_key,j_);
                        //add the status to the key (required for the next line to have the status encoded in the key)
                        apr.pc_data.pc_key_set_status(curr_key,pc_struct.pc_data.get_status(node_val));


                        //get all the neighbours
                        // (Neighbour directions are (+y,-y,+x,-x,+z,-z)

                        apr.pc_data.get_neighs_all(curr_key,node_val,neigh_keys);

                        float curr_val = apr.particles_int.get_val(curr_key);

                        for(int face = 0;face < 6;face++) {
                            //loop over the nieghbours cells = particles
                            for (int n = 0; n < neigh_keys.neigh_face[face].size(); n++) {
                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                uint64_t neigh_key = neigh_keys.neigh_face[face][n];

                                if (neigh_key > 0) {
                                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)

                                    //Get the neighbour status, we need to access the cell
                                    uint64_t neigh_node = apr.pc_data.get_val(neigh_key);
                                    //then we can get the status from this
                                    float neigh_val = apr.particles_int.get_val(neigh_key);

                                }

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


    // Example 5. Using Partnew Datastructure (PC using V, and Particles in a concurrent array)

    timer.start_timer("using curr_level structure and part new");


        for(uint64_t depth = (apr.part_new.access_data.depth_min);depth <= apr.part_new.access_data.depth_max;depth++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = apr.part_new.access_data.x_num[depth];
            const unsigned int z_num_ = apr.part_new.access_data.z_num[depth];

            CurrentLevel<float,uint64_t> curr_level;
            curr_level.set_new_depth(depth,apr.part_new);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure

                for(x_ = 0;x_ < x_num_;x_++){

                    curr_level.set_new_xz(x_,z_,apr.part_new);

                    for(j_ = 0;j_ < curr_level.j_num;j_++){

                        bool iscell = curr_level.new_j(j_,apr.part_new);

                        if (iscell){
                            //Indicates this is a particle cell node
                            curr_level.update_cell(apr.part_new);

                            int y_ = curr_level.y;

                            //give it the structure
                            float val = curr_level.get_part(apr.part_new.particle_data);

                            (void) val;

                        } else {

                            curr_level.update_gap();

                        }


                    }
                }
            }
        }

    timer.stop_timer();

    // Example 6 (Using the SARI structure from the PAPER)
    timer.start_timer("Current Level and Pc_data structure iterate through parts");

    for(uint64_t depth = (apr.pc_data.depth_min);depth <= apr.pc_data.depth_max;depth++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = apr.pc_data.x_num[depth];
        const unsigned int z_num_ = apr.pc_data.z_num[depth];

        CurrentLevel<float,uint64_t> curr_level;

        curr_level.set_new_depth(depth,apr.pc_data);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure

            for(x_ = 0;x_ < x_num_;x_++){

                curr_level.set_new_xz(x_,z_,apr.pc_data);

                for(j_ = 0;j_ < curr_level.j_num;j_++){

                    bool iscell = curr_level.new_j(j_,apr.pc_data);

                    if (iscell){
                        //Indicates this is a particle cell node
                        curr_level.update_cell(apr.part_new);

                        int y_ = curr_level.y;

                        //give it the structure
                        float val = curr_level.get_val(apr.particles_int);

                        (void) val;

                    } else {

                        curr_level.update_gap();

                    }


                }
            }
        }
    }

    timer.stop_timer();


    // Example 7: Get particle neighbours using the current level structure and the APR.pc_data (This is the SARI datastructure of V)

    //create a dataset to store it
    ExtraPartCellData<float> neigh_sum;
    neigh_sum.initialize_structure_cells(apr.pc_data); //initialize the layout to be the same as the Particle Cells/Particles

    timer.start_timer("Current Level and Pc_data structure iterate through parts");

    for(uint64_t depth = (apr.pc_data.depth_min);depth <= apr.pc_data.depth_max;depth++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = apr.pc_data.x_num[depth];
        const unsigned int z_num_ = apr.pc_data.z_num[depth];

        CurrentLevel<float,uint64_t> curr_level(apr.pc_data);

        curr_level.set_new_depth(depth,apr.pc_data);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure

            for(x_ = 0;x_ < x_num_;x_++){

                curr_level.set_new_xz(x_,z_,apr.pc_data);

                for(j_ = 0;j_ < curr_level.j_num;j_++){

                    bool iscell = curr_level.new_j(j_,apr.pc_data);

                    if (iscell){

                        //Indicates this is a particle cell node
                        curr_level.update_cell(apr.pc_data);

                        //give it the structure
                        float val = curr_level.get_val(apr.particles_int);

                        std::vector<std::vector<float>> neigh_vals;

                        curr_level.update_and_get_neigh_all_avg(apr.particles_int,apr.pc_data,neigh_vals);

                        for (int i = 0; i < neigh_vals.size(); ++i) {
                            if (neigh_vals[i].size() > 0){
                                curr_level.get_val(neigh_sum) += neigh_vals[i][0];
                            }
                        }


                    } else {

                        curr_level.update_gap();

                    }


                }
            }
        }
    }

    timer.stop_timer();


    // Example 8. Demo Iterator

    ExtraPartCellData<float> test;
    test.initialize_structure_cells(apr.pc_data);

    timer.start_timer("APR iterator");

    for (apr.begin(); apr.end() == true ; apr.it_forward()) {
        //
        //  Demo APR iterator
        //

        //access and assignment
        apr(test)= apr(apr.particles_int);

    }

    timer.stop_timer();

    ExtraPartCellData<float> neigh_avg;
    neigh_avg.initialize_structure_cells(apr.pc_data);

    timer.start_timer("APR iterator neighbours");

    std::vector<std::vector<float>> neigh_vals;

    for (apr.begin(); apr.end() == true ; apr.it_forward()) {
        //
        //  Demo APR iterator
        //

        //
        // Neighbour definitions
        // [+y,-y,+x,-x,+z,-z]
        //  [0,1,2,3,4,5]
        //

        //update all
        apr.get_neigh_all(apr.particles_int,neigh_vals);

        float counter = 0;

        for (int i = 0; i < neigh_vals.size(); ++i) {

            for (int j = 0; j < neigh_vals[i].size(); ++j) {
                apr(neigh_avg) += neigh_vals[i][j];
                counter++;
            }
        }

        apr(neigh_avg) = apr(neigh_avg)/(counter);

    }

    timer.stop_timer();

    Mesh_data<float> output_img;

    interp_img(output_img, apr.pc_data, apr.part_new, test,true);

    debug_write(output_img,"it_test");

    interp_img(output_img, apr.pc_data, apr.part_new, neigh_avg,true);

    debug_write(output_img,"it_test_neigh");


    ExtraPartCellData<float> info;
    info.initialize_structure_cells(apr.pc_data);

    timer.start_timer("APR iterator info");

    for (apr.begin(); apr.end() == true ; apr.it_forward()) {
        //
        //  Demo APR iterator
        //

        //access and info
        apr(info)= apr.depth();

    }

    timer.stop_timer();

    interp_img(output_img, apr.pc_data, apr.part_new, info,true);

    debug_write(output_img,"it_test_info");


    //
    //  Benchmarks
    //

    AnalysisData analysis_data;

    particle_linear_neigh_access_alt_1(pc_struct);

    lin_access_parts(pc_struct);

    particle_linear_neigh_access(pc_struct,50,analysis_data);

    pixels_linear_neigh_access(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],50,analysis_data);

}


