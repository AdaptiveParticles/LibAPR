////////////////////////
//
//  Mateusz Susik 2016
//
//  Utility functions for the tests
//
////////////////////////

#include "utils.h"

#include <algorithm>
#include <functional>
#include <iostream>

#include "../src/io/readimage.h"
#include "../src/io/write_parts.h"
#include "../src/io/read_parts.h"

bool compare_two_images(const Mesh_data<uint16_t>& in_memory, std::string filename) {

    /* Returns true iff two images are the same with tolerance of 1 per pixel. */

    Mesh_data<uint16_t > input_image;

    load_image_tiff(input_image, filename);

    auto it2 = input_image.mesh.begin();
    for(auto it1 = in_memory.mesh.begin(); it1 != in_memory.mesh.end(); it1++, it2++)
    {
        // 60000 is the threshold introduced in variance computations. When a value reaches zero when mean is
        // computed, it is changed to 60000 afterwards. It is caused by the set of parameters in test case.
        if(std::abs(*it1 - *it2) > 1 && std::abs(*it1 - *it2) != 60000) {
            std::cout << std::distance(it1, in_memory.mesh.begin()) << " " << *it1 << " " << *it2 << std::endl;
            return false;
        }
    }
    return true;

}

bool compare_two_ks(const Particle_map<float>& in_memory, std::string filename) {

    for (int k = in_memory.k_min;k <= in_memory.k_max;k++) {

        Mesh_data<uint8_t > to_compare;

        // in_memory.layers[k]
        load_image_tiff(to_compare, filename + "_" + std::to_string(k) + ".tif");

        auto it2 = to_compare.mesh.begin();
        for(auto it1 = in_memory.layers[k].mesh.begin();
                 it1 != in_memory.layers[k].mesh.end(); it1++, it2++)
        {
            if(*it1 != *it2) {
                std::cout << std::distance(it1, in_memory.layers[k].mesh.begin()) <<
                             " " << (int)*it1 << " " << (int)*it2 << std::endl;
                return false;
            }
        }
    }

    return true;
}

bool compare_part_rep_with_particle_map(const Particle_map<float>& in_memory, std::string filename) {
    Part_rep p_rep;
    read_parts_from_full_hdf5(p_rep, filename);


    // Check

    for(int i = 0; i < p_rep.status.data.size(); i++) {
        //count Intensity as well

        if(true) {

            if (p_rep.status.data[i] != EMPTY) {


                int x = p_rep.pl_map.cells[i].x;
                int y = p_rep.pl_map.cells[i].y;
                int z = p_rep.pl_map.cells[i].z;
                int k = p_rep.pl_map.cells[i].k;

                int x_num = in_memory.layers[k].x_num;
                int y_num = in_memory.layers[k].y_num;

                if (x <= p_rep.org_dims[1] / 2 &&
                    y <= p_rep.org_dims[0] / 2 &&
                    z <= p_rep.org_dims[2] / 2) {
                    // add if it is in domain
                    if (p_rep.status.data[i] == 2) {

                        if(in_memory.layers[k].mesh[(z-1) * x_num * y_num + (x-1) * y_num + y - 1] != TAKENSTATUS)
                        {
                            std::cout << "Different status: INITIALIZED" << std::endl;
                            return false;
                        }




                    } else if (p_rep.status.data[i] >= 4) {
                        //check if has the same status

                        if(p_rep.status.data[i] == 4 &&
                            in_memory.layers[k].mesh[(z-1) * x_num * y_num + (x-1) * y_num + y - 1] != NEIGHBOURSTATUS)
                        {
                            std::cout << "Different status: NEIGHBOUR " << std::endl;
                            return false;
                        }

                        if(p_rep.status.data[i] == 5 &&
                           in_memory.layers[k].mesh[(z-1) * x_num * y_num + (x-1) * y_num + y - 1] != SLOPESTATUS)
                        {
                            std::cout << "Different status: SLOPE" << (int)z << " " <<
                                         (int)x << " " << (int)y << std::endl;
                            //return false;
                        }

                    }

                }
            }
        }
    }


    return true;
}

Mesh_data<uint16_t> create_random_test_example(unsigned int size_y, unsigned int size_x,
                                                unsigned int size_z, unsigned int seed) {
    // creates the input image of a given size with given seed
    // uses ranlux48 random number generator
    // the seed used in 2016 for generation was 5489u

    std::ranlux48 generator(seed);
    std::normal_distribution<float> distribution(1000, 250);

    Mesh_data<uint16_t> test_example(size_y, size_x, size_z);

    std::generate(test_example.mesh.begin(), test_example.mesh.end(),
                  // partial application of generator and distribution to get_random_number function
                  std::bind(get_random_number, generator, distribution));

    return test_example;

}

Mesh_data<uint16_t> generate_random_ktest_example(unsigned int size_y, unsigned int size_x,
                                                  unsigned int size_z, unsigned int seed,
                                                  float mean_fraction, float sd_fraction) {

    // creates the input image of a given size with given seed
    // the image should be used as a source of benchmarking for the get_k step
    // dx, dy and dz should all be set to 1, rel_error to 1000
    // the seed used in 2016 for generation was 5489u

    std::ranlux48 generator(seed);

    int max_dim = std::max(size_x, std::max(size_y, size_z));
    float k_max = ceil(M_LOG2E*log(max_dim)) - 1;

    std::normal_distribution<float> distribution(k_max * mean_fraction, k_max * sd_fraction);

    Mesh_data<uint16_t> test_example(size_y, size_x, size_z);

#pragma omp parallel for default(shared)
    for(int i = 0; i < test_example.mesh.size(); i++){
        test_example.mesh[i] = get_random_number_k(generator, distribution, k_max);
    }

    std::generate(test_example.mesh.begin(), test_example.mesh.end(),
                  // partial application of generator and distribution to get_random_number function
                  std::bind(get_random_number_k, generator, distribution, k_max));

    return test_example;

}

uint16_t get_random_number(std::ranlux48& generator, std::normal_distribution<float>& distribution){

    float val = distribution(generator);
    //there should be no values below zero.
    return val < 0 ? 1 : val;

}

uint16_t get_random_number_k(std::ranlux48& generator,
                             std::normal_distribution<float>& distribution, float k_max){

    float val = distribution(generator);
    //there should be no values below zero.
    return std::max(K_BENCHMARK_REL_ERROR * pow(2, val - k_max), 0.01);

}

std::string get_source_directory(){
    // returns path to the directory where utils.cpp is stored

    std::string tests_directory = std::string(__FILE__);
    tests_directory = tests_directory.substr(0, tests_directory.find_last_of("\\/") + 1);

    return tests_directory;
}


bool compare_sparse_rep_with_part_map(const Particle_map<float>& part_map,PartCellStructure<float,uint64_t>& pc_struct,bool status_flag){
    //
    //  Compares the sparse representation with the particle map original data used to generate it
    //
    
    //initialize
    uint64_t node_val;
    uint64_t y_coord;
    int x_;
    int z_;
    uint64_t y_;
    uint64_t j_;
    uint64_t status;
    uint64_t status_org;
    
    
    uint64_t type;
    uint64_t yp_index;
    uint64_t yp_depth;
    uint64_t ym_index;
    uint64_t ym_depth;
    uint64_t next_coord;
    uint64_t prev_coord;
    
    uint64_t xp_index;
    uint64_t xp_depth;
    uint64_t zp_index;
    uint64_t zp_depth;
    uint64_t xm_index;
    uint64_t xm_depth;
    uint64_t zm_index;
    uint64_t zm_depth;
    
    std::cout << "Start Status Test" << std::endl;
    
    bool pass_test = true;
    
    //basic tests of status and coordinates
    
    for(int i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num = pc_struct.pc_data.x_num[i];
        const unsigned int z_num = pc_struct.pc_data.z_num[i];
        
        
        for(z_ = 0;z_ < z_num;z_++){
            
            for(x_ = 0;x_ < x_num;x_++){
                
                const size_t offset_pc_data = x_num*z_ + x_;
                y_coord = 0;
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                const size_t offset_part_map_data_0 = part_map.downsampled[i].y_num*part_map.downsampled[i].x_num*z_ + part_map.downsampled[i].y_num*x_;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (node_val&1){
                        //get the index gap node
                        type = (node_val & TYPE_MASK) >> TYPE_SHIFT;
                        yp_index = (node_val & YP_INDEX_MASK) >> YP_INDEX_SHIFT;
                        yp_depth = (node_val & YP_DEPTH_MASK) >> YP_DEPTH_SHIFT;
                        
                        ym_index = (node_val & YM_INDEX_MASK) >> YM_INDEX_SHIFT;
                        ym_depth = (node_val & YM_DEPTH_MASK) >> YM_DEPTH_SHIFT;
                        
                        next_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        prev_coord = (node_val & PREV_COORD_MASK) >> PREV_COORD_SHIFT;
                        
                        y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--;
                        
                        
                    } else {
                        //normal node
                        y_coord++;
                        
                        type = (node_val & TYPE_MASK) >> TYPE_SHIFT;
                        xp_index = (node_val & XP_INDEX_MASK) >> XP_INDEX_SHIFT;
                        xp_depth = (node_val & XP_DEPTH_MASK) >> XP_DEPTH_SHIFT;
                        zp_index = (node_val & ZP_INDEX_MASK) >> ZP_INDEX_SHIFT;
                        zp_depth = (node_val & ZP_DEPTH_MASK) >> ZP_DEPTH_SHIFT;
                        xm_index = (node_val & XM_INDEX_MASK) >> XM_INDEX_SHIFT;
                        xm_depth = (node_val & XM_DEPTH_MASK) >> XM_DEPTH_SHIFT;
                        zm_index = (node_val & ZM_INDEX_MASK) >> ZM_INDEX_SHIFT;
                        zm_depth = (node_val & ZM_DEPTH_MASK) >> ZM_DEPTH_SHIFT;
                        
                        //get and check status
                        status = (node_val & STATUS_MASK) >> STATUS_SHIFT;
                        status_org = part_map.layers[i].mesh[offset_part_map_data_0 + y_coord];
                        
                        if (status_flag == true){
                            //set the status (using old definitions of status)
                            switch(status_org){
                                case TAKENSTATUS:
                                {
                                    if(status != SEED){
                                        std::cout << "STATUS SEED BUG" << std::endl;
                                        pass_test = false;
                                    }
                                    break;
                                }
                                case NEIGHBOURSTATUS:
                                {
                                    if(status != BOUNDARY){
                                        std::cout << "STATUS BOUNDARY BUG" << std::endl;
                                        pass_test = false;
                                    }
                                    break;
                                }
                                case SLOPESTATUS:
                                {
                                    if(status != FILLER){
                                        std::cout << "STATUS FILLER BUG" << std::endl;
                                        pass_test = false;
                                    }
                                    
                                    break;
                                }
                                    
                            }
                        } else {
                            //using new
                            switch(status_org){
                                case SEED:
                                {
                                    if(status != SEED){
                                        std::cout << "STATUS SEED BUG" << std::endl;
                                        pass_test = false;
                                    }
                                    break;
                                }
                                case BOUNDARY:
                                {
                                    if(status != BOUNDARY){
                                        std::cout << "STATUS BOUNDARY BUG" << std::endl;
                                        pass_test = false;
                                    }
                                    break;
                                }
                                case FILLER:
                                {
                                    if(status != FILLER){
                                        std::cout << "STATUS FILLER BUG" << std::endl;
                                        pass_test = false;
                                    }
                                    
                                    break;
                                }
                                    
                            }
                            
                            
                        }
                    
                    }
                }
            
            }
        }
    
    }

    std::cout << "Finished Status Test" << std::endl;
    
    return pass_test;
    
    
    
}


bool compare_sparse_rep_neighcell_with_part_map(const Particle_map<float>& part_map,PartCellStructure<float,uint64_t>& pc_struct){
    //
    //
    //  Checks the
    //
    //
    //
    
    //initialize
    uint64_t node_val;
    uint64_t y_coord;
    int x_;
    int z_;
    uint64_t y_;
    uint64_t j_;
    uint64_t status;
    uint64_t status_org;
    
    
    uint64_t type;
    uint64_t yp_index;
    uint64_t yp_depth;
    uint64_t ym_index;
    uint64_t ym_depth;
    uint64_t next_coord;
    uint64_t prev_coord;
    
    uint64_t xp_index;
    uint64_t xp_depth;
    uint64_t zp_index;
    uint64_t zp_depth;
    uint64_t xm_index;
    uint64_t xm_depth;
    uint64_t zm_index;
    uint64_t zm_depth;
    
    bool pass_test = true;
    
    
    //Neighbour Routine Checking
    
    for(int i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num = pc_struct.pc_data.x_num[i];
        const unsigned int z_num = pc_struct.pc_data.z_num[i];
        
        
        for(z_ = 0;z_ < z_num;z_++){
            
            for(x_ = 0;x_ < x_num;x_++){
                
                const size_t offset_pc_data = x_num*z_ + x_;
                y_coord = 0;
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                const size_t offset_part_map_data_0 = part_map.downsampled[i].y_num*part_map.downsampled[i].x_num*z_ + part_map.downsampled[i].y_num*x_;
                
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (node_val&1){
                        //get the index gap node
                        type = (node_val & TYPE_MASK) >> TYPE_SHIFT;
                        yp_index = (node_val & YP_INDEX_MASK) >> YP_INDEX_SHIFT;
                        yp_depth = (node_val & YP_DEPTH_MASK) >> YP_DEPTH_SHIFT;
                        
                        ym_index = (node_val & YM_INDEX_MASK) >> YM_INDEX_SHIFT;
                        ym_depth = (node_val & YM_DEPTH_MASK) >> YM_DEPTH_SHIFT;
                        
                        next_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        prev_coord = (node_val & PREV_COORD_MASK) >> PREV_COORD_SHIFT;
                        
                        
                        y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        
                        y_coord--;
                        
                        
                        
                    } else {
                        //normal node
                        y_coord++;
                        
                        type = (node_val & TYPE_MASK) >> TYPE_SHIFT;
                        xp_index = (node_val & XP_INDEX_MASK) >> XP_INDEX_SHIFT;
                        xp_depth = (node_val & XP_DEPTH_MASK) >> XP_DEPTH_SHIFT;
                        zp_index = (node_val & ZP_INDEX_MASK) >> ZP_INDEX_SHIFT;
                        zp_depth = (node_val & ZP_DEPTH_MASK) >> ZP_DEPTH_SHIFT;
                        xm_index = (node_val & XM_INDEX_MASK) >> XM_INDEX_SHIFT;
                        xm_depth = (node_val & XM_DEPTH_MASK) >> XM_DEPTH_SHIFT;
                        zm_index = (node_val & ZM_INDEX_MASK) >> ZM_INDEX_SHIFT;
                        zm_depth = (node_val & ZM_DEPTH_MASK) >> ZM_DEPTH_SHIFT;
                        
                        //get and check status
                        status = (node_val & STATUS_MASK) >> STATUS_SHIFT;
                        status_org = part_map.layers[i].mesh[offset_part_map_data_0 + y_coord];
                        
                        
                        
                        //Check the x and z nieghbours, do they exist?
                        for(int face = 0;face < 6;face++){
                            
                            uint64_t x_n = 0;
                            uint64_t z_n = 0;
                            uint64_t y_n = 0;
                            uint64_t depth = 0;
                            uint64_t j_n = 0;
                            uint64_t status_n = 1;
                            uint64_t node_n = 0;
                            
                            std::vector<uint64_t> neigh_keys;
                            PartCellNeigh<uint64_t> neigh_keys_;
                            
                            uint64_t curr_key = 0;
                            curr_key |= ((uint64_t)i) << PC_KEY_DEPTH_SHIFT;
                            curr_key |= z_ << PC_KEY_Z_SHIFT;
                            curr_key |= x_ << PC_KEY_X_SHIFT;
                            curr_key |= j_ << PC_KEY_J_SHIFT;
                            
                            pc_struct.pc_data.get_neighs_face(curr_key,node_val,face,neigh_keys_);
                            neigh_keys = neigh_keys_.neigh_face[face];
                            
                            if (neigh_keys.size() > 0){
                                depth = (neigh_keys[0] & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
                                
                                if(i == depth){
                                    y_n = y_coord + pc_struct.pc_data.von_neumann_y_cells[face];
                                } else if (depth > i){
                                    //neighbours are on layer down (4)
                                    y_n = (y_coord + pc_struct.pc_data.von_neumann_y_cells[face])*2 + (pc_struct.pc_data.von_neumann_y_cells[face] < 0);
                                } else {
                                    //neighbour is parent
                                    y_n =  (y_coord + pc_struct.pc_data.von_neumann_y_cells[face])/2;
                                }
                                
                            } else {
                                //check that it is on a boundary and should have no neighbours
                                
                                
                            }
                            
                            int y_org = y_n;
                            
                            for(int n = 0;n < neigh_keys.size();n++){
                                
                                x_n = (neigh_keys[n] & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
                                z_n = (neigh_keys[n] & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
                                j_n = (neigh_keys[n] & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
                                depth = (neigh_keys[n] & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
                                
                                if ((n == 1) | (n == 3)){
                                    y_n = y_n + pc_struct.pc_data.von_neumann_y_cells[pc_struct.pc_data.neigh_child_dir[face][n-1]];
                                } else if (n ==2){
                                    y_n = y_org + pc_struct.pc_data.von_neumann_y_cells[pc_struct.pc_data.neigh_child_dir[face][n-1]];
                                }
                                int dir = pc_struct.pc_data.neigh_child_dir[face][n-1];
                                int shift = pc_struct.pc_data.von_neumann_y_cells[pc_struct.pc_data.neigh_child_dir[face][n-1]];
                                
                                if (neigh_keys[n] > 0){
                                    
                                    //calculate y so you can check back in the original structure
                                    const size_t offset_pc_data_loc = pc_struct.pc_data.x_num[depth]*z_n + x_n;
                                    node_n = pc_struct.pc_data.data[depth][offset_pc_data_loc][j_n];
                                    const size_t offset_part_map = part_map.downsampled[depth].y_num*part_map.downsampled[depth].x_num*z_n + part_map.downsampled[depth].y_num*x_n;
                                    status_n = part_map.layers[depth].mesh[offset_part_map + y_n];
                                    
                                    if((status_n> 0) & (status_n < 8)){
                                        //fine
                                    } else {
                                        std::cout << "NEIGHBOUR LEVEL BUG" << std::endl;
                                        pass_test = false;
                                    }
                                    
                                    if (node_n&1){
                                        //points to gap node
                                        std::cout << "INDEX BUG" << std::endl;
                                        pass_test = false;
                                    } else {
                                        //points to real node, correct
                                    }
                                }
                            }
                            
                        }
                        
                        
                    }
                }
                
            }
        }
        
    }
    
    std::cout << "Finished Neigh Cell test" << std::endl;
    
    
    
    
    return pass_test;
    
}

bool compare_sparse_rep_neighpart_with_part_map(const Particle_map<float>& part_map,PartCellStructure<float,uint64_t>& pc_struct){
    //
    //
    //  Tests the particle sampling and particle neighbours;
    //
    //  Bevan Cheeseman 2016
    //
    //
    
    
    //initialize
    uint64_t node_val;
    uint64_t y_coord;
    int x_;
    int z_;
    uint64_t y_;
    uint64_t j_;
    uint64_t status;
    uint64_t status_org;
    
    
    bool pass_test = true;
    
    //Neighbour Routine Checking
    
    for(int i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num = pc_struct.pc_data.x_num[i];
        const unsigned int z_num = pc_struct.pc_data.z_num[i];
        
        
        for(z_ = 0;z_ < z_num;z_++){
            
            for(x_ = 0;x_ < x_num;x_++){
                
                const size_t offset_pc_data = x_num*z_ + x_;
                y_coord = 0;
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                const size_t offset_part_map_data_0 = part_map.downsampled[i].y_num*part_map.downsampled[i].x_num*z_ + part_map.downsampled[i].y_num*x_;
                
                
                for(j_ = 0;j_ < j_num;j_++){
           
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    uint64_t node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if (node_val&1){
                        
                        //get the index gap node
                        y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        y_coord--;
                        
                        
                    } else {
                        //normal node
                        y_coord++;
                        
                        
                        //get and check status
                        status = (node_val & STATUS_MASK) >> STATUS_SHIFT;
                        status_org = part_map.layers[i].mesh[offset_part_map_data_0 + y_coord];
                        
                        
                        //get the index gap node
                        
                        uint64_t curr_key = 0;
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        pc_struct.part_data.access_data.pc_key_set_z(curr_key,z_);
                        pc_struct.part_data.access_data.pc_key_set_x(curr_key,x_);
                        pc_struct.part_data.access_data.pc_key_set_depth(curr_key,i);
                        
                        PartCellNeigh<uint64_t> neigh_keys;
                        PartCellNeigh<uint64_t> neigh_cell_keys;
                        
                        
                        //neigh_keys.resize(0);
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        uint64_t part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        //loop over the particles
                        for(int p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            pc_struct.part_data.access_data.pc_key_set_partnum(curr_key,p);
                            pc_struct.part_data.get_part_neighs_all(p,node_val,curr_key,status,part_offset,neigh_cell_keys,neigh_keys,pc_struct.pc_data);
                            
                            //First check your own intensity;
                            uint16_t own_int = pc_struct.part_data.get_part(curr_key);
                            
                            if(status == SEED){
                                
                                uint64_t curr_depth = pc_struct.pc_data.pc_key_get_depth(curr_key) + 1;
                                
                                uint64_t part_num = pc_struct.pc_data.pc_key_get_partnum(curr_key);
                                
                                uint64_t curr_x = pc_struct.pc_data.pc_key_get_x(curr_key)*2 + pc_struct.pc_data.seed_part_x[part_num];
                                uint64_t curr_z = pc_struct.pc_data.pc_key_get_z(curr_key)*2 + pc_struct.pc_data.seed_part_z[part_num];
                                uint64_t curr_y = y_coord*2 + pc_struct.pc_data.seed_part_y[part_num];
                                
                                curr_x = std::min(curr_x,(uint64_t)(part_map.downsampled[curr_depth].x_num-1));
                                curr_z = std::min(curr_z,(uint64_t)(part_map.downsampled[curr_depth].z_num-1));
                                curr_y = std::min(curr_y,(uint64_t)(part_map.downsampled[curr_depth].y_num-1));
                                
                                const size_t offset_part_map = part_map.downsampled[curr_depth].y_num*part_map.downsampled[curr_depth].x_num*curr_z + part_map.downsampled[curr_depth].y_num*curr_x;
                                uint16_t corr_val = (offset_part_map + curr_y);
                                
                                if(own_int == corr_val){
                                    //correct value
                                } else {
                                    
                                    
                                    
                                    std::cout << "Particle Intensity Error" << std::endl;
                                    pass_test = false;
                                }
                                
                                
                            } else {
                                
                                uint64_t curr_depth = pc_struct.pc_data.pc_key_get_depth(curr_key);
                                
                                
                                
                                uint64_t curr_x = pc_struct.pc_data.pc_key_get_x(curr_key);
                                uint64_t curr_z = pc_struct.pc_data.pc_key_get_z(curr_key);
                                uint64_t curr_y = y_coord;
                                
                                const size_t offset_part_map = part_map.downsampled[curr_depth].y_num*part_map.downsampled[curr_depth].x_num*curr_z + part_map.downsampled[curr_depth].y_num*curr_x;
                                uint16_t corr_val = (offset_part_map + curr_y);
                                
                                if(own_int == corr_val){
                                    //correct value
                                } else {
                                    std::cout << "Particle Intensity Error" << std::endl;
                                    pass_test = false;
                                }
                                
                            }
                            
                            
                            //Check the x and z nieghbours, do they exist?
                            for(int face = 0;face < 6;face++){
                                
                                uint64_t x_n = 0;
                                uint64_t z_n = 0;
                                uint64_t y_n = 0;
                                uint64_t depth = 0;
                                uint64_t j_n = 0;
                                uint64_t status_n = 0;
                                uint64_t intensity = 1;
                                uint64_t node_n = 0;
                                
                                
                                
                                
                                
                                for(int n = 0;n < neigh_keys.neigh_face[face].size();n++){
                                    
                                    
                                    pc_struct.pc_data.get_neigh_coordinates_part(neigh_keys,face,n,y_coord,y_n,x_n,z_n,depth);
                                    j_n = (neigh_keys.neigh_face[face][n] & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
                                    status_n = pc_struct.pc_data.pc_key_get_status(neigh_keys.neigh_face[face][n]);
                                    
                                    if (neigh_keys.neigh_face[face][n] > 0){
                                        
                                        uint16_t own_int = pc_struct.part_data.get_part(neigh_keys.neigh_face[face][n]);
                                        
                                        //calculate y so you can check back in the original structure
                                        
                                        uint64_t depth_ind = pc_struct.pc_data.pc_key_get_depth(neigh_keys.neigh_face[face][n]);
                                        if(status_n == SEED){
                                            depth_ind = depth_ind + 1;
                                        }
                                        
                                        
                                        x_n = std::min(x_n,(uint64_t)(part_map.downsampled[depth_ind].x_num-1));
                                        z_n = std::min(z_n,(uint64_t)(part_map.downsampled[depth_ind].z_num-1));
                                        y_n = std::min(y_n,(uint64_t)(part_map.downsampled[depth_ind].y_num-1));
                                        
                                        
                                        const size_t offset_part_map = part_map.downsampled[depth].y_num*part_map.downsampled[depth].x_num*z_n + part_map.downsampled[depth].y_num*x_n;
                                        //status_n = part_map.layers[depth].mesh[offset_part_map + y_n];
                                        
                                        uint16_t corr_val = (offset_part_map + y_n);
                                        
                                        
                                        
                                        if(own_int == corr_val){
                                            //correct value
                                        } else {
                                            
                                            std::cout << "Neighbour Particle Intensity Error" << std::endl;
                                            
                                            own_int = pc_struct.part_data.get_part(neigh_keys.neigh_face[face][n]);
                                            uint64_t index = pc_struct.pc_data.pc_key_get_index(pc_struct.part_data.access_data.get_val(neigh_keys.neigh_face[face][n]));
                                            uint64_t offset = x_num*z_n + x_n;
                                            //pc_struct.part_data.access_data.get_val(neigh_cell_keys.neigh_face[2][1]);
                                            pc_struct.pc_data.get_neigh_coordinates_part(neigh_keys,face,n,y_coord,y_n,x_n,z_n,depth);
                                            pass_test = false;
                                        }
                                        
                                        
                                    }
                                }
                                
                            }
                            
                            
                            
                            
                        }
                        
                        
                        
                        
                    }
                }
                
            }
        }
        
    }
    
    std::cout << "Finished Neigh Part test" << std::endl;
    
    
    
    return pass_test;
    
    
    
}
bool compare_y_coords(PartCellStructure<float,uint64_t>& pc_struct){
    
    //initialize
    uint64_t node_val;
    uint64_t y_coord;
    int x_;
    int z_;
    uint64_t y_;
    uint64_t j_;
    uint64_t status;
    uint64_t status_org;
    
    
    bool pass_test = true;
    
    //Neighbour Routine Checking
    
    for(int i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num = pc_struct.pc_data.x_num[i];
        const unsigned int z_num = pc_struct.pc_data.z_num[i];
        
        
        for(z_ = 0;z_ < z_num;z_++){
            
            for(x_ = 0;x_ < x_num;x_++){
                
                const size_t offset_pc_data = x_num*z_ + x_;
                y_coord = 0;
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    uint64_t node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (node_val&1){
                        
                        uint64_t next_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        uint64_t prev_coord = (node_val & PREV_COORD_MASK) >> PREV_COORD_SHIFT;
                        
                        uint64_t y_coord_prev = y_coord;
                        uint64_t y_coord_diff = y_coord +  ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        
                        //get the index gap node
                        y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        
                        if((y_coord_diff != (y_coord)) & (y_coord > 0)){
                            std::cout << " FAILED COORDINATE TEST" << std::endl;
                            pass_test = false;
                        }
                        
                        
                        y_coord--;
                        
                        
                        
                    } else {
                        //normal node
                        y_coord++;
                        
                        
                        
                    }
                }
                
            }
        }
        
    }
    
    std::cout << "Y_coordinate test" << std::endl;
    
    return pass_test;
}
bool read_write_structure_test(PartCellStructure<float,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2016
    //
    //  Test for the reading and writing of the particle cell sparse structure
    //
    //
    
    
    uint64_t x_;
    uint64_t z_;
    uint64_t j_;
    uint64_t curr_key;
    
    bool pass_test = true;
    
    
    std::string save_loc = "";
    std::string file_name = "io_test_file";
    
    write_apr_pc_struct(pc_struct,save_loc,file_name);
    
    PartCellStructure<float,uint64_t> pc_struct_read;
    read_apr_pc_struct(pc_struct_read,save_loc + file_name + "_pcstruct_part.h5");
    
    //compare all the different thigns and check they are correct;
    
    return(compare_two_structures_test(pc_struct,pc_struct_read));
    
    
}
bool compare_two_structures_test(PartCellStructure<float,uint64_t>& pc_struct,PartCellStructure<float,uint64_t>& pc_struct_read){
    //
    //  Bevan Cheeseman 2016
    //
    //  Test for the reading and writing of the particle cell sparse structure
    //
    //
    
    
    uint64_t x_;
    uint64_t z_;
    uint64_t j_;
    uint64_t curr_key;
    
    bool pass_test = true;
    
    
    //compare all the different thigns and check they are correct;
    
    
    //
    //  Check the particle data (need to account for casting)
    //
    
    for(uint64_t i = pc_struct_read.pc_data.depth_min;i <= pc_struct_read.pc_data.depth_max;i++){
        
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        
        //write the vals
        
        for(z_ = 0;z_ < z_num_;z_++){
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.part_data.particle_data.data[i][offset_pc_data].size();
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    uint16_t org_val = pc_struct.part_data.particle_data.data[i][offset_pc_data][j_];
                    uint16_t read_val = pc_struct_read.part_data.particle_data.data[i][offset_pc_data][j_];
                    
                    if(org_val != read_val){
                        pass_test = false;
                        std::cout << "Particle Intensity Read Error" << std::endl;
                    }
                    
                    
                }
                
                
                
            }
            
        }
    }
    
    for(uint64_t i = pc_struct_read.pc_data.depth_min;i <= pc_struct_read.pc_data.depth_max;i++){
        
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        
        //write the vals
        
        for(z_ = 0;z_ < z_num_;z_++){
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.part_data.particle_data.data[i][offset_pc_data].size();
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    uint16_t org_val = pc_struct.part_data.particle_data.data[i][offset_pc_data][j_];
                    uint16_t read_val = pc_struct_read.part_data.particle_data.data[i][offset_pc_data][j_];
                    
                    if(org_val != read_val){
                        pass_test = false;
                        std::cout << "Particle Intensity Read Error" << std::endl;
                    }
                    
                    
                }
                
                
                
            }
            
        }
    }
    
    
    uint64_t node_val_pc;
    uint64_t node_val_part;
    uint64_t status;
    uint64_t part_offset;
    uint64_t p;
    
    //other way round
    
    for(int i = pc_struct_read.pc_data.depth_min;i <= pc_struct_read.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);
            
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.pc_data.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            
                            uint16_t org_val = pc_struct.part_data.get_part(curr_key);
                            uint16_t read_val = pc_struct_read.part_data.get_part(curr_key);
                            
                            if(org_val != read_val){
                                pass_test = false;
                                std::cout << "Particle Intensity Access Error" << std::endl;
                            }
                            
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                    }
                    
                }
                
            }
            
        }
    }
    
    //second loop using the read particle cell structures access
    
    for(int i = pc_struct_read.pc_data.depth_min;i <= pc_struct_read.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct_read.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct_read.pc_data.z_num[i];
        
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_struct_read.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct_read.pc_data.pc_key_set_depth(curr_key,i);
            
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct_read.pc_data.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct_read.pc_data.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_struct_read.pc_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        node_val_part = pc_struct_read.part_data.access_data.data[i][offset_pc_data][j_];
                        
                        pc_struct_read.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct_read.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct_read.part_data.access_node_get_part_offset(node_val_part);
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct_read.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct_read.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            
                            uint16_t org_val = pc_struct.part_data.get_part(curr_key);
                            uint16_t read_val = pc_struct_read.part_data.get_part(curr_key);
                            
                            if(org_val != read_val){
                                pass_test = false;
                                std::cout << "Particle Intensity Access Error (READ)" << std::endl;
                            }
                            
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                    }
                    
                }
                
            }
            
        }
    }
    
    
    
    
    //
    //  Check the particle access data
    //
    
    for(uint64_t i = pc_struct_read.pc_data.depth_min;i <= pc_struct_read.pc_data.depth_max;i++){
        
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        
        //write the vals
        
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.part_data.access_data.data[i][offset_pc_data].size();
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    uint16_t org_val = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    uint16_t read_val = pc_struct_read.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if(org_val != read_val){
                        pass_test = false;
                        std::cout << "Particle Access Read Error" << std::endl;
                    }
                    
                    
                }
                
                
                
            }
            
        }
    }
    
    
    //
    //  Check the part cell data
    //
    
    for(uint64_t i = pc_struct_read.pc_data.depth_min;i <= pc_struct_read.pc_data.depth_max;i++){
        
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        
        //write the vals
        
        for(z_ = 0;z_ < z_num_;z_++){
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    uint64_t org_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    uint64_t read_val = pc_struct_read.pc_data.data[i][offset_pc_data][j_];
                    
                    if(org_val != read_val){
                        pass_test = false;
                        std::cout << "Particle Access Read Error" << std::endl;
                    }
                    
                    
                }
                
                
                
            }
            
        }
    }
    
    
    std::cout << "io_test_complete" << std::endl;
    
    return pass_test;
    
    
}
bool parent_structure_test(PartCellStructure<float,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2016
    //
    //  Tests the parent structure is working correctly by constructing arrays to check relationships
    //
    
    
    ///////////////////////////
    //
    //  Particle Cell Information
    //
    ///////////////////////////
    
    bool pass_test = true;
    
    //initialize
    uint64_t node_val_part;
    uint64_t y_coord;
    int x_;
    int z_;
    
    uint64_t j_;
    uint64_t status;
    uint64_t curr_key=0;
    
    std::vector<std::vector<uint8_t>> p_map;
    p_map.resize(pc_struct.pc_data.depth_max+1);
    
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        p_map[i].resize(x_num_*z_num_*y_num_,0);
        
        // First create the particle map
        for(z_ = 0;z_ < z_num_;z_++){
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                const size_t offset_p_map = y_num_*x_num_*z_ + y_num_*x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                y_coord = 0;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_part&1)){
                        //get the index gap node
                        y_coord++;
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        p_map[i][offset_p_map + y_coord] = status;
                        
                    } else {
                        
                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }
                    
                }
                
            }
            
        }
        
    }
    
    ///////////////////////////
    //
    //  Parent Information
    //
    ///////////////////////////
    
    std::vector<std::vector<uint8_t>> parent_map;
    
    parent_map.resize(pc_struct.depth_max);
    
    for(int i = pc_struct.depth_min;i <= (pc_struct.depth_max-1);i++){
        
        parent_map[i].resize(pc_struct.x_num[i]*pc_struct.y_num[i]*pc_struct.z_num[i],0);
    }
    
    uint64_t node_val;
    
    
    for(int i = (pc_struct.pc_data.depth_max);i > pc_struct.pc_data.depth_min;i--){
        
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        
        //#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,y_coord) if(z_num_*x_num_ > 100)
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
                
                uint64_t parent_x;
                uint64_t parent_y;
                uint64_t parent_z;
                
                y_coord = 0;
                
                uint64_t depth;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //this value encodes the state and neighbour locations of the particle cell
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val&1)){
                        y_coord++; //iterate y
                        
                        parent_x = x_/2;
                        parent_z = z_/2;
                        parent_y = y_coord/2;
                        depth = i - 1;
                        
                        if(parent_map[depth][parent_z*pc_struct.y_num[depth]*pc_struct.x_num[depth] + parent_x*pc_struct.y_num[depth] + parent_y] ==0){
                            
                            parent_map[depth][parent_z*pc_struct.y_num[depth]*pc_struct.x_num[depth] + parent_x*pc_struct.y_num[depth] + parent_y] = 2;
                            
                            parent_x = parent_x/2;
                            parent_z = parent_z/2;
                            parent_y = parent_y/2;
                            depth = depth - 1;
                            
                            if(depth > pc_struct.pc_data.depth_min){
                                
                                while(parent_map[depth][parent_z*pc_struct.y_num[depth]*pc_struct.x_num[depth] + parent_x*pc_struct.y_num[depth] + parent_y] ==0 ){
                                    
                                    parent_map[depth][parent_z*pc_struct.y_num[depth]*pc_struct.x_num[depth] + parent_x*pc_struct.y_num[depth] + parent_y] = 1;
                                    
                                    parent_x = parent_x/2;
                                    parent_z = parent_z/2;
                                    parent_y = parent_y/2;
                                    depth = depth - 1;
                                    
                                    if  (depth < pc_struct.pc_data.depth_min){
                                        break;
                                    }
                                    
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
 
    
    
    
    
    /////////////////////////////////
    //
    //  Create parent structure and check it
    //
    /////////////////////////////////
    
    PartCellParent<uint64_t> parent_cells(pc_struct);
    PartCellNeigh<uint64_t> neigh_keys;
    
    
//    Mesh_data<uint8_t> temp;
//    for(int i = parent_cells.neigh_info.depth_min; i <= parent_cells.neigh_info.depth_max;i++){
//        temp.y_num = pc_struct.y_num[i];
//        temp.x_num = pc_struct.x_num[i];
//        temp.z_num = pc_struct.z_num[i];
//        temp.mesh = parent_map[i];
//        
//        debug_write(temp,"test_parent_" + std::to_string(i));
//        
//    }
//    
    ////////////////////////////////
    //
    //  Neigh Info Checks
    //
    ////////////////////////////////
    
    for(int i = parent_cells.neigh_info.depth_min;i <= parent_cells.neigh_info.depth_max;i++){
        
        const unsigned int x_num_ = parent_cells.neigh_info.x_num[i];
        const unsigned int z_num_ = parent_cells.neigh_info.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            //set the key values
            parent_cells.neigh_info.pc_key_set_z(curr_key,z_);
            parent_cells.neigh_info.pc_key_set_depth(curr_key,i);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                parent_cells.neigh_info.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                const size_t offset_p_map = y_num_*x_num_*z_ + y_num_*x_;
                
                //number of nodes on the level
                const size_t j_num = parent_cells.neigh_info.data[i][offset_pc_data].size();
                
                y_coord = 0;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //this value encodes the state and neighbour locations of the particle cell
                    node_val = parent_cells.neigh_info.data[i][offset_pc_data][j_];
                    
                    if (!(node_val&1)){
                        //This node represents a particle cell
                        y_coord++;
                        //set the key index
                        parent_cells.neigh_info.pc_key_set_j(curr_key,j_);
                        
                        //get all the neighbours
                        
                        //First get the status and check that the values are consistent with the arrays
                        status = parent_cells.neigh_info.get_status(node_val);
                        
                        
                        if(status == GHOST_CHILDREN){
                            
                            if((parent_map[i][offset_p_map + y_coord] > 0) & (p_map[i][offset_p_map + y_coord] == 0)){
                                
                            } else {
                                uint64_t p_val = p_map[i][offset_p_map + y_coord];
                                uint64_t parent_val = parent_map[i][offset_p_map + y_coord];
                                
                                
                                std::cout << "GHOST PARENT BUG" << std::endl;
                                pass_test = false;
                            }
                            
                            
                        } else if (status == REAL_CHILDREN){
                            
                            if((parent_map[i][offset_p_map + y_coord] > 0) & (p_map[i][offset_p_map + y_coord] == 0)){
                                
                            } else {
                                std::cout << "REAL PARENT BUG" << std::endl;
                                pass_test = false;
                            }
                            
                            // additional check are the children real?
                            
                        } else {
                            std::cout << "Incorrect status bug" << std::endl;
                            pass_test = false;
                        }
                        
                        // Check all the neighbours (they should be parents and on the same depth)
                        parent_cells.get_neighs_parent_all(curr_key,node_val,neigh_keys);
                        
                        for(uint64_t face = 0; face < neigh_keys.neigh_face.size();face++){
                            
                            for(uint64_t n = 0; n < neigh_keys.neigh_face[face].size();n++){
                                uint64_t neigh_key = neigh_keys.neigh_face[face][n];
                                
                                uint64_t neigh_y;
                                uint64_t neigh_x;
                                uint64_t neigh_z;
                                uint64_t neigh_depth;
                                
                                if(neigh_key > 0){
                                    parent_cells.neigh_info.get_neigh_coordinates_cell(neigh_keys,face,n,y_coord,neigh_y,neigh_x,neigh_z,neigh_depth);
                                
                                    if(neigh_depth != i){
                                        std::cout << "Neighbour depth bug" << std::endl;
                                        pass_test = false;
                                    }
                                    
                                    uint64_t offset = neigh_z*pc_struct.y_num[neigh_depth]*pc_struct.x_num[neigh_depth] + neigh_x*pc_struct.y_num[neigh_depth] + neigh_y;
                                    
                                    
                                    uint64_t p_map_val = p_map[i][offset];
                                    uint64_t parent_map_val = parent_map[i][offset];
                                    
                                    if((parent_map[i][offset] > 0) & (p_map[i][offset] == 0)){
                                        //correct
                                    } else {
                                        std::cout << "REAL PARENT BUG" << std::endl;
                                        pass_test = false;
                                    }
                                    
                                    
                                }
                            }
                            
                            
                        }
                        
                    } else {
                        //This is a gap node
                        y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                    }
                    
                }
                
            }
            
        }
    }
    
    
    
    ////////////////////////////////
    //
    //  Parent Info Checks
    //
    ////////////////////////////////
    
    uint64_t node_val_parent;
    
    for(int i = parent_cells.neigh_info.depth_min;i <= parent_cells.neigh_info.depth_max;i++){
        
        const unsigned int x_num_ = parent_cells.neigh_info.x_num[i];
        const unsigned int z_num_ = parent_cells.neigh_info.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            //set the key values
            parent_cells.neigh_info.pc_key_set_z(curr_key,z_);
            parent_cells.neigh_info.pc_key_set_depth(curr_key,i);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                parent_cells.neigh_info.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                const size_t offset_p_map = y_num_*x_num_*z_ + y_num_*x_;
                
                //number of nodes on the level
                const size_t j_num = parent_cells.neigh_info.data[i][offset_pc_data].size();
                
                y_coord = 0;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //this value encodes the state and neighbour locations of the particle cell
                    node_val = parent_cells.neigh_info.data[i][offset_pc_data][j_];
                    
                    if (!(node_val&1)){
                        //This node represents a particle cell
                        y_coord++;
                        //set the key index
                        parent_cells.neigh_info.pc_key_set_j(curr_key,j_);
                        
                        status = parent_cells.neigh_info.get_status(node_val);
                        
                        node_val_parent = parent_cells.parent_info.data[i][offset_pc_data][j_];
                        
                        // Get parent method
                        uint64_t parent_key = parent_cells.get_parent_key(node_val_parent,curr_key);
                        
                        if(parent_key > 0){
                            
                            try{
                                parent_cells.neigh_info.get_val(parent_key);
                            } catch(const std::exception& e) {
                                std::cout << "parent error" << std::endl;
                                pass_test = false;
                            }
                            
                            try{
                                parent_cells.parent_info.get_val(parent_key);
                            } catch(const std::exception& e) {
                                std::cout << "parent error" << std::endl;
                                pass_test = false;
                            }
                        }
                        
                        //check this guy
                        
                        // Get children method
                        std::vector<uint64_t> children_keys;
                        std::vector<uint64_t> children_ind;
                        
                        parent_cells.get_children_keys(curr_key,children_keys,children_ind);
                        
                        //loop over the children
                        for(uint64_t c = 0; c < children_keys.size();c++){
                            
                            uint64_t child = children_keys[c];
                            //first question is does it exist
                            if(child > 0){
                                
                                uint64_t child_y = 0;
                                uint64_t child_x = 0;
                                uint64_t child_z = 0;
                                uint64_t child_depth = 0;
                                
                                //get the coordinates
                                parent_cells.get_child_coordinates_cell(children_keys,c,y_coord,child_y,child_x,child_z,child_depth);
                                
                                if(children_ind[c] == 0){
                                    // it is a parent node,
                                    
                                    uint64_t child_j = parent_cells.parent_info.pc_key_get_j(child);
                                    uint64_t child_parent = parent_cells.parent_info.get_val(child);
                                    uint64_t child_parent2 = parent_cells.parent_info2.get_val(child);
                                    uint64_t child_neigh = parent_cells.neigh_info.get_val(child);
                                    
                                    (void) child_parent2;
                                    (void) child_neigh; //force execution
                                    
                                    uint64_t parent_j = parent_cells.get_parent_j(child_parent);
                                    
                                    if (parent_j == j_){
                                        //check if the child points to the correct parent
                                    } else {
                                        std::cout << "child error" << std::endl;
                                        parent_cells.get_children_keys(curr_key,children_keys,children_ind);
                                        pass_test = false;
                                    }
                                    
                                    //check it in the parent_map structure
                                    uint64_t offset = child_z*pc_struct.y_num[child_depth]*pc_struct.x_num[child_depth] + child_x*pc_struct.y_num[child_depth] + child_y;
                                    
                                    if((parent_map[child_depth][offset] > 0) & (p_map[child_depth][offset] == 0)){
                                        //correct
                                    } else {
                                        std::cout << "CHILD PARENT BUG" << std::endl;
                                        pass_test = false;
                                    }
                                    
                                } else {
                                    //it is a real node check different things here
                                    
                                    
                                    //check it in the parent_map structure
                                    uint64_t offset = child_z*pc_struct.y_num[child_depth]*pc_struct.x_num[child_depth] + child_x*pc_struct.y_num[child_depth] + child_y;
                                    
                                    uint64_t pmap_val = p_map[child_depth][offset];
                                    
                                    if(pmap_val > 0){
                                        //correct
                                    } else {
                                        std::cout << "CHILD REAL BUG" << offset << std::endl;
                                        parent_cells.get_child_coordinates_cell(children_keys,c,y_coord,child_y,child_x,child_z,child_depth);
                                        pass_test = false;
                                    }
                                    
                                    

                                }
                            } else {
                                // check if it returns not in the structure, that it actually isn't in the structure!
                                
                                uint64_t x_c = 2*x_ + parent_cells.parent_info.seed_part_x[c];
                                uint64_t z_c = 2*z_ + parent_cells.parent_info.seed_part_z[c];
                                uint64_t y_c = 2*y_coord + parent_cells.parent_info.seed_part_y[c];
                                uint64_t depth_c = i +1;
                                
                                if ((x_c < pc_struct.x_num[depth_c]) & (z_c < pc_struct.z_num[depth_c]) & (y_c < pc_struct.y_num[depth_c]) ){
                                    
                                    //check it in the parent_map structure
                                    uint64_t offset = z_c*pc_struct.y_num[depth_c]*pc_struct.x_num[depth_c] + x_c*pc_struct.y_num[depth_c] + y_c;
                                    
                                    uint64_t parent_map_val = 0;
                                    
                                    uint64_t pmap_val = p_map[depth_c][offset];
                                    
                                    if(depth_c > parent_cells.neigh_info.depth_max){
                                        parent_map_val = 0;
                                    } else {
                                        parent_map_val = parent_map[depth_c][offset];
                                    }
                                    
                                    
                                    if ((pmap_val > 0) | (parent_map_val > 0)){
                                        std::cout << "MISSING CHLID" << std::endl;
                                        pass_test = false;
                                    }
                                    
                                }
                                
                            }
                            
                            
                        }
                        
                        
                    } else {
                        //This is a gap node
                        y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                    }
                    
                }
                
            }
            
        }
    }

    
    
    
    
    
    
    return pass_test;
    
}
void create_test_dataset_from_hdf5(Particle_map<float>& particle_map,PartCellStructure<float, uint64_t>& pc_struct,std::string name){
    
    std::string test_dir =  get_source_directory();
    
    //output
    std::string file_name = test_dir + name;
    
    read_apr_pc_struct(pc_struct,file_name);
    
    //Now we need to generate the particle map
    particle_map.k_max = pc_struct.depth_max;
    particle_map.k_min = pc_struct.depth_min;
    
    //initialize looping vars
    uint64_t x_;
    uint64_t y_coord;
    uint64_t z_;
    uint64_t j_;
    uint64_t node_val_part;
    uint64_t status;
    
    particle_map.layers.resize(pc_struct.depth_max+1);
    particle_map.downsampled.resize(pc_struct.depth_max+2);
    
    particle_map.downsampled[pc_struct.depth_max + 1].x_num = pc_struct.org_dims[1];
    particle_map.downsampled[pc_struct.depth_max + 1].y_num = pc_struct.org_dims[0];
    particle_map.downsampled[pc_struct.depth_max + 1].z_num = pc_struct.org_dims[2];
    particle_map.downsampled[pc_struct.depth_max + 1].mesh.resize(pc_struct.org_dims[1]*pc_struct.org_dims[0]*pc_struct.org_dims[2]);
    
    std::cout << "DIM1: " << pc_struct.org_dims[1] << std::endl;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        particle_map.layers[i].mesh.resize(x_num_*z_num_*y_num_,0);
        particle_map.layers[i].x_num = x_num_;
        particle_map.layers[i].y_num = y_num_;
        particle_map.layers[i].z_num = z_num_;
        
        particle_map.downsampled[i].x_num = x_num_;
        particle_map.downsampled[i].y_num = y_num_;
        particle_map.downsampled[i].z_num = z_num_;
        particle_map.downsampled[i].mesh.resize(x_num_*z_num_*y_num_,0);
        
        // First create the particle map
        for(z_ = 0;z_ < z_num_;z_++){
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                const size_t offset_p_map = y_num_*x_num_*z_ + y_num_*x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                y_coord = 0;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_part&1)){
                        //get the index gap node
                        y_coord++;
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        particle_map.layers[i].mesh[offset_p_map + y_coord] = status;
                        
                    } else {
                        
                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }
                    
                }
                
            }
            
        }
        
    }
    
    
    //intensity set up
    // Set the intensities
    for(int depth = particle_map.k_min; depth <= (particle_map.k_max+1);depth++){
        
        for(int i = 0; i < particle_map.downsampled[depth].mesh.size();i++){
            particle_map.downsampled[depth].mesh[i] = (uint16_t) i;
        }
        
    }
    
    
}
void create_reference_structure(PartCellStructure<float,uint64_t>& pc_struct,std::vector<Mesh_data<uint64_t>>& link_array){
    //
    //  Creates an array that can be used to link the new particle data structure for the filtering with the newer one.
    //
    //  Bevan Cheeseman 2017
    //
    //
    
    
    link_array.resize(pc_struct.pc_data.depth_max + 2);
    
    for(int i = pc_struct.depth_min; i <= pc_struct.depth_max;i++){
        link_array[i].initialize(pc_struct.y_num[i],pc_struct.x_num[i],pc_struct.z_num[i],0);
    }
    
    link_array[pc_struct.depth_max+1].initialize(pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],0);
    
    uint64_t y_coord; // y coordinate needs to be tracked and is not explicitly stored in the structure
  
    uint64_t p;
    uint64_t z_;
    uint64_t x_;
    uint64_t j_;
    uint64_t node_val_pc;
    uint64_t node_val_part;
    uint64_t curr_key;
    uint64_t status;
    uint64_t part_offset;
    
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        
//#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset)  if(z_num_*x_num_ > 100)
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
                uint64_t y_current = 0;
                uint64_t z_current;
                uint64_t depth_current;
                y_coord= 0;
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        y_coord++;
                        
                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        pc_struct.part_data.access_data.pc_key_set_status(curr_key,status);
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            
                            pc_struct.part_data.access_data.pc_key_set_partnum(curr_key,p);
                            
                            // First get some details about the current part
                            pc_struct.part_data.access_data.get_coordinates_part(y_coord,curr_key,x_current,z_current,y_current,depth_current,status_current);
                            
                            link_array[depth_current](y_current,x_current,z_current) = curr_key;
                            
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                        y_coord = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--;
                    }
                    
                }
                
            }
            
        }
    }
    
    
    
    
    
    
}
void create_intensity_reference_structure(PartCellStructure<float,uint64_t>& pc_struct,std::vector<Mesh_data<float>>& int_array){
    //
    //  Creates an array that can be used to link the new particle data structure for the filtering with the newer one.
    //
    //  Bevan Cheeseman 2017
    //
    //
    
    
    int_array.resize(pc_struct.pc_data.depth_max + 2);
    
    for(int i = pc_struct.depth_min; i <= pc_struct.depth_max;i++){
        int_array[i].initialize(pc_struct.y_num[i],pc_struct.x_num[i],pc_struct.z_num[i],0);
    }
    
    int_array[pc_struct.depth_max+1].initialize(pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],0);
    
    uint64_t y_coord; // y coordinate needs to be tracked and is not explicitly stored in the structure
    
    uint64_t p;
    uint64_t z_;
    uint64_t x_;
    uint64_t j_;
    uint64_t node_val_pc;
    uint64_t node_val_part;
    uint64_t curr_key;
    uint64_t status;
    uint64_t part_offset;
    
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        
        //#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset)  if(z_num_*x_num_ > 100)
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
                uint64_t y_current = 0;
                uint64_t z_current;
                uint64_t depth_current;
                y_coord= 0;
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        y_coord++;
                        
                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        pc_struct.part_data.access_data.pc_key_set_status(curr_key,status);
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            
                            pc_struct.part_data.access_data.pc_key_set_partnum(curr_key,p);
                            
                            // First get some details about the current part
                            pc_struct.part_data.access_data.get_coordinates_part(y_coord,curr_key,x_current,z_current,y_current,depth_current,status_current);
                            
                            int_array[depth_current](y_current,x_current,z_current) = pc_struct.part_data.particle_data.get_part(curr_key);
                            
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                        y_coord = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--;
                    }
                    
                }
                
            }
            
        }
    }
    
    
    
    
    
    
}



bool find_part_cell_test(PartCellStructure<float,uint64_t>& pc_struct){
    
    PartCellParent<uint64_t> parent_cells(pc_struct);
    
    int num_cells = pc_struct.get_number_cells();
    int num_parts = pc_struct.get_number_parts();
    
    std::cout << "Number cells: " << num_cells << std::endl;
    std::cout << "Number parts: " << num_parts << std::endl;
    
    //initialize looping vars
    uint64_t x_;
    uint64_t y_coord;
    uint64_t z_;
    uint64_t j_;
    uint64_t node_val;
    
    uint64_t curr_key = 0;
    
    bool pass_test = true;
    
    // FIND POINT X,Y,Z  in structure
    
    //loop over all the particle cells and then
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        
        // First create the particle map
        for(z_ = 0;z_ < z_num_;z_++){
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                y_coord = 0;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val&1)){
                        //get the index gap node
                        y_coord++;
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        pc_struct.part_data.access_data.pc_key_set_x(curr_key,x_);
                        pc_struct.part_data.access_data.pc_key_set_z(curr_key,z_);
                        pc_struct.part_data.access_data.pc_key_set_depth(curr_key,i);
                        
                        uint64_t factor = pow(2,pc_struct.depth_max + 1 - i);
                        
                        uint64_t y_f = y_coord*factor;
                        uint64_t x_f = x_*factor;
                        uint64_t z_f = z_*factor;
                        
                        uint64_t found_key = parent_cells.find_partcell(x_f,y_f,z_f,pc_struct);
                        
                        if(pc_struct.pc_data.pc_key_cell_isequal(found_key,curr_key)){
                            // success
                        } else {
                            
                            uint64_t x_t;
                            uint64_t z_t;
                            uint64_t j_t;
                            uint64_t depth_t;
                           
                            // get coordinates compare whats wrong
                            pc_struct.pc_data.get_details_cell(found_key,x_t,z_t,j_t,depth_t);
                            pass_test = false;
                            std::cout << "Found Key Doesn't Match" << std::endl;
                        }
                        
                        
                    } else {
                        
                         y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--;
                    }
                    
                }
                
            }
            
        }
        
    }
    
    return pass_test;
}
