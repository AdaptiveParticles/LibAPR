////////////////////////
//
//  Mateusz Susik 2016
//
//  Utility functions for the tests
//
////////////////////////

#include "utils.h"

bool compare_two_images(const MeshData<uint16_t>& in_memory, std::string filename) {

    /* Returns true iff two images are the same with tolerance of 1 per pixel. */

    MeshData<uint16_t > input_image;

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

        MeshData<uint8_t > to_compare;

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

MeshData<uint16_t> create_random_test_example(unsigned int size_y, unsigned int size_x,
                                               unsigned int size_z, unsigned int seed) {
    // creates the input image of a given size with given seed
    // uses ranlux48 random number generator
    // the seed used in 2016 for generation was 5489u

    std::ranlux48 generator(seed);
    std::normal_distribution<float> distribution(1000, 250);

    MeshData<uint16_t> test_example(size_y, size_x, size_z);

    std::generate(test_example.mesh.begin(), test_example.mesh.end(),
            // partial application of generator and distribution to get_random_number function
                  std::bind(get_random_number, generator, distribution));

    return test_example;

}

MeshData<uint16_t> generate_random_ktest_example(unsigned int size_y, unsigned int size_x,
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

    MeshData<uint16_t> test_example(size_y, size_x, size_z);

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
    //  Compares the sparse representation with the particle map_inplace original data used to generate it
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

bool read_write_structure_test(PartCellStructure<float,uint64_t>& aParticleCells, const std::string& aTestFileSuffix) {
    //
    // Bevan Cheeseman 2016
    //
    // Test saves provided Particle Cells and then reads saved file. Compares if both - provided and saved -
    // have same contents.
    //

    std::string saveLocation = "";
    std::string fileName = "io_test_file" + aTestFileSuffix;

    write_apr_pc_struct(aParticleCells, saveLocation, fileName);

    PartCellStructure<float, uint64_t> readParticleCells;
    // TODO: File suffix "_pcstruct_part.h5" should be defined in a one place
    read_apr_pc_struct(readParticleCells, saveLocation + fileName + "_pcstruct_part.h5");

    //compare all the different things and check they are correct
    return compare_two_structures_test(aParticleCells, readParticleCells);
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

        // First create the particle map_inplace
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


//    MeshData<uint8_t> temp;
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

    //Now we need to generate the particle map_inplace
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

        // First create the particle map_inplace
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
void create_reference_structure(PartCellStructure<float,uint64_t>& pc_struct,std::vector<MeshData<uint64_t>>& link_array){
    //
    //  Creates an array that can be used to link the new particle data structure for the filtering with the newer one.
    //
    //  Bevan Cheeseman 2017
    //
    //


    link_array.resize(pc_struct.pc_data.depth_max + 2);

    for(int i = pc_struct.depth_min; i <= pc_struct.depth_max;i++){
        link_array[i].initialize(pc_struct.y_num[i]+1,pc_struct.x_num[i]+1,pc_struct.z_num[i]+1,0);
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

                            pc_key neigh_key;
                            neigh_key.update_part(curr_key);

                            if(x_current != neigh_key.x_p){
                                int stop = 1;
                            }



                            if(z_current != neigh_key.z_p){
                                int stop = 3;
                            }


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
void create_j_reference_structure(PartCellStructure<float,uint64_t>& pc_struct,std::vector<MeshData<uint64_t>>& j_array){
    //
    //  Creates an array that can be used to link the new particle data structure for the filtering with the newer one.
    //
    //  Bevan Cheeseman 2017
    //
    //


    j_array.resize(pc_struct.pc_data.depth_max + 1);

    for(int i = pc_struct.depth_min; i <= pc_struct.depth_max;i++){
        j_array[i].initialize(pc_struct.y_num[i],pc_struct.x_num[i],pc_struct.z_num[i],0);
    }

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


                        j_array[i](y_coord,x_,z_) = j_;


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
void create_j_reference_structure(PartCellData<uint64_t>& pc_data,std::vector<MeshData<uint64_t>>& j_array){
    //
    //  Creates an array that can be used to link the new particle data structure for the filtering with the newer one.
    //
    //  Bevan Cheeseman 2017
    //
    //


    j_array.resize(pc_data.depth_max + 1);

    for(int i = pc_data.depth_min; i <= pc_data.depth_max;i++){
        j_array[i].initialize(pc_data.y_num[i],pc_data.x_num[i],pc_data.z_num[i],0);
    }

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


    for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_data.x_num[i];
        const unsigned int z_num_ = pc_data.z_num[i];


        //#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset)  if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;

            pc_data.pc_key_set_z(curr_key,z_);
            pc_data.pc_key_set_depth(curr_key,i);


            for(x_ = 0;x_ < x_num_;x_++){

                pc_data.pc_key_set_x(curr_key,x_);

                const size_t offset_pc_data = x_num_*z_ + x_;

                const size_t j_num = pc_data.data[i][offset_pc_data].size();

                uint64_t status_current;
                uint64_t x_current;
                uint64_t y_current = 0;
                uint64_t z_current;
                uint64_t depth_current;
                y_coord= 0;

                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){

                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_data.data[i][offset_pc_data][j_];

                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        y_coord++;

                        pc_data.pc_key_set_j(curr_key,j_);


                        j_array[i](y_coord,x_,z_) = j_;


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
pc_key find_neigh_cell(pc_key curr_cell,int dir,std::vector<MeshData<uint64_t>>& j_array){
    //
    //  Bevan Cheeseman 2017
    //
    //  Debugging Code, takes a node and brute force finds its neighbour
    //
    //

    pc_key neigh_key;

    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

    int offset_y = dir_y[dir];
    int offset_x = dir_x[dir];
    int offset_z = dir_z[dir];

    neigh_key.j = 0;

    //same level
    int depth = curr_cell.depth;
    int x = curr_cell.x + offset_x;
    int y = curr_cell.y + offset_y;
    int z = curr_cell.z + offset_z;

    uint64_t j= 0;

    if(x >= 0 & x < j_array[depth].x_num){
        if(y >= 0 & y < j_array[depth].y_num){
            if(z >= 0 & z < j_array[depth].z_num){


                j = j_array[depth](y,x,z);

            }
        }
    }

    if(j != 0){
        //neighbour is on same level
        neigh_key.y = y;
        neigh_key.x = x;
        neigh_key.z = z;
        neigh_key.j = (int)j;
        neigh_key.depth = depth;

    } else {

        //parent check
        depth = curr_cell.depth - 1;
        x = (curr_cell.x + offset_x)/2;
        y = (curr_cell.y + offset_y)/2;
        z = (curr_cell.z + offset_z)/2;
        j = 0;

        if(x >= 0 & x < j_array[depth].x_num){
            if(y >= 0 & y < j_array[depth].y_num){
                if(z >= 0 & z < j_array[depth].z_num){

                    j = j_array[depth](y,x,z);

                }
            }
        }

        if(j != 0){
            //neighbour is on same level
            neigh_key.y = y;
            neigh_key.x = x;
            neigh_key.z = z;
            neigh_key.j = (int)j;
            neigh_key.depth = depth;

        } else {

            //child check
            depth = curr_cell.depth + 1;
            x = 2*(curr_cell.x + offset_x) + (offset_x < 0);
            y = 2*(curr_cell.y + offset_y) + (offset_y < 0);
            z = 2*(curr_cell.z + offset_z) + (offset_z < 0);
            j = 0;

            if ( depth < j_array.size()){

                if(x >= 0 & x < j_array[depth].x_num){
                    if(y >= 0 & y < j_array[depth].y_num){
                        if(z >= 0 & z < j_array[depth].z_num){


                            j = j_array[depth](y,x,z);

                        }
                    }
                }
            }

            if(j != 0){
                //neighbour is on same level
                neigh_key.y = y;
                neigh_key.x = x;
                neigh_key.z = z;
                neigh_key.j = (int)j;
                neigh_key.depth = depth;
            }


        }

    }

    //

    return neigh_key;

}
pc_key find_neigh_cell_edge(pc_key curr_cell,int dir,std::vector<MeshData<uint64_t>>& j_array){
    //
    //  Bevan Cheeseman 2017
    //
    //  Debugging Code, takes a node and brute force finds its neighbour
    //
    //

    pc_key neigh_key;

    neigh_key.j = 0;

    const int8_t dir_y[12] = { 1, -1, 1, -1, 1, -1, 0, 0, 1, -1, 0, 0};
    const int8_t dir_x[12] = { 1, 1, -1, -1, 0, 0, 1, -1, 0, 0, 1, -1};
    const int8_t dir_z[12] = { 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1};

    int offset_y = dir_y[dir];
    int offset_x = dir_x[dir];
    int offset_z = dir_z[dir];

    //same level
    int depth = curr_cell.depth;
    int x = curr_cell.x + offset_x;
    int y = curr_cell.y + offset_y;
    int z = curr_cell.z + offset_z;

    uint64_t j= 0;

    if(x >= 0 & x < j_array[depth].x_num){
        if(y >= 0 & y < j_array[depth].y_num){
            if(z >= 0 & z < j_array[depth].z_num){


                j = j_array[depth](y,x,z);

            }
        }
    }

    if(j != 0){
        //neighbour is on same level
        neigh_key.y = y;
        neigh_key.x = x;
        neigh_key.z = z;
        neigh_key.j = (int)j;
        neigh_key.depth = depth;

    } else {

        //parent check
        depth = curr_cell.depth - 1;
        x = (curr_cell.x + offset_x)/2;
        y = (curr_cell.y + offset_y)/2;
        z = (curr_cell.z + offset_z)/2;
        j = 0;

        if(x >= 0 & x < j_array[depth].x_num){
            if(y >= 0 & y < j_array[depth].y_num){
                if(z >= 0 & z < j_array[depth].z_num){

                    j = j_array[depth](y,x,z);

                }
            }
        }

        if(j != 0){
            //neighbour is on same level
            neigh_key.y = y;
            neigh_key.x = x;
            neigh_key.z = z;
            neigh_key.j = (int)j;
            neigh_key.depth = depth;

        } else {

            //child check
            depth = curr_cell.depth + 1;
            x = 2*(curr_cell.x + offset_x) + (offset_x < 0);
            y = 2*(curr_cell.y + offset_y) + (offset_y < 0);
            z = 2*(curr_cell.z + offset_z) + (offset_z < 0);
            j = 0;

            if ( depth < j_array.size()){

                if(x >= 0 & x < j_array[depth].x_num){
                    if(y >= 0 & y < j_array[depth].y_num){
                        if(z >= 0 & z < j_array[depth].z_num){


                            j = j_array[depth](y,x,z);

                        }
                    }
                }
            }

            if(j != 0){
                //neighbour is on same level
                neigh_key.y = y;
                neigh_key.x = x;
                neigh_key.z = z;
                neigh_key.j = (int)j;
                neigh_key.depth = depth;
            }


        }

    }

    //

    return neigh_key;

}
pc_key find_neigh_cell_corner(pc_key curr_cell,int dir,std::vector<MeshData<uint64_t>>& j_array){
    //
    //  Bevan Cheeseman 2017
    //
    //  Debugging Code, takes a node and brute force finds its neighbour
    //
    //

    pc_key neigh_key;

    const int8_t dir_y[8] = { 1, -1, 1, -1, 1, -1, 1, -1};
    const int8_t dir_x[8] = { 1, 1, -1, -1, 1, 1, -1, -1};
    const int8_t dir_z[8] = { 1, 1, 1, 1, -1, -1, -1, -1};

    int offset_y = dir_y[dir];
    int offset_x = dir_x[dir];
    int offset_z = dir_z[dir];

    neigh_key.j = 0;

    //same level
    int depth = curr_cell.depth;
    int x = curr_cell.x + offset_x;
    int y = curr_cell.y + offset_y;
    int z = curr_cell.z + offset_z;

    uint64_t j= 0;

    if(x > 0 & x < j_array[depth].x_num){
        if(y > 0 & y < j_array[depth].y_num){
            if(z > 0 & z < j_array[depth].z_num){


                j = j_array[depth](y,x,z);

            }
        }
    }

    if(j != 0){
        //neighbour is on same level
        neigh_key.y = y;
        neigh_key.x = x;
        neigh_key.z = z;
        neigh_key.j = (int)j;
        neigh_key.depth = depth;

    } else {

        //parent check
        depth = curr_cell.depth - 1;
        x = (curr_cell.x + offset_x)/2;
        y = (curr_cell.y + offset_y)/2;
        z = (curr_cell.z + offset_z)/2;
        j = 0;

        if(x > 0 & x < j_array[depth].x_num){
            if(y > 0 & y < j_array[depth].y_num){
                if(z > 0 & z < j_array[depth].z_num){

                    j = j_array[depth](y,x,z);

                }
            }
        }

        if(j != 0){
            //neighbour is on same level
            neigh_key.y = y;
            neigh_key.x = x;
            neigh_key.z = z;
            neigh_key.j = (int)j;
            neigh_key.depth = depth;

        } else {

            //child check
            depth = curr_cell.depth + 1;
            x = 2*(curr_cell.x + offset_x) + (offset_x < 0);
            y = 2*(curr_cell.y + offset_y) + (offset_y < 0);
            z = 2*(curr_cell.z + offset_z) + (offset_z < 0);
            j = 0;

            if ( depth < j_array.size()){

                if(x > 0 & x < j_array[depth].x_num){
                    if(y > 0 & y < j_array[depth].y_num){
                        if(z > 0 & z < j_array[depth].z_num){


                            j = j_array[depth](y,x,z);

                        }
                    }
                }
            }

            if(j != 0){
                //neighbour is on same level
                neigh_key.y = y;
                neigh_key.x = x;
                neigh_key.z = z;
                neigh_key.j = (int)j;
                neigh_key.depth = depth;
            }


        }

    }

    //

    return neigh_key;

}



void create_intensity_reference_structure(PartCellStructure<float,uint64_t>& pc_struct,std::vector<MeshData<float>>& int_array){
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

        // First create the particle map_inplace
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
bool compare_neigh_cells(pc_key curr,pc_key neigh,int dir){

    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

    int offset_x = dir_x[dir];
    int offset_y = dir_y[dir];
    int offset_z = dir_z[dir];

    float k_diff = curr.depth - neigh.depth;

    float y_diff,x_diff,z_diff;

    bool success = true;

    if(neigh.depth_p > 20){
        //no neighbour set
        return true;
    }


    if(std::abs(k_diff) > 1){
        std::cout << "k mismatch " << std::endl;
        success = 0;

    } else {
        //check the distance between the cells (They should be bounded as border neighbors)
        if( k_diff == 0){
            // same level
            y_diff = (curr.y - neigh.y);

            x_diff = (curr.x - neigh.x);

            z_diff = (curr.z - neigh.z);


            if( y_diff != -offset_y){
                std::cout << "y error" << std::endl;
                success = 0;
            }

            if( x_diff != -offset_x){
                std::cout << "x error" << std::endl;
                success = 0;
            }

            if( z_diff != -offset_z){
                std::cout << "z error" << std::endl;
                success = 0;
            }


        } else if (k_diff == -1){
            // - 1case
            y_diff = (curr.y*2 +0.5 - neigh.y);

            x_diff = (curr.x*2 + 0.5 - neigh.x);

            z_diff = (curr.z*2 + 0.5 - neigh.z);

            if(offset_y != 0){

                if( y_diff != -1.5*offset_y ){
                    std::cout << "y error" << std::endl;
                    success = 0;
                }
            }
            else {
                y_diff = (curr.y - neigh.y/2);

                if( y_diff != 0 ){
                    std::cout << "y error" << std::endl;
                    success = 0;
                }

            }

            if(offset_x != 0){

                if( x_diff != -1.5*offset_x ){
                    std::cout << "x error" << std::endl;
                    success = 0;
                }
            }
            else {
                x_diff = (curr.x - neigh.x/2);

                if( x_diff != 0 ){
                    std::cout << "x error" << std::endl;
                    success = 0;
                }

            }


            if(offset_z != 0){

                if( z_diff != -1.5*offset_z ){
                    std::cout << "z error" << std::endl;
                    success = 0;
                }
            }
            else {
                z_diff = (curr.z - neigh.z/2);

                if( z_diff != 0 ){
                    std::cout << "z error" << std::endl;
                    success = 0;
                }

            }



        } else {
            /// + 1 case

            y_diff = (((curr.y - (neigh.y*2 + 0.5))));

            x_diff = (((curr.x - (neigh.x*2 + 0.5))));

            z_diff = (((curr.z - (neigh.z*2 + 0.5))));

            if(offset_y != 0){

                if( y_diff != -1.5*offset_y ){
                    std::cout << "y error" << std::endl;
                    success = 0;
                }
            }
            else {

                y_diff = (((curr.y/2 - (neigh.y))));

                if( y_diff != 0 ){
                    std::cout << "y error" << std::endl;
                    success = 0;
                }

            }

            if(offset_x != 0){

                if( x_diff != -1.5*offset_x ){
                    std::cout << "x error" << std::endl;
                    std::cout << curr.y << std::endl;
                    success = 0;
                }
            }
            else {
                x_diff = (((curr.x/2 - (neigh.x))));

                if( x_diff != 0 ){
                    std::cout << "x error" << std::endl;
                    success = 0;
                }

            }


            if(offset_z != 0){

                if( z_diff != -1.5*offset_z ){
                    std::cout << "z error" << std::endl;
                    std::cout << curr.y << std::endl;
                    success = 0;
                }
            }
            else {
                z_diff = (((curr.z/2 - (neigh.z))));

                if( z_diff != 0 ){
                    std::cout << "z error" << std::endl;
                    success = 0;
                }

            }
        }
    }

    return success;

}
bool compare_neigh(pc_key curr,pc_key neigh,int dir){

    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

    int offset_x = dir_x[dir];
    int offset_y = dir_y[dir];
    int offset_z = dir_z[dir];

    float k_diff = curr.depth_p - neigh.depth_p;

    float y_diff,x_diff,z_diff;

    bool success = true;

    if(neigh.depth_p > 20){
        //no neighbour set
        return true;
    }


    if(std::abs(k_diff) > 1){
        std::cout << "k mismatch " << std::endl;
        success = 0;

    } else {
        //check the distance between the cells (They should be bounded as border neighbors)
        if( k_diff == 0){
            // same level
            y_diff = (curr.y_p - neigh.y_p);

            x_diff = (curr.x_p - neigh.x_p);

            z_diff = (curr.z_p - neigh.z_p);


            if( y_diff != -offset_y){
                std::cout << "y error" << std::endl;
                success = 0;
            }

            if( x_diff != -offset_x){
                std::cout << "x error" << std::endl;
                success = 0;
            }

            if( z_diff != -offset_z){
                std::cout << "z error" << std::endl;
                success = 0;
            }


        } else if (k_diff == -1){
            // - 1case
            y_diff = (curr.y_p*2 +0.5 - neigh.y_p);

            x_diff = (curr.x_p*2 + 0.5 - neigh.x_p);

            z_diff = (curr.z_p*2 + 0.5 - neigh.z_p);

            if(offset_y != 0){

                if( y_diff != -1.5*offset_y ){
                    std::cout << "y error" << std::endl;
                    success = 0;
                }
            }
            else {
                y_diff = (curr.y_p - neigh.y_p/2);

                if( y_diff != 0 ){
                    std::cout << "y error" << std::endl;
                    success = 0;
                }

            }

            if(offset_x != 0){

                if( x_diff != -1.5*offset_x ){
                    std::cout << "x error" << std::endl;
                    success = 0;
                }
            }
            else {
                x_diff = (curr.x_p - neigh.x_p/2);

                if( x_diff != 0 ){
                    std::cout << "x error" << std::endl;
                    success = 0;
                }

            }


            if(offset_z != 0){

                if( z_diff != -1.5*offset_z ){
                    std::cout << "z error" << std::endl;
                    success = 0;
                }
            }
            else {
                z_diff = (curr.z_p - neigh.z_p/2);

                if( z_diff != 0 ){
                    std::cout << "z error" << std::endl;
                    success = 0;
                }

            }



        } else {
            /// + 1 case

            y_diff = (((curr.y_p - (neigh.y_p*2 + 0.5))));

            x_diff = (((curr.x_p - (neigh.x_p*2 + 0.5))));

            z_diff = (((curr.z_p - (neigh.z_p*2 + 0.5))));

            if(offset_y != 0){

                if( y_diff != -1.5*offset_y ){
                    std::cout << "y error" << std::endl;
                    success = 0;
                }
            }
            else {

                y_diff = (((curr.y_p/2 - (neigh.y_p))));

                if( y_diff != 0 ){
                    std::cout << "y error" << std::endl;
                    success = 0;
                }

            }

            if(offset_x != 0){

                if( x_diff != -1.5*offset_x ){
                    std::cout << "x error" << std::endl;
                    std::cout << curr.y_p << std::endl;
                    success = 0;
                }
            }
            else {
                x_diff = (((curr.x_p/2 - (neigh.x_p))));

                if( x_diff != 0 ){
                    std::cout << "x error" << std::endl;
                    success = 0;
                }

            }


            if(offset_z != 0){

                if( z_diff != -1.5*offset_z ){
                    std::cout << "z error" << std::endl;
                    std::cout << curr.y_p << std::endl;
                    success = 0;
                }
            }
            else {
                z_diff = (((curr.z_p/2 - (neigh.z_p))));

                if( z_diff != 0 ){
                    std::cout << "z error" << std::endl;
                    success = 0;
                }

            }
        }
    }

    if (success == 0){
        std::cout << dir << " y: " << curr.y_p << " x: " << curr.x_p <<  " z: " <<curr.z_p << " depth: " <<curr.depth_p <<  std::endl;
        std::cout << dir << " y: " << neigh.y_p << " x: " << neigh.x_p <<  " z: " <<neigh.z_p << " depth: " <<neigh.depth_p <<  std::endl;
    }

    return success;

}
bool utest_alt_part_struct(PartCellStructure<float,uint64_t>& pc_struct){
    //
    //  Calculate Neighbours Using Iterators
    //
    //

    bool success = true;

    std::vector<MeshData<float>> int_array;

    create_intensity_reference_structure(pc_struct,int_array);

    ParticleDataNew<float, uint64_t> part_new;

    part_new.initialize_from_structure(pc_struct);

    Part_timer timer;

    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable

    //
    // Extra variables required
    //

    std::vector<int> labels;
    labels.resize(1,0);

    std::vector<int> neigh_labels;
    neigh_labels.reserve(10);

    PartCellNeigh<uint64_t> neigh_part_keys; // data structure for holding particle or cell neighbours
    PartCellNeigh<uint64_t> neigh_cell_keys;

    float neigh;

    timer.verbose_flag = true;

    timer.start_timer("iterate parts");

    for(int direction = 0; direction < 6;direction++){


        for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = part_new.access_data.x_num[depth];
            const unsigned int z_num_ = part_new.access_data.z_num[depth];

            CurrentLevel<float,uint64_t> curr_level;
            curr_level.set_new_depth(depth,part_new);

            NeighOffset<float,uint64_t> neigh_y(direction);

            neigh_y.set_new_depth(curr_level,part_new);

            //#pragma omp parallel for default(shared) private(p,z_,x_,j_,neigh) firstprivate(curr_level,neigh_x,neigh_z,neigh_y) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure

                for(x_ = 0;x_ < x_num_;x_++){

                    curr_level.set_new_xz(x_,z_,part_new);
                    neigh_y.set_new_row(curr_level,part_new);

                    for(j_ = 0;j_ < curr_level.j_num;j_++){

                        bool iscell = curr_level.new_j(j_,part_new);

                        if (iscell){
                            //Indicates this is a particle cell node
                            curr_level.update_cell(part_new);

                            neigh_y.iterate(curr_level,part_new);

                            std::vector<pc_key> neigh_keys = neigh_y.get_key();

                            pc_key curr_level_key = curr_level.get_key();

                            neigh = neigh_y.get_part_single(part_new.particle_data);

                            for(int c = 0; c < neigh_keys.size();c++){

                                bool error = compare_neigh(curr_level_key,neigh_keys[c],direction);

                                if(!error){
                                    success = false;
                                }

                            }


                            float int_comp = neigh_y.get_int(int_array);



                            if(int_comp != neigh){
                                std::cout << "Neighbour Intensity Error" << std::endl;
                                success = false;
                            }



                        } else {

                            curr_level.update_gap();

                        }


                    }
                }
            }
        }

    }

    timer.stop_timer();

    if(success){
        std::cout << " New Part Scheme Neigh Success" << std::endl;
    } else {
        std::cout << " ***New Part Scheme Neigh Fail***" << std::endl;
    }

    return success;

}
bool utest_apr_serial_iterate(PartCellStructure<float,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2018
    //
    //  Test for the serial APR iterator
    //


    bool success = true;

    APR<float> apr;

    create_pc_data_new(apr,pc_struct);


    std::vector<MeshData<float>> int_array;

    create_intensity_reference_structure(pc_struct,int_array);

    APRIterator<float> apr_iterator(apr);
    uint64_t particle_number = 0;

    for (particle_number = 0; particle_number < apr_iterator.num_parts_total; ++particle_number) {
        float apr_val = apr_iterator(apr.particles_int);

        float check_val = int_array[apr_iterator.depth()](apr_iterator.y(),apr_iterator.x(),apr_iterator.z());

        if(check_val!=apr_val){
            success = false;
        }

    }

    return success;
}

bool utest_apr_parallel_iterate(PartCellStructure<float,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2018
    //
    //  Test of the APR openmp parallel iteration
    //
    bool success = true;

    APR<float> apr;

    create_pc_data_new(apr,pc_struct);

    std::vector<MeshData<float>> int_array;

    create_intensity_reference_structure(pc_struct,int_array);

    APRIterator<float> apr_iterator(apr);
    uint64_t particle_number = 0;

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
    for (particle_number = 0; particle_number < apr_iterator.num_parts_total; ++particle_number) {
        //needed step for any parallel loop (update to the next particle_number)
        apr_iterator.set_iterator_to_particle_by_number(particle_number);

        float apr_val = apr_iterator(apr.particles_int);

        float check_val = int_array[apr_iterator.depth()](apr_iterator.y(),apr_iterator.x(),apr_iterator.z());

        if(check_val!=apr_val){
            success = false;
        }

    }


    uint64_t counter = 0;

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator) reduction(+:counter)
        for (particle_number = apr_iterator.particles_level_begin(level); particle_number <  apr_iterator.particles_level_end(level); ++particle_number) {
            //
            //  Parallel loop over level
            //
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            counter++;

            if(apr_iterator.level() == level){

            } else{
                success = false;
            }
        }
    }

    if(counter != apr_iterator.num_parts_total){
        success = false;
    }


    ///
    ///  By level and z slice (loop over x and then y)
    ///

    counter = 0;

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        for(unsigned int z = 0; z < apr_iterator.spatial_index_z_max(level); ++z) {

            uint64_t  begin = apr_iterator.particles_z_begin(level,z);
            uint64_t  end = apr_iterator.particles_z_end(level,z);

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator) reduction(+:counter)
            for (particle_number = apr_iterator.particles_z_begin(level,z);
                 particle_number < apr_iterator.particles_z_end(level,z); ++particle_number) {
                //
                //  Parallel loop over level
                //
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                counter++;

                if (apr_iterator.z() == z) {

                } else {
                    success = false;
                }
            }
        }
    }

    if(counter != apr_iterator.num_parts_total){
        success = false;
    }


    ///
    ///  By level and z slice and x (loop over particles in y (memory order))
    ///

    counter = 0;

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        for (unsigned int z = 0; z < apr_iterator.spatial_index_z_max(level); ++z) {
            for (unsigned int x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

                uint64_t begin = apr_iterator.particles_zx_begin(level, z, x);
                uint64_t end = apr_iterator.particles_zx_end(level, z, x);

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator) reduction(+:counter)
                for (particle_number = apr_iterator.particles_zx_begin(level, z, x);
                     particle_number < apr_iterator.particles_zx_end(level, z, x); ++particle_number) {
                    //
                    //  Parallel loop over level
                    //
                    apr_iterator.set_iterator_to_particle_by_number(particle_number);

                    counter++;

                    if (apr_iterator.x() == x) {

                    } else {
                        success = false;
                    }
                }
            }
        }
    }

    if(counter != apr_iterator.num_parts_total){
        success = false;
    }


    return success;

}


bool check_neighbours(APR<float>& apr,APRIteratorNew<float>& current,APRIteratorNew<float>& neigh){


    bool success = true;

    if(abs((float)neigh.level() - (float)current.level())>1){
        success = false;
    }

    float delta_x = current.x_global() - neigh.x_global();
    float delta_y = current.y_global() - neigh.y_global();
    float delta_z = current.z_global() - neigh.z_global();

    float resolution_max = 1.11*(0.5*pow(2,current.level_max()-current.level()) + 0.5*pow(2,neigh.level_max()-neigh.level()));

    float distance = sqrt(pow(delta_x,2)+pow(delta_y,2)+pow(delta_z,2));

    if(distance > resolution_max){
        success = false;
    }

    return success;
}

bool check_neighbour_out_of_bounds(APRIteratorNew<float>& current,uint8_t face){


    uint64_t num_neigh = current.number_neighbours_in_direction(face);

    if(num_neigh ==0){
        ParticleCell neigh = current.get_neigh_particle_cell();

        if( (neigh.x >= current.spatial_index_x_max(neigh.level) ) | (neigh.y >= current.spatial_index_y_max(neigh.level) ) | (neigh.z >= current.spatial_index_z_max(neigh.level) )  ){
            return true;
        } else {
            return false;
        }
    }

    return true;
}

bool utest_apr_serial_neigh(PartCellStructure<float,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2018
    //
    //  Test for the serial APR iterator neighbours
    //


    bool success = true;

    APR<float> apr;

    create_pc_data_new(apr,pc_struct);

    std::vector<MeshData<float>> int_array;

    APRIterator<float> neigh_it(apr);

    create_intensity_reference_structure(pc_struct,int_array);

    APRIterator<float> apr_iterator_1(apr);

    uint64_t particle_number;
    for (particle_number = 0; particle_number < apr_iterator_1.num_parts_total; ++particle_number) {
        apr_iterator_1.set_iterator_to_particle_by_number(particle_number);
        //now we only update the neighbours, and directly access them through a neighbour iterator
        apr_iterator_1.update_all_neighbours();

        //loop over all the neighbours and set the neighbour iterator to it
        for (int dir = 0; dir < 6; ++dir) {
            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

            for (int index = 0; index < apr_iterator_1.number_neighbours_in_direction(dir); ++index) {
                // on each face, there can be 0-4 neighbours accessed by index
                if(neigh_it.set_neighbour_iterator(apr_iterator_1, dir, index)){
                    //will return true if there is a neighbour defined
                    float apr_val = neigh_it(apr.particles_int);
                    float check_val = int_array[neigh_it.depth()](neigh_it.y(),neigh_it.x(),neigh_it.z());

                    if(check_val!=apr_val){
                        success = false;
                    }

                }
            }
        }

    }


    ExtraParticleData<float> index_check;
    uint64_t counter = 0;

    for (particle_number = 0; particle_number < apr_iterator_1.num_parts_total; ++particle_number) {
        apr_iterator_1.set_iterator_to_particle_by_number(particle_number);

        //now we only update the neighbours, and directly access them through a neighbour iterator

        index_check.data.push_back(apr_iterator_1(apr.particles_int));
        counter++;

        //loop over all the neighbours and set the neighbour iterator to it
        for (int dir = 0; dir < 6; ++dir) {
            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
            apr_iterator_1.update_direction_neighbours(dir);

            for (int index = 0; index < apr_iterator_1.number_neighbours_in_direction(dir); ++index) {
                // on each face, there can be 0-4 neighbours accessed by index
                if(neigh_it.set_neighbour_iterator(apr_iterator_1, dir, index)){
                    //will return true if there is a neighbour defined
                    float apr_val = neigh_it(apr.particles_int);
                    float check_val = int_array[neigh_it.depth()](neigh_it.y(),neigh_it.x(),neigh_it.z());

                    if(check_val!=apr_val){
                        success = false;
                    }

                }
            }
        }

    }


    APRAccess apr_access2;
    std::vector<std::vector<uint8_t>> p_map;

    apr_access2.generate_pmap(apr,p_map);

    apr_access2.initialize_structure_from_particle_cell_tree(apr,p_map);


    APRIteratorNew<float> neighbour_iterator(apr_access2);
    APRIteratorNew<float> apr_iterator(apr_access2);


    for (particle_number = 0; particle_number < apr_iterator.total_number_parts(); ++particle_number) {

        apr_iterator.set_iterator_to_particle_by_number(particle_number);
        //counter++;

        float temp = 0;

        //loop over all the neighbours and set the neighbour iterator to it
        for (int direction = 0; direction < 6; ++direction) {
            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
            apr_iterator.find_neighbours_in_direction(direction);

            success = check_neighbour_out_of_bounds(apr_iterator,direction);

            for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {

                // on each face, there can be 0-4 neighbours accessed by index
                if(neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)){
                    //will return true if there is a neighbour defined
                    float apr_val = neighbour_iterator(index_check);
                    float check_val = int_array[neighbour_iterator.level()](neighbour_iterator.y(),neighbour_iterator.x(),neighbour_iterator.z());

                    if(check_val!=apr_val){
                        success = false;
                    }

                    if(!check_neighbours(apr,apr_iterator,neighbour_iterator)){
                        success = false;
                    }

                }

            }

        }
    }


    return success;


};

bool utest_apr_parallel_neigh(PartCellStructure<float,uint64_t>& pc_struct){

    bool success = true;

    APR<float> apr;

    create_pc_data_new(apr,pc_struct);

    std::vector<MeshData<float>> int_array;

    APRIterator<float> neigh_it(apr);

    create_intensity_reference_structure(pc_struct,int_array);

    APRIterator<float> apr_it(apr);
    unsigned int part = 0;

#pragma omp parallel for schedule(static) private(part) firstprivate(apr_it,neigh_it)
    for (part = 0; part < apr_it.num_parts_total; ++part) {
        //needed step for any parallel loop (update to the next part)
        apr_it.set_iterator_to_particle_by_number(part);

        //now we only update the neighbours, and directly access them through a neighbour iterator
        apr_it.update_all_neighbours();

        //loop over all the neighbours and set the neighbour iterator to it
        for (int dir = 0; dir < 6; ++dir) {
            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

            for (int index = 0; index < apr_it.number_neighbours_in_direction(dir); ++index) {
                // on each face, there can be 0-4 neighbours accessed by index
                if(neigh_it.set_neighbour_iterator(apr_it, dir, index)){
                    //will return true if there is a neighbour defined
                    float apr_val = neigh_it(apr.particles_int);
                    float check_val = int_array[neigh_it.depth()](neigh_it.y(),neigh_it.x(),neigh_it.z());

                    if(check_val!=apr_val){
                        success = false;
                    }

                }
            }
        }

    }


#pragma omp parallel for schedule(static) private(part) firstprivate(apr_it,neigh_it)
    for (part = 0; part < apr_it.num_parts_total; ++part) {
        //needed step for any parallel loop (update to the next part)
        apr_it.set_iterator_to_particle_by_number(part);

        //now we only update the neighbours, and directly access them through a neighbour iterator

        //loop over all the neighbours and set the neighbour iterator to it
        for (int dir = 0; dir < 6; ++dir) {
            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
            apr_it.update_direction_neighbours(dir);

            for (int index = 0; index < apr_it.number_neighbours_in_direction(dir); ++index) {
                // on each face, there can be 0-4 neighbours accessed by index
                if(neigh_it.set_neighbour_iterator(apr_it, dir, index)){
                    //will return true if there is a neighbour defined
                    float apr_val = neigh_it(apr.particles_int);
                    float check_val = int_array[neigh_it.depth()](neigh_it.y(),neigh_it.x(),neigh_it.z());

                    if(check_val!=apr_val){
                        success = false;
                    }

                }
            }
        }

    }

    return success;

}

void create_pc_data_new(APR<float>& apr,PartCellStructure<float,uint64_t>& pc_struct){
    //
    //
    //  Moves from the old data structure to the new datastructure (PC = particles, stores V)
    //
    //

    apr.pc_data.org_dims = pc_struct.org_dims;

    apr.pc_data.y_num = pc_struct.y_num;

    //first add the layers
    apr.pc_data.depth_max = pc_struct.depth_max + 1;
    apr.pc_data.depth_min = pc_struct.depth_min;

    apr.pc_data.z_num.resize(apr.pc_data.depth_max+1);
    apr.pc_data.x_num.resize(apr.pc_data.depth_max+1);

    apr.pc_data.y_num.resize(apr.pc_data.depth_max+1);

    apr.pc_data.y_num[apr.pc_data.depth_max] = pc_struct.org_dims[0];

    apr.pc_data.data.resize(apr.pc_data.depth_max+1);

    for(uint64_t i = apr.pc_data.depth_min;i < apr.pc_data.depth_max;i++){
        apr.pc_data.z_num[i] = pc_struct.z_num[i];
        apr.pc_data.x_num[i] = pc_struct.x_num[i];
        apr.pc_data.y_num[i] = pc_struct.y_num[i];
        apr.pc_data.data[i].resize(pc_struct.z_num[i]*pc_struct.x_num[i]);
    }

    apr.pc_data.z_num[apr.pc_data.depth_max] = pc_struct.org_dims[2];
    apr.pc_data.x_num[apr.pc_data.depth_max] = pc_struct.org_dims[1];
    apr.pc_data.y_num[apr.pc_data.depth_max] = pc_struct.org_dims[0];
    apr.pc_data.data[apr.pc_data.depth_max].resize(apr.pc_data.z_num[apr.pc_data.depth_max]*apr.pc_data.x_num[apr.pc_data.depth_max]);

    apr.particles_int.org_dims = pc_struct.org_dims;

    apr.particles_int.y_num = pc_struct.y_num;

    //first add the layers
    apr.particles_int.depth_max = pc_struct.depth_max + 1;
    apr.particles_int.depth_min = pc_struct.depth_min;

    apr.particles_int.z_num.resize(apr.pc_data.depth_max+1);
    apr.particles_int.x_num.resize(apr.pc_data.depth_max+1);

    apr.particles_int.y_num.resize(apr.pc_data.depth_max+1);

    apr.particles_int.y_num[apr.pc_data.depth_max] = pc_struct.org_dims[0];

    apr.particles_int.data.resize(apr.pc_data.depth_max+1);

    for(uint64_t i = apr.particles_int.depth_min;i <= apr.particles_int.depth_max;i++){
        apr.particles_int.z_num[i] = apr.pc_data.z_num[i];
        apr.particles_int.x_num[i] = apr.pc_data.x_num[i];
        apr.particles_int.y_num[i] = apr.pc_data.y_num[i];
        apr.particles_int.data[i].resize(apr.pc_data.z_num[i]*apr.pc_data.x_num[i]);
    }

    //now initialize the entries of the two data sets, access structure

    //initialize loop variables
    int x_;
    int z_;
    int y_;

    int x_seed;
    int z_seed;
    int y_seed;

    uint64_t j_;

    uint64_t status;
    uint64_t node_val;
    uint16_t node_val_part;

    //next initialize the entries;
    Part_timer timer;
    timer.verbose_flag = false;

    std::vector<uint16_t> temp_exist;
    std::vector<uint16_t> temp_location;

    std::vector<float> temp_int;

    timer.start_timer("intiialize access data structure");

    for(uint64_t i = apr.pc_data.depth_max;i >= apr.pc_data.depth_min;i--){

        const unsigned int x_num = apr.pc_data.x_num[i];
        const unsigned int z_num = apr.pc_data.z_num[i];


        const unsigned int x_num_seed = apr.pc_data.x_num[i-1];
        const unsigned int z_num_seed = apr.pc_data.z_num[i-1];

        temp_exist.resize(apr.pc_data.y_num[i]);
        temp_location.resize(apr.pc_data.y_num[i]);
        temp_int.resize(apr.pc_data.y_num[i]);

#pragma omp parallel for default(shared) private(j_,z_,x_,y_,node_val,status,z_seed,x_seed,node_val_part) firstprivate(temp_exist,temp_location) if(z_num*x_num > 100)
        for(z_ = 0;z_ < z_num;z_++){

            for(x_ = 0;x_ < x_num;x_++){

                std::fill(temp_exist.begin(), temp_exist.end(), 0);
                std::fill(temp_location.begin(), temp_location.end(), 0);

                std::fill(temp_int.begin(), temp_int.end(), 0);

                if( i < apr.pc_data.depth_max){
                    //access variables
                    const size_t offset_pc_data = x_num*z_ + x_;
                    const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();

                    y_ = 0;

                    //first loop over
                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                        node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];

                        if(!(node_val&1)){
                            //normal node
                            y_++;
                            //create pindex, and create status (0,1,2,3) and type
                            status = (node_val & STATUS_MASK) >> STATUS_SHIFT;  //need the status masks here, need to move them into the datastructure I think so that they are correctly accessible then to these routines.
                            uint16_t part_offset = (node_val_part & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;

                            if(status > SEED){
                                temp_exist[y_] = status;
                                temp_location[y_] = part_offset;
                            }

                        } else {

                            y_ = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_--;
                        }
                    }
                }

                x_seed = x_/2;
                z_seed = z_/2;

                if( i > apr.pc_data.depth_min){
                    //access variables
                    size_t offset_pc_data = x_num_seed*z_seed + x_seed;
                    const size_t j_num = pc_struct.pc_data.data[i-1][offset_pc_data].size();


                    y_ = 0;

                    //first loop over
                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                        node_val_part = pc_struct.part_data.access_data.data[i-1][offset_pc_data][j_];
                        node_val = pc_struct.pc_data.data[i-1][offset_pc_data][j_];

                        if(!(node_val&1)){
                            //normal node
                            y_++;
                            //create pindex, and create status (0,1,2,3) and type
                            status = (node_val & STATUS_MASK) >> STATUS_SHIFT;  //need the status masks here, need to move them into the datastructure I think so that they are correctly accessible then to these routines.
                            uint16_t part_offset = (node_val_part & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;

                            if(status == SEED){
                                temp_exist[2*y_] = status;
                                temp_exist[2*y_+1] = status;

                                temp_location[2*y_] = part_offset + (z_&1)*4 + (x_&1)*2;
                                temp_location[2*y_+1] = part_offset + (z_&1)*4 + (x_&1)*2 + 1;

                            }

                        } else {

                            y_ = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_--;
                        }
                    }
                }


                size_t first_empty = 0;

                size_t offset_pc_data = x_num*z_ + x_;
                size_t offset_pc_data_seed = x_num_seed*z_seed + x_seed;
                size_t curr_index = 0;
                size_t prev_ind = 0;

                //first value handle the duplication of the gap node

                status = temp_exist[0];

                if((status> 0)){
                    first_empty = 0;
                } else {
                    first_empty = 1;
                }

                size_t part_total= 0;

                for(y_ = 0;y_ < temp_exist.size();y_++){

                    status = temp_exist[y_];

                    if(status> 0){
                        curr_index+= 1 + prev_ind;
                        prev_ind = 0;
                        part_total++;
                    } else {
                        prev_ind = 1;
                    }
                }

                if(curr_index == 0){
                    apr.pc_data.data[i][offset_pc_data].resize(1); //always first adds an extra entry for intialization and extra info
                } else {
                    apr.pc_data.data[i][offset_pc_data].resize(curr_index + 2 - first_empty,0); //gap node to begin, already finishes with a gap node

                }



                curr_index = 0;
                prev_ind = 1;
                size_t prev_coord = 0;

                size_t part_counter=0;

                apr.pc_data.data[i][offset_pc_data][0] = 1;
                apr.pc_data.data[i][offset_pc_data].back() = 1;

                //initialize particles
                apr.particles_int.data[i][offset_pc_data].resize(apr.pc_data.data[i][offset_pc_data].size());

                for(y_ = 0;y_ < temp_exist.size();y_++){

                    status = temp_exist[y_];

                    if((status> 0)){

                        curr_index++;

                        //set starting type
                        if(prev_ind == 1){
                            //gap node
                            //set type

                            apr.pc_data.data[i][offset_pc_data][curr_index-1] = TYPE_GAP;
                            apr.pc_data.data[i][offset_pc_data][curr_index-1] |= (((uint64_t)y_) << NEXT_COORD_SHIFT);
                            apr.pc_data.data[i][offset_pc_data][curr_index-1] |= ( prev_coord << PREV_COORD_SHIFT);
                            apr.pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                            apr.pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);
                            //CHANGE HERE

                            //gap node
                            // pc_data.data[i][offset_pc_data][curr_index-1] = 1; //set type to gap
                            // pc_data.data[i][offset_pc_data][curr_index-1] |= ((y_ - prev_coord) << COORD_DIFF_SHIFT_PARTICLE); //set the coordinate difference

                            curr_index++;
                        }
                        prev_coord = y_;
                        //set type


                        apr.pc_data.data[i][offset_pc_data][curr_index-1] = TYPE_PC;

                        //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                        apr.pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                        apr.pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                        apr.pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                        apr.pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);

                        apr.pc_data.data[i][offset_pc_data][curr_index-1] |= (status << STATUS_SHIFT);

                        //lastly retrieve the intensities
                        if(status == SEED){
                            //seed from up one level
                            apr.particles_int.data[i][offset_pc_data][curr_index-1] = pc_struct.part_data.particle_data.data[i-1][offset_pc_data_seed][temp_location[y_]];
                        }
                        else {
                            //non seed same level
                            apr.particles_int.data[i][offset_pc_data][curr_index-1] = pc_struct.part_data.particle_data.data[i][offset_pc_data][temp_location[y_]];
                        }

                        part_counter++;


                        prev_ind = 0;
                    } else {
                        //store for setting above
                        if(prev_ind == 0){
                            //prev_coord = y_;
                        }

                        prev_ind = 1;

                    }
                }


                int stop = 1;



            }

        }
    }

    timer.stop_timer();


    ///////////////////////////////////
    //
    //  Calculate neighbours
    //
    /////////////////////////////////

    //(+y,-y,+x,-x,+z,-z)
    apr.pc_data.set_neighbor_relationships();


}


bool utest_apr_read_write(PartCellStructure<float,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2018
    //
    //  Checks the IO of the APR hdf5 datasets, also of the ExtraPartCellData
    //
    //


    bool success = true;

    APR<float> apr;

    create_pc_data_new(apr,pc_struct);

    std::vector<std::vector<uint8_t>> p_map;

    APRIterator<float> apr_iterator_old(apr);

    apr.apr_access.generate_pmap(apr,p_map);

    apr.apr_access.initialize_structure_from_particle_cell_tree(apr,p_map);

    APRIteratorNew<float> apr_iterator(apr);
    apr.particles_int_new.data.resize(apr_iterator.total_number_parts());

    uint64_t counter = 0;


    ExtraParticleData<float> index_check;
    std::vector<MeshData<float>> int_array;
    create_intensity_reference_structure(pc_struct,int_array);

    uint64_t particle_number;
    for (particle_number = 0; particle_number < apr_iterator_old.num_parts_total; ++particle_number) {

        apr_iterator_old.set_iterator_to_particle_by_number(particle_number);
        apr.particles_int_new.data[particle_number]=apr_iterator_old(apr.particles_int);
    }



    std::string save_loc = "";
    std::string file_name = "read_write_test";

    new_iterator::APRWriter writer;

    //write the APR
    writer.write_apr(apr,save_loc,file_name);

    APR<float> apr_read;

    writer.read_apr(apr_read,save_loc + file_name + "_apr.h5");
    APRIteratorNew<float> apr_iterator_read(apr_read);

    for (particle_number = 0; particle_number < apr_iterator.total_number_parts(); ++particle_number) {

        apr_iterator.set_iterator_to_particle_by_number(particle_number);
        apr_iterator_read.set_iterator_to_particle_by_number(particle_number);
        //counter++;

        //check the functionality
        if(apr_iterator(apr.particles_int_new)!=apr_iterator_read(apr_read.particles_int_new)){
            success = false;
        }

        if(apr_iterator.level()!=apr_iterator_read.level()){
            success = false;
        }

        if(apr_iterator.x()!=apr_iterator_read.x()){
            success = false;
        }

        if(apr_iterator.y()!=apr_iterator_read.y()){
            success = false;
        }

        if(apr_iterator.z()!=apr_iterator_read.z()){
            success = false;
        }

        if(apr_iterator.type()!=apr_iterator_read.type()){
            success = false;
        }

    }


    //
    // Now check the Extra Part Cell Data
    //



    APRIteratorNew<float> neighbour_iterator(apr_read);
    APRIteratorNew<float> apr_iterator_read2(apr_read);

    for (particle_number = 0; particle_number < apr_iterator_read.total_number_parts(); ++particle_number) {

        apr_iterator_read2.set_iterator_to_particle_by_number(particle_number);
        //counter++;

        float temp = 0;

        //loop over all the neighbours and set the neighbour iterator to it
        for (int direction = 0; direction < 6; ++direction) {
            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
            apr_iterator_read2.find_neighbours_in_direction(direction);

            success = check_neighbour_out_of_bounds(apr_iterator_read2,direction);

            for (int index = 0; index < apr_iterator_read2.number_neighbours_in_direction(direction); ++index) {

                // on each face, there can be 0-4 neighbours accessed by index
                if(neighbour_iterator.set_neighbour_iterator(apr_iterator_read2, direction, index)){
                    //will return true if there is a neighbour defined
                    float apr_val = neighbour_iterator(apr_read.particles_int_new);
                    float check_val = int_array[neighbour_iterator.level()](neighbour_iterator.y(),neighbour_iterator.x(),neighbour_iterator.z());

                    if(check_val!=apr_val){
                        success = false;
                    }

                    if(!check_neighbours(apr,apr_iterator_read2,neighbour_iterator)){
                        success = false;
                    }

                }

            }

        }
    }


    ExtraParticleData<float> extra_data(apr);

    for (particle_number = 0; particle_number < apr_iterator_read.total_number_parts(); ++particle_number) {
        apr_iterator_read.set_iterator_to_particle_by_number(particle_number);
        apr_iterator_read(extra_data) = apr_iterator_read.type();

    }

    //write one of the above results to file
    writer.write_particles_only(save_loc,"example_output",extra_data);

    std::string extra_file_name = save_loc + "example_output" + "_apr_extra_parts.h5";

    ExtraParticleData<float> extra_data_read;

    //you need the same apr used to write it to load it (doesn't save location data)
    writer.read_parts_only(extra_file_name,extra_data_read);

    for (particle_number = 0; particle_number < apr_iterator_read.total_number_parts(); ++particle_number) {
        apr_iterator_read.set_iterator_to_particle_by_number(particle_number);

        apr_iterator_read(extra_data) = apr_iterator_read.type();
        if(apr_iterator_read(extra_data) != apr_iterator_read(extra_data_read)){

            success = false;
        }
    }

    //Repeat with different data-type
    ExtraParticleData<uint16_t> extra_data16(apr);

    for (particle_number = 0; particle_number < apr_iterator_read.total_number_parts(); ++particle_number) {
        apr_iterator_read.set_iterator_to_particle_by_number(particle_number);

        apr_iterator_read(extra_data16) = apr_iterator_read.level();

    }

    //write one of the above results to file
    writer.write_particles_only(save_loc,"example_output16",extra_data16);

    std::string extra_file_name16 = save_loc + "example_output16" + "_apr_extra_parts.h5";

    ExtraParticleData<uint16_t> extra_data_read16;

    //you need the same apr used to write it to load it (doesn't save location data)
    writer.read_parts_only(extra_file_name16,extra_data_read16);

    for (particle_number = 0; particle_number < apr_iterator_read.total_number_parts(); ++particle_number) {
        apr_iterator_read.set_iterator_to_particle_by_number(particle_number);

        if(apr_iterator_read(extra_data16) != apr_iterator_read(extra_data_read16)){
            success = false;
        }
    }

    return success;


}
bool utest_neigh_cells(PartCellStructure<float,uint64_t>& pc_struct){   //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //

    std::vector<MeshData<uint64_t>> j_array;

    create_j_reference_structure(pc_struct,j_array);

    bool success = true;

    Part_timer timer;

    //initialize variables required
    uint64_t node_val_pc; // node variable encoding neighbour and cell information
    uint64_t node_val_part; // node variable encoding part offset status information
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    PartCellNeigh<uint64_t> neigh_part_keys; // data structure for holding particle or cell neighbours
    PartCellNeigh<uint64_t> neigh_cell_keys;
    //
    // Extra variables required
    //

    uint64_t status=0;


    float neigh;

    timer.verbose_flag = false;

    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};


    timer.start_timer("test neigh cells");

    for(int direction = 0;direction < 6;direction++){

        int counter = 0;
        int counter_same = 0;
        int counter_child = 0;
        int counter_parent = 0;

        for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
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


                    int y_coord = 0;

                    pc_key curr_cell;

                    //the y direction loop however is sparse, and must be accessed accordinagly
                    for(j_ = 0;j_ < j_num;j_++){



                        //particle cell node value, used here as it is requried for getting the particle neighbours
                        node_val_pc = pc_struct.pc_data.data[i][offset_pc_data][j_];

                        if (!(node_val_pc&1)){
                            //Indicates this is a particle cell node
                            y_coord++;

                            pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);

                            status = pc_struct.part_data.access_node_get_status(node_val_part);



                            curr_cell.update_cell(curr_key);

                            curr_cell.y = y_coord;

                            pc_struct.part_data.access_data.pc_key_set_status(curr_key,status);

                            pc_key neigh_miss = find_neigh_cell(curr_cell,direction,j_array);


                            pc_struct.pc_data.get_neighs_face(curr_key,node_val_pc,direction,neigh_cell_keys);


                            //loop over the nieghbours
                            for(int n = 0; n < neigh_cell_keys.neigh_face[direction].size();n++){
                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                uint64_t neigh_key = neigh_cell_keys.neigh_face[direction][n];

                                if(neigh_key > 0){
                                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                    //pc_struct.pc_data.get_neigh_coordinates_cell(neigh_cell_keys,direction,n,y_coord,y_neigh,x_neigh,z_neigh,depth_neigh);
                                    pc_key neigh_cell = pc_struct.pc_data.get_neigh_coordinates_cell(neigh_cell_keys,direction,n,y_coord);
                                    //Get the neighbour status, we need to access the cell

                                    //then we can get the status from this

                                    bool error = compare_neigh_cells(curr_cell,neigh_cell,direction);

                                    if(!error){
                                        success = false;
                                    }
                                }


                            }

                            if(neigh_cell_keys.neigh_face[direction].size() == 0){

                                bool out_bounds = false;

                                int x_numc = pc_struct.x_num[i];
                                int y_numc = pc_struct.y_num[i];
                                int z_numc = pc_struct.z_num[i];

                                int neigh_x = x_ + dir_x[direction];
                                int neigh_y = y_coord + dir_y[direction];
                                int neigh_z = z_ + dir_z[direction];

                                if(neigh_x < 0 | neigh_x >= pc_struct.x_num[i]){
                                    out_bounds = true;
                                }

                                if(neigh_y < 0 | neigh_y >= pc_struct.y_num[i]){
                                    out_bounds = true;
                                }

                                if(neigh_z < 0 | neigh_z >= pc_struct.z_num[i]){
                                    out_bounds = true;
                                }

                                if(!out_bounds){

                                    pc_key neigh_miss = find_neigh_cell(curr_cell,direction,j_array);

                                    int neigh_type = 0;

                                    if(neigh_miss.depth == i){
                                        //same level
                                        counter_same++;
                                        neigh_type = 1;

                                    } else if(neigh_miss.depth == (i - 1)){
                                        neigh_type = 2;
                                        counter_parent++;

                                    } else if(neigh_miss.depth == (i + 1)){
                                        neigh_type = 3;
                                        counter_child++;

                                    }

                                    node_key debug_node;

                                    if(direction == 0){
                                        uint64_t node_val = pc_struct.pc_data.data[i][offset_pc_data][j_ + 1];


                                        debug_node.update_node(node_val);

                                        std::cout << j_ + 1 << " " << j_num << " " << neigh_type << std::endl;
                                        std::cout << x_ << " " << y_coord << " " << z_ << " " << i << std::endl;

                                    } else if (direction == 1){
                                        uint64_t node_val = pc_struct.pc_data.data[i][offset_pc_data][j_ - 1];

                                        debug_node.update_node(node_val);


                                    } else {
                                        debug_node.update_node(node_val_pc);
                                    }



                                    if(neigh_type > 0){
                                        counter++;
                                        success = false;
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
        std::cout << direction << " " << counter << " s " << counter_same << " c " << counter_child << " p " << counter_parent << std::endl;
    }


    if(success){
        std::cout << " Cell Neighbour Success" << std::endl;
    } else {
        std::cout << " Cell Neighbour Fail" << std::endl;
    }

    timer.stop_timer();

    return success;

}


bool compare_cells(pc_key key1,pc_key key2){
    int same = (key1.x == key2.x) + (key1.z == key2.z) + (key1.j == key2.j) + (key1.depth == key2.depth);

    if(same == 4){
        return true;
    } else {
        return false;
    }


}



bool utest_moore_neighbours(PartCellStructure<float,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2017
    //
    //  Testing for full moore neighbourhood, it is seprated into face, edge, and corner.
    //

    //std::cout << "Should break here" << std::endl;
    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    ExtraPartCellData<uint16_t> y_vec;

    create_y_data(y_vec,part_new);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    std::vector<MeshData<uint64_t>> j_array;

    create_j_reference_structure(pc_data,j_array);

    std::vector<PartCellNeigh<uint64_t>> neigh_keys;

    PartCellNeigh<uint64_t> face_neigh;
    PartCellNeigh<uint64_t> edge_neigh(NUM_EDGES);
    PartCellNeigh<uint64_t> corner_neigh(NUM_CORNERS);

    neigh_keys.push_back(face_neigh);
    neigh_keys.push_back(edge_neigh);
    neigh_keys.push_back(corner_neigh);

    std::vector<uint16_t> coords = {0,0,0};

    uint64_t curr_key;

    bool success = true;

    uint64_t z_, x_, j_, y_, i, k;

    for (uint64_t depth = (y_vec.depth_min); depth <= y_vec.depth_max; depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = y_vec.x_num[depth];
        const unsigned int z_num_ = y_vec.z_num[depth];



        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure
            coords[2] = z_;
            pc_data.pc_key_set_z(curr_key,z_);
            pc_data.pc_key_set_depth(curr_key,depth);

            for (x_ = 0; x_ < x_num_; x_++) {
                coords[1] = x_;
                pc_data.pc_key_set_x(curr_key,x_);
                const unsigned int pc_offset = x_num_ * z_ + x_;

                uint16_t y_counter = 0;

                for (j_ = 0; j_ < pc_data.data[depth][pc_offset].size(); j_++) {

                    uint64_t node_val = pc_data.data[depth][pc_offset][j_];

                    if (!(node_val&1)) {

                        const int y = y_vec.data[depth][pc_offset][y_counter];

                        pc_data.pc_key_set_j(curr_key, j_);

                        coords[0] = y;

                        pc_data.get_neighs_all_diag(curr_key, node_val, neigh_keys, coords);

                        pc_key curr_cell;
                        curr_cell.update_cell(curr_key);
                        curr_cell.y = y;

                        ////////////////////////////////////////////////////////
                        //
                        //  Tsting edges
                        //
                        /////////////////////////////////////////////////////////

                        std::vector<pc_key> j_neigh;

                        std::vector<pc_key> neigh_moore;

                        unsigned int counter_edges = 0;

                        for (int n = 0; n < neigh_keys[1].neigh_face.size(); ++n) {

                            pc_key neigh_miss = find_neigh_cell_edge(curr_cell, n, j_array);

                            if(neigh_miss.j > 0){
                                j_neigh.push_back(neigh_miss);
                            }

                            if(neigh_keys[1].neigh_face[n].size() > 0) {

                                pc_key neigh_edge;
                                uint64_t neigh = neigh_keys[1].neigh_face[n][0];
                                neigh_edge.update_cell(neigh);

                                uint64_t test_i = pc_data.get_val(neigh);

                                bool same = neigh_edge.compare_cell(neigh_miss);

                                neigh_moore.push_back(neigh_edge);


                                counter_edges++;

                                if(!same){
                                    success = false;

                                }
                            }
                        }

                        ///////////////////////////////////////////////////////////////
                        //
                        //  Testing corners
                        //
                        /////////////////////////////////////////////////////////////

                        unsigned int counter_corners = 0;


                        for (int n = 0; n < neigh_keys[2].neigh_face.size(); ++n) {

                            pc_key neigh_miss = find_neigh_cell_corner(curr_cell, n, j_array);

                            if(neigh_miss.j > 0){
                                j_neigh.push_back(neigh_miss);
                            }


                            if(neigh_keys[2].neigh_face[n].size() > 0) {



                                pc_key neigh_corner;
                                uint64_t neigh = neigh_keys[2].neigh_face[n][0];
                                neigh_corner.update_cell(neigh);

                                uint64_t test_i = pc_data.get_val(neigh);

                                bool same = neigh_corner.compare_cell(neigh_miss);


                                neigh_moore.push_back(neigh_corner);

                                if(!same){
                                    success = false;

                                }

                                counter_corners++;

                            }
                        }


                        //faces

                        unsigned int counter_face = 0;

                        for (int n = 0; n < neigh_keys[0].neigh_face.size(); ++n) {

                            pc_key neigh_miss = find_neigh_cell(curr_cell, n, j_array);

                            if(neigh_miss.j > 0){
                                j_neigh.push_back(neigh_miss);
                            }

                            if(neigh_keys[0].neigh_face[n].size() > 0) {

                                pc_key neigh_face;
                                uint64_t neigh = neigh_keys[0].neigh_face[n][0];
                                neigh_face.update_cell(neigh);

                                uint64_t test_i = pc_data.get_val(neigh);

                                bool same = neigh_face.compare_cell(neigh_miss);

                                neigh_moore.push_back(neigh_face);

                                counter_face++;

                                if(!same){
                                    success = false;

                                }
                            }
                        }

                        pc_key comp;

                        unsigned int size_before = neigh_moore.size();


                        std::unique(neigh_moore.begin(),neigh_moore.end(),compare_cells);

                        unsigned int size_after = neigh_moore.size();

                        unsigned int num_j_before = j_neigh.size();

                        std::unique(j_neigh.begin(),j_neigh.end(),compare_cells);

                        unsigned int num_j = j_neigh.size();


                        if((size_before == size_after) & (num_j == size_after) ){

                        } else {
                            success = false;
                        }


                        y_counter++;
                    }

                }
            }
        }
    }

    std::cout << " Finished MOORE neighbourhood test" << std::endl;


    return success;


}

bool utest_neigh_parts(PartCellStructure<float,uint64_t>& pc_struct){   //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //

    //part_new.initialize_from_structure(pc_struct.pc_data);
    //part_new.transfer_intensities(pc_struct.part_data);




    Part_timer timer;

    //initialize variables required
    uint64_t node_val_pc; // node variable encoding neighbour and cell information
    uint64_t node_val_part; // node variable encoding part offset status information
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    PartCellNeigh<uint64_t> neigh_part_keys; // data structure for holding particle or cell neighbours
    PartCellNeigh<uint64_t> neigh_cell_keys;
    //
    // Extra variables required
    //

    uint64_t status=0;
    uint64_t part_offset=0;
    uint64_t p;



    timer.verbose_flag = false;


    bool success = true;


    timer.start_timer("iterate parts old");

    for(int direction = 0;direction < 6;direction++){

        for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
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


                    int y_coord = 0;

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

                            pc_key curr_part;
                            curr_part.update_cell(curr_key);

                            curr_part.y = y_coord;

                            pc_struct.part_data.access_data.pc_key_set_status(curr_key,status);

                            //loop over the particles
                            for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                                //first set the particle index value in the particle_data array (stores the intensities)
                                pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                                pc_struct.part_data.access_data.pc_key_set_partnum(curr_key,p);
                                //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering

                                pc_struct.part_data.get_part_neighs_face(direction,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                                curr_part.update_part(curr_key);

                                bool error = true;

                                for(int n = 0; n < neigh_part_keys.neigh_face[direction].size();n++){
                                    uint64_t neigh_part = neigh_part_keys.neigh_face[direction][n];


                                    if(neigh_part > 0){
                                        pc_key neigh_part_k =  pc_struct.pc_data.get_neigh_coordinates_part(neigh_part_keys,direction,n,y_coord);

                                        error = compare_neigh(curr_part,neigh_part_k,direction);

                                        if (!error){
                                            success = false;
                                            pc_struct.part_data.get_part_neighs_face(direction,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                                            pc_key neigh_part_d =  pc_struct.pc_data.get_neigh_coordinates_part(neigh_part_keys,direction,n,y_coord);

                                            uint64_t neigh_node = pc_struct.pc_data.get_val(neigh_part);

                                            node_key neigh_node_key;
                                            neigh_node_key.update_node(neigh_node);



                                            std::cout << status << " " << neigh_node_key.status << std::endl;

                                            for(int i = 0; i < neigh_cell_keys.neigh_face[direction].size();i++){
                                                pc_key neigh_c;
                                                neigh_c.update_cell(neigh_cell_keys.neigh_face[direction][i]);

                                            }

                                            int stop = 1;

                                        }


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

    }

    timer.stop_timer();

    if(success){
        std::cout << " Part Neigh Check Original - Success " << std::endl;
    } else {
        std::cout << "***** Part Neigh Check Original - Fail ******" << std::endl;

    }

    return success;


}



