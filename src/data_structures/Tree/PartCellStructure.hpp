///////////////////
//
//  Bevan Cheeseman 2016
//
//  Part cell base class, specifies interface
//
///////////////

#ifndef PARTPLAY_PARTCELLSTRUCTURE_HPP
#define PARTPLAY_PARTCELLSTRUCTURE_HPP

#include "PartCellBase.hpp"
#include "PartCellData.hpp"
#include "ParticleData.hpp"
#include "../particle_map.hpp"
#include "../meshclass.h"
#include "../../io/writeimage.h"

#include <vector>
#include <algorithm>
#include <array>



//parent node defitions




template <typename T,typename S> // type T is the image type, type S is the data structure base type
class PartCellStructure: public PartCellBase<T,S> {
    
    
    
private:
    
    PartCellData<S> pc_data; //particle cell data
    ParticleData<T,S> part_data; //individual particle data
    
    unsigned int depth_min;
    unsigned int depth_max;
    
    
    inline void add_status(uint8_t part_map_status,uint64_t* node_val){
        //
        //  takes in a node value and encodes the new status value
        //
        
        switch(part_map_status){
            case TAKENSTATUS:
            {
               
                *node_val = *node_val | SEED_SHIFTED;
                break;
            }
            case NEIGHBOURSTATUS:
            {
                
                 *node_val = *node_val | BOUNDARY_SHIFTED;
                break;
            }
            case SLOPESTATUS:
            {
                
                *node_val = *node_val | FILLER_SHIFTED;
                break;
            }
                
        }
        
    }
    
    void create_partcell_structure(Particle_map<T>& part_map){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes an optimal part_map configuration from the pushing scheme and creates an efficient data structure for procesing.
        //
        
        Part_timer timer;
        timer.verbose_flag = true;
        
        
        timer.start_timer("intiialize base");
        pc_data.initialize_base_structure(part_map);
        timer.stop_timer();
        
        
        
        //initialize loop variables
        int x_;
        int z_;
        uint64_t y_;
        uint64_t j_;
        
        //next initialize the entries;
        
        uint16_t curr_index;
        coords3d curr_coords;
        uint8_t status;
        uint8_t prev_ind = 0;
        
        timer.start_timer("intiialize part_cells");
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            const unsigned int y_num = part_map.layers[i].y_num;
            
            
#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){

                for(x_ = 0;x_ < x_num;x_++){
                    
                    size_t first_empty = 0;
                    const size_t offset_part_map = x_*y_num + z_*y_num*x_num;
                    const size_t offset_pc_data = x_num*z_ + x_;
                    curr_index = 0;
                    prev_ind = 0;
                    
                    //first value handle the duplication of the gap node
                    
                    status = part_map.layers[i].mesh[offset_part_map];
                    if((status> 0) & (status < 8)){
                        first_empty = 0;
                    } else {
                        first_empty = 1;
                    }
                    
                    for(y_ = 0;y_ < y_num;y_++){
                        
                        status = part_map.layers[i].mesh[offset_part_map + y_];
                        
                        if((status> 0) & (status < 8)){
                            curr_index+= 1 + prev_ind;
                            prev_ind = 0;
                        } else {
                            prev_ind = 1;
                        }
                    }
                    
                    if(curr_index == 0){
                        pc_data.data[i][offset_pc_data].resize(1); //always first adds an extra entry for intialization and extra info
                    } else {
                       
                        pc_data.data[i][offset_pc_data].resize(curr_index + 2 - first_empty,TYPE_GAP_END); //gap node to begin, already finishes with a gap node
                    
                    }
                }
            }
            
        }

        timer.stop_timer();
        
        timer.start_timer("First initialization step");
        
        //
        //  In this loop we set, prev_coord, next_coord, type and status
        //
        
        uint64_t prev_coord = 0;
        
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            const unsigned int y_num = part_map.layers[i].y_num;
            
#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind,prev_coord) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    
                    const size_t offset_part_map = x_*y_num + z_*y_num*x_num;
                    const size_t offset_pc_data = x_num*z_ + x_;
                    curr_index = 0;
                    prev_ind = 1;
                    prev_coord = 0;
                    
                    //initialize the first values type
                    pc_data.data[i][offset_pc_data][0] = TYPE_GAP_END;
                    
                    for(y_ = 0;y_ < y_num;y_++){
                        
                        status = part_map.layers[i].mesh[offset_part_map + y_];
                        
                        if((status> 0) & (status < 8)){
                            
                            curr_index++;
                            
                            //set starting type
                            if(prev_ind == 1){
                                //gap node
                                //set type
                                pc_data.data[i][offset_pc_data][curr_index-1] = TYPE_GAP;
                                pc_data.data[i][offset_pc_data][curr_index-1] |= (y_ << NEXT_COORD_SHIFT);
                                pc_data.data[i][offset_pc_data][curr_index-1] |= ( prev_coord << PREV_COORD_SHIFT);
                                pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                                pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);
                                curr_index++;
                            }
                            
                            //set type
                            pc_data.data[i][offset_pc_data][curr_index-1] = TYPE_PC;
                            
                            //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);
                            
                            //set the status
                            switch(status){
                                case TAKENSTATUS:
                                {
                                    pc_data.data[i][offset_pc_data][curr_index-1] |= SEED_SHIFTED;
                                    break;
                                }
                                case NEIGHBOURSTATUS:
                                {
                                    pc_data.data[i][offset_pc_data][curr_index-1] |= BOUNDARY_SHIFTED;
                                    break;
                                }
                                case SLOPESTATUS:
                                {
                                    pc_data.data[i][offset_pc_data][curr_index-1] |= FILLER_SHIFTED;
                                    break;
                                }
                                    
                            }
                            
                            prev_ind = 0;
                        } else {
                            //store for setting above
                            if(prev_ind == 0){
                                prev_coord = y_;
                            }
                            
                            prev_ind = 1;
                            
                        }
                    }
                    
                    //Initialize the last value to GAP END
                    //pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size()-1] = TYPE_GAP_END;
                }
            }
            
        }
        
        timer.stop_timer();
        
        ///////////////////////////////////
        //
        //  Calculate neighbours
        //
        /////////////////////////////////
        
        set_neighbor_relationsships();
        
        
        /////////////////////////////////////
        //
        //  PARTICLE DATA STRUCTURES
        //
        //////////////////////////////////////
        
        // Initialize the particle data access and intensity structures
        part_data.initialize_from_structure(pc_data);
        
        // Estimate the intensities from the down sampled images
        
        timer.start_timer("Get the intensities");
        
        // Particles are ordered as ((-y,-x,-z),(+y,-x,-z),(-y,+x,-z),(+y,+x,-z),(-y,-x,+z),(+y,-x,+z),(-y,+x,+z),(+y,+x,+z))

        uint64_t part_offset;
        uint64_t node_val;
        uint64_t y_coord;
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord,part_offset) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    const size_t offset_pc_data = x_num*z_ + x_;
                    y_coord = 0;
                    const size_t j_num = part_data.access_data.data[i][offset_pc_data].size();
                    
                    
                    
                    const size_t offset_part_map_data_0 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*2*z_ + part_map.downsampled[i+1].y_num*2*x_;
                    const size_t offset_part_map_data_1 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*2*z_ + part_map.downsampled[i+1].y_num*2*(x_+1);
                    const size_t offset_part_map_data_2 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*2*(z_+1) + part_map.downsampled[i+1].y_num*2*x_;
                    const size_t offset_part_map_data_3 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*2*(z_+1) + part_map.downsampled[i+1].y_num*2*(x_+1);
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val = part_data.access_data.data[i][offset_pc_data][j_];
                        
                        if (node_val&1){
                            //get the index gap node
                            y_coord += (node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE;
                            y_coord--;
                            
                        } else {
                            //normal node
                            y_coord++;
                            
                            //get and check status
                            status = (node_val & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE;
                            
                            if(status == SEED){
                                //need to sampled the image at depth + 1
                                part_offset = (node_val & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                                
                                //0
                                part_data.particle_data.data[i][offset_pc_data][part_offset] = part_map.downsampled[i+1].mesh[offset_part_map_data_0 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+1] = part_map.downsampled[i+1].mesh[offset_part_map_data_0 + 2*y_coord + 1];
                                //1
                                part_data.particle_data.data[i][offset_pc_data][part_offset+2] = part_map.downsampled[i+1].mesh[offset_part_map_data_1 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+3] = part_map.downsampled[i+1].mesh[offset_part_map_data_1 + 2*y_coord + 1];
                                //3
                                part_data.particle_data.data[i][offset_pc_data][part_offset+4] = part_map.downsampled[i+1].mesh[offset_part_map_data_2 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+5] = part_map.downsampled[i+1].mesh[offset_part_map_data_2 + 2*y_coord + 1];
                                //4
                                part_data.particle_data.data[i][offset_pc_data][part_offset+6] = part_map.downsampled[i+1].mesh[offset_part_map_data_3 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+7] = part_map.downsampled[i+1].mesh[offset_part_map_data_3 + 2*y_coord + 1];
                                
                            }
                            
                            
                        }
                    }
                    
                }
            }
            
            // BOUNDARY/FILLER PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord,part_offset) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    const size_t offset_pc_data = x_num*z_ + x_;
                    y_coord = 0;
                    const size_t j_num = part_data.access_data.data[i][offset_pc_data].size();
                    
                    const size_t offset_part_map_data_0 = part_map.downsampled[i].y_num*part_map.downsampled[i].x_num*z_ + part_map.downsampled[i].y_num*x_;
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        
                        
                        node_val = part_data.access_data.data[i][offset_pc_data][j_];
                        
                        if (node_val&1){
                            //get the index gap node
                            y_coord += (node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE;
                            y_coord--;
                            
                            
                        } else {
                            //normal node
                            y_coord++;
                            
                            //get and check status
                            status = (node_val & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE;
                            
                            //boundary and filler cells only have one particle
                            if(status > SEED){
                                //need to sampled the image at depth + 1
                                part_offset = (node_val & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                                
                                //0
                                part_data.particle_data.data[i][offset_pc_data][part_offset] = part_map.downsampled[i].mesh[offset_part_map_data_0 + y_coord];
                               
                                
                            }
                            
                            
                        }
                    }
                    
                }
            }
            
        }
        
        timer.stop_timer();

        
        
    }
    
    
     // Particles are ordered as ((-y,-x,-z),(+y,-x,-z),(-y,+x,-z),(+y,+x,-z),(-y,-x,+z),(+y,-x,+z),(-y,+x,+z),(+y,+x,+z))
    uint8_t seed_part_y[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    uint8_t seed_part_x[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    uint8_t seed_part_z[8] = {0, 0, 0, 0, 1, 1, 1, 1};
                                                               
    int8_t von_neumann_y[6] = {0, 0, -1, 1, 0, 0};
    int8_t von_neumann_x[6] = {0, -1, 0, 0, 1, 0};
    int8_t von_neumann_z[6] = {-1, 0, 0, 0, 0, 1};
    
public:
    
    uint8_t get_status(uint64_t node_val){
        //
        //  Extracts the status
        //
        
        return (node_val & STATUS_MASK) >> STATUS_SHIFT;
    }
    
    
      
    //decleration
    void initialize_structure(Particle_map<T>& particle_map){
        
        
        for(int i = particle_map.k_min;i <= particle_map.k_max;i++){
            debug_write(particle_map.layers[i],"kmap" + std::to_string(i));
        }
        
        
        //create_sparse_graph_format(particle_map);
        create_partcell_structure(particle_map);
        
        pc_data.test_get_neigh_dir();
    }
    
    
    
    PartCellStructure(Particle_map<T> &particle_map){
        //
        //  initialization of the tree structure
        //
        
        depth_min = particle_map.k_min;
        depth_max = particle_map.k_max;
        
        initialize_structure(particle_map);
    }

    void set_neighbor_relationsships(){
        
        Part_timer timer;
        timer.verbose_flag = true;
        uint64_t z_;
        uint64_t x_;
        uint64_t j_;
        
        timer.start_timer("parent child with child loop x");
        
        unsigned int y_neigh;
        unsigned int y_parent;
        uint64_t j_parent;
        uint64_t j_neigh;
        
        unsigned int y_child;
        uint64_t j_child;
        uint64_t node_val;
        uint64_t y_coord;
        
        
        for(int i = (pc_data.depth_min);i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            
            const unsigned int x_num_parent = pc_data.x_num[i-1];
            
            if(i == pc_data.depth_max){
                
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord) if(z_num*x_num > 100)
                for(z_ = 0;z_ < z_num;z_++){
                    
                    for(x_ = 0;x_ < (x_num-1);x_++){
                        
                        const size_t z_parent = z_/2;
                        const size_t x_parent = (x_+1)/2;
                        
                        const size_t z_neigh = z_;
                        const size_t x_neigh = x_+1;
                        
                        const size_t offset_pc_data = x_num*z_ + x_;
                        const size_t offset_pc_data_parent = x_num_parent*z_parent + x_parent;
                        const size_t offset_pc_data_neigh = x_num*z_neigh + x_neigh;
                        
                        //initialization
                        y_coord = (pc_data.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_neigh = (pc_data.data[i][offset_pc_data_neigh][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_parent = (pc_data.data[i-1][offset_pc_data_parent][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        j_parent = 1;
                        j_neigh = 1;
                        
                        if (pc_data.data[i-1][offset_pc_data_parent].size() == 1){
                            //set to max so its not checked
                            y_parent = 64000;
                        }
                        
                        if (pc_data.data[i][offset_pc_data_neigh].size() == 1){
                            //set to max so its not checked
                            y_neigh = 64000;
                        }
                        
                        y_coord--;
                        
                        const size_t j_num = pc_data.data[i][offset_pc_data].size();
                        const size_t j_num_parent = pc_data.data[i-1][offset_pc_data_parent].size();
                        const size_t j_num_neigh = pc_data.data[i][offset_pc_data_neigh].size();
                        
                        for(j_ = 1;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = pc_data.data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_coord--;
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                while ((y_neigh < y_coord) & (j_neigh < (j_num_neigh-1))){
                                    
                                    j_neigh++;
                                    node_val = pc_data.data[i][offset_pc_data_neigh][j_neigh];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_neigh = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_neigh++;
                                        
                                    } else {
                                        //normal node
                                        y_neigh++;
                                    }
                                    
                                }
                                
                                while ((y_parent < y_coord/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val = pc_data.data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_parent++;
                                        
                                    } else {
                                        //normal node
                                        y_parent++;
                                        
                                    }
                                    
                                }
                                
                                if(y_coord == y_neigh){
                                    pc_data.data[i][offset_pc_data][j_] |= (j_neigh << XP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_SAME << XP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (j_neigh << XM_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (LEVEL_SAME << XM_DEPTH_SHIFT);
                                    
                                } else if (y_coord/2 == y_parent) {
                                    pc_data.data[i][offset_pc_data][j_] |= (j_parent << XP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_DOWN << XP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << XM_INDEX_SHIFT);
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent] |= (LEVEL_UP << XM_DEPTH_SHIFT);
                                } else {
                                    std::cout << "BUG" << std::endl;
                                }
                                
                            }
                            
                        }
                        
                    }
                }
                
            } else if (i == pc_data.depth_min) {
                
                const unsigned int x_num_child = pc_data.x_num[i+1];
                
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord,y_child,j_child) if(z_num*x_num > 100)
                for(z_ = 0;z_ < z_num;z_++){
                    
                    for(x_ = 0;x_ < (x_num-1);x_++){
                        
                        
                        const size_t z_child = z_*2;
                        const size_t x_child = (x_+1)*2;
                        
                        const size_t z_neigh = z_;
                        const size_t x_neigh = x_+1;
                        
                        const size_t offset_pc_data = x_num*z_ + x_;
                        const size_t offset_pc_data_neigh = x_num*z_neigh + x_neigh;
                        const size_t offset_pc_data_child = x_num_child*z_child + x_child;
                        
                        //initialization
                        y_coord = (pc_data.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_neigh = (pc_data.data[i][offset_pc_data_neigh][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_child = (pc_data.data[i+1][offset_pc_data_child][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        
                        j_neigh = 1;
                        j_child = 1;
                        
                        
                        if (pc_data.data[i][offset_pc_data_neigh].size() == 1){
                            //set to max so its not checked
                            y_neigh = 64000;
                        }
                        
                        if (pc_data.data[i+1][offset_pc_data_child].size() == 1){
                            //set to max so its not checked
                            y_child = 64000;
                        }
                        
                        y_coord--;
                        
                        const size_t j_num = pc_data.data[i][offset_pc_data].size();
                        const size_t j_num_neigh = pc_data.data[i][offset_pc_data_neigh].size();
                        const size_t j_num_child = pc_data.data[i+1][offset_pc_data_child].size();
                        
                        for(j_ = 1;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = pc_data.data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_coord--;
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                while ((y_neigh < y_coord) & (j_neigh < (j_num_neigh-1))){
                                    
                                    j_neigh++;
                                    node_val = pc_data.data[i][offset_pc_data_neigh][j_neigh];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_neigh = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_neigh++;
                                        
                                    } else {
                                        //normal node
                                        y_neigh++;
                                        
                                    }
                                    
                                }
                                
                                while ((y_child < y_coord*2) & (j_child < (j_num_child-1))){
                                    
                                    j_child++;
                                    node_val = pc_data.data[i+1][offset_pc_data_child][j_child];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_child = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_child++;
                                        
                                    } else {
                                        //normal node
                                        y_child++;
                                        
                                    }
                                    
                                }
                                
                                if(y_coord == y_neigh){
                                    pc_data.data[i][offset_pc_data][j_] |= (j_neigh << XP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_SAME << XP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (j_ << XM_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (LEVEL_SAME << XM_DEPTH_SHIFT);
                                } else if (y_coord*2 == y_child) {
                                    pc_data.data[i][offset_pc_data][j_] |= (j_child << XP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_UP << XP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i+1][offset_pc_data_child][j_child] |= (j_ << XM_INDEX_SHIFT);
                                    pc_data.data[i+1][offset_pc_data_child][j_child] |= (LEVEL_DOWN << XM_DEPTH_SHIFT);
                                } else {
                                    std::cout << "BUG" << std::endl;
                                }
                                
                            }
                            
                        }
                        
                    }
                }
                
                
                
            } else {
                
                const unsigned int x_num_child = pc_data.x_num[i+1];
                
                
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord,j_child,y_child) if(z_num*x_num > 100)
                for(z_ = 0;z_ < z_num;z_++){
                    
                    for(x_ = 0;x_ < (x_num-1);x_++){
                        
                        const size_t z_parent = z_/2;
                        const size_t x_parent = (x_+1)/2;
                        
                        const size_t z_child = z_*2;
                        const size_t x_child = (x_+1)*2;
                        
                        const size_t z_neigh = z_;
                        const size_t x_neigh = x_+1;
                        
                        const size_t offset_pc_data = x_num*z_ + x_;
                        const size_t offset_pc_data_parent = x_num_parent*z_parent + x_parent;
                        const size_t offset_pc_data_neigh = x_num*z_neigh + x_neigh;
                        const size_t offset_pc_data_child = x_num_child*z_child + x_child;
                        
                        //initialization
                        y_coord = (pc_data.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT ;
                        y_neigh = (pc_data.data[i][offset_pc_data_neigh][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_parent = (pc_data.data[i-1][offset_pc_data_parent][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_child = (pc_data.data[i+1][offset_pc_data_child][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        j_parent = 1;
                        j_neigh = 1;
                        j_child = 1;
                        
                        if (pc_data.data[i-1][offset_pc_data_parent].size() == 1){
                            //set to max so its not checked
                            y_parent = 64000;
                        }
                        
                        if (pc_data.data[i][offset_pc_data_neigh].size() == 1){
                            //set to max so its not checked
                            y_neigh = 64000;
                        }
                        
                        if (pc_data.data[i+1][offset_pc_data_child].size() == 1){
                            //set to max so its not checked
                            y_child= 64000;
                        }
                        
                        y_coord--;
                        
                        const size_t j_num = pc_data.data[i][offset_pc_data].size();
                        const size_t j_num_parent = pc_data.data[i-1][offset_pc_data_parent].size();
                        const size_t j_num_neigh = pc_data.data[i][offset_pc_data_neigh].size();
                        const size_t j_num_child = pc_data.data[i+1][offset_pc_data_child].size();
                        
                        
                        for(j_ = 1;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = pc_data.data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_coord--;
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                while ((y_neigh < y_coord) & (j_neigh < (j_num_neigh-1))){
                                    
                                    j_neigh++;
                                    node_val = pc_data.data[i][offset_pc_data_neigh][j_neigh];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_neigh = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_neigh++;
                                        
                                    } else {
                                        //normal node
                                        y_neigh++;
                                        
                                    }
                                    
                                }
                                
                                while ((y_parent < y_coord/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val = pc_data.data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_parent++;
                                        
                                    } else {
                                        //normal node
                                        y_parent++;
                                        
                                    }
                                    
                                }
                                
                                while ((y_child < y_coord*2) & (j_child < (j_num_child-1))){
                                    
                                    j_child++;
                                    node_val = pc_data.data[i+1][offset_pc_data_child][j_child];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_child = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_child++;
                                        
                                    } else {
                                        //normal node
                                        y_child++;
                                        
                                    }
                                    
                                }
                                
                                if(y_coord == y_neigh){
                                    pc_data.data[i][offset_pc_data][j_] |= (j_neigh << XP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_SAME << XP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (j_ << XM_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (LEVEL_SAME << XM_DEPTH_SHIFT);
                                } else if (y_coord*2 == y_child) {
                                    pc_data.data[i][offset_pc_data][j_] |= (j_child << XP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_UP << XP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i+1][offset_pc_data_child][j_child] |= (j_ << XM_INDEX_SHIFT);
                                    pc_data.data[i+1][offset_pc_data_child][j_child] |= (LEVEL_DOWN << XM_DEPTH_SHIFT);
                                } else if (y_coord/2 == y_parent){
                                    pc_data.data[i][offset_pc_data][j_] |= (j_parent << XP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_DOWN << XP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << XM_INDEX_SHIFT);
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent] |= (LEVEL_UP << XM_DEPTH_SHIFT);
                                } else {
                                    std::cout << "BUG" << std::endl;
                                }
                                
                                
                                
                                
                            }
                            
                        }
                        
                    }
                }
                
                
                
            }
            
            
        }
        
        timer.stop_timer();
        
        
        timer.start_timer("parent child with child loop z");
        
        for(int i = (pc_data.depth_min);i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            
            const unsigned int x_num_parent = pc_data.x_num[i-1];
            
            if(i == pc_data.depth_max){
                
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord) if(z_num*x_num > 100)
                for(z_ = 0;z_ < (z_num-1);z_++){
                    
                    for(x_ = 0;x_ < (x_num);x_++){
                        
                        const size_t z_parent = (z_+1)/2;
                        const size_t x_parent = (x_)/2;
                        
                        const size_t z_neigh = (z_+1);
                        const size_t x_neigh = x_;
                        
                        const size_t offset_pc_data = x_num*z_ + x_;
                        const size_t offset_pc_data_parent = x_num_parent*z_parent + x_parent;
                        const size_t offset_pc_data_neigh = x_num*z_neigh + x_neigh;
                        
                        //initialization
                        y_coord = (pc_data.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_neigh = (pc_data.data[i][offset_pc_data_neigh][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_parent = (pc_data.data[i-1][offset_pc_data_parent][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        j_parent = 1;
                        j_neigh = 1;
                        
                        if (pc_data.data[i-1][offset_pc_data_parent].size() == 1){
                            //set to max so its not checked
                            y_parent = 64000;
                        }
                        
                        if (pc_data.data[i][offset_pc_data_neigh].size() == 1){
                            //set to max so its not checked
                            y_neigh = 64000;
                        }
                        
                        y_coord--;
                        
                        const size_t j_num = pc_data.data[i][offset_pc_data].size();
                        const size_t j_num_parent = pc_data.data[i-1][offset_pc_data_parent].size();
                        const size_t j_num_neigh = pc_data.data[i][offset_pc_data_neigh].size();
                        
                        for(j_ = 1;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = pc_data.data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_coord--;
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                while ((y_neigh < y_coord) & (j_neigh < (j_num_neigh-1))){
                                    
                                    j_neigh++;
                                    node_val = pc_data.data[i][offset_pc_data_neigh][j_neigh];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_neigh = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_neigh++;
                                        
                                    } else {
                                        //normal node
                                        y_neigh++;
                                        
                                    }
                                    
                                }
                                
                                while ((y_parent < y_coord/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val = pc_data.data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_parent++;
                                        
                                    } else {
                                        //normal node
                                        y_parent++;
                                        
                                    }
                                    
                                }
                                
                                if(y_coord == y_neigh){
                                    pc_data.data[i][offset_pc_data][j_] |= (j_neigh << ZP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_SAME << ZP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (j_neigh << ZM_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (LEVEL_SAME << ZM_DEPTH_SHIFT);
                                    
                                } else if (y_coord/2 == y_parent) {
                                    pc_data.data[i][offset_pc_data][j_] |= (j_parent << ZP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_DOWN << ZP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << ZM_INDEX_SHIFT);
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent] |= (LEVEL_UP << ZM_DEPTH_SHIFT);
                                } else {
                                    std::cout << "BUG" << std::endl;
                                }
                            }
                            
                        }
                        
                    }
                }
                
            } else if (i == pc_data.depth_min) {
                
                const unsigned int x_num_child = pc_data.x_num[i+1];
                
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord,y_child,j_child) if(z_num*x_num > 100)
                for(z_ = 0;z_ < (z_num-1);z_++){
                    
                    for(x_ = 0;x_ < (x_num);x_++){
                        
                        
                        const size_t z_child = (z_+1)*2;
                        const size_t x_child = (x_)*2;
                        
                        const size_t z_neigh = (z_+1);
                        const size_t x_neigh = x_;
                        
                        const size_t offset_pc_data = x_num*z_ + x_;
                        const size_t offset_pc_data_neigh = x_num*z_neigh + x_neigh;
                        const size_t offset_pc_data_child = x_num_child*z_child + x_child;
                        
                        //initialization
                        y_coord = (pc_data.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_neigh = (pc_data.data[i][offset_pc_data_neigh][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_child = (pc_data.data[i+1][offset_pc_data_child][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        
                        j_neigh = 1;
                        j_child = 1;
                        
                        
                        if (pc_data.data[i][offset_pc_data_neigh].size() == 1){
                            //set to max so its not checked
                            y_neigh = 64000;
                        }
                        
                        if (pc_data.data[i+1][offset_pc_data_child].size() == 1){
                            //set to max so its not checked
                            y_child = 64000;
                        }
                        
                        y_coord--;
                        
                        const size_t j_num = pc_data.data[i][offset_pc_data].size();
                        const size_t j_num_neigh = pc_data.data[i][offset_pc_data_neigh].size();
                        const size_t j_num_child = pc_data.data[i+1][offset_pc_data_child].size();
                        
                        for(j_ = 1;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = pc_data.data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_coord--;
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                while ((y_neigh < y_coord) & (j_neigh < (j_num_neigh-1))){
                                    
                                    j_neigh++;
                                    node_val = pc_data.data[i][offset_pc_data_neigh][j_neigh];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_neigh = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_neigh++;
                                        
                                    } else {
                                        //normal node
                                        y_neigh++;
                                        
                                    }
                                    
                                }
                                
                                while ((y_child < y_coord*2) & (j_child < (j_num_child-1))){
                                    
                                    j_child++;
                                    node_val = pc_data.data[i+1][offset_pc_data_child][j_child];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_child = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_child++;
                                        
                                    } else {
                                        //normal node
                                        y_child++;
                                        
                                    }
                                    
                                }
                                
                                if(y_coord == y_neigh){
                                    pc_data.data[i][offset_pc_data][j_] |= (j_neigh << ZP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_SAME << ZP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (j_ << ZM_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (LEVEL_SAME << ZM_DEPTH_SHIFT);
                                } else if (y_coord*2 == y_child) {
                                    pc_data.data[i][offset_pc_data][j_] |= (j_child << ZP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_UP << ZP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i+1][offset_pc_data_child][j_child] |= (j_ << ZM_INDEX_SHIFT);
                                    pc_data.data[i+1][offset_pc_data_child][j_child] |= (LEVEL_DOWN << ZM_DEPTH_SHIFT);
                                } else {
                                    std::cout << "BUG" << std::endl;
                                }
                                
                            }
                            
                        }
                        
                    }
                }
                
                
                
            } else {
                
                const unsigned int x_num_child = pc_data.x_num[i+1];
                
                
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord,j_child,y_child) if(z_num*x_num > 100)
                for(z_ = 0;z_ < (z_num-1);z_++){
                    
                    for(x_ = 0;x_ < (x_num);x_++){
                        
                        const size_t z_parent = (z_+1)/2;
                        const size_t x_parent = (x_)/2;
                        
                        const size_t z_child = (z_+1)*2;
                        const size_t x_child = (x_)*2;
                        
                        const size_t z_neigh = z_+1;
                        const size_t x_neigh = x_;
                        
                        const size_t offset_pc_data = x_num*z_ + x_;
                        const size_t offset_pc_data_parent = x_num_parent*z_parent + x_parent;
                        const size_t offset_pc_data_neigh = x_num*z_neigh + x_neigh;
                        const size_t offset_pc_data_child = x_num_child*z_child + x_child;
                        
                        //initialization
                        y_coord = (pc_data.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_neigh = (pc_data.data[i][offset_pc_data_neigh][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_parent = (pc_data.data[i-1][offset_pc_data_parent][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_child = (pc_data.data[i+1][offset_pc_data_child][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        j_parent = 1;
                        j_neigh = 1;
                        j_child = 1;
                        
                        if (pc_data.data[i-1][offset_pc_data_parent].size() == 1){
                            //set to max so its not checked
                            y_parent = 64000;
                        }
                        
                        if (pc_data.data[i][offset_pc_data_neigh].size() == 1){
                            //set to max so its not checked
                            y_neigh = 64000;
                        }
                        
                        if (pc_data.data[i+1][offset_pc_data_child].size() == 1){
                            //set to max so its not checked
                            y_child = 64000;
                        }
                        
                        const size_t j_num = pc_data.data[i][offset_pc_data].size();
                        const size_t j_num_parent = pc_data.data[i-1][offset_pc_data_parent].size();
                        const size_t j_num_neigh = pc_data.data[i][offset_pc_data_neigh].size();
                        const size_t j_num_child = pc_data.data[i+1][offset_pc_data_child].size();
                        
                        y_coord--;
                        
                        for(j_ = 1;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = pc_data.data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_coord--;
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                while ((y_neigh < y_coord) & (j_neigh < (j_num_neigh-1))){
                                    
                                    j_neigh++;
                                    node_val = pc_data.data[i][offset_pc_data_neigh][j_neigh];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_neigh = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        
                                        j_neigh++;
                                        
                                    } else {
                                        //normal node
                                        y_neigh++;
                                        
                                    }
                                    
                                }
                                
                                while ((y_parent < y_coord/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val = pc_data.data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_parent++;
                                        
                                    } else {
                                        //normal node
                                        y_parent++;
                                        
                                    }
                                    
                                }
                                
                                while ((y_child < y_coord*2) & (j_child < (j_num_child-1))){
                                    
                                    j_child++;
                                    node_val = pc_data.data[i+1][offset_pc_data_child][j_child];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_child = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_child++;
                                        
                                    } else {
                                        //normal node
                                        y_child++;
                                        
                                    }
                                    
                                }
                                
                                if(y_coord == y_neigh){
                                    pc_data.data[i][offset_pc_data][j_] |= (j_neigh << ZP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_SAME << ZP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (j_ << ZM_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data_neigh][j_neigh] |= (LEVEL_SAME << ZM_DEPTH_SHIFT);
                                } else if (y_coord*2 == y_child) {
                                    pc_data.data[i][offset_pc_data][j_] |= (j_child << ZP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_UP << ZP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i+1][offset_pc_data_child][j_child] |= (j_ << ZM_INDEX_SHIFT);
                                    pc_data.data[i+1][offset_pc_data_child][j_child] |= (LEVEL_DOWN << ZM_DEPTH_SHIFT);
                                } else if (y_coord/2 == y_parent){
                                    pc_data.data[i][offset_pc_data][j_] |= (j_parent << ZP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_DOWN << ZP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << ZM_INDEX_SHIFT);
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent] |= (LEVEL_UP << ZM_DEPTH_SHIFT);
                                } else {
                                    std::cout << "BUG" << std::endl;
                                }
                                
                                
                                
                                
                            }
                            
                        }
                        
                    }
                }
                
                
                
            }
            
            
        }
        
        timer.stop_timer();

        /////////////////////////////////////////////////////////////
        //
        //
        // Y direction neigh loop
        //
        //
        /////////////////////////////////////////////////////////////
        
        
        timer.start_timer("parent child with child loop y");
        
        for(int i = (pc_data.depth_min);i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            
            const unsigned int x_num_parent = pc_data.x_num[i-1];
            const unsigned int x_num_child = pc_data.x_num[i+1];
            
            if(i == pc_data.depth_max){
                
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,y_coord) if(z_num*x_num > 100)
                for(z_ = 0;z_ < (z_num);z_++){
                    
                    for(x_ = 0;x_ < (x_num);x_++){
                        
                        const size_t z_parent = (z_)/2;
                        const size_t x_parent = (x_)/2;
                        
                        const size_t offset_pc_data = x_num*z_ + x_;
                        const size_t offset_pc_data_parent = x_num_parent*z_parent + x_parent;
                        
                        //initialization
                        y_coord = (pc_data.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_parent = (pc_data.data[i-1][offset_pc_data_parent][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        j_parent = 1;
                        
                        y_coord--;
                        
                        if (pc_data.data[i-1][offset_pc_data_parent].size() == 1){
                            //set to max so its not checked
                            y_parent = 64000;
                        }
                        
                        const size_t j_num = pc_data.data[i][offset_pc_data].size();
                        const size_t j_num_parent = pc_data.data[i-1][offset_pc_data_parent].size();
                        
                        for(j_ = 1;j_ < (j_num);j_++){
                            
                            // Parent relation
                            
                            node_val = pc_data.data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                                
                                //here is where the code for the access has to go.
                                if((y_coord+1)/2 == y_parent){
                                    pc_data.data[i][offset_pc_data][j_] |= (j_parent << YP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_DOWN << YP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent-1] |= ((j_-1) << YM_INDEX_SHIFT);
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent-1] |= (LEVEL_UP << YM_DEPTH_SHIFT);
                                } else {
                                    //end node
                                }
                                
                                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_coord--;

                                
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                
                                while ((y_parent < (y_coord+1)/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val = pc_data.data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_parent++;
                                        
                                    } else {
                                        //normal node
                                        y_parent++;
                                        
                                    }
                                    
                                }
                                
                                
                                
                            }
                            
                        }
                        
                    }
                }
                
            } else if (i == pc_data.depth_min) {
                
                
                
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord,y_child,j_child) if(z_num*x_num > 100)
                for(z_ = 0;z_ < (z_num);z_++){
                    
                    for(x_ = 0;x_ < (x_num);x_++){
                        
                        
                        const size_t z_child = (z_)*2;
                        const size_t x_child = (x_)*2;
                        
                        const size_t offset_pc_data = x_num*z_ + x_;
                        
                        const size_t offset_pc_data_child = x_num_child*z_child + x_child;
                        
                        //initialization
                        y_coord = (pc_data.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_child = (pc_data.data[i+1][offset_pc_data_child][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        
                        j_child = 1;
                        
                        
                        if (pc_data.data[i+1][offset_pc_data_child].size() == 1){
                            //set to max so its not checked
                            y_child = 64000;
                        }
                        
                        y_coord--;
                        
                        const size_t j_num = pc_data.data[i][offset_pc_data].size();
                        
                        const size_t j_num_child = pc_data.data[i+1][offset_pc_data_child].size();
                        
                        for(j_ = 1;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = pc_data.data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                                
                                
                                if((y_coord+1)*2 == y_child){
                                    
                                    pc_data.data[i][offset_pc_data][j_] |= (j_child << YP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (LEVEL_UP << YP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i+1][offset_pc_data_child][j_child-1] |= ((j_-1) << YM_INDEX_SHIFT);
                                    pc_data.data[i+1][offset_pc_data_child][j_child-1] |= (LEVEL_DOWN << YM_DEPTH_SHIFT);
                                } else {
                                    //end node
                                }
                                
                                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_coord--;
                                
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                
                                while ((y_child < (y_coord+1)*2) & (j_child < (j_num_child-1))){
                                    
                                    j_child++;
                                    node_val = pc_data.data[i+1][offset_pc_data_child][j_child];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_child = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_child++;
                                        
                                    } else {
                                        //normal node
                                        y_child++;
                                        
                                    }
                                    
                                }
                                
                            }
                            
                        }
                        
                    }
                }
                
                
                
            } else {
                
            
      
                
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,y_coord,j_child,y_child) if(z_num*x_num > 100)
                for(z_ = 0;z_ < (z_num);z_++){
                    
                    for(x_ = 0;x_ < (x_num);x_++){
                        
                        const size_t z_parent = (z_)/2;
                        const size_t x_parent = (x_)/2;
                        
                        const size_t z_child = (z_)*2;
                        const size_t x_child = (x_)*2;
                        
                        
                        const size_t offset_pc_data = x_num*z_ + x_;
                        const size_t offset_pc_data_parent = x_num_parent*z_parent + x_parent;
                        const size_t offset_pc_data_child = x_num_child*z_child + x_child;
                        
                        //initialization
                        y_coord = (pc_data.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_parent = (pc_data.data[i-1][offset_pc_data_parent][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_child = (pc_data.data[i+1][offset_pc_data_child][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        j_parent = 1;
                        j_child = 1;
                        
                        if (pc_data.data[i-1][offset_pc_data_parent].size() == 1){
                            //set to max so its not checked
                            y_parent = 64000;
                        }
                        
                        if (pc_data.data[i+1][offset_pc_data_child].size() == 1){
                            //set to max so its not checked
                            y_child = 64000;
                        }
                        
                        y_coord--;
                        
                        const size_t j_num = pc_data.data[i][offset_pc_data].size();
                        const size_t j_num_parent = pc_data.data[i-1][offset_pc_data_parent].size();
                        const size_t j_num_child = pc_data.data[i+1][offset_pc_data_child].size();
                        
                        for(j_ = 1;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = pc_data.data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                               
                                
                                if ((y_coord+1)*2 == y_child) {
                                    pc_data.data[i][offset_pc_data][j_] |= (j_child << YP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= ( LEVEL_DOWN << YP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i+1][offset_pc_data_child][j_child-1] |= ( (j_-1) << YM_INDEX_SHIFT);
                                    pc_data.data[i+1][offset_pc_data_child][j_child-1] |= (LEVEL_UP  << YM_DEPTH_SHIFT);
                                } else if((y_coord+1)/2 == y_parent){
                                    pc_data.data[i][offset_pc_data][j_] |= (j_parent << YP_INDEX_SHIFT);
                                    pc_data.data[i][offset_pc_data][j_] |= (  LEVEL_DOWN  << YP_DEPTH_SHIFT);
                                    //symmetric
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent-1] |= ( (j_-1) << YM_INDEX_SHIFT);
                                    pc_data.data[i-1][offset_pc_data_parent][j_parent-1] |= ( LEVEL_UP  << YM_DEPTH_SHIFT);
                                } else {
                                    //end node
                                }
                                
                                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_coord--;
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                
                                while ((y_parent < (y_coord+1)/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val = pc_data.data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_parent++;
                                        
                                    } else {
                                        //normal node
                                        y_parent++;
                                        
                                    }
                                    
                                }
                                
                                while ((y_child < (y_coord+1)*2) & (j_child < (j_num_child-1))){
                                    
                                    j_child++;
                                    node_val = pc_data.data[i+1][offset_pc_data_child][j_child];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_child = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_child++;
                                        
                                    } else {
                                        //normal node
                                        y_child++;
                                        
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        timer.stop_timer();

        
    }
    
};


#endif //PARTPLAY_PARTCELLSTRUCTURE_HPP