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
#include "../particle_map.hpp"
#include "../meshclass.h"
#include "../../io/writeimage.h"

#include <vector>
#include <algorithm>
#include <array>

// Bit manipulation defitinitions
//masks for storing the neighbour indices (for 13 bit indices and 64 node)
#define TYPE_MASK (((uint64_t)1 << 0) - 1) << 1
#define TYPE_SHIFT 1
#define STATUS_MASK (((uint64_t)1 << 3) - 1) << 2
#define STATUS_SHIFT 2

//xp is x + 1 neigh
#define XP_DEPTH_MASK (((uint64_t)1 << 5) - 1) << 4
#define XP_DEPTH_SHIFT 4
#define XP_INDEX_MASK (((uint64_t)1 << 18) - 1) << 6
#define XP_INDEX_SHIFT 6
//xm is x - 1 neigh
#define XM_DEPTH_MASK (((uint64_t)1 << 20) - 1) << 19
#define XM_DEPTH_SHIFT 19
#define XM_INDEX_MASK (((uint64_t)1 << 33) - 1) << 21
#define XM_INDEX_SHIFT 21

#define ZP_DEPTH_MASK (((uint64_t)1 << 35) - 1) << 34
#define ZP_DEPTH_SHIFT 34
#define ZP_INDEX_MASK (((uint64_t)1 << 48) - 1) << 36
#define ZP_INDEX_SHIFT 36

#define ZM_DEPTH_MASK  (((uint64_t)1 << 50) - 1) << 49
#define ZM_DEPTH_SHIFT 49
#define ZM_INDEX_MASK (((uint64_t)1 << 63) - 1) << 51
#define ZM_INDEX_SHIFT 51
//gap node defs

#define YP_DEPTH_MASK (((uint64_t)1 << 3) - 1) << 2
#define YP_DEPTH_SHIFT 2
#define YP_INDEX_MASK (((uint64_t)1 << 16) - 1) << 4
#define YP_INDEX_SHIFT 4

#define YM_DEPTH_MASK (((uint64_t)1 << 18) - 1) << 17
#define YM_DEPTH_SHIFT 17
#define YM_INDEX_MASK (((uint64_t)1 << 31) - 1) << 19
#define YM_INDEX_SHIFT 19

#define NEXT_COORD_MASK (((uint64_t)1 << 44) - 1) << 32
#define NEXT_COORD_SHIFT 32
#define PREV_COORD_MASK (((uint64_t)1 << 57) - 1) << 45
#define PREV_COORD_SHIFT 45

#define FREE_MEM_MASK (((uint64_t)1 << 63) - 1) << 58
#define FREE_MEM_SHIFT 58

//parent node defitions


//Neighbour definitions
#define NO_NEIGHBOUR 4
#define LEVEL_SAME 1
#define LEVEL_DOWN 0
#define LEVEL_UP 2

//Define Status definitions
#define SEED 1
#define BOUNDARY 2
#define FILLER 3

#define SEED_SHIFTED (uint64_t)1 << 2
#define BOUNDARY_SHIFTED (uint64_t)2 << 2
#define FILLER_SHIFTED (uint64_t)3 << 2

template <typename T,typename S> // type T is the image type, type S is the data structure base type
class PartCellStructure: public PartCellBase<T,S> {
    
    
    
private:
    
    PartCellData<S> pc_data;
    
    unsigned int k_min;
    unsigned int k_max;
    
    void get_neigh_in_dir(std::vector<Mesh_data<uint16_t>>& layer_index,coords3d coords,uint64_t* index,uint64_t* indicator,unsigned int level,const std::vector<int>& dir){
        //
        //  Bevan Cheeseman 2016
        //
        //  Get the index of the neighbour, and whether it is up or down
        //
        
        coords = {coords.x + dir[0],coords.y + dir[1],coords.z + dir[2]};
        
        //need to check if this new point is in the domain
        coords3d dim_p = {layer_index[level].x_num,layer_index[level].y_num,layer_index[level].z_num};
        coords3d dim_m = {0,0,0};
        
        
        if (coords < dim_p && coords < dim_m){
            
            uint16_t neigh = layer_index[level](coords.y,coords.x,coords.z);
            
            if(neigh > 0){
                *index = neigh;
                *indicator = 1;
                
            } else {
                //check parent (do first, smaller array, so faster)
                if ((level - 1)>=k_min){
                    neigh = layer_index[level-1](coords.y/2,coords.x/2,coords.z/2);
                
                }
                
                if(neigh > 0){
                    //parent is your neighbour
                    *index = neigh;
                    *indicator = 0;
                    
                } else {
                    //add the first of the children, you can then raster from that point to your neighbour using the pointers given.
                    *index = layer_index[level+1](2*coords.y-((dir[0] ==0) | (dir[0] == 1)),2*coords.x -((dir[1] ==0) | (dir[1] == 1)),2*coords.z -((dir[2] ==0) | (dir[2] == 1)));
                    *indicator = 2;
                }
                
            }
            
        } else {
            //neigh doesn't exist
            *index = 0;
            *indicator = 4;
        }
    }
    
    void compute_map_key(std::vector<Mesh_data<std::vector<uint64_t>>>& new_map,std::vector<Mesh_data<uint16_t>>& layers_large,coords3d& coords,unsigned int depth){
        //
        //
        //  Bevan Cheeseman 2016
        //
        //
        //  Computes the neighbour indices, and up down key for the access tree.
        //
        //   (type 2 bit) (status 2 bit) (+-= y forward (2bits)) (+-=, x up coord 15bits), (+-=, x down coord 15bits) (+-=, z up coord 15bits), (+-=, z down coord 15bits) = 64
        //
        //   Allows for cell indices up to 13 bits = 8192, therefore, maximum single dim of 16384. (17.6 TB full image size)
        //
        //   status is stored in an 8bit or less array
        //
        
        uint64_t node_val = 0;
        
        uint64_t index = 0;
        uint64_t indicator = 0;
        
        std::vector<int> dir = {0,0,0};
        
        //set type
        node_val = 1;
        
        //xp
        dir = {1,0,0};
        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
        
        //shift bits
        index = index << XP_INDEX_SHIFT;
        node_val = node_val | index;
        
        indicator = indicator << XP_DEPTH_SHIFT;
        node_val = node_val | indicator;
        
//        //xm
//        dir = {-1,0,0};
//        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
//        //shift bits
//        index = index << XM_INDEX_SHIFT;
//        node_val = node_val | index;
//        
//        indicator = indicator << XM_DEPTH_SHIFT;
//        node_val = node_val | indicator;
//        
//        //zp
//        dir = {0,0,1};
//        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
//        //shift bits
//        index = index << ZP_INDEX_SHIFT;
//        node_val = node_val | index;
//        
//        indicator = indicator << ZP_DEPTH_SHIFT;
//        node_val = node_val | indicator;
//        
//        //zm
//        dir = {0,0,-1};
//        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
//        //shift bits
//        index = index << ZM_INDEX_SHIFT;
//        node_val = node_val | index;
//        
//        indicator = indicator << ZM_DEPTH_SHIFT;
//        node_val = node_val | indicator;
//        
//        //status is still unset
//        
//        //add value to structure
//        new_map[depth](coords.z,coords.x,0).push_back(node_val);
//        
//        
//        ////////////////////////////////
//        //Y direction back case check
//        dir = {0,-1,0};
//        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
//        
//        if(indicator!=1){
//            //then get the previous node value (which should be a gap node)
//            node_val = new_map[depth](coords.z,coords.x,1)[new_map[depth](coords.z,coords.x,0).size()-2];
//            
//            index = index << YM_INDEX_SHIFT;
//            node_val = node_val | index;
//            
//            indicator = indicator << YM_DEPTH_SHIFT;
//            node_val = node_val | indicator;
//            
//            //previous coord
//            index = coords.y;
//            node_val = node_val | (index << NEXT_COORD_SHIFT);
//            
//            new_map[depth](coords.z,coords.x,0)[new_map[depth](coords.z,coords.x,0).size()-2] = node_val;
//            
//        }
//        
//        ////////////////////////////////
//        //Y direction forward case check
//        
//        //yp
//        dir = {0,1,0};
//        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
//        
//        if(indicator!=1 & indicator!= NO_NEIGHBOUR){
//            node_val = 0;
//            index = index << YP_INDEX_SHIFT;
//            node_val = node_val | index;
//            
//            indicator = indicator << YP_DEPTH_SHIFT;
//            node_val = node_val | indicator;
//            
//            //previous coord
//            index = coords.y;
//            node_val = node_val | (index << PREV_COORD_SHIFT);
//            
//            //add the gap node
//            new_map[depth](coords.z,coords.x,1).push_back(node_val);
//        }
    }
    
    void add_status(uint8_t part_map_status,uint64_t* node_val){
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
        int y_;
        int j_;
        
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
                    
                    const size_t offset_part_map = x_*y_num + z_*y_num*x_num;
                    const size_t offset_pc_data = x_num*z_ + x_;
                    curr_index = 0;
                    prev_ind = 0;
                    
                    for(y_ = 0;y_ < y_num;y_++){
                        
                        status = part_map.layers[i].mesh[offset_part_map + y_];
                        
                        if((status> 0) & (status < 8)){
                            curr_index+= 1 + prev_ind;
                            prev_ind = 0;
                        } else {
                            prev_ind = 1;
                        }
                    }
                    
                    pc_data.data[i][offset_pc_data].resize(curr_index+1,0); //always first adds an extra entry for intialization and extra info
                }
            }
            
        }

        timer.stop_timer();
        
        timer.start_timer("First initialization step");
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            const unsigned int y_num = part_map.layers[i].y_num;
            
            
#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    const size_t offset_part_map = x_*y_num + z_*y_num*x_num;
                    const size_t offset_pc_data = x_num*z_ + x_;
                    curr_index = 0;
                    
                    for(y_ = 0;y_ < y_num;y_++){
                        
                        status = part_map.layers[i].mesh[offset_part_map + y_];
                        
                        if((status> 0) & (status < 8)){
                            curr_index++;
                            pc_data.data[i][offset_pc_data][curr_index-1] = 2;
                        }
                    }
                    
                }
            }
            
        }
        
        timer.stop_timer();
        
        timer.start_timer("access pc data");
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            const unsigned int y_num = part_map.layers[i].y_num;
            
            
#pragma omp parallel for default(shared) private(z_,x_,j_,curr_index) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    
                    const size_t offset_pc_data = x_num*z_ + x_;
                    curr_index = 0;
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                      
                        pc_data.data[i][offset_pc_data][j_] = 2;
                        
                    }
                    
                }
            }
            
        }
        
        timer.stop_timer();


        
       
        
        
    }
    
    
    void create_sparse_graph_format(Particle_map<T>& part_map){
        //
        // Playing with a new data-structure (access paradigm)
        //
        //
        Part_timer timer;
        timer.verbose_flag = true;
        
        timer.start_timer("init1");
        
        //need larger layers structure
        std::vector<Mesh_data<uint16_t>> layers_large;
        layers_large.resize(part_map.layers.size());
        
        for(int i = 0;i < part_map.layers.size();i++){
            layers_large[i].initialize(part_map.layers[i].y_num,part_map.layers[i].x_num,part_map.layers[i].z_num,0);
        }
        
        timer.stop_timer();
        
        //then loop over the structure and create the indices
        
        timer.start_timer("init2");
        
        uint16_t curr_index;
        coords3d curr_coords;
        
        for(int i = part_map.k_min;i < (part_map.k_max +1) ;i++){
            
            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
                for(int x_ = 0;x_ < part_map.layers[i].x_num;x_++){
                    
                    curr_index = 0;
                    
                    for(int y_ = 0;y_ < part_map.layers[i].y_num;y_++){
                        
                        if((part_map.layers[i](y_,x_,z_) > 0) & (part_map.layers[i](y_,x_,z_) < 8)){
                            curr_index++;
                            layers_large[i](y_,x_,z_)=curr_index;
                        }
                    }
                }
            }
        }
        
         timer.stop_timer();
        
        
        timer.start_timer("initnew");
        
        
        unsigned int x_child;
        unsigned int x_parent;
        unsigned int z_child;
        unsigned int z_parent;
        unsigned int y_child;
        unsigned int y_parent;
        
        unsigned int curr_level;
        unsigned int curr_parent;
        unsigned int curr_child;
        
        int x_;
        int y_;
        
        
        for(int i = (part_map.k_min +1);i < (part_map.k_max +1) ;i++){
            
            if (i < part_map.k_max){
#pragma omp parallel for default(shared) private(x_,y_,y_child,y_parent,z_parent,x_parent,z_child,x_child,curr_level,curr_parent,curr_child) if(part_map.layers[i].z_num*part_map.layers[i].x_num > 100)
                for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
                    for(x_ = 0;x_ < part_map.layers[i].x_num;x_++){
                        
                        z_child = z_*2;
                        x_child = x_*2;
                        
                        z_parent = z_/2;
                        x_parent = x_/2;
                        
                        
                        for(y_ = 0;y_ < part_map.layers[i].y_num;y_++){
                            
                            y_child = y_*2;
                            y_parent = y_/2;
                            
                            curr_level = layers_large[i](y_,x_,z_);
                            curr_parent = layers_large[i-1](y_parent,x_parent,z_parent);
                            curr_child = layers_large[i+1](y_parent,x_parent,z_parent);
                            
                            
                            layers_large[i](y_,x_,z_) = curr_level + curr_parent + curr_child;
                        }
                    }
                }
            } else {
                
                for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
                    
#pragma omp parallel for default(shared) private(x_,y_,y_child,y_parent,z_parent,x_parent,curr_level,curr_parent)
                    for(x_ = 0;x_ < part_map.layers[i].x_num;x_++){
                        
                        
                        z_parent = z_/2;
                        x_parent = x_/2;
                        
                        
                        for(y_ = 0;y_ < part_map.layers[i].y_num;y_++){
                            
                            y_child = y_*2;
                            y_parent = y_/2;
                            
                            curr_level = layers_large[i](y_,x_,z_);
                            curr_parent = layers_large[i-1](y_parent,x_parent,z_parent);
                            
                            layers_large[i](y_,x_,z_) = curr_level + curr_parent;
                        }
                    }
                }
            }
        }
        
        timer.stop_timer();
        
        
        
        //then create the new structure
        
//        timer.start_timer("init3");
//        
//        std::vector<Mesh_data<std::vector<uint64_t>>> new_map;
//        
//        new_map.resize(part_map.k_max+1);
//        
//        int count = 0;
//        
//        for(int i = part_map.k_min;i < part_map.k_max +1 ;i++){
//            new_map[i].set_size(part_map.layers[i].z_num,part_map.layers[i].x_num,1);
//            new_map[i].mesh.resize(part_map.layers[i].z_num*part_map.layers[i].x_num);
//            
//            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
//                for(int x_ = 0;x_ < part_map.layers[i].x_num;x_++){
//                    
//                    //each has one extra element, this can encode starting information, or at what layer you an iterator should start from at accessing at this level
//                    new_map[i](z_,x_,0).push_back(0);
//                    
//                    for(int y_ = 0;y_ < part_map.layers[i].y_num;y_++){
//                        if(layers_large[i](y_,x_,z_) > 0){
//                            count++;
//                            
//                            curr_coords = {x_,y_,z_};
//                            
//                            compute_map_key(new_map,layers_large,curr_coords,i);
//                            
//                            
//                        }
//                    }
//                }
//            }
//        }
//        
//        timer.stop_timer();
        
//        timer.start_timer("init4");
//        
//        for(int i = part_map.k_min;i < part_map.k_max +1 ;i++){
//            
//            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
//                for(int x_ = 0;x_ < part_map.layers[i].x_num;x_++){
//                    
//                    curr_index = 0;
//                    
//                    for(int y_ = 0;y_ < part_map.layers[i].y_num;y_++){
//                        
//                        if((part_map.layers[i](y_,x_,z_) > 0) & (part_map.layers[i](y_,x_,z_) < 8)){
//                            
//                            add_status(part_map.layers[i](y_,x_,z_),&new_map[i](z_,x_,0)[curr_index]);
//                            curr_index++;
//                        }
//                    }
//                }
//            }
//        }
//
//        timer.stop_timer();
        
        
//        //////////////
//        //
//        // Analysis, lets have a look at some of the properties of new part_cell_structure (PCS)
//        //
//        //////////////
//        
//        std::vector<float> counter_l;
//        counter_l.resize(part_map.k_max+1,0);
//        float counter = 0;
//        float counter_z = 0;
//        
//        for(int i = part_map.k_min;i < part_map.k_max +1 ;i++){
//            
//            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
//                for(int x_ = 0;x_ < part_map.layers[i].x_num;x_++){
//                    counter_z++;
//                    //loop ove relements in structure
//                    counter += new_map[i](z_,x_,0).size();
//                    counter_l[i] += new_map[i](z_,x_,0).size();
//                
//                }
//            }
//        }
//        
//        
//        std::cout << "Size without padding: " << (count
//                                                  *8.0/1000000.0) << " MB" << std::endl;
//        std::cout << "New Structure Size (Neighbor O(1) access) Estmate: " << (counter*8.0/1000000.0) << " MB" << std::endl;
//        std::cout << "New Cells: " << count << " Number Nodes: " << counter << std::endl;
//        std::cout << "New Structure Size  (No Neighbor Access) Estmate: " << (counter/1000000.0) << " MB" << std::endl;
//        
//        
//     
//        
//        timer.start_timer("get_status_crs");
//        
//        uint8_t status;
//        uint64_t node_val;
//        
//        for(int i = part_map.k_min;i < part_map.k_max +1 ;i++){
//            
//            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
//                for(int x_ = 0;x_ < part_map.layers[i].x_num;x_++){
//                    for(int j_ = 0;j_ < new_map[i](z_,x_,0).size();j_++){
//                        
//                        node_val = new_map[i](z_,x_,0)[j_];
//                        
//                        if (!node_val&1){
//                            status = get_status(node_val);
//                        }
//                    }
//                }
//            }
//        }
//        
//        timer.stop_timer();
//        
//        
//        timer.start_timer("compute_coordinates_crs");
//        
//        coords3d coords;
//        uint16_t y_coord;
//        
//        int j_;
//        int x_;
//        
//        int x_num;
//        int z_num;
//        
//        
//        for(int i = part_map.k_min;i < part_map.k_max +1 ;i++){
//            
//            x_num = part_map.layers[i].x_num;
//            z_num = part_map.layers[i].z_num;
//            
//#pragma omp parallel for default(shared) private(x_,j_,coords,node_val,coords) if(z_num*x_num > 1000)
//            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
//                for(x_ = 0;x_ < part_map.layers[i].x_num;x_++){
//                    
//                    coords.x = x_;
//                    coords.z = z_;
//                    y_coord = 0;
//                    
//                    for(j_ = 0;j_ < new_map[i](z_,x_,0).size();j_++){
//                        
//                        node_val = new_map[i](z_,x_,0)[j_];
//                        
//                        if (node_val&1){
//                            //get the index
//                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
//                            y_coord--;
//                            
//                        } else {
//                            y_coord++;
//                            coords.y = y_coord;
//                        }
//                    }
//                }
//            }
//        }
//        
//        timer.stop_timer();
//        
//        timer.start_timer("set up memory");
//        
//        std::vector<Mesh_data<std::vector<float>>> new_map_parts;
//        
//        new_map_parts.resize(k_max+1);
//        
//        
//        for(int i = part_map.k_min;i < part_map.k_max +1 ;i++){
//            new_map_parts[i].set_size(part_map.layers[i].z_num,part_map.layers[i].x_num,1);
//            new_map_parts[i].mesh.resize(part_map.layers[i].z_num*part_map.layers[i].x_num);
//            
//            x_num = part_map.layers[i].x_num;
//            z_num = part_map.layers[i].z_num;
//            uint8_t level = i;
//            
//            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
//                for(x_ = 0;x_ < part_map.layers[i].x_num;x_++){
//                    
//                    new_map_parts[i](z_,x_,0).resize(new_map[i](z_,x_,0).size(),0);
//                }
//            }
//        }
//
//        timer.stop_timer();
//        
//        timer.start_timer("get_intensity");
//        //play with getting intensities.. (first won't get the seed sample correctly), lets just try this hack first
//      
//        
//        for(int i = part_map.k_min;i < part_map.k_max +1 ;i++){
//            
//            x_num = part_map.layers[i].x_num;
//            z_num = part_map.layers[i].z_num;
//            uint8_t level = i;
//            
//#pragma omp parallel for default(shared) private(x_,j_,node_val,y_coord) if(z_num*x_num > 60)
//            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
//                for(x_ = 0;x_ < part_map.layers[i].x_num;x_++){
//
//                    y_coord = 0;
//                    
//                    for(j_ = 0;j_ < new_map[level](z_,x_,0).size();j_++){
//                        
//                        node_val = new_map[level](z_,x_,0)[j_];
//                        
//                        if (node_val&1){
//                            //get the index
//                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
//                            y_coord--;
//                            
//                        } else {
//                            y_coord++;
//                            
//                            new_map_parts[level](z_,x_,0)[j_] = part_map.downsampled[level](y_coord,x_,z_);
//                        
//                        }
//                    }
//                }
//            }
//        }
//        
//        
//        timer.stop_timer();
//        
//        
//        timer.start_timer("iterate intensities");
//     
//        
//        float intensity;
//        
//        
//        for(int i = part_map.k_min;i < part_map.k_max +1 ;i++){
//            
//            x_num = part_map.layers[i].x_num;
//            z_num = part_map.layers[i].z_num;
//            uint8_t level = i;
//            
//#pragma omp parallel for default(shared) private(x_,j_,node_val) if(z_num*x_num > 100)
//            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
//                for(x_ = 0;x_ < part_map.layers[i].x_num;x_++){
//                    
//#pragma omp simd
//                    for(j_ = 0;j_ < new_map[level](z_,x_,0).size();j_++){
//                        
//                        if (new_map[level](z_,x_,0)[j_]&1){
//                            //get the index
//                            
//                            
//                        } else {
//                            
//                            new_map_parts[level](z_,x_,0)[j_] += 5;
//                            
//                        }
//                    }
//                }
//            }
//        }
//        
//        
//        timer.stop_timer();
        
        
    }
    
    
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
    
    
    void get_face_neighbours(coords3d& coords,uint64_t* node_val, uint8_t face,uint8_t* depth)
    {
        /** Get neighbours of a cell in one of the directions.
         *
         *  @param node_val    input: curr node, output: neighbour node
         *  @param face        direction to follow. Possible values are [0,5]
         *                     They stand for [-z,-x,-y,y,x,z]
         *  @param coords     input : coordinates of the node output: coordinates of the neighbour
         *  @param depth      input : depth of the node  output: depth of the neighbour
         */
        bool faces[6] = {0,0,0,0,0,0};
        faces[face] = true;
        
        uint64_t neigh_index;
        uint64_t neigh_indicator;
        
//        switch(face){
//            case 0:{
//                //-z
//                neigh_indicator = (node_val & ZM_DEPTH_MASK) >> ZM_DEPTH_SHIFT;
//                
//                switch(neigh_indicator){
//                    case(LEVEL_SAME):{
//                        neigh_index = (node_val & ZM_INDEX_MASK) >> ZM_INDEX_SHIFT;
//                        coords = {coords.x + von_neumann_x[face],coords.y + von_neumann_y[face],coords.z + von_neumann_z[face]};
//                        *node_val = part_cells[*depth](coords.z,coords.x,0)[neigh_index];
//                        
//                        break;
//                    }
//                    case(LEVEL_UP):{
//                        neigh_index = (node_val & ZM_INDEX_MASK) >> ZM_INDEX_SHIFT;
//                        coords = {2*coords.x + von_neumann_x[face],2*coords.y + von_neumann_y[face],2*coords.z + von_neumann_z[face]};
//                        *depth += 1;
//                        *node_val = part_cells[*depth](coords.z,coords.x,0)[neigh_index];
//                        
//                        break;
//                    }
//                    case(LEVEL_DOWN):{
//                        //first of four children (this needs to be extended, and need to think about how to return the extra children)
//                        neigh_index = (node_val & ZM_INDEX_MASK) >> ZM_INDEX_SHIFT;
//                        coords = {2*coords.x-(von_neumann_x[face] > -1),2*coords.y-(von_neumann_y[face] > -1),2*coords.z-(von_neumann_z[face] > -1)}
//                        *depth += -1;
//                        *node_val = part_cells[*depth](coords.z,coords.x,0)[neigh_index];
//                        
//                        break;
//                    }
//                }
//                
//                return;
//            }
//            case 1:{
//                //-x
//                neigh_indicator = (node_val & XM_DEPTH_MASK) >> XM_DEPTH_SHIFT;
//                
//                if(neigh_indicator!=NO_NEIGHBOUR){
//                    neigh_index = (node_val & XM_INDEX_MASK) >> XM_INDEX_SHIFT;
//                    coords = {coords.x + von_neumann_x[face],coords.y + von_neumann_y[face],coords.z + von_neumann_z[face]};
//                    *depth += *depth + neigh_indicator - 1;
//                    *node_val = part_cells[depth](coords.z,coords.x,0)[neigh_index];
//                }
//                
//                
//                return;
//            }
//            case 2:{
//                //-y
//                uint64_t node_next = part_cells[depth](coords.z,coords.x,0)[curr_index - 1];
//                
//                if(node_next&1){
//                    // it is a gap node, so there is a shift or it doens't exist and you've reach the edge
//                    
//                    neigh_indicator = (node_next & YM_DEPTH_MASK) >> YM_DEPTH_SHIFT;
//                    
//                    if(neigh_indicator!=NO_NEIGHBOUR){
//                        neigh_index = (node_val & YM_INDEX_MASK) >> YM_INDEX_SHIFT;
//                        coords = {coords.x + von_neumann_x[face],coords.y + von_neumann_y[face],coords.z + von_neumann_z[face]};
//                        *depth += *depth + neigh_indicator - 1;
//                        *node_val = part_cells[depth](coords.z,coords.x,0)[neigh_index];
//                    }
//                    
//                } else {
//                    *node_val = node_next;
//                    coords = {coords.x + von_neumann_x[face],coords.y + von_neumann_y[face],coords.z + von_neumann_z[face]};
//                }
//                
//                return;
//            }
//            case 3:{
//                //y
//                uint64_t node_next = part_cells[depth](coords.z,coords.x,0)[curr_index + 1];
//                
//                if(node_next&1){
//                    // it is a gap node, so there is a shift or it doens't exist and you've reach the edge
//                    
//                    neigh_indicator = (node_next & YP_DEPTH_MASK) >> YP_DEPTH_SHIFT;
//                    
//                    if(neigh_indicator!=NO_NEIGHBOUR){
//                        neigh_index = (node_val & YP_INDEX_MASK) >> YP_INDEX_SHIFT;
//                        coords = {coords.x + von_neumann_x[face],coords.y + von_neumann_y[face],coords.z + von_neumann_z[face]};
//                        *depth += *depth + neigh_indicator - 1;
//                        *node_val = part_cells[depth](coords.z,coords.x,0)[neigh_index];
//                    }
//                    
//                } else {
//                    *node_val = node_next;
//                    coords = {coords.x + von_neumann_x[face],coords.y + von_neumann_y[face],coords.z + von_neumann_z[face]};
//                }
//                
//                return;
//            }
//            case 4:{
//                //x
//                neigh_indicator = (node_val & XP_DEPTH_MASK) >> XP_DEPTH_SHIFT;
//                
//                if(neigh_indicator!=NO_NEIGHBOUR){
//                    neigh_index = (node_val & XP_INDEX_MASK) >> XP_INDEX_SHIFT;
//                    coords = {coords.x + von_neumann_x[face],coords.y + von_neumann_y[face],coords.z + von_neumann_z[face]};
//                    *depth += *depth + neigh_indicator - 1;
//                    *node_val = part_cells[depth](coords.z,coords.x,0)[neigh_index];
//                }
//                
//                return;
//            }
//            case 5:{
//                //z
//                neigh_indicator = (node_val & ZP_DEPTH_MASK) >> ZP_DEPTH_SHIFT;
//                
//                if(neigh_indicator!=NO_NEIGHBOUR){
//                    neigh_index = (node_val & ZP_INDEX_MASK) >> ZP_INDEX_SHIFT;
//                    coords = {coords.x + von_neumann_x[face],coords.y + von_neumann_y[face],coords.z + von_neumann_z[face]};
//                    *depth += *depth + neigh_indicator - 1;
//                    *node_val = part_cells[depth](coords.z,coords.x,0)[neigh_index];
//                }
//                
//                
//                return;
//            }
//        }
        
        
        
    }
    
    //decleration
    void initialize_structure(Particle_map<T>& particle_map){
        
        create_sparse_graph_format(particle_map);
        create_partcell_structure(particle_map);
    }
    
    
    
    PartCellStructure(Particle_map<T> &particle_map){
        //
        //  initialization of the tree structure
        //
        
        k_min = particle_map.k_min;
        k_max = particle_map.k_max;
        
        initialize_structure(particle_map);
    }

    
    
};


#endif //PARTPLAY_PARTCELLSTRUCTURE_HPP