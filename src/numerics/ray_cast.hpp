#ifndef _ray_cast_num_h
#define _ray_cast_num_h
//////////////////////////////////////////////////
//
//
//  Bevan Cheeseman 2016
//
//  Ray casting numerics
//
//
//////////////////////////////////////////////////

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "../data_structures/Tree/PartCellParent.hpp"

#include "filter_help/CurrLevel.hpp"
#include "filter_help/NeighOffset.hpp"

#include "../../test/utils.h"


struct move {
    int dir;
    int index;
};

struct coord {
    float x,y,z;
    
    void set(float x_,float y_,float z_){
        x=x_;
        z=z_;
        y=y_;
    }
};

template<typename S>
move calculate_dir_index_parralell(CurrentLevel<S,uint64_t>& curr_level,coord& curr_loc,coord& new_loc){
    //
    //  Bevan Cheeseman 2017
    //
    //  Takes in new coordinate and figure out the direction and index for it
    //
    //  Only handles non-diagonal currently
    //
    
    move next_move;
    
    unsigned int depth_offset = pow(2,curr_level.depth_max - curr_level.depth + 1);
    
    int  yc = (curr_loc.y/depth_offset);
    int  xc = (curr_loc.x/depth_offset);
    int  zc = (curr_loc.z/depth_offset);
    
    int  yn = (new_loc.y/depth_offset);
    int  xn = (new_loc.x/depth_offset);
    int  zn = (new_loc.z/depth_offset);
    
    int offset_y = yn - yc;
    int offset_x = xn - xc;
    int offset_z = zn - zc;
    
    next_move.dir = (offset_y == -1) + (offset_x == 1)*2 + (offset_x == -1)*3 + (offset_z == 1)*4 + (offset_z == -1)*5;
    
    int move_sum = ((abs(offset_y) > 0) + (abs(offset_x) > 0) + (abs(offset_z) > 0));
    
    if (next_move.dir > 5 | (move_sum > 1)){
        std::cout << "dir not parallel" << std::endl;
        next_move.dir = 0;
        next_move.index = 0;
        return next_move;
    } else if (move_sum ==1) {
        
        int child_x = new_loc.x/(depth_offset*2);
        int child_y = new_loc.y/(depth_offset*2);
        int child_z = new_loc.z/(depth_offset*2);
        
        if(next_move.dir > 2){
            // ydirections
            next_move.index = (child_x&1)*2 + (child_z&1);
        } else if (next_move.dir > 4){
            // xdirections
            next_move.index = (child_z&1)*2 + (child_y&1);
        } else {
            // zdirections
            next_move.index = (child_x&1)*2 + (child_y&1);
        }
        
    } else {
        next_move.dir = -1;
    }
    
    return next_move;
    
}
template<typename S>
coord new_position(CurrentLevel<S,uint64_t>& curr_level,unsigned int direction,coord& curr_coord){
    //
    //
    //  Demo Ray Case, has a direction and propogrates to next cell along it.
    //
    //
    
    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};
    
    float step_size = pow(2,curr_level.depth_max - curr_level.depth + 1)*.24999; //move quarter step then you can never hop a child.
    
    float offset_x = dir_x[direction];
    float offset_y = dir_y[direction];
    float offset_z = dir_z[direction];
    
    coord new_loc;
    new_loc.set(curr_coord.x + offset_x*step_size,curr_coord.y + offset_y*step_size,curr_coord.z + offset_z*step_size);
    
    return new_loc;
    
}

template<typename S>
void single_ray_parrallel(PartCellStructure<S,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2017
    //
    //  Simple ray case example, signle ray, accumulating, parralell projection
    //
    //
    
    
    //////////////////////////////
    //
    //  This creates data sets where each particle is a cell.
    //
    //  This same code can be used where there are multiple particles per cell as in original pc_struct, however, the particles have to be accessed in a different way.
    //
    //////////////////////////////
    
    
    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);
    
    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);
    
    //Genearate particle at cell locations, easier access
    ExtraPartCellData<float> particles_int;
    part_new.create_particles_at_cell_structure(particles_int);
    
    //used for finding a part cell
    PartCellParent<uint64_t> parent_cells(pc_data);
    
    //iterator
    CurrentLevel<float,uint64_t> curr_level(pc_data);
    
    //random seed
    srand ((unsigned int)time(NULL));
    
    //chose a point within the domain
    uint64_t x = rand()%(pc_struct.org_dims[1]*2), y = rand()%(pc_struct.org_dims[0]*2), z = rand()%(pc_struct.org_dims[2]*2);
    
    uint64_t init_key = parent_cells.find_partcell(x, y, z, pc_data);
    
    if(init_key > 0){
        //above zero means the location is inside the domain
        
        curr_level.init(init_key,pc_data);
        
        bool end_domain = false;
        
        //choose random direction to propogate along
        unsigned int direction = rand()%6;
        
        move next_move;
        //current and next location
        coord next_loc;
        coord curr_loc;
        next_loc.set(x,y,z);
        
        
        int counter =0;
        float accum_int = 0;
        float accum_depth = 0;
        float accum_status = 0;
        
        
        while(!end_domain){
            //iterate through domain until you hit the edge
            //next becomes current
            std::swap(next_loc,curr_loc);
            
            //get new position
            next_loc = new_position(curr_level,direction,curr_loc);
            
            //calculate the new move
            next_move =  calculate_dir_index_parralell(curr_level,curr_loc,next_loc);
            
            if(next_move.dir >= 0){
                //if its a new cell
                end_domain = curr_level.move_cell(next_move.dir,next_move.index,part_new,pc_data);
                
                //get the intensity of the particle
                accum_depth += curr_level.depth;
                accum_status += curr_level.status;
                accum_int += curr_level.get_val(particles_int);
                
                counter++;
                
            }
            
            
            
        }
        
        std::cout << "moved " << counter << " times through the domain" << std::endl;
        std::cout << "from x: " << x << " y: " << y << " z: " << z << std::endl;
        std::cout << "to x: " << curr_loc.x << " y: " << curr_loc.y << " z: " << curr_loc.z << std::endl;
        std::cout << "in direction " << direction << std::endl;
        std::cout << "accumulated " << accum_int << " intensity" << std::endl;
        std::cout << "accumulated " << accum_status << " status" << std::endl;
        std::cout << "accumulated " << accum_depth << " depth" << std::endl;
        
    } else {
        std::cout << "outside domain" << std::endl;
    }
    
    
}

























#endif