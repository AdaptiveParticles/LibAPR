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

const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

struct proj_par{
    int proj_type = 0;
    int status_th = 10;
    float Ip_th = 0;
    int start_th = 5;
    int direction = 0;
    bool avg_flag = true;

};

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

struct ray {

    float accum_int;
    int accum_status;
    int accum_depth;
    int counter;
    float value;

    void init(){
        accum_int = 0;
        accum_status = 0;
        accum_depth = 0;
        counter = 0;
        value=0;
    }

};

template<typename S>
bool compare_current_cell_location(CurrentLevel<S,uint64_t>& curr_level,coord& curr_loc){
    //
    //  Bevan Cheeseman 2017
    //
    //  Debug Asssit
    //
    //  Compares the current cell, with the current location.
    //

    bool same_cell = true;

    unsigned int depth_offset = pow(2,curr_level.depth_max - curr_level.depth + 1);

    int x = curr_loc.x/depth_offset;
    int y = curr_loc.y/depth_offset;
    int z = curr_loc.z/depth_offset;

    if(x != curr_level.x){
        same_cell = false;
    }

    if(z != curr_level.z){
        same_cell = false;
    }

    return same_cell;
}


template<typename S>
move calculate_dir_index_parralell(const CurrentLevel<S,uint64_t>& curr_level,coord& curr_loc,coord& new_loc){
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

    /*if (next_move.dir > 5 || (move_sum > 1)){
        std::cout << "dir not parallel" << std::endl;
        next_move.dir = 0;
        next_move.index = 0;
        return next_move;
    } else if (move_sum == 1) {*/

    uint64_t child_x = new_loc.x / (depth_offset / 2);
    uint64_t child_y = new_loc.y / (depth_offset / 2);
    uint64_t child_z = new_loc.z / (depth_offset / 2);

    if (next_move.dir < 2) {
        // ydirections
        next_move.index = (child_x & 1) * 2 + (child_z & 1);
    } else if (next_move.dir < 4) {
        // xdirections
        next_move.index = (child_z & 1) * 2 + (child_y & 1);
    } else {
        // zdirections
        next_move.index = (child_x & 1) * 2 + (child_y & 1);
    }

    if(move_sum != 1) {
        next_move.dir = -1;
    }

    return next_move;

}
template<typename S>
coord new_position(const CurrentLevel<S,uint64_t>& curr_level,unsigned int direction,coord& curr_coord){
    //
    //
    //  Demo Ray Case, has a direction and propogrates to next cell along it.
    //
    //

    float step_size = pow(2,curr_level.depth_max - curr_level.depth + 1)*.49999; //move half step then you can never hop a child.

    float offset_x = dir_x[direction];
    float offset_y = dir_y[direction];
    float offset_z = dir_z[direction];

    coord new_loc;
    new_loc.set(curr_coord.x + offset_x*step_size,curr_coord.y + offset_y*step_size,curr_coord.z + offset_z*step_size);

    return new_loc;

}




template<typename S>
bool proj_function(ray& curr_ray,CurrentLevel<S,uint64_t>& curr_level,ExtraPartCellData<float>& particles_int,bool stop_ray,proj_par& par){
    //get the intensity of the particle


    if(par.proj_type == 0){
        // maximum projection
        curr_ray.accum_depth += curr_level.depth;
        curr_ray.accum_status += curr_level.status;
        //accum_int += curr_level.get_val(particles_int);
        curr_ray.value = std::max(curr_ray.value,curr_level.get_val(particles_int));

        curr_ray.counter++;
    } else if(par.proj_type == 1){
        // content projection

        int start_th = par.start_th;
        int status_th = par.status_th;
        int Ip_th = par.Ip_th;

        float Ip = curr_level.get_val(particles_int);

        if((curr_level.depth == (curr_level.depth_max)) & (curr_level.status ==1) & (Ip > Ip_th)){
            curr_ray.accum_depth++;
            //curr_ray.accum_status += (curr_level.status == 1);
            //curr_ray.accum_int = 0;
            //curr_ray.counter = 0;

        }

        if(curr_ray.accum_depth > start_th){
            curr_ray.accum_status += ((curr_level.depth == (curr_level.depth_max)) & (curr_level.status ==1));
            curr_ray.accum_int += Ip;
            curr_ray.counter++;
            curr_ray.value = std::max(curr_ray.value,Ip);


        } else {
            curr_ray.accum_int += Ip;
            curr_ray.counter++;
            curr_ray.value = std::max(curr_ray.value,Ip);
        }

        if(curr_ray.accum_status >= status_th){
            stop_ray = true;
            curr_ray.value = curr_ray.accum_int/curr_ray.counter;
        }


    }

    return stop_ray;

}


template<typename S>
void multi_ray_parrallel(PartCellStructure<S,uint64_t>& pc_struct,proj_par& pars){
    //
    //  Bevan Cheeseman 2017
    //
    //  Simple ray case example, multi ray, accumulating, parralell projection
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

    //Need to add here a parameters here

    //unsigned int direction = 0;
    Mesh_data<S> proj_img;
    Mesh_data<uint64_t> seed_cells;

    float active_x = 2;
    float active_y = 2;
    float active_z = 2;

    float start = 1;

    switch(pars.direction){
        case(0):{
            //x//z
            proj_img.initialize(pc_struct.org_dims[1],pc_struct.org_dims[2],1,0);
            seed_cells.initialize(pc_struct.org_dims[1],pc_struct.org_dims[2],1,0);
            active_x = 0;
            active_z = 1;
            start = 1;
            break;
        }
        case(1): {
            //x//z
            proj_img.initialize(pc_struct.org_dims[1], pc_struct.org_dims[2], 1, 0);
            seed_cells.initialize(pc_struct.org_dims[1], pc_struct.org_dims[2], 1, 0);
            active_x = 0;
            active_z = 1;
            start = 2*(pc_struct.org_dims[0]-1) - 1;
            break;
        }
        case(2):{
            //y//z
            proj_img.initialize(pc_struct.org_dims[0], pc_struct.org_dims[2], 1, 0);
            seed_cells.initialize(pc_struct.org_dims[0], pc_struct.org_dims[2], 1, 0);
            active_y = 0;
            active_z = 1;
            start = 1;
            break;
        }
        case(3):{
            //yz
            proj_img.initialize(pc_struct.org_dims[0], pc_struct.org_dims[2], 1, 0);
            seed_cells.initialize(pc_struct.org_dims[0], pc_struct.org_dims[2], 1, 0);
            active_y = 0;
            active_z = 1;
            start = 2*(pc_struct.org_dims[1]-1) - 1;
            break;
        }
        case(4):{
            //yx
            proj_img.initialize(pc_struct.org_dims[0], pc_struct.org_dims[1], 1, 0);
            seed_cells.initialize(pc_struct.org_dims[0], pc_struct.org_dims[1], 1, 0);
            active_y = 0;
            active_x = 1;
            start = 1;
            break;
        }
        case(5):{
            //yx
            proj_img.initialize(pc_struct.org_dims[0], pc_struct.org_dims[1], 1, 0);
            seed_cells.initialize(pc_struct.org_dims[0], pc_struct.org_dims[1], 1, 0);
            active_y = 0;
            active_x = 1;
            start = 2*(pc_struct.org_dims[2]-1) - 1;
            break;
        }
    }



    bool end_domain = false;

    //choose random direction to propogate along


    int counter =0;
    float accum_int = 0;
    float accum_depth = 0;
    float accum_status = 0;

    ray curr_ray;
    curr_ray.init();

    Part_timer timer;

    timer.verbose_flag = true;

    timer.start_timer("init");

    move next_move;
    //current and next location
    coord next_loc;
    coord curr_loc;

    std::vector<float> coords = {0,0,0};

    //get starting points
    for (int dim1 = 0; dim1 < proj_img.y_num; ++dim1) {
        for (int dim2 = 0; dim2 < proj_img.x_num; ++dim2) {
            coords[0] = dim1*2 + 1;
            coords[1] = dim2*2 + 1;
            coords[2] = start;

            float x = coords[active_x];
            float y = coords[active_y];
            float z = coords[active_z];

            //initialize ray
            uint64_t init_key = parent_cells.find_partcell(x, y, z, pc_data);
            seed_cells(dim1,dim2,0) = init_key;
        }
    }

    timer.stop_timer();

    timer.start_timer("ray cast");
    next_move.dir =0;
    next_move.index =0;

    uint64_t counter1 = 0;
    uint64_t counter2 = 0;

    int dim1,dim2;

#pragma omp parallel for default(shared) private(dim1,dim2,end_domain) firstprivate(next_move,curr_ray,next_loc,curr_loc,curr_level,coords)
    for (dim1 = 0; dim1 < proj_img.y_num; ++dim1) {
        for (dim2 = 0; dim2 < proj_img.x_num; ++dim2) {
            coords[0] = dim1*2 + 1;
            coords[1] = dim2*2 + 1;
            coords[2] = start;

            float x = coords[active_x];
            float y = coords[active_y];
            float z = coords[active_z];

            end_domain = false;

            curr_ray.init();

            //initialize ray
            uint64_t init_key = seed_cells(dim1,dim2,0);

            if(init_key > 0){

                curr_level.init(init_key,pc_data);
                next_loc.set(x,y,z);

                while(!end_domain){
                    //iterate through domain until you hit the edge
                    //next becomes current
                    std::swap(next_loc,curr_loc);

                    //get new position
                    next_loc = new_position(curr_level,pars.direction,curr_loc);

                    //calculate the new move
                    next_move =  calculate_dir_index_parralell(curr_level,curr_loc,next_loc);

                    //counter1++;

                    if(next_move.dir >= 0){
                        //if its a new cell
                        end_domain = curr_level.move_cell(next_move.dir,next_move.index,part_new,pc_data);

                        //next_move =  calculate_dir_index_parralell(curr_level,curr_loc,next_loc);

                        end_domain = proj_function(curr_ray,curr_level,particles_int,end_domain,pars);

                        //counter2++;
                    }
                }

                if(pars.proj_type == 0){
                    proj_img(dim1,dim2,0) = curr_ray.value;
                } else {
                    if(pars.avg_flag) {
                        proj_img(dim1, dim2, 0) = curr_ray.accum_int / curr_ray.counter;
                    } else {
                        proj_img(dim1,dim2,0) = curr_ray.accum_depth;
                    }
                }
            }

        }
    }

    timer.stop_timer();

    debug_write(proj_img,"parllel_proj" + std::to_string(pars.proj_type));

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

                compare_current_cell_location(curr_level,next_loc);

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