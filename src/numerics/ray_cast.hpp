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
#include <ctime>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "../data_structures/Tree/PartCellParent.hpp"

#include "filter_help/CurrLevel.hpp"
#include "filter_help/NeighOffset.hpp"

#include "../../test/utils.h"

#include "misc_numerics.hpp"

#include "../../src/vis/Camera.h"
#include "../../src/vis/RaytracedObject.h"


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
    float counter;
    float value;

    float x_dir = 0;
    float y_dir = 0;
    float z_dir = 0;

    void init(){
        accum_int = 0;
        accum_status = 0;
        accum_depth = 0;
        counter = 0;
        value=0;
    }

};

template<typename S>
bool compare_current_cell_location(const CurrentLevel<S,uint64_t>& curr_level,coord& curr_loc){
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
move calculate_dir_index(const CurrentLevel<S,uint64_t>& curr_level,coord& curr_loc,coord& new_loc){
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

    if(offset_y == 1){
        next_move.dir = 0;
    } else if(offset_y == -1){
        next_move.dir = 1;
    } else if(offset_x == 1){
        next_move.dir = 2;
    } else if(offset_x == -1){
        next_move.dir = 3;
    } else if(offset_z == 1){
        next_move.dir = 4;
    } else if(offset_z == -1){
        next_move.dir = 5;
    } else {
        next_move.dir = -1;
    }

    if(next_move.dir > 0) {

        curr_loc = new_position(curr_level,next_move.dir,curr_loc);

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
coord new_position(const CurrentLevel<S,uint64_t>& curr_level,ray& curr_ray,coord& curr_coord){
    //
    //
    //  Demo Ray Case, has a direction and propogrates to next cell along it.
    //
    //

    float step_size = pow(2,curr_level.depth_max - curr_level.depth + 1)*.49999; //move half step then you can never hop a child.

    coord new_loc;
    new_loc.set(curr_coord.x + curr_ray.y_dir*step_size,curr_coord.y + curr_ray.x_dir*step_size,curr_coord.z + curr_ray.z_dir*step_size);

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

        float factor = 1.0/(1.0*pow(1,curr_level.depth_max - curr_level.depth));

        float Ip = curr_level.get_val(particles_int);

        if((curr_level.depth == (curr_level.depth_max)) & (curr_level.status ==1) & (Ip > Ip_th)){
            curr_ray.accum_depth++;
            //curr_ray.accum_status += (curr_level.status == 1);
            //curr_ray.accum_int = 0;
            //curr_ray.counter = 0;

        }

        if(curr_ray.accum_depth > start_th){
            curr_ray.accum_status += ((curr_level.depth == (curr_level.depth_max)) & (curr_level.status ==1));
            curr_ray.accum_int += Ip*factor;
            curr_ray.counter+= factor;
            //curr_ray.value = std::max(curr_ray.value,Ip);


        } else {
            curr_ray.accum_int += Ip*factor;
            curr_ray.counter+= factor;
           // curr_ray.value = std::max(curr_ray.value,Ip);
        }

        if(curr_ray.accum_status >= status_th){
            stop_ray = true;
            curr_ray.value = curr_ray.accum_int/curr_ray.counter;
        }

    }

    return stop_ray;

}


template<typename S>
void multi_ray_gen(PartCellStructure<S,uint64_t>& pc_struct,proj_par& pars){
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

    float offset_x = dir_x[pars.direction];
    float offset_y = dir_y[pars.direction];
    float offset_z = dir_z[pars.direction];

    bool end_domain = false;

    //choose random direction to propogate along

    int counter =0;
    float accum_int = 0;
    float accum_depth = 0;
    float accum_status = 0;

    ray curr_ray;
    curr_ray.init();

    curr_ray.x_dir = offset_x;
    curr_ray.y_dir = offset_y;
    curr_ray.z_dir = offset_z;

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

    coord camera;

    camera.x = -offset_x*pc_struct.org_dims[1]*2*0.25 + (offset_x == 0)*pc_struct.org_dims[1];
    camera.y = -offset_y*pc_struct.org_dims[0]*2*0.25 + (offset_y == 0)*pc_struct.org_dims[0];
    camera.z = -offset_z*pc_struct.org_dims[2]*2*0.25 + (offset_z == 0)*pc_struct.org_dims[2];


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

            float dist = sqrt(pow((x-camera.x),2) + pow((y-camera.y),2) + pow((z-camera.z),2));

            curr_ray.x_dir = -(x-camera.x)/dist;// + (rand() % 100 - 50)/8000.0 ;
            curr_ray.y_dir = -(y-camera.y)/dist;// + (rand() % 100 - 50)/8000.0 ;
            curr_ray.z_dir = -(z-camera.z)/dist;// (rand() % 100 - 50)/2000.0 ;

            end_domain = false;

            curr_ray.init();

            //initialize ray
            uint64_t init_key = seed_cells(dim1,dim2,0);

            if(init_key > 0){

                curr_level.init(init_key,pc_data);
                next_loc.set(x,y,z);
                curr_loc.set(x,y,z);

                while(!end_domain){
                    //iterate through domain until you hit the edge
                    //next becomes current

                    //calculate the new move
                    next_move =  calculate_dir_index(curr_level,curr_loc,next_loc);

                    if(next_move.dir >= 0){
                        //if its a new cell
                        end_domain = curr_level.move_cell(next_move.dir,next_move.index,part_new,pc_data);

                        if(compare_current_cell_location(curr_level,next_loc)) {
                            //reached the correct cell

                            end_domain = proj_function(curr_ray,curr_level,particles_int,end_domain,pars);

                            std::swap(curr_loc,next_loc);
                            //get new position
                            next_loc = new_position(curr_level, curr_ray, curr_loc);

                        }


                    } else {
                        //not escaped the cell
                        std::swap(curr_loc,next_loc);
                        next_loc = new_position(curr_level, curr_ray, curr_loc);
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
void find_ray_parallel(const CurrentLevel<S, uint64_t> &curr_level,const proj_par &pars,int &dim1,int &dim2,int &num){
    //
    //  Bevan Cheeseman 2017
    //

    //calculate real coordinate

    float step_size = pow(2,curr_level.depth_max - curr_level.depth);


    switch(pars.direction){
        case(0):{
            //x//z

            dim1 = curr_level.x * step_size;
            dim2 = curr_level.z * step_size;

            break;
        }
        case(1): {
            //x//z

            dim1 = curr_level.x * step_size;
            dim2 = curr_level.z * step_size;

            break;
        }
        case(2):{
            //y//z

            dim1 = curr_level.y * step_size;
            dim2 = curr_level.z * step_size;

            break;
        }
        case(3):{
            //yz

            dim1 = curr_level.y * step_size;
            dim2 = curr_level.z * step_size;

            break;
        }
        case(4):{
            //yx

            dim1 = curr_level.y * step_size;
            dim2 = curr_level.x * step_size;

            break;
        }
        case(5):{
            //yx

            dim1 = curr_level.y * step_size;
            dim2 = curr_level.x * step_size;

            break;
        }
    }

    num = step_size;

}



template<typename S>
void multi_ray_parrallel_raster(PartCellStructure<S,uint64_t>& pc_struct,proj_par& pars){
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


    timer.stop_timer();

    timer.start_timer("ray cast");
    next_move.dir =0;
    next_move.index =0;

    int z_,x_,j_,y_;

    for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = part_new.access_data.x_num[depth];
        const unsigned int z_num_ = part_new.access_data.z_num[depth];

        CurrentLevel<float, uint64_t> curr_level(pc_data);
        curr_level.set_new_depth(depth, part_new);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        curr_level.update_cell(part_new);

                        float temp_int =  curr_level.get_val(particles_int);

                        int dim1 = 0;
                        int dim2 = 0;
                        int size = 0;

                        find_ray_parallel(curr_level,pars,dim1,dim2,size);

                        //add to all the required rays

                        for (int k = 0; k < size; ++k) {
#pragma omp simd
                            for (int i = 0; i < size; ++i) {
                                proj_img.mesh[dim1 + i + (dim2 + k)*proj_img.y_num] = std::max(proj_img.mesh[dim1 + i + (dim2 + k)*proj_img.y_num],temp_int);
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

    debug_write(proj_img,"parllel_proj" + std::to_string(pars.proj_type));

}
void get_ray(const int& dir,const int& y,const int& x,const int& z,const float& step_size,int &dim1,int &dim2){
    //
    //  Bevan Cheeseman 2017
    //

    //calculate real coordinate


    if(dir < 2){
        //xz
        dim1 = x * step_size;
        dim2 = z * step_size;
    } else if (dir < 4){
        //yz
        dim1 = y * step_size;
        dim2 = z * step_size;
    } else {
        //yx
        dim1 = y * step_size;
        dim2 = x * step_size;
    }

}
template<typename S>
void multi_ray_parrallel_raster_alt(PartCellStructure<S,uint64_t>& pc_struct,proj_par& pars){
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

    ExtraPartCellData<uint16_t> y_vec;

    create_y_data(y_vec,part_new,pc_data);

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


    Part_timer timer;

    timer.verbose_flag = true;

    timer.start_timer("ray cast");

    const int dir = pars.direction;

    int z_,x_,j_,y_,i,k;

    for(uint64_t depth = (y_vec.depth_min);depth <= y_vec.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = y_vec.x_num[depth];
        const unsigned int z_num_ = y_vec.z_num[depth];
        const float step_size = pow(2,y_vec.depth_max - depth);

#pragma omp parallel for default(shared) private(z_,x_,j_,i,k)  if(z_num_*x_num_ > 100)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                for (j_ = 0; j_ < y_vec.data[depth][pc_offset].size(); j_++) {

                    int dim1 = 0;
                    int dim2 = 0;

                    const int y = y_vec.data[depth][pc_offset][j_];

                    get_ray(dir,y,x_,z_,step_size,dim1,dim2);

                    const float temp_int = part_new.particle_data.data[depth][pc_offset][j_];

                    //add to all the required rays
                    const int offset_max_dim1 = std::min((int)proj_img.y_num,(int)(dim1 + step_size));
                    const int offset_max_dim2 = std::min((int)proj_img.x_num,(int)(dim2 + step_size));

                    for ( k = dim2; k < offset_max_dim2; ++k) {
                        for (i = dim1; i < offset_max_dim1; ++i) {
                            proj_img.mesh[ i + (k)*proj_img.y_num] = std::max(temp_int,proj_img.mesh[ i + (k)*proj_img.y_num]);

                        }
                    }

                }
            }
        }
    }

    timer.stop_timer();

    debug_write(proj_img,"parllel_proj_alt" + std::to_string(pars.proj_type));

}
template<typename S>
void multi_ray_parrallel_raster_alt_d(PartCellStructure<S,uint64_t>& pc_struct,proj_par& pars){
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

    ExtraPartCellData<uint16_t> y_vec;

    create_y_data(y_vec,part_new,pc_data);

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


    std::vector<Mesh_data<S>> depth_slice;

    depth_slice.resize(y_vec.depth_max + 1);

    depth_slice[y_vec.depth_max].initialize(proj_img.y_num,proj_img.x_num,1,0);

    std::vector<int> depth_vec;
    depth_vec.resize(y_vec.depth_max + 1);

    for(int i = y_vec.depth_min;i < y_vec.depth_max;i++){
        float d = pow(2,y_vec.depth_max - i);
        depth_slice[i].initialize(ceil(proj_img.y_num/d),ceil(proj_img.x_num/d),1,0);
        depth_vec[i] = d;
    }


    //choose random direction to propogate along


    int counter =0;


    Part_timer timer;

    timer.verbose_flag = true;

    timer.start_timer("ray cast parts");

    const int dir = pars.direction;

    int z_,x_,j_,y_,i,k;

    for(uint64_t depth = (y_vec.depth_min);depth <= y_vec.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = y_vec.x_num[depth];
        const unsigned int z_num_ = y_vec.z_num[depth];
        const float step_size = 1;

#pragma omp parallel for default(shared) private(z_,x_,j_,i,k)  schedule(guided) if(z_num_*x_num_ > 100)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                for (j_ = 0; j_ < y_vec.data[depth][pc_offset].size(); j_++) {

                    int dim1 = 0;
                    int dim2 = 0;

                    const int y = y_vec.data[depth][pc_offset][j_];

                    get_ray(dir,y,x_,z_,step_size,dim1,dim2);

                    const float temp_int = part_new.particle_data.data[depth][pc_offset][j_];

                    depth_slice[depth].mesh[ dim1 + (dim2)*depth_slice[depth].y_num] = std::max(temp_int,depth_slice[depth].mesh[ dim1 + (dim2)*depth_slice[depth].y_num]);

                }
            }
        }
    }


//#pragma omp parallel for default(shared) private(z_,x_,j_,i,k)
//    for (x_ = 0; x_ < depth_slice[y_vec.depth_max].x_num; x_++) {
//        //both z and x are explicitly accessed in the structure
//
//        for (y_ = 0; y_ < depth_slice[y_vec.depth_max].y_num; y_++) {
//            for(uint64_t depth = (y_vec.depth_min);depth < y_vec.depth_max;depth++) {
//                //loop over the resolutions of the structure
//                i = y_/depth_vec[depth];
//                k = x_/depth_vec[depth];
//                depth_slice[y_vec.depth_max].mesh[ y_ + (x_)*depth_slice[y_vec.depth_max].y_num] =
//                        std::max(depth_slice[y_vec.depth_max].mesh[ y_ + (x_)*depth_slice[y_vec.depth_max].y_num],depth_slice[depth].mesh[ i + (k)*depth_slice[depth].y_num]);
//
//            }
//        }
//    }



    uint64_t depth;


    for(depth = (y_vec.depth_min);depth < y_vec.depth_max;depth++) {

        const int step_size = pow(2,y_vec.depth_max - depth);
#pragma omp parallel for default(shared) private(z_,x_,j_,i,k) schedule(guided) if (depth > 9)
        for (x_ = 0; x_ < depth_slice[depth].x_num; x_++) {
            //both z and x are explicitly accessed in the structure

            for (y_ = 0; y_ < depth_slice[depth].y_num; y_++) {

                const float curr_int = depth_slice[depth].mesh[ y_ + (x_)*depth_slice[depth].y_num];

                const int dim1  = y_*step_size;
                const int dim2 =  x_*step_size;

                //add to all the required rays
                const int offset_max_dim1 = std::min((int)depth_slice[y_vec.depth_max].y_num,(int)(dim1 + step_size));
                const int offset_max_dim2 = std::min((int)depth_slice[y_vec.depth_max].x_num,(int)(dim2 + step_size));

                if(curr_int > 0){

                    for ( k = dim2; k < offset_max_dim2; ++k) {
                        for (i = dim1; i < offset_max_dim1; ++i) {
                            depth_slice[y_vec.depth_max].mesh[ i + (k)*depth_slice[y_vec.depth_max].y_num] = std::max(curr_int,depth_slice[y_vec.depth_max].mesh[ i + (k)*depth_slice[y_vec.depth_max].y_num]);

                        }
                    }
                }

            }
        }
    }

    timer.stop_timer();



    debug_write(depth_slice[y_vec.depth_max],"parllel_proj_alt" + std::to_string(pars.proj_type));

}

template<typename S>
void multi_ray_parrallel_raster_alt_d_off(PartCellStructure<S,uint64_t>& pc_struct,proj_par& pars){
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

    ExtraPartCellData<uint16_t> y_off;

    create_y_offsets(y_off,part_new,pc_data);

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


    std::vector<Mesh_data<S>> depth_slice;

    depth_slice.resize(y_off.depth_max + 1);

    depth_slice[y_off.depth_max].initialize(proj_img.y_num,proj_img.x_num,1,0);

    std::vector<int> depth_vec;
    depth_vec.resize(y_off.depth_max + 1);

    for(int i = y_off.depth_min;i < y_off.depth_max;i++){
        float d = pow(2,y_off.depth_max - i);
        depth_slice[i].initialize(ceil(proj_img.y_num/d),ceil(proj_img.x_num/d),1,0);
        depth_vec[i] = d;
    }


    //choose random direction to propogate along


    int counter =0;


    Part_timer timer;

    timer.verbose_flag = true;

    timer.start_timer("ray cast parts");

    const int dir = pars.direction;

    int z_,x_,j_,y_,i,k;

    for(uint64_t depth = (y_off.depth_min);depth <= y_off.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = y_off.x_num[depth];
        const unsigned int z_num_ = y_off.z_num[depth];
        const float step_size = 1;

#pragma omp parallel for default(shared) private(z_,x_,j_,i,k)  schedule(guided) if(z_num_*x_num_ > 100)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                int y = 0;
                int y_next = 0;
                int y_prev = 0;

                for (j_ = 0; j_ < y_off.data[depth][pc_offset].size(); j_++) {

                    y_next = y_off.data[depth][pc_offset][j_];

                    int dim1 = 0;
                    int dim2 = 0;

                    get_ray(dir,y,x_,z_,step_size,dim1,dim2);

                    const float temp_int = part_new.particle_data.data[depth][pc_offset][j_];

                    depth_slice[depth].mesh[ dim1 + (dim2)*depth_slice[depth].y_num] = std::max(temp_int,depth_slice[depth].mesh[ dim1 + (dim2)*depth_slice[depth].y_num]);

                }
            }
        }
    }


    timer.stop_timer();

    uint64_t depth;


    for(depth = (y_off.depth_min);depth < y_off.depth_max;depth++) {

        const int step_size = pow(2,y_off.depth_max - depth);
#pragma omp parallel for default(shared) private(z_,x_,j_,i,k) schedule(guided) if (depth > 9)
        for (x_ = 0; x_ < depth_slice[depth].x_num; x_++) {
            //both z and x are explicitly accessed in the structure

            for (y_ = 0; y_ < depth_slice[depth].y_num; y_++) {

                const float curr_int = depth_slice[depth].mesh[ y_ + (x_)*depth_slice[depth].y_num];

                const int dim1  = y_*step_size;
                const int dim2 =  x_*step_size;

                //add to all the required rays
                const int offset_max_dim1 = std::min((int)depth_slice[y_off.depth_max].y_num,(int)(dim1 + step_size));
                const int offset_max_dim2 = std::min((int)depth_slice[y_off.depth_max].x_num,(int)(dim2 + step_size));

                if(curr_int > 0){

                    for ( k = dim2; k < offset_max_dim2; ++k) {
                        for (i = dim1; i < offset_max_dim1; ++i) {
                            depth_slice[y_off.depth_max].mesh[ i + (k)*depth_slice[y_off.depth_max].y_num] = std::max(curr_int,depth_slice[y_off.depth_max].mesh[ i + (k)*depth_slice[y_off.depth_max].y_num]);

                        }
                    }
                }

            }
        }
    }



    debug_write(depth_slice[y_off.depth_max],"parllel_proj_alt" + std::to_string(pars.proj_type));

}






template<typename S>
void multi_ray_parrallel_raster_mesh(PartCellStructure<S,uint64_t>& pc_struct,proj_par& pars){
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


    Mesh_data<S> image;

    pc_struct.interp_parts_to_pc(image,pc_struct.part_data.particle_data);

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


    Part_timer timer;

    timer.verbose_flag = true;

    timer.start_timer("ray cast mesh");

    const int dir = pars.direction;

    int z_,x_,j_,y_,i,k;

    //loop over the resolutions of the structure
    const unsigned int x_num_ = image.x_num;
    const unsigned int z_num_ = image.z_num;
    const float step_size = 1;
    const unsigned int y_num_ = image.y_num;

#pragma omp parallel for default(shared) private(z_,x_,j_,i,k)  if(z_num_*x_num_ > 100)
    for (z_ = 0; z_ < z_num_; z_++) {
        //both z and x are explicitly accessed in the structure

        for (x_ = 0; x_ < x_num_; x_++) {

            const unsigned int pc_offset = x_num_*z_ + x_;

            for (j_ = 0; j_ < y_num_; j_++) {

                int dim1 = 0;
                int dim2 = 0;

                get_ray(dir,j_,x_,z_,step_size,dim1,dim2);

                const float temp_int = image.mesh[j_ + x_*image.y_num + z_*image.x_num*image.y_num];

                proj_img.mesh[dim1 + (dim2)*proj_img.y_num] = std::max(temp_int,proj_img.mesh[ dim1 + (dim2)*proj_img.y_num]);

            }
        }
    }


    timer.stop_timer();

    debug_write(proj_img,"parllel_proj_mesh" + std::to_string(pars.proj_type));

}



template<typename S>
void gen_raster_cast(PartCellStructure<S,uint64_t>& pc_struct,proj_par& pars){
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

    //iterator
    CurrentLevel<float,uint64_t> curr_level(pc_data);

    //Need to add here a parameters here

    //unsigned int direction = 0;
    Mesh_data<std::vector<S>> proj_img;

    float active_x = 2;
    float active_y = 2;
    float active_z = 2;

    float start = 1;

    switch(pars.direction){
        case(0):{
            //x//z
            proj_img.initialize(pc_struct.org_dims[1],pc_struct.org_dims[2],1);

            active_x = 0;
            active_z = 1;
            start = 1;
            break;
        }
        case(1): {
            //x//z
            proj_img.initialize(pc_struct.org_dims[1], pc_struct.org_dims[2], 1);

            active_x = 0;
            active_z = 1;
            start = 2*(pc_struct.org_dims[0]-1) - 1;
            break;
        }
        case(2):{
            //y//z
            proj_img.initialize(pc_struct.org_dims[0], pc_struct.org_dims[2], 1);

            active_y = 0;
            active_z = 1;
            start = 1;
            break;
        }
        case(3):{
            //yz
            proj_img.initialize(pc_struct.org_dims[0], pc_struct.org_dims[2], 1);

            active_y = 0;
            active_z = 1;
            start = 2*(pc_struct.org_dims[1]-1) - 1;
            break;
        }
        case(4):{
            //yx
            proj_img.initialize(pc_struct.org_dims[0], pc_struct.org_dims[1], 1);

            active_y = 0;
            active_x = 1;
            start = 1;
            break;
        }
        case(5):{
            //yx
            proj_img.initialize(pc_struct.org_dims[0], pc_struct.org_dims[1], 1);

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


    timer.stop_timer();

    timer.start_timer("ray cast");
    next_move.dir =0;
    next_move.index =0;

    int z_,x_,j_,y_;

    for(uint64_t depth = (part_new.access_data.depth_max-3);depth <= part_new.access_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = part_new.access_data.x_num[depth];
        const unsigned int z_num_ = part_new.access_data.z_num[depth];

        CurrentLevel<float, uint64_t> curr_level(pc_data);
        curr_level.set_new_depth(depth, part_new);

//#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        curr_level.update_cell(part_new);

                        float temp_int =  curr_level.get_val(particles_int);

                        int dim1 = 0;
                        int dim2 = 0;
                        int size = 0;

                        find_ray_parallel(curr_level,pars,dim1,dim2,size);

                        //add to all the required rays

                        for (int k = 0; k < size; ++k) {
                            for (int i = 0; i < size; ++i) {
                                proj_img(dim1 + i ,(dim2 + k),0).push_back(temp_int);
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

   // debug_write(proj_img,"parllel_proj" + std::to_string(pars.proj_type));

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

/////////////////////////////////////////////////////
//
//  New Non Parallell Projections
//
//
///////////////////////////////////////////////////


template<typename S>
void prospective_mesh_raycast(PartCellStructure<S,uint64_t>& pc_struct,proj_par& pars) {
    //
    //  Bevan Cheeseman 2017
    //
    //  Max Ray Cast Proposective Projection
    //
    //

    float height = 0.5;

    Mesh_data<S> image;

    pc_struct.interp_parts_to_pc(image, pc_struct.part_data.particle_data);

    float radius = 8.5 * image.z_num;

    ///////////////////////////////////////////
    //
    //  Set up Perspective
    //
    ////////////////////////////////////////////

    float x0 = -height * image.x_num;
    float y0 = -image.y_num * .5;
    float z0 = -image.z_num * .5;

    unsigned int imageWidth = image.x_num;
    unsigned int imageHeight = image.y_num;

    Part_timer timer;

    timer.verbose_flag = true;

    timer.start_timer("ray cast mesh prospective");

    for (float theta = 0.0f; theta < 0.4f; theta += 0.02f) {
        Camera cam = Camera(glm::vec3(x0 + radius*sin(theta), y0 , z0 + radius * cos(theta)),
                            glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));
        cam.setTargeted(glm::vec3(0.0f, 0.0f, 0.0f));

        cam.setPerspectiveCamera((float) imageWidth / (float) imageHeight, (float) (60.0f / 180.0f * M_PI), 0.5f,
                                 70.0f);

//    cam.setOrthographicCamera(imageWidth, imageHeight, 1.0f, 200.0f);
        // ray traced object, sitting on the origin, with no rotation applied
        RaytracedObject o = RaytracedObject(glm::vec3(0.0f, 0.0f, 0.0f), glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));

        auto start = std::chrono::high_resolution_clock::now();
        glm::mat4 inverse_projection = glm::inverse(*cam.getProjection());
        glm::mat4 inverse_modelview = glm::inverse((*cam.getView()) * (*o.getModel()));


        Mesh_data<S> proj_img;
        proj_img.initialize(imageHeight, imageWidth, 1, 0);

        //Need to add here a parameters here


        bool end_domain = false;

        //choose random direction to propogate along


        int counter = 0;




        const int dir = pars.direction;

        int z_, x_, j_, y_, i, k;

        //loop over the resolutions of the structure
        const unsigned int x_num_ = image.x_num;
        const unsigned int z_num_ = image.z_num;
        const float step_size = 1;
        const unsigned int y_num_ = image.y_num;

        const glm::mat4 mvp = (*cam.getProjection()) * (*cam.getView());

//#pragma omp parallel for default(shared) private(z_,x_,j_,i,k)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_ * z_ + x_;

                for (j_ = 0; j_ < y_num_; j_++) {


                    glm::vec2 pos = o.worldToScreen(mvp, glm::vec3((float) x_, (float) j_, (float) z_), imageWidth,
                                                    imageHeight);

                    const float temp_int = image.mesh[j_ + x_ * image.y_num + z_ * image.x_num * image.y_num];

                    const int dim1 = -floor(pos.y);
                    const int dim2 = -floor(pos.x);

                    if (dim1 > 0 & dim2 > 0 & (dim1 < proj_img.y_num) & (dim2 < proj_img.x_num)) {

                        proj_img.mesh[dim1 + (dim2) * proj_img.y_num] = std::max(temp_int, proj_img.mesh[dim1 + (dim2) *
                                                                                                                proj_img.y_num]);
                    }
                }
            }
        }

        debug_write(proj_img,"perspective_proj_mesh_" + std::to_string(theta) + "_" + std::to_string(pars.proj_type));
    }


    timer.stop_timer();


}



#endif