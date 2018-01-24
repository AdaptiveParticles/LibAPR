///////////////////
//
//  Bevan Cheeseman 2016
//
//  Part cell base class, specifies interface
//
///////////////

#ifndef PARTPLAY_PARTCELLSTRUCTURE_HPP
#define PARTPLAY_PARTCELLSTRUCTURE_HPP


#include <bitset>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <array>

#include "PartCellData.hpp"
#include "ParticleData.hpp"
#include "benchmarks/development/old_structures/particle_map.hpp"
#include "src/data_structures/Mesh/MeshData.hpp"
#include "benchmarks/development/old_io/writeimage.h"

#ifdef _MSC_VER
#define NOMINMAX
#include <intrin.h>
#include <Windows.h>

uint32_t __inline __builtin_ctz(uint32_t value)
{
	DWORD trailing_zero = 0;

	if (_BitScanForward(&trailing_zero, value))
	{
		return trailing_zero;
	}
	else
	{
		// This is undefined, I better choose 32 than 0
		return 32;
	}
}

uint32_t __inline __builtin_clz(uint32_t value)
{
	DWORD leading_zero = 0;

	if (_BitScanReverse(&leading_zero, value))
	{
		return 31 - leading_zero;
	}
	else
	{
		// Same remarks as above
		return 32;
	}
}
#endif

//parent node defitions

template<typename T>
class APR;

template <typename T,typename S> // type T is the image type, type S is the data structure base type
class PartCellStructure {
    
    
    
private:
    
    uint64_t number_parts = 0;
    uint64_t number_cells = 0;
    
    
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
    
    void create_partcell_structure(std::vector<std::vector<uint8_t>>& p_map){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes an optimal part_map configuration from the pushing scheme and creates an efficient data structure for procesing.
        //
        
        Part_timer timer;
        timer.verbose_flag = true;
        
        pc_data.depth_max = depth_max;
        pc_data.depth_min = depth_min;
        
        pc_data.data.resize(depth_max+1);
        
        pc_data.x_num.resize(depth_max+1);
        pc_data.z_num.resize(depth_max+1);
        
        for(int i = depth_min;i <= depth_max;i++){
            pc_data.x_num[i] = x_num[i];
            pc_data.z_num[i] = z_num[i];
            pc_data.data[i].resize(z_num[i]*x_num[i]);
        }
        
        //initialize loop variables
        uint64_t x_;
        uint64_t z_;
        uint64_t y_;

        
        //next initialize the entries;
        
        uint16_t curr_index;
        uint64_t status;
        uint8_t prev_ind = 0;
        
        timer.start_timer("intiialize part_cells");
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];
            
            
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind) if(z_num_*x_num_ > 100)
#endif
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){
                    
                    size_t first_empty = 0;
                    const size_t offset_part_map = x_*y_num_ + z_*y_num_*x_num_;
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    curr_index = 0;
                    prev_ind = 0;
                    
                    //first value handle the duplication of the gap node
                    
                    status = p_map[i][offset_part_map];
                    if((status> 0)){
                        first_empty = 0;
                    } else {
                        first_empty = 1;
                    }
                    
                    for(y_ = 0;y_ < y_num_;y_++){
                        
                        status = p_map[i][offset_part_map + y_];
                        
                        if(status> 0){
                            curr_index+= 1 + prev_ind;
                            prev_ind = 0;
                        } else {
                            prev_ind = 1;
                        }
                    }
                    
                    if(curr_index == 0){
                        pc_data.data[i][offset_pc_data].resize(1); //always first adds an extra entry for intialization and extra info
                    } else {
                       
                        pc_data.data[i][offset_pc_data].resize(curr_index + 2 - first_empty,0); //gap node to begin, already finishes with a gap node
                    
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
        
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];
            
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind,prev_coord) if(z_num_*x_num_ > 100)
#endif
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    
                    const size_t offset_part_map = x_*y_num_ + z_*y_num_*x_num_;
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    curr_index = 0;
                    prev_ind = 1;
                    prev_coord = 0;
                    
                    //initialize the first values type
                    pc_data.data[i][offset_pc_data][0] = TYPE_GAP_END;
                    
                    for(y_ = 0;y_ < y_num_;y_++){
                        
                        status = p_map[i][offset_part_map + y_];
                        
                        if(status> 0){
                            
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
                            prev_coord = y_;
                            //set type
                            pc_data.data[i][offset_pc_data][curr_index-1] = TYPE_PC;
                            
                            //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);
                            
                            //set the status
                            switch(status){
                                case SEED:
                                {
                                    pc_data.data[i][offset_pc_data][curr_index-1] |= SEED_SHIFTED;
                                    break;
                                }
                                case BOUNDARY:
                                {
                                    pc_data.data[i][offset_pc_data][curr_index-1] |= BOUNDARY_SHIFTED;
                                    break;
                                }
                                case FILLER:
                                {
                                    pc_data.data[i][offset_pc_data][curr_index-1] |= FILLER_SHIFTED;
                                    break;
                                }
                                    
                            }
                            
                            prev_ind = 0;
                        } else {
                            //store for setting above
                            if(prev_ind == 0){
                                //prev_coord = y_;
                            }
                            
                            prev_ind = 1;
                            
                        }
                    }
                    
                    //Initialize the last value GAP END indicators to no neighbour
                    pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size()-1] = TYPE_GAP_END;
                    pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size()-1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                    pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size()-1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);
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
        pc_data.set_neighbor_relationships();
        
        //std::cout << "Finished neighbour relationships" << std::endl;
        
        // Initialize the particle data access and intensity structures
        part_data.initialize_from_structure(pc_data);
        
        uint64_t num_cells = 0;
        uint64_t num_parts = 0;
        
        uint64_t j_;
        
        S node_val;
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status) reduction(+:num_cells,num_parts)
#endif
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = part_data.access_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        node_val = pc_data.data[i][offset_pc_data][j_];
                        
                        if (!(node_val&1)){
                            //in this loop there is a cell
                            num_cells++;
                            status = pc_data.get_status(node_val);
                            //determine how many particles in the cell
                            if(status==SEED){
                                num_parts=num_parts + 8;
                            } else {
                                num_parts= num_parts + 1;
                            }
                            
                        }
                    }
                    
                }
            }
        }
        
        number_cells = num_cells;
        number_parts = num_parts;
        


        
    }



    void create_particle_structures(std::vector<std::vector<uint16_t>>& Ip){
        
        /////////////////////////////////////
        //
        //  PARTICLE DATA INITIALIZATION
        //
        //////////////////////////////////////
        
        //initialize loop variables
        uint64_t x_;
        uint64_t z_;
 
        
        
        Part_timer timer;
        
        
        
        // Estimate the intensities from the down sampled images
        
        timer.start_timer("Get the intensities");
        
        // Particles are ordered as ((-y,-x,-z),(+y,-x,-z),(-y,+x,-z),(+y,+x,-z),(-y,-x,+z),(+y,-x,+z),(-y,+x,+z),(+y,+x,+z))
        
        S offset;
        
        //
        //  Takes the read in particles and places them back in the correct place in the structure;
        //
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            
            offset = 0;
            
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t j_num = part_data.particle_data.data[i][offset_pc_data].size();
                    
                    std::copy(Ip[i].begin()+offset,Ip[i].begin()+offset+j_num,part_data.particle_data.data[i][offset_pc_data].begin());
                    
                    offset += j_num;
                    
                }
            }
            
        }
        
        timer.stop_timer();
        
        
        
    }
    
    template<typename U>
    void create_partcell_structure(Particle_map<U>& part_map){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes an optimal part_map configuration from the pushing scheme and creates an efficient data structure for procesing.
        //
        
        Part_timer timer;
        timer.verbose_flag = false;
        
        
        timer.start_timer("intiialize base");
        pc_data.initialize_base_structure(part_map);
        timer.stop_timer();
        

        //initialize loop variables
        uint64_t x_;
        uint64_t z_;
        uint64_t y_;
        uint64_t j_;
        
        //next initialize the entries;
        
        uint16_t curr_index;
        uint8_t status;
        uint8_t prev_ind = 0;
        
        timer.start_timer("intiialize part_cells");
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            const unsigned int y_num = part_map.layers[i].y_num;
            
            
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind) if(z_num*x_num > 100)
#endif
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
                        
                        pc_data.data[i][offset_pc_data].resize(curr_index + 2 - first_empty,0); //gap node to begin, already finishes with a gap node
                        
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

        const bool sampling_type = false;

        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            const unsigned int y_num = part_map.layers[i].y_num;
            
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind,prev_coord) if(z_num*x_num > 100)
#endif
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
                            prev_coord = y_;
                            //set type
                            pc_data.data[i][offset_pc_data][curr_index-1] = TYPE_PC;
                            
                            //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);

                            if(sampling_type) {

                                //set the status
                                switch (status) {
                                    case TAKENSTATUS: {
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= SEED_SHIFTED;
                                        break;
                                    }
                                    case NEIGHBOURSTATUS: {
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= SEED_SHIFTED;
                                        break;
                                    }
                                    case SLOPESTATUS: {
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= FILLER_SHIFTED;
                                        break;
                                    }

                                }

                            } else {

                                //set the status
                                switch (status) {
                                    case TAKENSTATUS: {
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= SEED_SHIFTED;
                                        break;
                                    }
                                    case NEIGHBOURSTATUS: {
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= BOUNDARY_SHIFTED;
                                        break;
                                    }
                                    case SLOPESTATUS: {
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= FILLER_SHIFTED;
                                        break;
                                    }

                                }


                            }
                            
                            prev_ind = 0;
                        } else {
                            //store for setting above
                            if(prev_ind == 0){
                                //prev_coord = y_;
                            }
                            
                            prev_ind = 1;
                            
                        }
                    }
                    
                    //Initialize the last value GAP END indicators to no neighbour
                    pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size()-1] = TYPE_GAP_END;
                    pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size()-1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                    pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size()-1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);
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
        pc_data.set_neighbor_relationships();
        
       // std::cout << "Finished neighbour relationships" << std::endl;
        
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
        
        S part_offset;
        S node_val;
        S y_coord;
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord,part_offset) if(z_num*x_num > 100)
#endif
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    const size_t offset_pc_data = x_num*z_ + x_;
                    y_coord = 0;
                    const size_t j_num = part_data.access_data.data[i][offset_pc_data].size();
                    
                    
                    S x_2p = std::min(2*x_+1,(S)(part_map.downsampled[i+1].x_num-1));
                    S z_2p = std::min(2*z_+1,(S)(part_map.downsampled[i+1].z_num-1));
                    
                    const size_t offset_part_map_data_0 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*2*z_ + part_map.downsampled[i+1].y_num*2*x_;
                    const size_t offset_part_map_data_1 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*2*z_ + part_map.downsampled[i+1].y_num*x_2p;
                    const size_t offset_part_map_data_2 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*z_2p + part_map.downsampled[i+1].y_num*2*x_;
                    const size_t offset_part_map_data_3 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*z_2p + part_map.downsampled[i+1].y_num*x_2p;
                    
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
                                
                                
                                S y_2p = std::min(2*y_coord+1,(S)(part_map.downsampled[i+1].y_num-1));
                                //need to sampled the image at depth + 1
                                part_offset = (node_val & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                                
                                
                                part_data.particle_data.data[i][offset_pc_data][part_offset] = part_map.downsampled[i+1].mesh[offset_part_map_data_0 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+1] = part_map.downsampled[i+1].mesh[offset_part_map_data_0 + y_2p];
                                //1
                                part_data.particle_data.data[i][offset_pc_data][part_offset+2] = part_map.downsampled[i+1].mesh[offset_part_map_data_1 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+3] = part_map.downsampled[i+1].mesh[offset_part_map_data_1 + y_2p];
                                //3
                                part_data.particle_data.data[i][offset_pc_data][part_offset+4] = part_map.downsampled[i+1].mesh[offset_part_map_data_2 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+5] = part_map.downsampled[i+1].mesh[offset_part_map_data_2 + y_2p];
                                //4
                                part_data.particle_data.data[i][offset_pc_data][part_offset+6] = part_map.downsampled[i+1].mesh[offset_part_map_data_3 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+7] = part_map.downsampled[i+1].mesh[offset_part_map_data_3 + y_2p];
                                
                            }
                            
                            
                        }
                    }
                    
                }
            }
            
            // BOUNDARY/FILLER PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord,part_offset) if(z_num*x_num > 100)
#endif
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
        
        
        //Lastly calculate the number of particle and number of cells
        
        
        S num_cells = 0;
        S num_parts = 0;
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord,part_offset) reduction(+:num_cells,num_parts) if(z_num*x_num > 100)
#endif
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    const size_t offset_pc_data = x_num*z_ + x_;
                    
                    const size_t j_num = part_data.access_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        node_val = part_data.access_data.data[i][offset_pc_data][j_];
                        
                        if (!(node_val&1)){
                            //in this loop there is a cell
                            num_cells++;
                            
                            //determine how many particles in the cell
                            if(part_data.access_node_get_status(node_val)==SEED){
                                num_parts+=8;
                            } else {
                                num_parts+=1;
                            }
                            
                        }
                    }
                    
                }
            }
        }
        
        number_cells = num_cells;
        number_parts = num_parts;
        
        
    }
    

    
    void test_get_neigh_dir_memory(){
        //
        // Test the get neighbour direction code for speed
        //
        
        uint64_t z_;
        uint64_t x_;
        uint64_t j_;
        uint64_t node_val;
        
        
        Part_timer timer;
        
        timer.verbose_flag = 1;
        
        timer.start_timer("get neighbour cells memory all");
        
        uint64_t curr_key;
        std::vector<uint64_t> neigh_keys;
        
        PartCellData<std::vector<uint64_t>> neigh_vec_all;
        neigh_vec_all.initialize_from_partcelldata(pc_data);
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
#endif
            for(z_ = 0;z_ < z_num_;z_++){
                
                
                for(x_ = 0;x_ < x_num_;x_++){
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    neigh_vec_all.data[i][offset_pc_data].resize(j_num);
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        neigh_vec_all.data[i][offset_pc_data][j_].reserve(6);
                    }
                }
            }
            
        }
        
        
        
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key) if(z_num_*x_num_ > 100)
#endif
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                curr_key |= ((uint64_t)i) << PC_KEY_DEPTH_SHIFT;
                curr_key |= z_ << PC_KEY_Z_SHIFT;
                
                //neigh_keys.reserve(24);
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    curr_key &=  -((PC_KEY_X_MASK) + 1);
                    curr_key |= x_ << PC_KEY_X_SHIFT;
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val = pc_data.data[i][offset_pc_data][j_];
                        
                        if (!(node_val&1)){
                            //get the index gap node
                            
                            curr_key &= -((PC_KEY_J_MASK) + 1);
                            curr_key |= j_ << PC_KEY_J_SHIFT;
                            
                            pc_data.get_neigh_0(curr_key,node_val,neigh_vec_all.data[i][offset_pc_data][j_]);
                            pc_data.get_neigh_1(curr_key,node_val,neigh_vec_all.data[i][offset_pc_data][j_]);
                            pc_data.get_neigh_2(curr_key,node_val,neigh_vec_all.data[i][offset_pc_data][j_]);
                            pc_data.get_neigh_3(curr_key,node_val,neigh_vec_all.data[i][offset_pc_data][j_]);
                            pc_data.get_neigh_4(curr_key,node_val,neigh_vec_all.data[i][offset_pc_data][j_]);
                            pc_data.get_neigh_5(curr_key,node_val,neigh_vec_all.data[i][offset_pc_data][j_]);
                            
                        } else {
                            
                        }
                        
                    }
                    
                }
                
            }
        }
        
        timer.stop_timer();
        
        
    }

    
    template<typename U>
    void test_partcell_struct(Particle_map<U>& part_map){
        //
        //  Bevan Cheeseman 2016
        //
        //  Compares the PartCellStructure with the partmap and checks if things are correct
        //
        
        
        //initialize
        uint64_t node_val;
        uint64_t y_coord;
        int x_;
        int z_;
        uint64_t j_;
        uint64_t status;
        uint64_t status_org;
        
        
        S type;
        S yp_index;
        S yp_depth;
        S ym_index;
        S ym_depth;
        S next_coord;
        S prev_coord;
        
    
        S xp_index;
        S xp_depth;
        S zp_index;
        S zp_depth;
        S xm_index;
        S xm_depth;
        S zm_index;
        S zm_depth;
        
        //basic tests of status and coordinates
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            
            
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    const size_t offset_pc_data = x_num*z_ + x_;
                    y_coord = 0;
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    const size_t offset_part_map_data_0 = part_map.downsampled[i].y_num*part_map.downsampled[i].x_num*z_ + part_map.downsampled[i].y_num*x_;
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        
                        node_val = pc_data.data[i][offset_pc_data][j_];
                        
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
                            
                            //set the status
                            switch(status_org){
                                case TAKENSTATUS:
                                {
                                    if(status != SEED){
                                        std::cout << "STATUS SEED BUG" << std::endl;
                                    }
                                    break;
                                }
                                case NEIGHBOURSTATUS:
                                {
                                    if(status != BOUNDARY){
                                        std::cout << "STATUS BOUNDARY BUG" << std::endl;
                                    }
                                    break;
                                }
                                case SLOPESTATUS:
                                {
                                    if(status != FILLER){
                                        std::cout << "STATUS FILLER BUG" << std::endl;
                                    }

                                    break;
                                }
                                    
                            }
                            
                            
                            
                        }
                    }
                    
                }
            }
            
        }
        
        //Neighbour Checking
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            
            
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    const size_t offset_pc_data = x_num*z_ + x_;
                    y_coord = 0;
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    const size_t offset_part_map_data_0 = part_map.downsampled[i].y_num*part_map.downsampled[i].x_num*z_ + part_map.downsampled[i].y_num*x_;
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        
                        node_val = pc_data.data[i][offset_pc_data][j_];
                        
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
                                
                                node_val = pc_data.data[i][offset_pc_data][j_];
                                
                                S x_n = 0;
                                S z_n = 0;
                                S y_n = 0;
                                S depth = 0;
                                S index_n = 0;
                                S status_n = 1;
                                S node_n = 0;
                                S neigh_indicator = 0;
                                
                                if(face < 2){
                                    node_val = pc_data.data[i][offset_pc_data][j_+pc_data.von_neumann_y_cells[face]];
                                    
                                    if(!(node_val&1)){
                                        //same level
                                        
                                        neigh_indicator = LEVEL_SAME;
                                    } else {
                                        neigh_indicator = (node_val & pc_data.depth_mask_dir[face]) >> pc_data.depth_shift_dir[face];
                                    }
                                    
                                } else {
                                    neigh_indicator = (node_val & pc_data.depth_mask_dir[face]) >> pc_data.depth_shift_dir[face];
                                }
                                
                                
                                switch(neigh_indicator){
                                    case(LEVEL_SAME):{
                                        //same level return single neighbour
                                        
                                        depth = i;
                                        x_n = x_ + pc_data.von_neumann_x_cells[face];
                                        y_n = y_coord + pc_data.von_neumann_y_cells[face];
                                        z_n = z_ + pc_data.von_neumann_z_cells[face];
                                        
                                        const size_t offset_part_map = part_map.downsampled[depth].y_num*part_map.downsampled[depth].x_num*z_n + part_map.downsampled[depth].y_num*x_n;
                                        
                                        //get the value in the original array
                                        status_n = part_map.layers[depth].mesh[offset_part_map + y_n];
                                        
                                        if(face < 2){
                                            node_n = node_val;
                                            index_n = j_ +pc_data.von_neumann_y_cells[face];
                                        } else {
                                            index_n = (((node_val & pc_data.index_mask_dir[face]) >> pc_data.index_shift_dir[face]));
                                        
                                            const size_t offset_pc_data_loc = pc_data.x_num[depth]*z_n + x_n;
                                            node_n = pc_data.data[depth][offset_pc_data_loc][index_n];
                                        }
                                            
                                        break;
                                    }
                                    case(LEVEL_DOWN):{
                                        // Neighbour is on parent level (depth - 1)
                                        depth = i-1;
                                        
                                        x_n = (x_ + pc_data.von_neumann_x_cells[face])/2;
                                        y_n = (y_coord + pc_data.von_neumann_y_cells[face])/2;
                                        z_n = (z_ + pc_data.von_neumann_z_cells[face])/2;
                                        
                                        const size_t offset_part_map = part_map.downsampled[depth].y_num*part_map.downsampled[depth].x_num*z_n + part_map.downsampled[depth].y_num*x_n;
                                        
                                        status_n = part_map.layers[depth].mesh[offset_part_map + y_n];
                                        
                                        index_n = (((node_val & pc_data.index_mask_dir[face]) >> pc_data.index_shift_dir[face]));
                                        
                                        const size_t offset_pc_data_loc = pc_data.x_num[depth]*z_n + x_n;
                                        node_n = pc_data.data[depth][offset_pc_data_loc][index_n];
                                        
                                        break;
                                    }
                                    case(LEVEL_UP):{
                                        depth = i+1;
                                        
                                        x_n = (x_ + pc_data.von_neumann_x_cells[face])*2 + (pc_data.von_neumann_x_cells[face] < 0);
                                        y_n = (y_coord + pc_data.von_neumann_y_cells[face])*2 + (pc_data.von_neumann_y_cells[face] < 0);
                                        z_n = (z_ + pc_data.von_neumann_z_cells[face])*2 + (pc_data.von_neumann_z_cells[face] < 0);
                                        
                                        const size_t offset_part_map = part_map.downsampled[depth].y_num*part_map.downsampled[depth].x_num*z_n + part_map.downsampled[depth].y_num*x_n;
                                        
                                        status_n = part_map.layers[depth].mesh[offset_part_map + y_n];
                                        
                                        index_n = (((node_val & pc_data.index_mask_dir[face]) >> pc_data.index_shift_dir[face]));
                                        
                                        const size_t offset_pc_data_loc = pc_data.x_num[depth]*z_n + x_n;
                                        node_n = pc_data.data[depth][offset_pc_data_loc][index_n];

                                        break;
                                    }
                                }
                                
                                if((status_n> 0) & (status_n < 8)){
                                    //fine
                                } else {
                                    std::cout << "NEIGHBOUR LEVEL BUG" << std::endl;
                                }
                                
                                if (node_n&1){
                                    //points to gap node
                                    std::cout << "INDEX BUG" << std::endl;
                                } else {
                                    //points to real node, correct
                                }
                                
                                
                            }
                            
                            
                        }
                    }
                    
                }
            }
            
        }

        
        //Neighbour Routine Checking
    
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            
            
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    const size_t offset_pc_data = x_num*z_ + x_;
                    y_coord = 0;
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    const size_t offset_part_map_data_0 = part_map.downsampled[i].y_num*part_map.downsampled[i].x_num*z_ + part_map.downsampled[i].y_num*x_;
                    
                    
                    
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        
                        node_val = pc_data.data[i][offset_pc_data][j_];
                        
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
                                
                                S x_n = 0;
                                S z_n = 0;
                                S y_n = 0;
                                S depth = 0;
                                S j_n = 0;
                                S status_n = 1;
                                S node_n = 0;
                                
                                PartCellNeigh<S> neigh_keys_;
                                std::vector<S> neigh_keys;
                                S curr_key = 0;
                                curr_key |= ((uint64_t)i) << PC_KEY_DEPTH_SHIFT;
                                curr_key |= z_ << PC_KEY_Z_SHIFT;
                                curr_key |= x_ << PC_KEY_X_SHIFT;
                                curr_key |= j_ << PC_KEY_J_SHIFT;
                                
                                pc_data.get_neighs_face(curr_key,node_val,face,neigh_keys_);
                                
                                neigh_keys = neigh_keys_.neigh_face[face];
                                
                                if (neigh_keys.size() > 0){
                                    depth = (neigh_keys[0] & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
                                    
                                    if(i == depth){
                                        y_n = y_coord + pc_data.von_neumann_y_cells[face];
                                    } else if (depth > i){
                                        //neighbours are on layer down (4)
                                        y_n = (y_coord + pc_data.von_neumann_y_cells[face])*2 + (pc_data.von_neumann_y_cells[face] < 0);
                                    } else {
                                        //neighbour is parent
                                        y_n =  (y_coord + pc_data.von_neumann_y_cells[face])/2;
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
                                        y_n = y_n + pc_data.von_neumann_y_cells[pc_data.neigh_child_dir[face][n-1]];
                                    } else if (n ==2){
                                        y_n = y_org + pc_data.von_neumann_y_cells[pc_data.neigh_child_dir[face][n-1]];
                                    }
                                    
                                    S y_n2;S x_n2;S z_n2;S depth2;
                                    pc_data.get_neigh_coordinates_cell(neigh_keys_,face,n,y_coord,y_n2,x_n2,z_n2,depth2);
                                    
                                    
                                    if (neigh_keys[n] > 0){
                                        
                                        //calculate y so you can check back in the original structure
                                        const size_t offset_pc_data_loc = pc_data.x_num[depth]*z_n + x_n;
                                        node_n = pc_data.data[depth][offset_pc_data_loc][j_n];
                                        const size_t offset_part_map = part_map.downsampled[depth].y_num*part_map.downsampled[depth].x_num*z_n + part_map.downsampled[depth].y_num*x_n;
                                        status_n = part_map.layers[depth].mesh[offset_part_map + y_n];
                                        
                                        
                                        
                                        if((status_n> 0) & (status_n < 8)){
                                            //fine
                                        } else {
                                            std::cout << "NEIGHBOUR LEVEL BUG" << std::endl;
                                        }
                                        
                                        
                                        if (node_n&1){
                                            //points to gap node
                                            std::cout << "INDEX BUG" << std::endl;
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
    
        
        
        //Neighbour Routine Checking
        
        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];
            
            
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    const size_t offset_pc_data = x_num*z_ + x_;
                    y_coord = 0;
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    const size_t offset_part_map_data_0 = part_map.downsampled[i].y_num*part_map.downsampled[i].x_num*z_ + part_map.downsampled[i].y_num*x_;
                    
                    
                    
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        
                        node_val = pc_data.data[i][offset_pc_data][j_];
                        
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
                                
                                S x_n = 0;
                                S z_n = 0;
                                S y_n = 0;
                                S depth = 0;
                                S j_n = 0;
                                S status_n = 1;
                                S node_n = 0;
                                
                                PartCellNeigh<S> neigh_keys;
                                S curr_key = 0;
                                curr_key |= ((uint64_t)i) << PC_KEY_DEPTH_SHIFT;
                                curr_key |= z_ << PC_KEY_Z_SHIFT;
                                curr_key |= x_ << PC_KEY_X_SHIFT;
                                curr_key |= j_ << PC_KEY_J_SHIFT;
                                
                                pc_data.get_neighs_face(curr_key,node_val,face,neigh_keys);
                                
                                
                                for(int n = 0;n < neigh_keys.neigh_face[face].size();n++){
                                    
                                    
                                    pc_data.get_neigh_coordinates_cell(neigh_keys,face,n,y_coord,y_n,x_n,z_n,depth);
                                    j_n = (neigh_keys.neigh_face[face][n] & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
                                    
                                    if (neigh_keys.neigh_face[face][n] > 0){
                                        
                                        //calculate y so you can check back in the original structure
                                        const size_t offset_pc_data_loc = pc_data.x_num[depth]*z_n + x_n;
                                        node_n = pc_data.data[depth][offset_pc_data_loc][j_n];
                                        const size_t offset_part_map = part_map.downsampled[depth].y_num*part_map.downsampled[depth].x_num*z_n + part_map.downsampled[depth].y_num*x_n;
                                        status_n = part_map.layers[depth].mesh[offset_part_map + y_n];
                                        
                                        if((status_n> 0) & (status_n < 8)){
                                            //fine
                                        } else {
                                            std::cout << "NEIGHBOUR LEVEL BUG" << std::endl;
                                        }
                                        
                                        
                                        if (node_n&1){
                                            //points to gap node
                                            std::cout << "INDEX BUG" << std::endl;
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

        
    
    }



     // Particles are ordered as ((-y,-x,-z),(+y,-x,-z),(-y,+x,-z),(+y,+x,-z),(-y,-x,+z),(+y,-x,+z),(-y,+x,+z),(+y,+x,+z))
    const uint8_t seed_part_y[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    const uint8_t seed_part_x[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    const uint8_t seed_part_z[8] = {0, 0, 0, 0, 1, 1, 1, 1};
                                                               
    const int8_t von_neumann_y[6] = {0, 0, -1, 1, 0, 0};
    const int8_t von_neumann_x[6] = {0, -1, 0, 0, 1, 0};
    const int8_t von_neumann_z[6] = {-1, 0, 0, 0, 0, 1};
    
public:
    
    PartCellData<S> pc_data; //particle cell data
    ParticleData<T,S> part_data; //individual particle data
    
    unsigned int depth_min;
    unsigned int depth_max;

    std::string name;

    Proc_par pars;
    
    std::vector<unsigned int> x_num;
    std::vector<unsigned int> y_num;
    std::vector<unsigned int> z_num;
    
    std::vector<unsigned int> org_dims;

    PartCellStructure(){

        org_dims.resize(3);

    }

    PartCellStructure(Particle_map<T> &particle_map){
        //
        //  initialization of the tree structure
        //

        initialize_structure(particle_map);
    }
    
    uint64_t get_number_parts(){
        //calculated on initialization
        return number_parts;
    }
    
    
    uint64_t get_number_cells(){
        //calculated on intiialization
        return number_cells;
    }
    
    uint8_t get_status(uint64_t node_val){
        //
        //  Extracts the status
        //
        
        return (node_val & STATUS_MASK) >> STATUS_SHIFT;
    }
    
    
    void initialize_structure_read(std::vector<std::vector<uint8_t>>& p_map,std::vector<std::vector<uint16_t>>& Ip){
        //
        //  Re-creates the structure from the read in p_map and particle data
        //
        
        create_partcell_structure(p_map);
        create_particle_structures(Ip);
        
        pc_data.y_num = y_num;
    }
    
    void initialize_particle_read(std::vector<std::vector<uint8_t>>& p_map){
        //
        //  Re-creates the structure from the read in p_map and particle data
        //
        
        create_partcell_structure(p_map);
        pc_data.y_num = y_num;
    }
    
    void initialize_pc_read(std::vector<std::vector<uint8_t>>& p_map){
        //
        //  Re-creates the structure from the read in p_map and particle data
        //
        
        create_partcell_structure(p_map);
        pc_data.y_num = y_num;
    }





    template<typename U>
    void update_parts(Particle_map<U>& part_map){

        S part_offset;
        S node_val;
        S y_coord;


        //initialize loop variables
        uint64_t x_;
        uint64_t z_;
        uint64_t y_;
        uint64_t j_;

        //next initialize the entries;

        uint16_t curr_index;
        uint8_t status;
        uint8_t prev_ind = 0;



        for(int i = pc_data.depth_min;i <= pc_data.depth_max;i++){

            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];


            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.

            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord,part_offset) if(z_num*x_num > 100)
#endif
            for(z_ = 0;z_ < z_num;z_++){

                for(x_ = 0;x_ < x_num;x_++){

                    const size_t offset_pc_data = x_num*z_ + x_;
                    y_coord = 0;
                    const size_t j_num = part_data.access_data.data[i][offset_pc_data].size();


                    S x_2p = std::min(2*x_+1,(S)(part_map.downsampled[i+1].x_num-1));
                    S z_2p = std::min(2*z_+1,(S)(part_map.downsampled[i+1].z_num-1));

                    const size_t offset_part_map_data_0 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*2*z_ + part_map.downsampled[i+1].y_num*2*x_;
                    const size_t offset_part_map_data_1 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*2*z_ + part_map.downsampled[i+1].y_num*x_2p;
                    const size_t offset_part_map_data_2 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*z_2p + part_map.downsampled[i+1].y_num*2*x_;
                    const size_t offset_part_map_data_3 = part_map.downsampled[i+1].y_num*part_map.downsampled[i+1].x_num*z_2p + part_map.downsampled[i+1].y_num*x_2p;

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


                                S y_2p = std::min(2*y_coord+1,(S)(part_map.downsampled[i+1].y_num-1));
                                //need to sampled the image at depth + 1
                                part_offset = (node_val & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;


                                part_data.particle_data.data[i][offset_pc_data][part_offset] = part_map.downsampled[i+1].mesh[offset_part_map_data_0 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+1] = part_map.downsampled[i+1].mesh[offset_part_map_data_0 + y_2p];
                                //1
                                part_data.particle_data.data[i][offset_pc_data][part_offset+2] = part_map.downsampled[i+1].mesh[offset_part_map_data_1 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+3] = part_map.downsampled[i+1].mesh[offset_part_map_data_1 + y_2p];
                                //3
                                part_data.particle_data.data[i][offset_pc_data][part_offset+4] = part_map.downsampled[i+1].mesh[offset_part_map_data_2 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+5] = part_map.downsampled[i+1].mesh[offset_part_map_data_2 + y_2p];
                                //4
                                part_data.particle_data.data[i][offset_pc_data][part_offset+6] = part_map.downsampled[i+1].mesh[offset_part_map_data_3 + 2*y_coord];
                                part_data.particle_data.data[i][offset_pc_data][part_offset+7] = part_map.downsampled[i+1].mesh[offset_part_map_data_3 + y_2p];

                            }


                        }
                    }

                }
            }

            // BOUNDARY/FILLER PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord,part_offset) if(z_num*x_num > 100)
#endif
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



    }


    //decleration
    template<typename U>
    void initialize_structure(Particle_map<U>& particle_map){


        depth_min = particle_map.k_min;
        depth_max = particle_map.k_max;

        org_dims.resize(3);

        org_dims[0] = particle_map.downsampled[depth_max+1].y_num;
        org_dims[1] = particle_map.downsampled[depth_max+1].x_num;
        org_dims[2] = particle_map.downsampled[depth_max+1].z_num;

        x_num.resize(depth_max+1);
        y_num.resize(depth_max+1);
        z_num.resize(depth_max+1);

        for(int i = depth_min;i <= depth_max;i++){
            x_num[i] = particle_map.downsampled[i].x_num;
            y_num[i] = particle_map.downsampled[i].y_num;
            z_num[i] = particle_map.downsampled[i].z_num;

        }


        create_partcell_structure(particle_map);
    }
    
    template<typename U,typename V>
    void interp_parts_to_pc(MeshData<U>& out_image,ExtraPartCellData<V>& interp_data){
        MeshData<U> curr_k_img;
        MeshData<U> prev_k_img;
        
        int y_dim = ceil(org_dims[0]/2.0)*2;
        int x_dim = ceil(org_dims[1]/2.0)*2;
        int z_dim = ceil(org_dims[2]/2.0)*2;
        
        prev_k_img.mesh.resize(x_dim*z_dim*y_dim);
        curr_k_img.mesh.resize(x_dim*z_dim*y_dim);
        
        interp_parts_to_pc(interp_data,curr_k_img,prev_k_img);
        
        std::swap(out_image,curr_k_img);
    
    }
    
    template<typename U,typename V>
    void interp_parts_to_pc(ExtraPartCellData<V>& interp_data,MeshData<U>& curr_k_img,MeshData<U>& prev_k_img){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and creates piece-wise constant image
        //
        

       // MeshData<U> curr_k_img;
        //MeshData<U> prev_k_img;
        
        constexpr int y_incr[8] = {0,1,0,1,0,1,0,1};
        constexpr int x_incr[8] = {0,0,1,1,0,0,1,1};
        constexpr int z_incr[8] = {0,0,0,0,1,1,1,1};
        
        prev_k_img.set_size(pow(2,depth_min-1),pow(2,depth_min-1),pow(2,depth_min-1));
        
        Part_timer timer;
        timer.verbose_flag = false;
        
        Part_timer t_n;
        t_n.verbose_flag = false;
        t_n.start_timer("loop");
        
        uint64_t z_ = 0;
        uint64_t x_ = 0;
        uint64_t j_ = 0;
        uint64_t y_coord = 0;
        uint64_t status = 0;
        uint64_t part_offset = 0;
        uint64_t curr_key = 0;
        uint64_t node_val = 0;
        
        uint64_t x_p = 0;
        uint64_t y_p = 0;
        uint64_t z_p = 0;
        uint64_t depth_ = 0;
        uint64_t status_ = 0;
        
        //loop over all levels of k
        for (uint64_t d = depth_min; depth_max >= d; d++) {
            
            ///////////////////////////////////////////////////////////////
            //
            //  Transfer previous level to current level
            //
            ////////////////////////////////////////////////////////////////
            timer.start_timer("upsample");
            
            const_upsample_img(curr_k_img,prev_k_img,org_dims);
            
            timer.stop_timer();
            
            /////////////////////////////////////////////////////////////////////
            //
            //  Place seed particles
            //
            //
            /////////////////////////////////////////////////////////////////
            
            timer.start_timer("particle loop");
            
            if ( d > depth_min){
                
                
                const unsigned int x_num_ = x_num[d-1];
                const unsigned int z_num_ = z_num[d-1];
                
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,status,part_offset,x_p,y_p,z_p,depth_,status_,y_coord) if(z_num_*x_num_ > 100)
#endif
                for(z_ = 0;z_ < z_num_;z_++){
                    
                    curr_key = 0;
                    
                    //set the key values
                    part_data.access_data.pc_key_set_z(curr_key,z_);
                    part_data.access_data.pc_key_set_depth(curr_key,d-1);
                    
                    for(x_ = 0;x_ < x_num_;x_++){
                        
                        part_data.access_data.pc_key_set_x(curr_key,x_);
                        
                        const size_t offset_pc_data = x_num_*z_ + x_;
                        
                        //number of nodes on the level
                        const size_t j_num = part_data.access_data.data[d-1][offset_pc_data].size();
                        
                        y_coord = 0;
                        
                        for(j_ = 0;j_ < j_num;j_++){
                            
                            //this value encodes the state and neighbour locations of the particle cell
                            node_val = part_data.access_data.data[d-1][offset_pc_data][j_];
                            
                            if (!(node_val&1)){
                                //This node represents a particle cell
                                y_coord++;
                                
                                //set the key index
                                part_data.access_data.pc_key_set_j(curr_key,j_);
                                
                                //get all the neighbours
                                status = part_data.access_node_get_status(node_val);
                                
                                if(status == SEED){
                                    
                                    part_offset = part_data.access_node_get_part_offset(node_val);
                                    
                                    part_data.access_data.pc_key_set_status(curr_key,status);
                                    
                                    part_data.access_data.get_coordinates_cell(y_coord,curr_key,x_p,z_p,y_p,depth_,status_);

                                    
                                    //loop over the particles
                                    for(int p = 0;p < part_data.get_num_parts(status);p++){
                                        // get coordinates
                                        part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                                        
                                        curr_k_img.mesh[2*y_p+ y_incr[p] +  curr_k_img.y_num*(2*x_p + x_incr[p]) + curr_k_img.x_num*curr_k_img.y_num*(2*z_p + z_incr[p])] = interp_data.get_part(curr_key);
                                        
                                    }
                                    
                                } else {
                                    
                                }
                                
                            } else {
                                //This is a gap node
                                y_coord += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                                y_coord--;
                            }
                            
                        }
                    }
                }
                
            }
            
            timer.stop_timer();
            
      
            
            //////////////////////////////////////
            //
            //  Get cell info from representation
            //
            ///////////////////////////////////
            
            
            const unsigned int x_num_ = x_num[d];
            const unsigned int z_num_ = z_num[d];
            
            timer.start_timer("particle loop");
            
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,status,part_offset,x_p,y_p,z_p,depth_,status_,y_coord) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                //set the key values
                part_data.access_data.pc_key_set_z(curr_key,z_);
                part_data.access_data.pc_key_set_depth(curr_key,d);
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    part_data.access_data.pc_key_set_x(curr_key,x_);
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    //number of nodes on the level
                    const size_t j_num = part_data.access_data.data[d][offset_pc_data].size();
                    
                    y_coord = 0;
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        //this value encodes the state and neighbour locations of the particle cell
                        node_val = part_data.access_data.data[d][offset_pc_data][j_];
                        
                        if (!(node_val&1)){
                            //This node represents a particle cell
                            y_coord++;
                            
                            //set the key index
                            part_data.access_data.pc_key_set_j(curr_key,j_);
                            
                            //get all the neighbours
                            status = part_data.access_node_get_status(node_val);
                            
                            
                            if(status == SEED){
                                
                                
                            } else {
                               
                                part_offset = part_data.access_node_get_part_offset(node_val);
                                
                                part_data.access_data.pc_key_set_status(curr_key,status);
                                
                                part_data.access_data.get_coordinates_cell(y_coord,curr_key,x_p,z_p,y_p,depth_,status_);
                                
                                part_data.access_data.pc_key_set_index(curr_key,part_offset);
                                curr_k_img.mesh[y_p + curr_k_img.y_num*x_p + curr_k_img.x_num*curr_k_img.y_num*z_p] = interp_data.get_part(curr_key);

                            }
                            
                        } else {
                            //This is a gap node
                            y_coord += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                            y_coord--;
                        }
                        
                    }
                }
            }
            
            timer.stop_timer();
            
            /////////////////////////////////////////////////
            //
            //  Place single particles into image
            //
            /////////////////////////////////////////////////
            
            
            std::swap(prev_k_img,curr_k_img);
            
        }
        
        timer.start_timer("upsample");
        
        const_upsample_img(curr_k_img,prev_k_img,org_dims);
        
        timer.stop_timer();
        
        
        timer.start_timer("particle loop");
        
        const unsigned int x_num_ = x_num[depth_max];
        const unsigned int z_num_ = z_num[depth_max];
        
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,status,part_offset,x_p,y_p,z_p,depth_,status_,y_coord) if(z_num_*x_num_ > 100)
#endif
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            //set the key values
            part_data.access_data.pc_key_set_z(curr_key,z_);
            part_data.access_data.pc_key_set_depth(curr_key,depth_max);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                part_data.access_data.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                //number of nodes on the level
                const size_t j_num = part_data.access_data.data[depth_max][offset_pc_data].size();
                
                y_coord = 0;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //this value encodes the state and neighbour locations of the particle cell
                    node_val = part_data.access_data.data[depth_max][offset_pc_data][j_];
                    
                    if (!(node_val&1)){
                        //This node represents a particle cell
                        y_coord++;
                        
                        //set the key index
                        part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        //get all the neighbours
                        status = part_data.access_node_get_status(node_val);
                        
                        
                        if(status == SEED){
                            
                            part_offset = part_data.access_node_get_part_offset(node_val);
                            
                            part_data.access_data.pc_key_set_status(curr_key,status);
                            
                            part_data.access_data.get_coordinates_cell(y_coord,curr_key,x_p,z_p,y_p,depth_,status_);
                            
                            
                            //loop over the particles
                            for(int p = 0;p < part_data.get_num_parts(status);p++){
                                // get coordinates
                                part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                                
                                curr_k_img.mesh[2*y_p+ y_incr[p] +  curr_k_img.y_num*(2*x_p + x_incr[p]) + curr_k_img.x_num*curr_k_img.y_num*(2*z_p + z_incr[p])] = interp_data.get_part(curr_key);
                                
                            }
                            
                        } else {
                            
                        }
                        
                    } else {
                        //This is a gap node
                        y_coord += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }
                    
                }
            }
        }
        
    
        timer.stop_timer();
    
        t_n.stop_timer();
        
        
    }

    
};




#endif //PARTPLAY_PARTCELLSTRUCTURE_HPP