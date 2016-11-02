///////////////////
//
//  Bevan Cheeseman 2016
//
//  PartCellData class, the data container for CRS sparse format for APR
//
///////////////

#ifndef PARTPLAY_PARTICLEDATA_HPP
#define PARTPLAY_PARTICLEDATA_HPP

#include <stdint.h>

#include "PartCellData.hpp"
#include "PartKey.hpp"

#define SEED_NUM_PARTICLES 8
#define NON_SEED_NUM_PARTICLES 1

// bit shift definitions for particle data access
#define TYPE_MASK_PARTICLE (uint64_t) 1
#define TYPE_SHIFT_PARTICLE 0
#define STATUS_MASK_PARTICLE ((((uint64_t)1) << 2) - 1) << 1
#define STATUS_SHIFT_PARTICLE 1
// particle index offset y direction
#define Y_PINDEX_MASK_PARTICLE ((((uint64_t)1) << 13) - 1) << 3
#define Y_PINDEX_SHIFT_PARTICLE 3

// gap node definitions
#define Y_DEPTH_MASK_PARTICLE ((((uint64_t)1) << 2) - 1) << 1
#define Y_DEPTH_SHIFT_PARTICLE 1
#define COORD_DIFF_MASK_PARTICLE ((((uint64_t)1) << 13) - 1) << 3
#define COORD_DIFF_SHIFT_PARTICLE 3

template <typename T,typename S> // type T is the image type, type S is the data structure base type
class ParticleData {
    
public:
    
    /*
     * Number of layers without the root and the contents.
     */
    
    ParticleData(){};
    
    T& operator ()(int depth, int x_,int z_,int j_,int index){
        // data access
        return access_data[depth][x_num[depth]*z_ + x_][j_];
    }
    
    
    T& operator ()(const PartKey& key){
        // data access
        uint16_t offset = access_data[key.depth][access_data.x_num[key.depth]*key.z + key.x][key.j];
        return particle_data[key.depth][access_data.x_num[key.depth]*key.z + key.x][offset + key.index];
    }
    
    PartCellData<S> access_data;
    PartCellData<T> particle_data;
    
    uint8_t depth_max;
    uint8_t depth_min;
    
    std::vector<unsigned int> z_num;
    std::vector<unsigned int> x_num;
    
    template<typename U>
    void initialize_from_structure(PartCellData<U>& part_cell_data){
        //
        //  Initialize the two data structures
        //
        //
        
        access_data.initialize_from_partcelldata(part_cell_data);
        particle_data.initialize_from_partcelldata(part_cell_data);
        
        //now initialize the entries of the two data sets, access structure
        
        //initialize loop variables
        int x_;
        int z_;
        
        U j_;
        
        //next initialize the entries;
        Part_timer timer;
        timer.verbose_flag = 1;
        
        timer.start_timer("intiialize access data structure");
        
        for(int i = access_data.depth_min;i <= access_data.depth_max;i++){
            
            const unsigned int x_num = access_data.x_num[i];
            const unsigned int z_num = access_data.z_num[i];
            
#pragma omp parallel for default(shared) private(z_,x_) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    const size_t offset_pc_data = x_num*z_ + x_;
                    access_data.data[i][offset_pc_data].resize(part_cell_data.data[i][offset_pc_data].size());
                }

            }
        }
        
        timer.stop_timer();
        
        timer.start_timer("initialize structures");
        
        U status;
        U node_val;
        
        
        for(int i = access_data.depth_min;i <= access_data.depth_max;i++){
            
            const unsigned int x_num = access_data.x_num[i];
            const unsigned int z_num = access_data.z_num[i];
            
#pragma omp parallel for default(shared) private(z_,x_,j_,status,node_val) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    S part_counter = 0;
                    
                    //access variables
                    const size_t offset_pc_data = x_num*z_ + x_;
                    const size_t j_num = access_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff
                        
                        node_val = part_cell_data.data[i][offset_pc_data][j_];
                        
                        if(!(node_val&1)){
                            //normal node
                            
                            //create pindex, and create status (0,1,2,3) and type
                            status = (node_val & STATUS_MASK) >> STATUS_SHIFT;  //need the status masks here, need to move them into the datastructure I think so that they are correctly accessible then to these routines.
                            
                            access_data.data[i][offset_pc_data][j_] = 0; //set normal type
                            
                            access_data.data[i][offset_pc_data][j_] |= (status << STATUS_SHIFT_PARTICLE); //add the particle status
                            
                            access_data.data[i][offset_pc_data][j_] |= (part_counter << Y_PINDEX_SHIFT_PARTICLE); //add the particle starting index for the part cell
                            
                            //update the counter
                            if (status > SEED){
                                // Filler or Boundary 1 particle
                                part_counter += NON_SEED_NUM_PARTICLES;
                                
                            } else {
                                // Seed particle cell 8 particles
                                part_counter +=SEED_NUM_PARTICLES;
                            }
                            
                        } else {
                            
                        
                            //gap node
                            access_data.data[i][offset_pc_data][j_] = 1; //set type to gap
                            access_data.data[i][offset_pc_data][j_] |= (((node_val & YP_DEPTH_MASK) >> YP_DEPTH_SHIFT) << Y_DEPTH_SHIFT_PARTICLE); //set the depth change
                            access_data.data[i][offset_pc_data][j_] |= ((((node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT) - ((node_val & PREV_COORD_MASK) >> PREV_COORD_SHIFT)) << COORD_DIFF_SHIFT_PARTICLE); //set the coordinate difference
                            
                            
                        }
                        
                    }
                    
                    //then resize the particle data structure here..
                    particle_data.data[i][offset_pc_data].resize(part_counter);
                }
                
            }
        }
        
        timer.stop_timer();
        
        //then done for initialization, then need to get the intensities.. 
        
    }
    
    template<uint64_t face,uint64_t index,typename U>
    U get_part_cseed_nboundary(const U& curr_offset,const U& neigh_offset){
        //
        // current is seed and neighbour is boundary
        //
        
        //accessed face and then part
        constexpr uint64_t index_offset[6][8] = {{1,0,3,0,5,0,7,0},{0,0,0,2,0,4,0,6},{2,3,0,0,6,7,0,0},{0,0,0,1,0,0,4,5},{4,5,6,7,0,0,0,0},{0,0,0,0,0,1,2,3}};
        constexpr uint64_t sel_offset[6][8] = {{0,1,0,1,0,1,0,1},{1,0,1,0,1,0,1,0},{0,0,1,1,0,0,1,1},{1,1,0,0,1,1,0,0},{0,0,0,0,1,1,1,1},{1,1,1,1,0,0,0,0}};
        
        return (sel_offset[face][index] == 1)*neigh_offset + (sel_offset[face][index] == 0)*curr_offset + index_offset[face][index];
        
    }
    
    template<uint64_t face,uint64_t index,typename U>
    U get_part_cseed_nseed(const U& curr_offset,const U& neigh_offset){
        //
        // current is seed and neighbour is seed
        //
        
        //accessed face and then part
        constexpr uint64_t index_offset[6][8] = {{1,0,3,2,5,4,7,6},{1,0,3,2,5,4,7,6},{2,3,0,1,6,7,4,5},{2,3,0,1,6,7,4,5},{4,5,6,7,0,1,2,3},{4,5,6,7,0,1,2,3}};
        constexpr uint64_t sel_offset[6][8] = {{0,1,0,1,0,1,0,1},{1,0,1,0,1,0,1,0},{0,0,1,1,0,0,1,1},{1,1,0,0,1,1,0,0},{0,0,0,0,1,1,1,1},{1,1,1,1,0,0,0,0}};
        
        return (sel_offset[face][index] == 1)*neigh_offset + (sel_offset[face][index] == 0)*curr_offset + index_offset[face][index];
        
    }
    
    template<uint64_t face,uint64_t index,typename U>
    U get_part_cboundary_nseed(const U& neigh_offset){
        //
        // current is seed and neighbour is seed
        //
        
        //accessed face and then part
        constexpr uint64_t index_offset[6][4] = {{0,2,4,6},{1,3,5,7},{0,1,4,5},{2,3,6,7},{0,1,2,3},{4,5,6,7}};
        
        return neigh_offset + index_offset[face][index];
        
    }
    
    template<uint64_t face,typename U>
    U get_part_cfiller_nseed(const U& neigh_offset,uint64_t& index){
        //
        // current is seed and neighbour is seed
        //
        
        //accessed face and then part
        constexpr uint64_t index_offset[6][4] = {{0,2,4,6},{1,3,5,7},{0,1,4,5},{2,3,6,7},{0,1,2,3},{4,5,6,7}};
        
        return neigh_offset + index_offset[face][index];
        
    }
    
    
    
    template<typename U>
    U get_index_odd_even(U& coord0,U& coord1){
        //
        //  Calculates the index required for Filler seed pair for (x,z)
        //
        switch (coord0) {
            case 0:{
                //even
                switch ((coord1)) {
                    case 0:{
                        //even
                        return 0;
                    }
                    case 1:{
                        //odd
                        return 2;
                    }
                }
                break;
            }
            case 1:{
                //odd
                switch ((coord1)) {
                    case 0:{
                        //even
                        return 1;
                    }
                    case 1:{
                        //odd
                        return 3;
                    }
                }
                break;
            }
        }
        return 0;
        
    }
    template<typename U>
    U check_previous_val(const U& face,const U& curr_key,PartCellData<U>& pc_data){
        //
        //  Checks the previous node in the structure and determines which cell you are in, by checking if they share the neighbour
        //
        
        constexpr uint64_t index_mask_dir[6] = {YP_INDEX_MASK,YM_INDEX_MASK,XP_INDEX_MASK,XM_INDEX_MASK,ZP_INDEX_MASK,ZM_INDEX_MASK};
        constexpr uint64_t index_shift_dir[6] = {YP_INDEX_SHIFT,YM_INDEX_SHIFT,XP_INDEX_SHIFT,XM_INDEX_SHIFT,ZP_INDEX_SHIFT,ZM_INDEX_SHIFT};
        
        constexpr int8_t von_neumann_y_cells[6] = { 1,-1, 0, 0, 0, 0};
        
        U curr_neigh_index = (curr_key & index_mask_dir[face]) >> index_shift_dir[face];
        
        U prev_neigh_index = curr_key;
        
        prev_neigh_index &= -((PC_KEY_J_MASK) - 1);
        prev_neigh_index|=  (((curr_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT) + von_neumann_y_cells[face]) << PC_KEY_J_SHIFT;
        
        prev_neigh_index = pc_data.get_val(prev_neigh_index);
        
        if (prev_neigh_index&1){
            return 0;
        } else{
            prev_neigh_index = (prev_neigh_index & index_mask_dir[face]) >> index_shift_dir[face];
            
            if(prev_neigh_index == curr_neigh_index){
                return 1;
            } else {
                return 0;
            }
        }
    }
    
    template<uint64_t face,typename U>
    U calc_index_cfiller_nseed(const U& curr_key,PartCellData<U>& pc_data){
        //
        //  Determines the index for filler cells with a seed neighbour at lower resolution
        //
        //
        
        
        
        U coord1;
        U coord0;
        
        switch(face){
            case 0:{
                
                coord0 = (curr_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
                coord1 = (curr_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
                
                coord0 = coord0&1;
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            } case 1:{
                
                coord0 = (curr_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
                coord1 = (curr_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
                
                coord0 = coord0&1; //check if odd
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            } case 2:{
                
                //shift curr_key back
                
                //get the key that is currently pointed to
                
                
                coord0 = check_previous_val<U>(face,curr_key,pc_data);
                coord1 = (curr_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            } case 3:{
                
                coord0 = check_previous_val<U>(face,curr_key,pc_data);
                coord1 = (curr_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            } case 4:{
                
                coord0 = check_previous_val<U>(face,curr_key,pc_data);
                coord1 = (curr_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            } case 5:{
                
                coord0 = check_previous_val<U>(face,curr_key,pc_data);
                coord1 = (curr_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            }
        }
        return 0;
        
    }
    
    template<uint64_t face,uint64_t index,typename U>
    U get_part_cseed_nfiller_p1(const U& curr_offset,const std::vector<U>& neigh_cell_keys){
        //
        // current is seed and neighbour is filler on higher level depth
        //
        
        //accessed face and then part
        constexpr uint64_t index_offset[6][8] = {{1,0,3,1,5,2,7,3},{0,0,1,2,2,4,3,6},{2,3,0,1,6,7,2,3},{0,1,0,1,2,3,4,5},{4,5,6,7,0,1,2,3},{0,1,2,3,0,1,2,3}};
        constexpr uint64_t sel_offset[6][8] = {{0,1,0,1,0,1,0,1},{1,0,1,0,1,0,1,0},{0,0,1,1,0,0,1,1},{1,1,0,0,1,1,0,0},{0,0,0,0,1,1,1,1},{1,1,1,1,0,0,0,0}};
        
        if(sel_offset[face][index] == 1){
            
            return (access_data.get_val(neigh_cell_keys[index_offset[face][index]]) & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
            
        } else {
            
            return curr_offset + neigh_cell_keys[index_offset[face][index]];
            
            
        }
        
    }
    
    
    template<uint64_t face,uint64_t index,typename U>
    void get_part_neighs_face_t(const U& curr_key,const U& curr_status,const U& curr_offset,const std::vector<U>& neigh_cell_keys,std::vector<U>& neigh_part_keys,PartCellData<U>& pc_data){
        //
        //  Bevan Cheeseman (2016)
        //
        //  Get all the particle neighbours nieghbours in direction face
        //
        /** Get neighbours of a cell in one of the direction
         *
         *  @param curr_key    input: current key, output: neighbour key
         *  @param face        direction to follow. Possible values are [0,5]
         *                     They stand for [+y,-y,+x,-x,+z,-z] //change this ordering.. (y+ y-) are different,
         */
        //
        //
        //  Varaibes I will need are current offset, and neighbour offset, these are encoded in the tree?
        //
        //
        
        if(neigh_cell_keys.size() > 0){
            
            neigh_part_keys.resize(neigh_cell_keys.size());
            
            U neigh_status = (access_data.get_val(neigh_cell_keys[0]) & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE;
            U neigh_offset;
            
            switch(curr_status){
                case SEED:
                {
                    switch(neigh_status){
                        case SEED:
                        {
                            
                            neigh_offset = (access_data.get_val(neigh_cell_keys[0]) & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                            neigh_part_keys[0] = get_part_cseed_nseed<face,index,U>(curr_offset,neigh_offset);
                            
                            return;
                        }
                        case BOUNDARY:
                        {
                            
                            neigh_offset = (access_data.get_val(neigh_cell_keys[0]) & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                            // will have one neighbour
                            neigh_part_keys[0] =  get_part_cseed_nboundary<face,index,U>(curr_offset,neigh_offset);
                            
                            
                            return;
                        }
                        case FILLER:
                        {
                            //This is the case where the filler are higher resolution then the seed case,
                            neigh_part_keys[0] = get_part_cseed_nfiller_p1<face,index,U>(curr_offset,neigh_cell_keys);
                            
                            return;
                        }
                    }
                    
                    break;
                }
                case BOUNDARY:
                {
                    switch(neigh_status){
                        case SEED:
                        {
                            neigh_part_keys.resize(4);
                            neigh_offset = (access_data.get_val(neigh_cell_keys[0]) & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                            neigh_part_keys[0] = get_part_cboundary_nseed<face,0,U>(neigh_offset);
                            neigh_part_keys[1] = get_part_cboundary_nseed<face,1,U>(neigh_offset);
                            neigh_part_keys[2] = get_part_cboundary_nseed<face,2,U>(neigh_offset);
                            neigh_part_keys[3] = get_part_cboundary_nseed<face,3,U>(neigh_offset);
                            
                            return;
                        }
                        default:
                        {
                            //will possibly have more then one neighbour
                            for(int i = 0; i < neigh_part_keys.size();i++){
                                neigh_offset = (access_data.get_val(neigh_cell_keys[i]) & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                                neigh_part_keys[i] = neigh_offset;
                            }
                            return;
                        }
                    }
                    
                    break;
                }
                case FILLER:
                {
                    switch(neigh_status){
                        case SEED:
                        {
                            //more complicated case, have to first get the correct index, then the function can be run
                            neigh_offset = (access_data.get_val(neigh_cell_keys[0]) & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                            
                            U neigh_index = calc_index_cfiller_nseed<face,U>(curr_key,pc_data);
                            
                            neigh_part_keys[0] = get_part_cfiller_nseed<face,U>(neigh_offset,neigh_index);
                            
                            return;
                        }
                        default:
                        {
                            //will possibly have more then one neighbour
                            for(int i = 0; i < neigh_part_keys.size();i++){
                                neigh_offset = (access_data.get_val(neigh_cell_keys[i]) & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                                neigh_part_keys[i] = neigh_offset;
                            }
                            
                            return;
                        }
                    }
                    
                    break;
                }
            }
            
        }

        
    }
    
    template<typename U>
    void  get_part_neighs_face(const uint64_t face,const uint64_t index,const U& curr_key,const U& status,const U& part_offset,const std::vector<U>& neigh_cell_keys,std::vector<U>& neigh_part_keys,PartCellData<U>& pc_data){
        //
        //
        //  Forces the explicit compilation of the all the function templates
        //
        //
        
        switch(face){
            case 0:{
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<0,0,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<0,1,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<0,2,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<0,3,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<0,4,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<0,5,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<0,6,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<0,7,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                        
                }
                break;
            } case 1:{
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<1,0,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<1,1,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<1,2,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<1,3,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<1,4,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<1,5,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<1,6,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<1,7,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                        
                }
                break;
            } case 2:{
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<2,0,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<2,1,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<2,2,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<2,3,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<2,4,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<2,5,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<2,6,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<2,7,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                        
                }
                break;
            } case 3:{
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<3,0,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<3,1,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<3,2,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<3,3,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<3,4,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<3,5,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<3,6,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<3,7,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                        
                }
                break;
            } case 4:{
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<4,0,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<4,1,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<4,2,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<4,3,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<4,4,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<4,5,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<4,6,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<4,7,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                        
                }
                break;
            } case 5:{
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<5,0,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<5,1,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<5,2,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<5,3,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<5,4,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<5,5,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<5,6,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<5,7,U>(curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                        break;
                    }
                        
                }
                break;
            }


        }
        
        
    }
    
    template<typename U>
    void test_get_part_neigh_dir(PartCellData<U>& pc_data){
        //
        // Test the get neighbour direction code for speed
        //
        
        U z_;
        U x_;
        U j_;
        U node_val_pc;
        S node_val_part;
        S part_offset;
        
        Part_timer timer;
        
        timer.verbose_flag = 1;
        
        U status;
        U curr_key;
        std::vector<U> neigh_cell_keys;
        std::vector<U> neigh_part_keys;
        
        
        timer.start_timer("get neighbour parts");
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,node_val_part,curr_key,part_offset,status,neigh_cell_keys,neigh_part_keys) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                curr_key |= ((uint64_t)i) << PC_KEY_DEPTH_SHIFT;
                curr_key |= z_ << PC_KEY_Z_SHIFT;
                
                neigh_cell_keys.reserve(24);
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    curr_key &=  -((PC_KEY_X_MASK) + 1);
                    curr_key |= x_ << PC_KEY_X_SHIFT;
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val_pc = pc_data.data[i][offset_pc_data][j_];
                        
                        
                        if (!(node_val_pc&1)){
                            //get the index gap node
                            node_val_part = access_data.data[i][offset_pc_data][j_];
                            
                            curr_key &= -((PC_KEY_J_MASK) + 1);
                            curr_key |= j_ << PC_KEY_J_SHIFT;
                            
                            
                            //neigh_keys.resize(0);
                            status = (node_val_part & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE;
                            
                            part_offset = (node_val_part & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                            
                            neigh_cell_keys.resize(0);
                            pc_data.get_neigh_0(curr_key,node_val_pc,neigh_cell_keys);
                            
                            (void) neigh_cell_keys;
                            
                            for(int m = 0; m < neigh_cell_keys.size();m++){
                                neigh_cell_keys[m] = access_data.get_val(neigh_cell_keys[m]);
                            }
                            
                            uint64_t face = 0;
                            
                            switch(status){
                                case SEED:
                                {
                                    //loop over the 8 particles
                                    for(uint64_t p = 0;p < 8;p++){
                                        //curr_key &= -((PC_KEY_INDEX_MASK) + 1);
                                        //curr_key |= (part_offset+p) << PC_KEY_INDEX_SHIFT;
                                        
                                        get_part_neighs_face(face,p,curr_key,status,part_offset+p,neigh_cell_keys,neigh_part_keys,pc_data);
                                        
                                    }
                                    
                                    (void) neigh_part_keys;
                                    (void) neigh_cell_keys;
                                    
                                    //loop over neighborus and add the different part offsets
                                    
                                    break;
                                }
                                default:
                                {
                                    //one particle
                                    curr_key &= -((PC_KEY_INDEX_MASK) + 1);
                                    curr_key |= part_offset << PC_KEY_INDEX_SHIFT;
                                    
                                    //loop over neighbours, and add in the part offset
                                    //get_part_neighs_face(face,0,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                                    
                                    
                                    (void) neigh_part_keys;
                                    (void) neigh_cell_keys;
                                    
                                    break;
                                }
                                    
                            }
                            
                            
                        } else {
                            
                        }
                        
                    }
                    
                }
                
            }
        }
        
        timer.stop_timer();
        
        
    }

    
private:
    
    uint64_t num_particles;
    
    
    
};

#endif //PARTPLAY_PARTCELLDATA_HPP