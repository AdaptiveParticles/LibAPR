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
    
    PartCellData<S> access_data;
    PartCellData<T> particle_data;
    
    
    ParticleData(){};
    
    S& operator ()(int depth, int x_,int z_,int j_,int index){
        // data access
        return access_data[depth][x_num[depth]*z_ + x_][j_];
    }
    
    inline S get_num_parts(S& status){
        //
        //  Returns the number of particles for a cell;
        //
        return (1 + 7*(status == SEED));
    }
    
    inline S access_node_get_status(const S& node_val){
        return (node_val & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE;
    }
    
    inline S access_node_get_part_offset(const S& node_val){
        return (node_val & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
    }
    
    T&  get_part(const uint64_t part_key){
        // data access
        
        const uint64_t depth = (part_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (part_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (part_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t index = (part_key & PC_KEY_INDEX_MASK) >> PC_KEY_INDEX_SHIFT;
        
        return particle_data.data[depth][particle_data.x_num[depth]*z_ + x_][index];
        
        //return data[(pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT][x_num[(pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT]*((pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT) + ((pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT)][(pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT];
    }
    
    
    
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
    
   
    
    template<typename U>
    void  get_part_neighs_all(const uint64_t index,const U& node_val,U& curr_key,const U& status,const U& part_offset,PartCellNeigh<U>& neigh_cell_keys,PartCellNeigh<U>& neigh_part_keys,PartCellData<U>& pc_data){
        //
        //
        //  Forces the explicit compilation of the all the function templates
        //
        //
        
        //If the previous cell calculated on was not the current one, calculate the particle cell neighbours (calculates all directions, this is to allow for better coping for irregular access, without excessive checking)
        if(!access_data.pc_key_cell_isequal(curr_key,neigh_cell_keys.curr)){
            pc_data.get_neighs_all(curr_key,node_val,neigh_cell_keys);
            
            U temp = 0;
            
            // Get the status and offsets of all the neihbours here, to remove redundancy
            for(int i = 0;i < neigh_cell_keys.neigh_face.size();i++){
                for(int n = 0;n < neigh_cell_keys.neigh_face[i].size();n++){
                    if(neigh_cell_keys.neigh_face[i][n] > 0){
                        
                        temp = access_data.get_val(neigh_cell_keys.neigh_face[i][n]);
                        access_data.pc_key_set_status(neigh_cell_keys.neigh_face[i][n],access_node_get_status(temp));
                        access_data.pc_key_set_index(neigh_cell_keys.neigh_face[i][n],access_node_get_part_offset(temp));
                    }
                }
            }

        }
        
        //set the current key.
        access_data.pc_key_set_status(curr_key,status);
        neigh_part_keys.curr = curr_key;
        
        
        neigh_part_keys.neigh_face[0].resize(0);
        neigh_part_keys.neigh_face[1].resize(0);
        neigh_part_keys.neigh_face[2].resize(0);
        neigh_part_keys.neigh_face[3].resize(0);
        neigh_part_keys.neigh_face[4].resize(0);
        neigh_part_keys.neigh_face[5].resize(0);
        
        switch(index){
            case 0: {
                get_part_neighs_face_t<0,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                get_part_neighs_face_t<1,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                get_part_neighs_face_t<2,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                get_part_neighs_face_t<3,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                get_part_neighs_face_t<4,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                get_part_neighs_face_t<5,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                
                return;
            }
            case 1: {
                get_part_neighs_face_t<0,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                get_part_neighs_face_t<1,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                get_part_neighs_face_t<2,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                get_part_neighs_face_t<3,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                get_part_neighs_face_t<4,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                get_part_neighs_face_t<5,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                return;
            }
            case 2: {
                get_part_neighs_face_t<0,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                get_part_neighs_face_t<1,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                get_part_neighs_face_t<2,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                get_part_neighs_face_t<3,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                get_part_neighs_face_t<4,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                get_part_neighs_face_t<5,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                return;
            }
            case 3: {
                get_part_neighs_face_t<0,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                get_part_neighs_face_t<1,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                get_part_neighs_face_t<2,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                get_part_neighs_face_t<3,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                get_part_neighs_face_t<4,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                get_part_neighs_face_t<5,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                return;
            }
            case 4: {
                get_part_neighs_face_t<0,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                get_part_neighs_face_t<1,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                get_part_neighs_face_t<2,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                get_part_neighs_face_t<3,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                get_part_neighs_face_t<4,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                get_part_neighs_face_t<5,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                return;
            }
            case 5: {
                get_part_neighs_face_t<0,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                get_part_neighs_face_t<1,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                get_part_neighs_face_t<2,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                get_part_neighs_face_t<3,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                get_part_neighs_face_t<4,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                get_part_neighs_face_t<5,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                return;
            }
            case 6: {
                get_part_neighs_face_t<0,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                get_part_neighs_face_t<1,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                get_part_neighs_face_t<2,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                get_part_neighs_face_t<3,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                get_part_neighs_face_t<4,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                get_part_neighs_face_t<5,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                return;
            }
            case 7: {
                get_part_neighs_face_t<0,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                get_part_neighs_face_t<1,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                get_part_neighs_face_t<2,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                get_part_neighs_face_t<3,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                get_part_neighs_face_t<4,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                get_part_neighs_face_t<5,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                return;
            }
                
        }
        
        
        
    }
    
    template<typename U>
    void  get_part_neighs_face(const uint64_t face,const uint64_t index,const U& node_val,U& curr_key,const U& status,const U& part_offset,PartCellNeigh<U>& neigh_cell_keys,PartCellNeigh<U>& neigh_part_keys,PartCellData<U>& pc_data){
        //
        //
        //  Forces the explicit compilation of the all the function templates
        //
        //
        
        //If the previous cell calculated on was not the current one, calculate the particle cell neighbours (calculates all directions, this is to allow for better coping for irregular access, without excessive checking)
        if(!access_data.pc_key_cell_isequal(curr_key,neigh_cell_keys.curr)){
            pc_data.get_neighs_all(curr_key,node_val,neigh_cell_keys);
            //pc_data.get_neighs_face(curr_key,node_val,face,neigh_cell_keys);
            
            U temp = 0;
            
            // Get the status and offsets of all the neihbours here, to remove redundancy
            for(int i = 0;i < neigh_cell_keys.neigh_face.size();i++){
                for(int n = 0;n < neigh_cell_keys.neigh_face[i].size();n++){
                    if(neigh_cell_keys.neigh_face[i][n] > 0){
                    
                        temp = access_data.get_val(neigh_cell_keys.neigh_face[i][n]);
                        access_data.pc_key_set_status(neigh_cell_keys.neigh_face[i][n],access_node_get_status(temp));
                        access_data.pc_key_set_index(neigh_cell_keys.neigh_face[i][n],access_node_get_part_offset(temp));
                    }
                }
            }
            
            //
        }
        
        //set the current key.
        access_data.pc_key_set_status(curr_key,status);
        neigh_part_keys.curr = curr_key;
        
        
        switch(face){
            case 0:{
                neigh_part_keys.neigh_face[0].resize(0);
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<0,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<0,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<0,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<0,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<0,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<0,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<0,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<0,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[0],neigh_part_keys.neigh_face[0],pc_data);
                        break;
                    }
                        
                }
                break;
            } case 1:{
                neigh_part_keys.neigh_face[1].resize(0);
                switch(index){
                        
                    case 0: {
                        get_part_neighs_face_t<1,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<1,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<1,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<1,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<1,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<1,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<1,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<1,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[1],neigh_part_keys.neigh_face[1],pc_data);
                        break;
                    }
                        
                }
                break;
            } case 2:{
                neigh_part_keys.neigh_face[2].resize(0);
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<2,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<2,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<2,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<2,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<2,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<2,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<2,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<2,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[2],neigh_part_keys.neigh_face[2],pc_data);
                        break;
                    }
                        
                }
                break;
            } case 3:{
                neigh_part_keys.neigh_face[3].resize(0);
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<3,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<3,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<3,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<3,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<3,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<3,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<3,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<3,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[3],neigh_part_keys.neigh_face[3],pc_data);
                        break;
                    }
                        
                }
                break;
            } case 4:{
                neigh_part_keys.neigh_face[4].resize(0);
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<4,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<4,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<4,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<4,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<4,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<4,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<4,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<4,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[4],neigh_part_keys.neigh_face[4],pc_data);
                        break;
                    }
                        
                }
                break;
            } case 5:{
                neigh_part_keys.neigh_face[5].resize(0);
                switch(index){
                    case 0: {
                        get_part_neighs_face_t<5,0,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                        break;
                    }
                    case 1: {
                        get_part_neighs_face_t<5,1,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                        break;
                    }
                    case 2: {
                        get_part_neighs_face_t<5,2,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                        break;
                    }
                    case 3: {
                        get_part_neighs_face_t<5,3,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                        break;
                    }
                    case 4: {
                        get_part_neighs_face_t<5,4,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                        break;
                    }
                    case 5: {
                        get_part_neighs_face_t<5,5,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                        break;
                    }
                    case 6: {
                        get_part_neighs_face_t<5,6,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
                        break;
                    }
                    case 7: {
                        get_part_neighs_face_t<5,7,U>(curr_key,status,part_offset,neigh_cell_keys.neigh_face[5],neigh_part_keys.neigh_face[5],pc_data);
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
        PartCellNeigh<U> neigh_cell_keys;
        PartCellNeigh<U> neigh_part_keys;
        
        uint64_t face = 0;
        
        uint64_t p = 0;
        
        timer.start_timer("get neighbour parts dir");
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_pc,node_val_part,curr_key,part_offset,status,neigh_cell_keys,neigh_part_keys) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                access_data.pc_key_set_depth(curr_key,i);
                access_data.pc_key_set_z(curr_key,z_);
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    access_data.pc_key_set_x(curr_key,x_);
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val_pc = pc_data.data[i][offset_pc_data][j_];
                        
                        if (!(node_val_pc&1)){
                            //get the index gap node
                            node_val_part = access_data.data[i][offset_pc_data][j_];
                            
                            access_data.pc_key_set_j(curr_key,j_);
                            
                            status = access_node_get_status(node_val_part);
                            part_offset = access_node_get_part_offset(node_val_part);
                            
                            //loop over the particles
                            for(p = 0;p < get_num_parts(status);p++){
                                access_data.pc_key_set_index(curr_key,part_offset+p);
                                get_part_neighs_face(face,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                                        
                            }
                            
                            
                        } else {
                            
                        }
                        
                    }
                    
                }
                
            }
        }
        
        timer.stop_timer();
        
//        timer.start_timer("get neighbour parts dir + add parts");
//        
//        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
//            
//            const unsigned int x_num_ = pc_data.x_num[i];
//            const unsigned int z_num_ = pc_data.z_num[i];
//            
//            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
//            
//            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,node_val_part,curr_key,part_offset,status,neigh_cell_keys,neigh_part_keys) if(z_num_*x_num_ > 100)
//            for(z_ = 0;z_ < z_num_;z_++){
//                
//                curr_key = 0;
//                
//                access_data.pc_key_set_depth(curr_key,i);
//                access_data.pc_key_set_z(curr_key,z_);
//                
//                neigh_cell_keys.reserve(6);
//                neigh_part_keys.reserve(6);
//                
//                for(x_ = 0;x_ < x_num_;x_++){
//                    
//                    access_data.pc_key_set_x(curr_key,x_);
//                    
//                    const size_t offset_pc_data = x_num_*z_ + x_;
//                    
//                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
//                    
//                    for(j_ = 0;j_ < j_num;j_++){
//                        
//                        node_val_pc = pc_data.data[i][offset_pc_data][j_];
//                        
//                        if (!(node_val_pc&1)){
//                            //get the index gap node
//                            node_val_part = access_data.data[i][offset_pc_data][j_];
//                            
//                            
//                            access_data.pc_key_set_j(curr_key,j_);
//                            
//                            
//                            //neigh_keys.resize(0);
//                            status = access_node_get_status(node_val_part);
//                            part_offset = access_node_get_part_offset(node_val_part);
//
//                            
//                            neigh_cell_keys.resize(0);
//                            neigh_part_keys.resize(0);
//                            pc_data.get_neigh_0(curr_key,node_val_pc,neigh_cell_keys);
//                            
//                            (void) neigh_cell_keys;
//                            
//                            uint64_t face = 3;
//                            uint64_t val = 0;
//                            
//                            switch(status){
//                                case SEED:
//                                {
//                                    //loop over the 8 particles
//                                    for(uint64_t p = 0;p < 8;p++){
//                                        access_data.pc_key_set_index(curr_key,part_offset+p);
//                                        neigh_part_keys.resize(0);
//                                        get_part_neighs_face(face,p,curr_key,status,part_offset+p,neigh_cell_keys,neigh_part_keys,pc_data);
//                                        
//                                        for(uint64_t n = 0; n < neigh_part_keys.size();n++){
//                                            val = neigh_part_keys[n];
//                                            if (val > 0){
//                                                
//                                                get_part(curr_key) = get_part(val);
//                                            }
//                                        }
//                                        
//                                    }
//                                    
//                                    (void) neigh_part_keys;
//                                    (void) neigh_cell_keys;
//                                    
//                                    //loop over neighborus and add the different part offsets
//                                    
//                                    break;
//                                }
//                                default:
//                                {
//                                    //one particle
//                                    access_data.pc_key_set_index(curr_key,part_offset);
//                                    
//                                    //loop over neighbours, and add in the part offset
//                                    get_part_neighs_face(face,0,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
//                                    
//                                    for(uint64_t n = 0; n < neigh_part_keys.size();n++){
//                                        val = neigh_part_keys[n];
//                                        if (val > 0){
//                                            
//                                            get_part(curr_key) = get_part(val);
//                                        }
//                                    }
//                                    
//                                    
//                                    (void) neigh_part_keys;
//                                    (void) neigh_cell_keys;
//                                    
//                                    break;
//                                }
//                                    
//                                    
//
//                                    
//                            }
//                            
//                            
//                        } else {
//                            
//                        }
//                        
//                    }
//                    
//                }
//                
//            }
//        }
//        
//        timer.stop_timer();
        
        
    }
    
    template<typename U>
    void test_get_part_neigh_all(PartCellData<U>& pc_data){
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
        PartCellNeigh<U> neigh_cell_keys;
        PartCellNeigh<U> neigh_part_keys;
        
        
        uint64_t p;
        
        timer.start_timer("get neighbour parts all");
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_pc,node_val_part,curr_key,part_offset,status,neigh_cell_keys,neigh_part_keys) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                access_data.pc_key_set_depth(curr_key,i);
                access_data.pc_key_set_z(curr_key,z_);

                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    access_data.pc_key_set_x(curr_key,x_);
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val_pc = pc_data.data[i][offset_pc_data][j_];
                        
                        
                        if (!(node_val_pc&1)){
                            //get the index gap node
                            node_val_part = access_data.data[i][offset_pc_data][j_];
                            
                            access_data.pc_key_set_j(curr_key,j_);
                            
                            
                            //neigh_keys.resize(0);
                            status = access_node_get_status(node_val_part);
                            part_offset = access_node_get_part_offset(node_val_part);
                            
                            //loop over the particles
                            for(p = 0;p < get_num_parts(status);p++){
                                access_data.pc_key_set_index(curr_key,part_offset+p);
                                get_part_neighs_all(p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_data);
                                
                            }
                            
                        } else {
                            
                        }
                        
                    }
                    
                }
                
            }
        }
        
        timer.stop_timer();
        
        
    }

    
    
    template<typename U>
    void test_get_part_neigh_all_memory(PartCellData<U>& pc_data){
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
        
        PartCellNeigh<U> neigh_cell_keys;
        PartCellNeigh<U> neigh_part_keys;
        
        
        uint64_t p;

        
        
        PartCellData<PartCellNeigh<U>> neigh_vec_all;
        neigh_vec_all.initialize_from_partcelldata(pc_data);
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            
            for(z_ = 0;z_ < z_num_;z_++){
                
                
                for(x_ = 0;x_ < x_num_;x_++){
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = particle_data.data[i][offset_pc_data].size();
                    neigh_vec_all.data[i][offset_pc_data].resize(j_num);
                    
                }
            }
            
        }
        
        timer.start_timer("get neighbour parts all memory");
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_pc,node_val_part,curr_key,part_offset,status,neigh_cell_keys) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                access_data.pc_key_set_depth(curr_key,i);
                access_data.pc_key_set_z(curr_key,z_);
                
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    access_data.pc_key_set_x(curr_key,x_);
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val_pc = pc_data.data[i][offset_pc_data][j_];
                        
                        
                        if (!(node_val_pc&1)){
                            //get the index gap node
                            node_val_part = access_data.data[i][offset_pc_data][j_];
                            
                            access_data.pc_key_set_j(curr_key,j_);
                            
                            
                            //neigh_keys.resize(0);
                            status = access_node_get_status(node_val_part);
                            part_offset = access_node_get_part_offset(node_val_part);
                            
                            //loop over the particles
                            for(p = 0;p < get_num_parts(status);p++){
                                access_data.pc_key_set_index(curr_key,part_offset+p);
                                
                                get_part_neighs_all(p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_vec_all.data[i][offset_pc_data][part_offset+p],pc_data);
                                
                            }
                            
                            
                        } else {
                            
                        }
                        
                    }
                    
                }
                
            }
        }
        
        timer.stop_timer();
        
        
        timer.start_timer("loop neighbour parts all memory");
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_pc,node_val_part,curr_key,part_offset,status,neigh_cell_keys,neigh_part_keys) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                access_data.pc_key_set_depth(curr_key,i);
                access_data.pc_key_set_z(curr_key,z_);
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    access_data.pc_key_set_x(curr_key,x_);
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val_part = access_data.data[i][offset_pc_data][j_];
                        
                        
                        if (!(node_val_part&1)){
                            //get the index gap node
                            
                            access_data.pc_key_set_j(curr_key,j_);
                            
                            
                            //neigh_keys.resize(0);
                            status = access_node_get_status(node_val_part);
                            part_offset = access_node_get_part_offset(node_val_part);
                            
                            U val = 0;
                            
                            //loop over the particles
                            for(p = 0;p < get_num_parts(status);p++){
                                access_data.pc_key_set_index(curr_key,part_offset+p);
                                
                                for(uint64_t n = 0; n < neigh_vec_all.data[i][offset_pc_data][part_offset+p].neigh_face[0].size();n++){
                                    val = neigh_vec_all.data[i][offset_pc_data][part_offset+p].neigh_face[0][n];
                                    if (val > 0){
                                        
                                        get_part(curr_key) = get_part(val);
                                    }
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
    
    template<uint64_t face,uint64_t index,typename U>
    inline U get_part_cseed_nboundary(const U& curr_offset,const U& neigh_offset,const U& curr_key,U& neigh_key){
        //
        // current is seed and neighbour is boundary
        //
        
        //accessed face and then part
        constexpr uint64_t index_offset[6][8] = {{1,0,3,0,5,0,7,0},{0,0,0,2,0,4,0,6},{2,3,0,0,6,7,0,0},{0,0,0,1,0,0,4,5},{4,5,6,7,0,0,0,0},{0,0,0,0,0,1,2,3}};
        constexpr uint64_t sel_offset[6][8] = {{0,1,0,1,0,1,0,1},{1,0,1,0,1,0,1,0},{0,0,1,1,0,0,1,1},{1,1,0,0,1,1,0,0},{0,0,0,0,1,1,1,1},{1,1,1,1,0,0,0,0}};
        
        U temp;
        
        if(sel_offset[face][index]==1){
            temp = neigh_key;
            
            temp &= -((PC_KEY_PARTNUM_MASK) + 1); // clear the current value
            temp|= index_offset[face][index]  << PC_KEY_PARTNUM_SHIFT; //set  value
            
            temp &= -((PC_KEY_INDEX_MASK) + 1); // clear the current value
            temp|= (neigh_offset + index_offset[face][index])  << PC_KEY_INDEX_SHIFT; //set  value

        } else {
            temp = curr_key;
            temp &= -((PC_KEY_INDEX_MASK) + 1); // clear the current value
            temp|= (curr_offset + index_offset[face][index])  << PC_KEY_INDEX_SHIFT; //set  value
            
            temp &= -((PC_KEY_PARTNUM_MASK) + 1); // clear the current value
            temp|= index_offset[face][index]  << PC_KEY_PARTNUM_SHIFT; //set  value
            
        }
        
        return temp;
        
    }
    
    template<uint64_t face,uint64_t index,typename U>
    inline U get_part_cseed_nseed(const U& curr_offset,const U& neigh_offset,const U& curr_key,U& neigh_key){
        //
        // current is seed and neighbour is seed
        //
        
        //accessed face and then part
        constexpr uint64_t index_offset[6][8] = {{1,0,3,2,5,4,7,6},{1,0,3,2,5,4,7,6},{2,3,0,1,6,7,4,5},{2,3,0,1,6,7,4,5},{4,5,6,7,0,1,2,3},{4,5,6,7,0,1,2,3}};
        constexpr uint64_t sel_offset[6][8] = {{0,1,0,1,0,1,0,1},{1,0,1,0,1,0,1,0},{0,0,1,1,0,0,1,1},{1,1,0,0,1,1,0,0},{0,0,0,0,1,1,1,1},{1,1,1,1,0,0,0,0}};
        
        U temp;
        
        if(sel_offset[face][index]==1){
            temp = neigh_key;
            
            temp &= -((PC_KEY_PARTNUM_MASK) + 1); // clear the current value
            temp|= index_offset[face][index]  << PC_KEY_PARTNUM_SHIFT; //set  value
            
            temp &= -((PC_KEY_INDEX_MASK) + 1); // clear the current value
            temp|= (neigh_offset + index_offset[face][index])  << PC_KEY_INDEX_SHIFT; //set  value
            
        } else {
            temp = curr_key;
            
            temp &= -((PC_KEY_PARTNUM_MASK) + 1); // clear the current value
            temp|= index_offset[face][index]  << PC_KEY_PARTNUM_SHIFT; //set  value
            
            temp &= -((PC_KEY_INDEX_MASK) + 1); // clear the current value
            temp|= (curr_offset + index_offset[face][index])  << PC_KEY_INDEX_SHIFT; //set  value
        }
        
        return temp;
        
    }
    
    template<uint64_t face,uint64_t index,typename U>
    U get_part_cboundary_nseed(const U& neigh_offset,U& neigh_key){
        //
        // current is seed and neighbour is seed
        //
        
        //accessed face and then part
        constexpr uint64_t index_offset[6][4] = {{0,2,4,6},{1,3,5,7},{0,1,4,5},{2,3,6,7},{0,1,2,3},{4,5,6,7}};
        
        U temp;
        
        
        temp = neigh_key;
            
        temp &= -((PC_KEY_PARTNUM_MASK) + 1); // clear the current value
        temp|= index_offset[face][index]  << PC_KEY_PARTNUM_SHIFT; //set  value
            
        temp &= -((PC_KEY_INDEX_MASK) + 1); // clear the current value
        temp|= (neigh_offset + index_offset[face][index])  << PC_KEY_INDEX_SHIFT; //set  value
        
        return temp;
        
    }
    
    template<uint64_t face,typename U>
    U get_part_cfiller_nseed(const U& neigh_offset,uint64_t& index,U& neigh_key){
        //
        // current is seed and neighbour is seed
        //
        
        //accessed face and then part
        constexpr uint64_t index_offset[6][4] = {{0,2,4,6},{1,3,5,7},{0,1,4,5},{2,3,6,7},{0,1,2,3},{4,5,6,7}};
        
        U temp;
        
        
        temp = neigh_key;
        
        temp &= -((PC_KEY_PARTNUM_MASK) + 1); // clear the current value
        temp|= index_offset[face][index]  << PC_KEY_PARTNUM_SHIFT; //set  value
        
        temp &= -((PC_KEY_INDEX_MASK) + 1); // clear the current value
        temp|= (neigh_offset + index_offset[face][index])  << PC_KEY_INDEX_SHIFT; //set  value
        
        return temp;
        
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
        
        
        U curr_node = pc_data.get_val(curr_key);
        
        U curr_neigh_index = (curr_node & index_mask_dir[face]) >> index_shift_dir[face];
        
        U neigh_key = 0;
        
        access_data.pc_key_set_j(neigh_key,access_data.pc_key_get_j(curr_key)-1);
        
        access_data.pc_key_set_x(neigh_key,access_data.pc_key_get_x(curr_key));
        
        access_data.pc_key_set_z(neigh_key,access_data.pc_key_get_z(curr_key));
        
        access_data.pc_key_set_depth(neigh_key,access_data.pc_key_get_depth(curr_key));
        
        
        U prev_neigh_index = 0;
        
        
        prev_neigh_index = pc_data.get_val(neigh_key);
        
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
                
                coord0 = access_data.pc_key_get_x(curr_key);
                coord1 = access_data.pc_key_get_z(curr_key);
                
                coord0 = coord0&1;
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            } case 1:{
                
                coord0 = access_data.pc_key_get_x(curr_key);
                coord1 = access_data.pc_key_get_z(curr_key);
                coord0 = coord0&1; //check if odd
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            } case 2:{
                
                //shift curr_key back
                
                //get the key that is currently pointed to
                
                
                coord0 = check_previous_val<U>(face,curr_key,pc_data);
                coord1 = access_data.pc_key_get_z(curr_key);
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            } case 3:{
                
                coord0 = check_previous_val<U>(face,curr_key,pc_data);
                coord1 = access_data.pc_key_get_z(curr_key);
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            } case 4:{
                
                coord0 = check_previous_val<U>(face,curr_key,pc_data);
                coord1 = access_data.pc_key_get_z(curr_key);
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            } case 5:{
                
                coord0 = check_previous_val<U>(face,curr_key,pc_data);
                coord1 = access_data.pc_key_get_z(curr_key);
                coord1 = coord1&1;
                
                return get_index_odd_even<U>(coord0,coord1);
            }
        }
        return 0;
        
    }
    
    template<uint64_t face,uint64_t index,typename U>
    U get_part_cseed_nfiller_p1(const U& curr_offset,std::vector<U>& neigh_cell_keys,const U& curr_key){
        //
        // current is seed and neighbour is filler on higher level depth
        //
        
        //accessed face and then part
        constexpr uint64_t index_offset[6][8] = {{1,0,3,1,5,2,7,3},{0,0,1,2,2,4,3,6},{2,3,0,1,6,7,2,3},{0,1,0,1,2,3,4,5},{4,5,6,7,0,1,2,3},{0,1,2,3,0,1,2,3}};
        constexpr uint64_t sel_offset[6][8] = {{0,1,0,1,0,1,0,1},{1,0,1,0,1,0,1,0},{0,0,1,1,0,0,1,1},{1,1,0,0,1,1,0,0},{0,0,0,0,1,1,1,1},{1,1,1,1,0,0,0,0}};
        
        U temp;
        
        if(sel_offset[face][index] == 1){
            if (neigh_cell_keys[index_offset[face][index]] > 0){
                
                U temp2;
                
                //neighbour cell
                temp2 = (access_data.get_val(neigh_cell_keys[index_offset[face][index]]) & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                temp = neigh_cell_keys[index_offset[face][index]];
                                           
                temp &= -((PC_KEY_INDEX_MASK) + 1); // clear the current value
                temp|= temp2  << PC_KEY_INDEX_SHIFT; //set  value
                                       
                return temp;
            } else {
                return 0;
            }
        } else {
            
            temp = curr_key;
                                       
            temp &= -((PC_KEY_PARTNUM_MASK) + 1); // clear the current value
            temp|= index_offset[face][index]  << PC_KEY_PARTNUM_SHIFT; //set  value
                                       
            temp &= -((PC_KEY_INDEX_MASK) + 1); // clear the current value
            temp|= (curr_offset + index_offset[face][index])  << PC_KEY_INDEX_SHIFT; //set  value
            
        }
        
        return temp;
        
    }
    
    
    template<uint64_t face,uint64_t index,typename U>
    void get_part_neighs_face_t(const U& curr_key,const U& curr_status,const U& curr_offset,std::vector<U>& neigh_cell_keys,std::vector<U>& neigh_part_keys,PartCellData<U>& pc_data){
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
            
            size_t curr_size = neigh_part_keys.size();
        
            
           // U neigh_node = access_data.get_val(neigh_cell_keys[0]);
            U neigh_status = access_data.pc_key_get_status(neigh_cell_keys[0]);
            U neigh_offset;
            
            switch(curr_status){
                case SEED:
                {
                    switch(neigh_status){
                        case SEED:
                        {
                            neigh_part_keys.push_back(neigh_cell_keys[0]);
                            
                            neigh_offset = access_data.pc_key_get_index(neigh_part_keys[curr_size]);
                            neigh_part_keys[curr_size] = get_part_cseed_nseed<face,index,U>(curr_offset,neigh_offset,curr_key,neigh_part_keys[curr_size]);
                            
                            
                            return;
                        }
                        case BOUNDARY:
                        {
                            neigh_part_keys.push_back(neigh_cell_keys[0]);
                            neigh_offset = access_data.pc_key_get_index(neigh_part_keys[curr_size]);
                            // will have one neighbour
                            neigh_part_keys[curr_size] =  get_part_cseed_nboundary<face,index,U>(curr_offset,neigh_offset,curr_key,neigh_part_keys[curr_size]);
                            
                            return;
                        }
                        case FILLER:
                        {
                            //This is the case where the filler are higher resolution then the seed case,
                            neigh_part_keys.push_back(neigh_cell_keys[0]);
                            neigh_part_keys[curr_size] = get_part_cseed_nfiller_p1<face,index,U>(curr_offset,neigh_cell_keys,curr_key);
                            
                            
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
                            
                            //this is the resize case..
                            //all others can just use the previous, and this then just needs to duplicate
                            neigh_part_keys.push_back(neigh_cell_keys[0]);
                            neigh_offset = access_data.pc_key_get_index(neigh_part_keys[curr_size]);
                            neigh_part_keys.resize(curr_size + 4);
                            
                            
                            U temp = neigh_cell_keys[0];
                                       
                            neigh_part_keys[curr_size] = get_part_cboundary_nseed<face,0,U>(neigh_offset,temp);
                            
                            neigh_part_keys[curr_size+1] = get_part_cboundary_nseed<face,1,U>(neigh_offset,temp);
                            
                            neigh_part_keys[curr_size+2] = get_part_cboundary_nseed<face,2,U>(neigh_offset,temp);
                           
                            neigh_part_keys[curr_size+3] = get_part_cboundary_nseed<face,3,U>(neigh_offset,temp);
                            
                            
                            return;
                        }
                        default:
                        {
                            neigh_part_keys.resize(curr_size + neigh_cell_keys.size());
                            std::copy(neigh_cell_keys.begin(),neigh_cell_keys.end(),neigh_part_keys.begin() + curr_size);
                            //will possibly have more then one neighbour
                            for(int i = 0; i < neigh_cell_keys.size();i++){
                                if(neigh_cell_keys[i] > 0){
                                    neigh_offset = access_data.pc_key_get_index(neigh_cell_keys[i]);
                                    
                                    access_data.pc_key_set_index(neigh_part_keys[curr_size+i],neigh_offset);
                                   
                                }
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
                            neigh_part_keys.push_back(neigh_cell_keys[0]);
                            //more complicated case, have to first get the correct index, then the function can be run
                            neigh_offset = access_data.pc_key_get_index(neigh_part_keys[curr_size]);
                            
                            U neigh_index = calc_index_cfiller_nseed<face,U>(curr_key,pc_data);
                            
                            neigh_part_keys[curr_size] = get_part_cfiller_nseed<face,U>(neigh_offset,neigh_index,neigh_part_keys[curr_size]);
                           
                            
                            return;
                        }
                        default:
                        {
                            neigh_part_keys.resize(curr_size + neigh_cell_keys.size());
                            std::copy(neigh_cell_keys.begin(),neigh_cell_keys.end(),neigh_part_keys.begin() + curr_size);

                            
                            //will possibly have more then one neighbour
                            for(int i = 0; i < neigh_cell_keys.size();i++){
                                if(neigh_cell_keys[i] > 0){
                                    neigh_offset = access_data.pc_key_get_index(neigh_cell_keys[i]);
                                    
                                    access_data.pc_key_set_index(neigh_part_keys[curr_size+i],neigh_offset);
                                    
                                }
                            }
                            return;
                        }
                    }
                    
                    break;
                }
            }
            
        }
        
        
    }
    
    
};

#endif //PARTPLAY_PARTCELLDATA_HPP