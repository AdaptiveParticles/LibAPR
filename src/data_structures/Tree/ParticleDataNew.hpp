///////////////////
//
//  Bevan Cheeseman 2016
//
//  PartCellData class, the data container for CRS sparse format for APR
//
///////////////

#ifndef PARTPLAY_PARTICLEDATA_NEW_HPP
#define PARTPLAY_PARTICLEDATA_NEW_HPP

#include <stdint.h>

#include "PartCellData.hpp"
#include "ExtraPartCellData.hpp"
#include "PartCellStructure.hpp"


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
class ParticleDataNew {
    
public:
    
    /*
     * Number of layers without the root and the contents.
     */
    
    PartCellData<S> access_data;
    ExtraPartCellData<T> particle_data;
    std::vector<std::vector<uint64_t>> global_index_offset;
    
    ParticleDataNew(){};
    
    S& operator ()(int depth, int x_,int z_,int j_,int index){
        // data access
        return access_data[depth][x_num[depth]*z_ + x_][j_];
    }
    
    inline S get_num_parts(S& status){
        //
        //  Returns the number of particles for a cell;
        //
        
        //        if(status == SEED){
        //            return 8;
        //        } else {
        //            return 1;
        //        }
        
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
    
    uint64_t get_global_index(const uint64_t part_key){
        
        const uint64_t depth = (part_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (part_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (part_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t index = (part_key & PC_KEY_INDEX_MASK) >> PC_KEY_INDEX_SHIFT;
        
        return global_index_offset[depth][particle_data.x_num[depth]*z_ + x_] + index;
        
    }
    
    uint64_t depth_max;
    uint64_t depth_min;
    
    std::vector<unsigned int> z_num;
    std::vector<unsigned int> x_num;
    std::vector<unsigned int> y_num;
    
    template<typename U>
    void initialize_from_structure(PartCellStructure<U,uint64_t>& pc_struct){
        //
        //  Initialize the two data structures
        //
        //
        
        x_num = pc_struct.x_num;
        y_num = pc_struct.y_num;
        z_num = pc_struct.z_num;
        

        access_data.org_dims = pc_struct.org_dims;
        particle_data.org_dims = pc_struct.org_dims;

        //first add the layers
        particle_data.depth_max = pc_struct.depth_max + 1;
        particle_data.depth_min = pc_struct.depth_min;
        
        depth_max = particle_data.depth_max;
        depth_min = particle_data.depth_min;
        
        particle_data.z_num.resize(particle_data.depth_max+1);
        particle_data.x_num.resize(particle_data.depth_max+1);
        
        particle_data.data.resize(particle_data.depth_max+1);
        
        for(uint64_t i = particle_data.depth_min;i < particle_data.depth_max;i++){
            particle_data.z_num[i] = pc_struct.z_num[i];
            particle_data.x_num[i] = pc_struct.x_num[i];
            particle_data.data[i].resize(particle_data.z_num[i]*particle_data.x_num[i]);
        }
        
        particle_data.z_num[particle_data.depth_max] = pc_struct.org_dims[2];
        particle_data.x_num[particle_data.depth_max] = pc_struct.org_dims[1];
        particle_data.data[particle_data.depth_max].resize(particle_data.z_num[particle_data.depth_max]*particle_data.x_num[particle_data.depth_max]);
        
        //first add the layers
        access_data.depth_max = pc_struct.depth_max + 1;
        access_data.depth_min = pc_struct.depth_min;
        
        access_data.z_num.resize(access_data.depth_max+1);
        access_data.x_num.resize(access_data.depth_max+1);
        access_data.y_num.resize(access_data.depth_max+1);
        
        access_data.data.resize(access_data.depth_max + 1);
        
        for(uint64_t i = access_data.depth_min;i < access_data.depth_max;i++){
            access_data.z_num[i] = pc_struct.z_num[i];
            access_data.x_num[i] = pc_struct.x_num[i];
            access_data.y_num[i] = pc_struct.y_num[i];
            access_data.data[i].resize(access_data.z_num[i]*access_data.x_num[i]);
        }
        
        access_data.z_num[access_data.depth_max] = pc_struct.org_dims[2];
        access_data.x_num[access_data.depth_max] = pc_struct.org_dims[1];
        access_data.y_num[access_data.depth_max] = pc_struct.org_dims[0];
        access_data.data[access_data.depth_max].resize(access_data.z_num[access_data.depth_max]*access_data.x_num[access_data.depth_max]);
        
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
        
        timer.start_timer("intiialize access data structure");
        
        for(uint64_t i = access_data.depth_max;i >= access_data.depth_min;i--){
            
            const unsigned int x_num = access_data.x_num[i];
            const unsigned int z_num = access_data.z_num[i];
            
            
            const unsigned int x_num_seed = access_data.x_num[i-1];
            const unsigned int z_num_seed = access_data.z_num[i-1];
            
            temp_exist.resize(access_data.y_num[i]);
            temp_location.resize(access_data.y_num[i]);
            
#pragma omp parallel for default(shared) private(j_,z_,x_,y_,node_val,status,z_seed,x_seed,node_val_part) firstprivate(temp_exist,temp_location) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    
                    std::fill(temp_exist.begin(), temp_exist.end(), 0);
                    std::fill(temp_location.begin(), temp_location.end(), 0);
                    
                    if( i < access_data.depth_max){
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
                    
                    if( i > access_data.depth_min){
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
                        access_data.data[i][offset_pc_data].resize(1); //always first adds an extra entry for intialization and extra info
                    } else {
                        access_data.data[i][offset_pc_data].resize(curr_index + 2 - first_empty,0); //gap node to begin, already finishes with a gap node
                        
                    }
                    
                    //initialize particles
                    particle_data.data[i][offset_pc_data].resize(part_total);
                    
                    curr_index = 0;
                    prev_ind = 1;
                    size_t prev_coord = 0;
                    
                    size_t part_counter=0;
                    
                    access_data.data[i][offset_pc_data][0] = 1;
                    access_data.data[i][offset_pc_data].back() = 1;
                    
                    for(y_ = 0;y_ < temp_exist.size();y_++){
                        
                        status = temp_exist[y_];
                        
                        if((status> 0)){
                            
                            curr_index++;
                            
                            //set starting type
                            if(prev_ind == 1){
                                //gap node
                                //set type
                                
                                //gap node
                                access_data.data[i][offset_pc_data][curr_index-1] = 1; //set type to gap
                                access_data.data[i][offset_pc_data][curr_index-1] |= ((y_ - prev_coord) << COORD_DIFF_SHIFT_PARTICLE); //set the coordinate difference
                                
                                curr_index++;
                            }
                            prev_coord = y_;
                            //set type
                            
                            
                            access_data.data[i][offset_pc_data][curr_index-1] = 0; //set normal type
                            
                            access_data.data[i][offset_pc_data][curr_index-1] |= (status << STATUS_SHIFT_PARTICLE); //add the particle status
                            
                            access_data.data[i][offset_pc_data][curr_index-1] |= (part_counter << Y_PINDEX_SHIFT_PARTICLE); //add the particle starting index for the part cell
                            
                            //lastly retrieve the intensities
                            if(status == SEED){
                                //seed from up one level
                                particle_data.data[i][offset_pc_data][part_counter] = pc_struct.part_data.particle_data.data[i-1][offset_pc_data_seed][temp_location[y_]];
                            }
                            else {
                                //non seed same level
                                particle_data.data[i][offset_pc_data][part_counter] = pc_struct.part_data.particle_data.data[i][offset_pc_data][temp_location[y_]];
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
        
        
        
        
    }
    template<typename U>
    void create_pc_data_new(PartCellData<U>& pc_data_new){
        

        pc_data_new.org_dims = access_data.org_dims;

        pc_data_new.y_num = y_num;
        
        //first add the layers
        pc_data_new.depth_max = access_data.depth_max;
        pc_data_new.depth_min = access_data.depth_min;
        
        pc_data_new.z_num.resize(access_data.depth_max+1);
        pc_data_new.x_num.resize(access_data.depth_max+1);
        
        pc_data_new.data.resize(access_data.depth_max+1);
        
        for(uint64_t i = access_data.depth_min;i <= access_data.depth_max;i++){
            pc_data_new.z_num[i] = access_data.z_num[i];
            pc_data_new.x_num[i] = access_data.x_num[i];
            pc_data_new.data[i].resize(access_data.z_num[i]*access_data.x_num[i]);
        }
        
        
        //initialize all the structure arrays
        int j_,x_,z_;
        
        for(uint64_t i = access_data.depth_min;i <= access_data.depth_max;i++){
            
            const unsigned int x_num = access_data.x_num[i];
            const unsigned int z_num = access_data.z_num[i];
            
#pragma omp parallel for default(shared) private(z_,x_) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    const size_t offset_pc_data = x_num*z_ + x_;
                    pc_data_new.data[i][offset_pc_data].resize(access_data.data[i][offset_pc_data].size());
                }
                
            }
        }
        
        
        //then initialize the values;
        
        for(uint64_t depth = (access_data.depth_min);depth <= access_data.depth_max;depth++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = access_data.x_num[depth];
            const unsigned int z_num_ = access_data.z_num[depth];
            
            
#pragma omp parallel for default(shared) private(z_,x_,j_)  if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    int y_=0;
                    uint64_t prev_coord=0;
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    uint64_t node_val_pc;
                    
                    const size_t j_num = pc_data_new.data[depth][offset_pc_data].size();
                    
                    //the y direction loop however is sparse, and must be accessed accordinagly
                    for(j_ = 0;j_ < j_num-1;j_++){
                        
                        //particle cell node value, used here as it is requried for getting the particle neighbours
                        node_val_pc = access_data.data[depth][offset_pc_data][j_];
                        
                        if (!(node_val_pc&1)){
                            
                            
                            //Indicates this is a particle cell node
                            y_++;
                            
                            pc_data_new.data[depth][offset_pc_data][j_] = TYPE_PC;
                            
                            //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                            pc_data_new.data[depth][offset_pc_data][j_] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                            pc_data_new.data[depth][offset_pc_data][j_] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                            pc_data_new.data[depth][offset_pc_data][j_] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                            pc_data_new.data[depth][offset_pc_data][j_] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);
                            
                            //add status here
                            uint64_t status = access_node_get_status(node_val_pc);
                            pc_data_new.data[depth][offset_pc_data][j_] |= (status << STATUS_SHIFT);
                            
                        } else {
                            
                            prev_coord = y_;
                            
                            y_ += ((node_val_pc & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                            
                            pc_data_new.data[depth][offset_pc_data][j_] = TYPE_GAP;
                            pc_data_new.data[depth][offset_pc_data][j_] |= (((uint64_t)y_) << NEXT_COORD_SHIFT);
                            pc_data_new.data[depth][offset_pc_data][j_] |= ( prev_coord << PREV_COORD_SHIFT);
                            pc_data_new.data[depth][offset_pc_data][j_] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                            pc_data_new.data[depth][offset_pc_data][j_] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);
                            
                            y_--;
                        }
                        
                        
                    }
                    
                    //Initialize the last value GAP END indicators to no neighbour
                    pc_data_new.data[depth][offset_pc_data][j_num-1] = TYPE_GAP_END;
                    pc_data_new.data[depth][offset_pc_data][j_num-1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                    pc_data_new.data[depth][offset_pc_data][j_num-1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);
                    
                    
                }
            }
        }
    
    
        ///////////////////////////////////
        //
        //  Calculate neighbours
        //
        /////////////////////////////////
        
        //(+y,-y,+x,-x,+z,-z)
        pc_data_new.set_neighbor_relationships();
        
    
        
    }
    
    template<typename U>
    void create_particles_at_cell_structure(ExtraPartCellData<U>& pdata_new){
        //
        //  Bevan Cheesean 2017
        //
        //  This places particles so they are no longer contiguous, with gaps mirroring those in the cell data structure, such that no additional key is needed
        //
        //  Uses more memory, but has quicker access
        //
        
        pdata_new.initialize_structure_cells(access_data);
        
        uint64_t z_,x_,j_,node_val;
        uint64_t part_offset;
        
        for(uint64_t i = access_data.depth_min;i <= access_data.depth_max;i++){
            
            const unsigned int x_num_ = access_data.x_num[i];
            const unsigned int z_num_ = access_data.z_num[i];
            
#pragma omp parallel for default(shared) private(z_,x_,j_,part_offset,node_val)  if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t j_num = access_data.data[i][offset_pc_data].size();
                    
                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff
                        
                        node_val = access_data.data[i][offset_pc_data][j_];
                        
                        if(!(node_val&1)){
                            
                            part_offset = access_node_get_part_offset(node_val);
                            
                            pdata_new.data[i][offset_pc_data][j_] = particle_data.data[i][offset_pc_data][part_offset];
                            
                        } else {
                            
                        }
                        
                    }
                }
            }
        }
        
        
        
    }
    
    
    
    template<typename U>
    void utest_structure(PartCellStructure<U,uint64_t>& pc_struct,std::vector<Mesh_data<uint64_t>> link_array){
        //
        //  Bevan Cheeseman 2017
        //
        //  Compare the new particle structure, of this class with the old one, using the link array which is a matrix giving the references to the old structure
        //
        //
        
        uint64_t z_,x_,j_,y_,node_val;
        uint64_t old_key;
        uint64_t old_node_val;
        
        
        for(uint64_t i = access_data.depth_min;i <= access_data.depth_max;i++){
            
            const unsigned int x_num_ = access_data.x_num[i];
            const unsigned int z_num_ = access_data.z_num[i];
            
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t j_num = access_data.data[i][offset_pc_data].size();
                    y_ = 0;
                    
                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff
                        
                        node_val = access_data.data[i][offset_pc_data][j_];
                        
                        if(!(node_val&1)){
                            y_++;
                            old_key = link_array[i](y_,x_,z_);
                            
                            pc_key curr_key;
                            curr_key.update_part(old_key);
                            
                            if( x_ == curr_key.x_p){
                                // all good
                            } else {
                               std::cout << "ERROR x" << curr_key.status << " " << curr_key.depth << " " << curr_key.p << std::endl;
                            }
                            
                            if( z_ == curr_key.z_p){
                                // all good
                            } else {
                                std::cout << "ERROR z" << curr_key.status << " " << curr_key.depth << " " << curr_key.p << std::endl;
                            }
                            
                            old_node_val = pc_struct.pc_data.get_val(old_key);
                            
                            uint16_t part_offset =  (node_val& Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;
                            
                            float current_int = particle_data.data[i][offset_pc_data][part_offset];
                            
                            float old_int = pc_struct.part_data.get_part(old_key);
                            
                            if(current_int != old_int){
                                std::cout << "incorrect" << std::endl;
                            }
                            
                            
                        } else {
                            y_ += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                            y_--;
                            
                        }
                    }
                    
                }
                
            }
            
            
        }
        
        
        
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
    
    void initialize_global_index(){
        //
        //  Bevan Cheeseman 2016
        //
        //  Offset vector used for calculating a global index for each particle
        //
        //  (Can be useful for certain parallel routines)
        //
        //  Global particle index = index_offset + j;
        //
        
        uint64_t x_;
        uint64_t z_;
        uint64_t counter = 0;
        
        //initialize
        global_index_offset.resize(particle_data.depth_max+1);
        
        for(uint64_t i = particle_data.depth_min;i <= particle_data.depth_max;i++){
            
            size_t x_num_ = particle_data.x_num[i];
            size_t z_num_ = particle_data.z_num[i];
            
            global_index_offset[i].resize(x_num_*z_num_);
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = particle_data.data[i][offset_pc_data].size();
                    
                    
                    global_index_offset[i][offset_pc_data] = counter;
                    
                    counter += j_num;
                    
                    
                    
                }
            }
        }
        
        
        
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