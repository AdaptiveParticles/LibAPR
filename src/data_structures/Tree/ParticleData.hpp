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
        
        U temp3;
        
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
    
private:
    
    uint64_t num_particles;
    
    
    
};

#endif //PARTPLAY_PARTCELLDATA_HPP