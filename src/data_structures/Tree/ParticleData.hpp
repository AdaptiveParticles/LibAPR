///////////////////
//
//  Bevan Cheeseman 2016
//
//  PartCellData class, the data container for CRS sparse format for APR
//
///////////////
'
#ifndef PARTPLAY_PARTICLEDATA_HPP
#define PARTPLAY_PARTICLEDATA_HPP

#include "PartCellData.hpp"
#include "PartKey.hpp"


template <typename T,typename S> // type T data structure base type, type S particle data type
class ParticleData {
    
public:
    
    /*
     * Number of layers without the root and the contents.
     */
    
    ParticleData(){};
    
    T& operator ()(int depth, int x_,int z_,int j_,int index){
        // data access
        return data[depth][x_num[depth]*z_ + x_][j_];
    }
    
    
    T& operator ()(const PartKey& key){
        // data access
        uint16_t offset = access_data[key.depth][access_data.x_num[key.depth]*key.z + key.x][key.j]
        return particle_data[key.depth][access_data.x_num[key.depth]*key.z + key.x][offset + key.index];
    }
    
    PartCellData<T> access_data;
    PartCellData<S> particle_data;
    
    uint8_t depth_max;
    uint8_t depth_min;
    
    std::vector<unsigned int> z_num;
    std::vector<unsigned int> x_num;
    
    template<typename U>
    void initialize_from_structure(PartCellData<S>& part_cell_data){
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
        int y_;
        uint64_t j_;
        
        //next initialize the entries;
        Part_timer timer;
        
        timer.start_timer("intiialize access data structure");
        
        for(int i = access_data.depth_min;i <= access_data.depth_max;i++){
            
            const unsigned int x_num = access_data.x_num[i];
            const unsigned int z_num = access_data.z_num[i];
            
#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    const size_t offset_pc_data = x_num*z_ + x_;
                    access_data.data[i][offset_pc_data].resize(part_cell_data.data[i][offset_pc_data].size());
                }

            }
        }
        
        timer.stop_timer();
        
        for(int i = access_data.depth_min;i <= access_data.depth_max;i++){
            
            const unsigned int x_num = access_data.x_num[i];
            const unsigned int z_num = access_data.z_num[i];
            
#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){
                
                for(x_ = 0;x_ < x_num;x_++){
                    const size_t offset_pc_data = x_num*z_ + x_;
                    const size_t j_num = acess_data[i][offset_pc_data].size();
                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff
                        
                        //finish here
                    }
                    
                    //then resize the particle data structure here..
                }
                
            }
        }
        
        //then done for initialization, then need to get the intensities.. 
        
    }
    
private:
    
    uint64_t num_particles;
    
    
    
};

#endif //PARTPLAY_PARTCELLDATA_HPP