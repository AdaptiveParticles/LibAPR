///////////////////
//
//  Bevan Cheeseman 2016
//
//     Class for storing extra cell or particle data, that can then be accessed using the access data from pc_structure or parent cells
//
///////////////

#ifndef PARTPLAY_EXTRAPARTCELLDATA_HPP
#define PARTPLAY_EXTRAPARTCELLDATA_HPP
// type T data structure base type

#include "PartCellData.hpp"
#include "ParticleData.hpp"



template<typename T>
class ExtraPartCellData {
    
public:
    
    //the neighbours arranged by face
    
    
    
    ExtraPartCellData(){
    };
    
    template<typename S>
    ExtraPartCellData(PartCellData<S>& pc_data){
        initialize_structure_cells(pc_data);
    };
    
    template<typename S>
    ExtraPartCellData(ExtraPartCellData<S>& part_data){
        initialize_structure_parts(part_data);
    };

    
    
    uint64_t depth_max;
    uint64_t depth_min;
    
    std::vector<unsigned int> z_num;
    std::vector<unsigned int> x_num;
    
    std::vector<std::vector<std::vector<T>>> data;

    std::vector<unsigned int> org_dims;

    std::vector<std::vector<uint64_t>> global_index_offset;
    
    template<typename S>
    void initialize_structure_cells(PartCellData<S>& pc_data){
        //
        //  Initialize the structure to the same size as the given structure
        //

        org_dims = pc_data.org_dims;

        //first add the layers
        depth_max = pc_data.depth_max;
        depth_min = pc_data.depth_min;
        
        z_num.resize(depth_max+1);
        x_num.resize(depth_max+1);
        
        data.resize(depth_max+1);
        
        for(uint64_t i = depth_min;i <= depth_max;i++){
            z_num[i] = pc_data.z_num[i];
            x_num[i] = pc_data.x_num[i];
            data[i].resize(z_num[i]*x_num[i]);
            
            for(int j = 0;j < pc_data.data[i].size();j++){
                data[i][j].resize(pc_data.data[i][j].size(),0);
            }
            
        }
        
    }
    
    template<typename S>
    void initialize_data(std::vector<std::vector<S>>& input_data){
        //
        //  Initializes the data, from an existing array that is stored by depth
        //
        
        uint64_t x_;
        uint64_t z_;
        uint64_t offset;
        
        
        for(uint64_t i = depth_min;i <= depth_max;i++){
            
            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            
            offset = 0;
            
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t j_num = data[i][offset_pc_data].size();
                    
                    std::copy(input_data[i].begin()+offset,input_data[i].begin()+offset+j_num,data[i][offset_pc_data].begin());
                    
                    offset += j_num;
                    
                }
            }
            
        }
        
    }
    
    template<typename S>
    void initialize_structure_parts(ExtraPartCellData<S>& part_data){
        //
        //  Initialize the structure to the same size as the given structure
        //
        
        //first add the layers
        depth_max = part_data.depth_max;
        depth_min = part_data.depth_min;
        
        z_num.resize(depth_max+1);
        x_num.resize(depth_max+1);
        
        data.resize(depth_max+1);

        org_dims = part_data.org_dims;
        
        for(uint64_t i = depth_min;i <= depth_max;i++){
            z_num[i] = part_data.z_num[i];
            x_num[i] = part_data.x_num[i];
            data[i].resize(z_num[i]*x_num[i]);
            
            for(int j = 0;j < part_data.data[i].size();j++){
                data[i][j].resize(part_data.data[i][j].size(),0);
            }
            
        }
        
    }




    
    T&  get_val(const uint64_t& pc_key){
        // data access
        
        const uint64_t depth = (pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t j_ = (pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
        
        
        return data[depth][x_num[depth]*z_ + x_][j_];
    }
    T&  get_part(const uint64_t part_key){
        // data access
        
        const uint64_t depth = (part_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (part_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (part_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t index = (part_key & PC_KEY_INDEX_MASK) >> PC_KEY_INDEX_SHIFT;
        
        return data[depth][x_num[depth]*z_ + x_][index];
        
    }

    uint64_t get_global_index(const uint64_t part_key){

        const uint64_t depth = (part_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (part_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (part_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t index = (part_key & PC_KEY_INDEX_MASK) >> PC_KEY_INDEX_SHIFT;

        return global_index_offset[depth][x_num[depth]*z_ + x_] + index;

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
        global_index_offset.resize(depth_max+1);

        for(uint64_t i = depth_min;i <= depth_max;i++){

            size_t x_num_ = x_num[i];
            size_t z_num_ = z_num[i];

            global_index_offset[i].resize(x_num_*z_num_);

            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;

                    const size_t j_num = data[i][offset_pc_data].size();

                    global_index_offset[i][offset_pc_data] = counter;

                    counter += j_num;


                }
            }
        }



    }


    
private:
    
};

#endif //PARTPLAY_PARTNEIGH_HPP