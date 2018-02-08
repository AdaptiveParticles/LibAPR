///////////////////
//
//  Bevan Cheeseman 2016
//
//     Class for storing extra cell or particle data, that can then be accessed using the access data from pc_structure or parent cells
//
///////////////

#ifndef PARTPLAY_EXTRAPARTCELLDATA_HPP
#define PARTPLAY_EXTRAPARTCELLDATA_HPP

#include <functional>

template<typename V>
class APR;

class APRAccess;

template<typename T>
class ExtraPartCellData {
    
public:
    
    //the neighbours arranged by face

    ExtraPartCellData() {};
    
    template<typename S>
    ExtraPartCellData(ExtraPartCellData<S>& part_data) {
        initialize_structure_parts(part_data);
    };

    uint64_t depth_max;
    uint64_t depth_min;

    std::vector<unsigned int> z_num;
    std::vector<unsigned int> x_num;
    std::vector<unsigned int> y_num;

    std::vector<unsigned int> org_dims;
    std::vector<std::vector<std::vector<T>>> data;
    std::vector<std::vector<uint64_t>> global_index_offset;

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

    template<typename S>
    void initialize_structure_parts_empty(APR<S>& apr){
        //
        //  Initialize the structure to the same size as the given structure
        //

        //first add the layers
        depth_max = apr.level_max();
        depth_min = apr.level_min();

        z_num.resize(depth_max+1);
        x_num.resize(depth_max+1);
        y_num.resize(depth_max+1);

        data.resize(depth_max+1);

        org_dims.resize(3);
        org_dims[0] = apr.orginal_dimensions(0);
        org_dims[1] = apr.orginal_dimensions(1);
        org_dims[2] = apr.orginal_dimensions(2);

        for(uint64_t i = depth_min;i <= depth_max;i++){
            z_num[i] = apr.spatial_index_z_max(i);
            x_num[i] = apr.spatial_index_x_max(i);
            y_num[i] = apr.spatial_index_y_max(i);

            data[i].resize(z_num[i]*x_num[i]);
        }
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

    uint64_t structure_size(){
        //
        //  Simply counts the amount of data in the structure
        //
        //
        uint64_t x_;
        uint64_t z_;
        uint64_t counter = 0;

        for(uint64_t i = depth_min;i <= depth_max;i++){
            size_t x_num_ = x_num[i];
            size_t z_num_ = z_num[i];

            for(z_ = 0;z_ < z_num_;z_++){
                for(x_ = 0;x_ < x_num_;x_++){
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t j_num = data[i][offset_pc_data].size();
                    counter += j_num;
                }
            }
        }

        return counter;
    }
};


#endif //PARTPLAY_PARTNEIGH_HPP
