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
#include "benchmarks/development/Tree/ParticleData.hpp"

#include <functional>

template<typename V>
class APR;

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

    template<typename S>
    ExtraPartCellData(APR<S>& apr){
        // intialize from apr
        initialize_structure_cells(apr.pc_data);
    }

    uint64_t depth_max;
    uint64_t depth_min;
    
    std::vector<unsigned int> z_num;
    std::vector<unsigned int> x_num;
    std::vector<unsigned int> y_num;
    
    std::vector<std::vector<std::vector<T>>> data;

    std::vector<unsigned int> org_dims;

    std::vector<std::vector<uint64_t>> global_index_offset;

    template<typename S>
    void init(APR<S>& apr){
        // do nothing
        initialize_structure_cells(apr.pc_data);
    }


    template<typename S>
    void copy_parts(ExtraPartCellData<S>& parts_to_copy){
        //
        //  Copy's the data from one particle dataset to another, assumes it is already intialized.
        //

        uint64_t x_;
        uint64_t z_;

        for(uint64_t i = depth_min;i <= depth_max;i++){

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];

#pragma omp parallel for private(z_,x_)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t j_num = data[i][offset_pc_data].size();

                    std::copy(parts_to_copy.data[i][offset_pc_data].begin(),parts_to_copy.data[i][offset_pc_data].end(),data[i][offset_pc_data].begin());

                }
            }

        }


    }

    template<typename S>
    void copy_parts(ExtraPartCellData<S>& parts_to_copy,const unsigned int level){
        //
        //  Copy's the data from one particle dataset to another, assumes it is already intialized, for a specific level
        //

        uint64_t x_;
        uint64_t z_;

        const unsigned int x_num_ = x_num[level];
        const unsigned int z_num_ = z_num[level];

#pragma omp parallel for private(z_,x_)
        for(z_ = 0;z_ < z_num_;z_++){

            for(x_ = 0;x_ < x_num_;x_++){

                const size_t offset_pc_data = x_num_*z_ + x_;
                const size_t j_num = data[level][offset_pc_data].size();

                std::copy(parts_to_copy.data[level][offset_pc_data].begin(),parts_to_copy.data[level][offset_pc_data].end(),data[level][offset_pc_data].begin());

            }
        }


    }


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

            int j = 0;
#pragma omp parallel for private(j)
            for(j = 0;j < pc_data.data[i].size();j++){
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

    template<typename S>
    void initialize_structure_parts_empty(ExtraPartCellData<S>& part_data){
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

    template<typename V>
    void set_val(V val){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and fills all particles with a value V.
        //

        int z_,x_,j_,y_;

        for(uint64_t depth = (depth_min);depth <= depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = x_num[depth];
            const unsigned int z_num_ = z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                    std::fill(data[depth][pc_offset].begin(),data[depth][pc_offset].end(),val);

                }
            }
        }
    }
    template<typename V,class BinaryOperation>
    void zip_inplace(ExtraPartCellData<V> &parts2, BinaryOperation op){
        //
        //  Bevan Cheeseman 2017
        //
        //  Takes two particle data sets and adds them, and puts it in the first one
        //
        //  See std::transform for examples of Unary Operators
        //
        //

        int z_,x_,j_,y_;

        for(uint64_t depth = (depth_min);depth <= depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = x_num[depth];
            const unsigned int z_num_ = z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                    std::transform(data[depth][pc_offset].begin(), data[depth][pc_offset].end(), parts2.data[depth][pc_offset].begin(), data[depth][pc_offset].begin(), op);

                }
            }
        }

    }

    template<typename V,class BinaryOperation>
    ExtraPartCellData<V> zip(ExtraPartCellData<V> &parts2, BinaryOperation op){
        //
        //  Bevan Cheeseman 2017
        //
        //  Takes two particle data sets and adds them, and puts it in the first one
        //
        //  See std::transform for examples of BinaryOperation
        //
        //  Returns the result to another particle dataset
        //

        ExtraPartCellData<V> output;
        output.initialize_structure_parts(*this);

        int z_,x_,j_,y_;

        for(uint64_t depth = (depth_min);depth <= depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = x_num[depth];
            const unsigned int z_num_ = z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                    std::transform(data[depth][pc_offset].begin(), data[depth][pc_offset].end(), parts2.data[depth][pc_offset].begin(), output.data[depth][pc_offset].begin(), op);

                }
            }
        }

        return output;

    }



    template<typename U,class UnaryOperator>
    ExtraPartCellData<U> map(UnaryOperator op){
        //
        //  Bevan Cheeseman 2018
        //
        //  Performs a unary operator on a particle dataset in parrallel and returns a new dataset with the result
        //
        //  See std::transform for examples of different operators to use
        //
        //

        ExtraPartCellData<U> output;
        output.initialize_structure_parts(*this);

        int z_,x_,j_,y_;

        for(uint64_t depth = (depth_min);depth <= depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = x_num[depth];
            const unsigned int z_num_ = z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                    std::transform(data[depth][pc_offset].begin(),data[depth][pc_offset].end(),output.data[depth][pc_offset].begin(),op);

                }
            }
        }

        return output;

    }

    template<class UnaryOperator>
    void map_inplace(UnaryOperator op){
        //
        //  Bevan Cheeseman 2018
        //
        //  Performs a unary operator on a particle dataset in parrallel and returns a new dataset with the result
        //
        //  See std::transform for examples of different operators to use
        //

        int z_,x_,j_,y_;

        for(uint64_t depth = (depth_min);depth <= depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = x_num[depth];
            const unsigned int z_num_ = z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                    std::transform(data[depth][pc_offset].begin(),data[depth][pc_offset].end(),data[depth][pc_offset].begin(),op);

                }
            }
        }

    }


    template<class UnaryOperator>
    void map_inplace(UnaryOperator op,unsigned int level){
        //
        //  Bevan Cheeseman 2018
        //
        //  Performs a unary operator on a particle dataset in parrallel and returns a new dataset with the result
        //
        //  See std::transform for examples of different operators to use
        //

        int z_,x_,j_,y_;

        //loop over the resolutions of the structure
        const unsigned int x_num_ = x_num[level];
        const unsigned int z_num_ = z_num[level];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;

#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                std::transform(data[level][pc_offset].begin(),data[level][pc_offset].end(),data[level][pc_offset].begin(),op);

            }
        }


    }




    /// some helpers functions for the transform_parts above
    template<typename V>
    V square(V input){
        return pow(input,2);
    }
    template<typename V>
    V square_root(V input){
        return sqrt(input);
    }


private:
    
};





#endif //PARTPLAY_PARTNEIGH_HPP