//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_EXTRAPARTICLEDATA_HPP
#define PARTPLAY_EXTRAPARTICLEDATA_HPP


#include <functional>
#include "src/data_structures/APR/APR.hpp"


template<typename V>
class APR;

template<typename DataType>
class ExtraParticleData {

public:

    //the neighbours arranged by face

    ExtraParticleData(){
    };

    template<typename S>
    ExtraParticleData(ExtraParticleData<S>& part_data){
        //initialize_structure_parts(part_data);
    };

    template<typename S>
    ExtraParticleData(APR<S>& apr){
        // intialize from apr
        data.resize(apr.num_parts_total);
    }

    std::vector<DataType> data;


    template<typename S>
    void init(APR<S>& apr){
        // do nothing
        initialize_structure_cells(apr.pc_data);
    }


    template<typename S>
    void copy_parts(ExtraParticleData<S>& parts_to_copy){
        //
        //  Copy's the data from one particle dataset to another, assumes it is already intialized.
        //

//        uint64_t x_;
//        uint64_t z_;
//
//        for(uint64_t i = depth_min;i <= depth_max;i++){
//
//            const unsigned int x_num_ = x_num[i];
//            const unsigned int z_num_ = z_num[i];
//
//#pragma omp parallel for private(z_,x_)
//            for(z_ = 0;z_ < z_num_;z_++){
//
//                for(x_ = 0;x_ < x_num_;x_++){
//
//                    const size_t offset_pc_data = x_num_*z_ + x_;
//                    const size_t j_num = data[i][offset_pc_data].size();
//
//                    std::copy(parts_to_copy.data[i][offset_pc_data].begin(),parts_to_copy.data[i][offset_pc_data].end(),data[i][offset_pc_data].begin());
//
//                }
//            }
//
//        }


    }

    template<typename S>
    void copy_parts(ExtraParticleData<S>& parts_to_copy,const unsigned int level){
        //
        //  Copy's the data from one particle dataset to another, assumes it is already intialized, for a specific level
        //

//        uint64_t x_;
//        uint64_t z_;
//
//        const unsigned int x_num_ = x_num[level];
//        const unsigned int z_num_ = z_num[level];
//
//#pragma omp parallel for private(z_,x_)
//        for(z_ = 0;z_ < z_num_;z_++){
//
//            for(x_ = 0;x_ < x_num_;x_++){
//
//                const size_t offset_pc_data = x_num_*z_ + x_;
//                const size_t j_num = data[level][offset_pc_data].size();
//
//                std::copy(parts_to_copy.data[level][offset_pc_data].begin(),parts_to_copy.data[level][offset_pc_data].end(),data[level][offset_pc_data].begin());
//
//            }
//        }


    }


    template<typename S>
    void initialize_data(std::vector<std::vector<S>>& input_data){
        //
        //  Initializes the data, from an existing array that is stored by depth
        //

//        uint64_t x_;
//        uint64_t z_;
//        uint64_t offset;
//
//
//        for(uint64_t i = depth_min;i <= depth_max;i++){
//
//            const unsigned int x_num_ = x_num[i];
//            const unsigned int z_num_ = z_num[i];
//
//            offset = 0;
//
//            for(z_ = 0;z_ < z_num_;z_++){
//
//                for(x_ = 0;x_ < x_num_;x_++){
//
//                    const size_t offset_pc_data = x_num_*z_ + x_;
//                    const size_t j_num = data[i][offset_pc_data].size();
//
//                    std::copy(input_data[i].begin()+offset,input_data[i].begin()+offset+j_num,data[i][offset_pc_data].begin());
//
//                    offset += j_num;
//
//                }
//            }
//
//        }

    }

    template<typename S>
    void initialize_structure_parts(ExtraPartCellData<S>& part_data){
        //
        //  Initialize the structure to the same size as the given structure
        //
//
//        //first add the layers
//        depth_max = part_data.depth_max;
//        depth_min = part_data.depth_min;
//
//        z_num.resize(depth_max+1);
//        x_num.resize(depth_max+1);
//
//        data.resize(depth_max+1);
//
//        org_dims = part_data.org_dims;
//
//        for(uint64_t i = depth_min;i <= depth_max;i++){
//            z_num[i] = part_data.z_num[i];
//            x_num[i] = part_data.x_num[i];
//            data[i].resize(z_num[i]*x_num[i]);
//
//            for(int j = 0;j < part_data.data[i].size();j++){
//                data[i][j].resize(part_data.data[i][j].size(),0);
//            }
//
//        }

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

//        int z_,x_,j_,y_;
//
//        for(uint64_t depth = (depth_min);depth <= depth_max;depth++) {
//            //loop over the resolutions of the structure
//            const unsigned int x_num_ = x_num[depth];
//            const unsigned int z_num_ = z_num[depth];
//
//            const unsigned int x_num_min_ = 0;
//            const unsigned int z_num_min_ = 0;
//
//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
//            for (z_ = z_num_min_; z_ < z_num_; z_++) {
//                //both z and x are explicitly accessed in the structure
//
//                for (x_ = x_num_min_; x_ < x_num_; x_++) {
//
//                    const unsigned int pc_offset = x_num_*z_ + x_;
//
//                    std::transform(data[depth][pc_offset].begin(), data[depth][pc_offset].end(), parts2.data[depth][pc_offset].begin(), data[depth][pc_offset].begin(), op);
//
//                }
//            }
//        }

    }

    template<typename V,class BinaryOperation>
    ExtraParticleData<V> zip(ExtraParticleData<V> &parts2, BinaryOperation op){
        //
        //  Bevan Cheeseman 2017
        //
        //  Takes two particle data sets and adds them, and puts it in the first one
        //
        //  See std::transform for examples of BinaryOperation
        //
        //  Returns the result to another particle dataset
        //

//        ExtraPartCellData<V> output;
//        output.initialize_structure_parts(*this);
//
//        int z_,x_,j_,y_;
//
//        for(uint64_t depth = (depth_min);depth <= depth_max;depth++) {
//            //loop over the resolutions of the structure
//            const unsigned int x_num_ = x_num[depth];
//            const unsigned int z_num_ = z_num[depth];
//
//            const unsigned int x_num_min_ = 0;
//            const unsigned int z_num_min_ = 0;
//
//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
//            for (z_ = z_num_min_; z_ < z_num_; z_++) {
//                //both z and x are explicitly accessed in the structure
//
//                for (x_ = x_num_min_; x_ < x_num_; x_++) {
//
//                    const unsigned int pc_offset = x_num_*z_ + x_;
//
//                    std::transform(data[depth][pc_offset].begin(), data[depth][pc_offset].end(), parts2.data[depth][pc_offset].begin(), output.data[depth][pc_offset].begin(), op);
//
//                }
//            }
//        }
//
//        return output;

    }



    template<typename U,class UnaryOperator>
    ExtraParticleData<U> map(UnaryOperator op){
        //
        //  Bevan Cheeseman 2018
        //
        //  Performs a unary operator on a particle dataset in parrallel and returns a new dataset with the result
        //
        //  See std::transform for examples of different operators to use
        //
        //

//        ExtraPartCellData<U> output;
//        output.initialize_structure_parts(*this);
//
//        int z_,x_,j_,y_;
//
//        for(uint64_t depth = (depth_min);depth <= depth_max;depth++) {
//            //loop over the resolutions of the structure
//            const unsigned int x_num_ = x_num[depth];
//            const unsigned int z_num_ = z_num[depth];
//
//            const unsigned int x_num_min_ = 0;
//            const unsigned int z_num_min_ = 0;
//
//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
//            for (z_ = z_num_min_; z_ < z_num_; z_++) {
//                //both z and x are explicitly accessed in the structure
//
//                for (x_ = x_num_min_; x_ < x_num_; x_++) {
//
//                    const unsigned int pc_offset = x_num_*z_ + x_;
//
//                    std::transform(data[depth][pc_offset].begin(),data[depth][pc_offset].end(),output.data[depth][pc_offset].begin(),op);
//
//                }
//            }
//        }
//
//        return output;

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

//        int z_,x_,j_,y_;
//
//        for(uint64_t depth = (depth_min);depth <= depth_max;depth++) {
//            //loop over the resolutions of the structure
//            const unsigned int x_num_ = x_num[depth];
//            const unsigned int z_num_ = z_num[depth];
//
//            const unsigned int x_num_min_ = 0;
//            const unsigned int z_num_min_ = 0;
//
//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
//            for (z_ = z_num_min_; z_ < z_num_; z_++) {
//                //both z and x are explicitly accessed in the structure
//
//                for (x_ = x_num_min_; x_ < x_num_; x_++) {
//
//                    const unsigned int pc_offset = x_num_*z_ + x_;
//
//                    std::transform(data[depth][pc_offset].begin(),data[depth][pc_offset].end(),data[depth][pc_offset].begin(),op);
//
//                }
//            }
//        }

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

//        int z_,x_,j_,y_;
//
//        //loop over the resolutions of the structure
//        const unsigned int x_num_ = x_num[level];
//        const unsigned int z_num_ = z_num[level];
//
//        const unsigned int x_num_min_ = 0;
//        const unsigned int z_num_min_ = 0;
//
//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
//        for (z_ = z_num_min_; z_ < z_num_; z_++) {
//            //both z and x are explicitly accessed in the structure
//
//            for (x_ = x_num_min_; x_ < x_num_; x_++) {
//
//                const unsigned int pc_offset = x_num_*z_ + x_;
//
//                std::transform(data[level][pc_offset].begin(),data[level][pc_offset].end(),data[level][pc_offset].begin(),op);
//
//            }
//        }


    }


};


#endif //PARTPLAY_EXTRAPARTICLEDATA_HPP
