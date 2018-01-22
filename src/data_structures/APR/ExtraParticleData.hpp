//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_EXTRAPARTICLEDATA_HPP
#define PARTPLAY_EXTRAPARTICLEDATA_HPP

template<typename V>
class APR;

template<typename V>
class APRIterator;

#include <functional>
#include "src/data_structures/APR/APR.hpp"
//#include "src/data_structures/APR/APRIterator.hpp"


template<typename DataType>
class ExtraParticleData {

private:

    uint64_t parallel_particle_number_threshold;

public:

    //the neighbours arranged by face

    ExtraParticleData():parallel_particle_number_threshold(5000000l){
    };


    template<typename S>
    ExtraParticleData(APR<S>& apr):parallel_particle_number_threshold(5000000l){
        // intialize from apr
        data.resize(apr.total_number_particles());
    }




    std::vector<DataType> data;

    template<typename S>
    void init(APR<S>& apr){
        data.resize(apr.total_number_particles());
    }

    uint64_t total_number_particles(){
        return data.size();
    }

    /////////////////
    /// ////
    /// \tparam S
    /// \param apr_iterator
    /// \return access to particle data

    template<typename S>
    DataType& operator[](APRIterator<S>& apr_iterator){
        return data[apr_iterator.global_index()];
    }

    template<typename S>
    DataType get_particle(APRIterator<S>& apr_iterator){
        return data[apr_iterator.global_index()];
    }

    template<typename S>
    void set_particle(APRIterator<S>& apr_iterator,DataType set_val){
        data[apr_iterator.global_index()] = set_val;
    }

    template<typename S,typename T>
    void copy_parts(APR<T>& apr,ExtraParticleData<S>& parts_to_copy,const unsigned int level = 0,unsigned int aNumberOfBlocks = 10){
        //
        //  Copy's the data from one particle dataset to another
        //

        const uint64_t total_number_of_particles = parts_to_copy.data.size();

        //checking if its the right size, if it is, this should do nothing.
        this->data.resize(total_number_of_particles);

        APRIterator<T> apr_iterator(apr);

        size_t particle_number_start;
        size_t particle_number_stop;
        size_t total_particles_to_iterate;

        if(level==0){
            particle_number_start=0;
            particle_number_stop = total_number_of_particles;

        } else {
            particle_number_start = apr_iterator.particles_level_begin(level);
            particle_number_stop = apr_iterator.particles_level_end(level);
        }

        total_particles_to_iterate = particle_number_stop - particle_number_start;

        //determine if openMP should be used.
        if(total_particles_to_iterate < parallel_particle_number_threshold){
            aNumberOfBlocks = 1;
        }

        const size_t numOfElementsPerBlock = total_particles_to_iterate/aNumberOfBlocks;

        unsigned int blockNum;
#pragma omp parallel for private(blockNum) schedule(static)
        for (blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
            size_t offsetBegin = particle_number_start + blockNum * numOfElementsPerBlock;
            size_t offsetEnd = particle_number_start + offsetBegin + numOfElementsPerBlock;
            if (blockNum == aNumberOfBlocks - 1) {
                // Handle tailing elements if number of blocks does not divide.
                offsetEnd = particle_number_stop;
            }

            //Operation to be performed on the chunk
            std::copy(parts_to_copy.data.begin() + offsetBegin, parts_to_copy.data.begin() + offsetEnd, this->data.begin() + offsetBegin);
        }

    }



    template<typename V,class BinaryOperation,typename T>
    void zip_inplace(APR<T>& apr,ExtraParticleData<V> &parts2, BinaryOperation op,const unsigned int level = 0,unsigned int aNumberOfBlocks = 10){
        //
        //  Bevan Cheeseman 2017
        //
        //  Takes two particle data sets and adds them, and puts it in the first one
        //
        //  See std::transform for examples of Unary Operators
        //
        //

        APRIterator<T> apr_iterator(apr);

        size_t particle_number_start;
        size_t particle_number_stop;
        size_t total_particles_to_iterate;

        if(level==0){
            particle_number_start=0;
            particle_number_stop = total_number_particles();

        } else {
            particle_number_start = apr_iterator.particles_level_begin(level);
            particle_number_stop = apr_iterator.particles_level_end(level);
        }

        total_particles_to_iterate = particle_number_stop - particle_number_start;

        //determine if openMP should be used.
        if(total_particles_to_iterate < parallel_particle_number_threshold){
            aNumberOfBlocks = 1;
        }

        const size_t numOfElementsPerBlock = total_particles_to_iterate/aNumberOfBlocks;

        unsigned int blockNum;
#pragma omp parallel for private(blockNum) schedule(static)
        for (blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
            size_t offsetBegin = particle_number_start + blockNum * numOfElementsPerBlock;
            size_t offsetEnd = particle_number_start + offsetBegin + numOfElementsPerBlock;
            if (blockNum == aNumberOfBlocks - 1) {
                // Handle tailing elements if number of blocks does not divide.
                offsetEnd = particle_number_stop;
            }

            //Operation to be performed on the chunk
            std::transform(data.begin() + offsetBegin,data.begin() + offsetEnd, parts2.data.begin() + offsetBegin, data.begin() + offsetBegin, op);
        }



    }

    template<typename V,class BinaryOperation,typename T>
    void zip(APR<T>& apr,ExtraParticleData<V> &parts2,ExtraParticleData<V>& output, BinaryOperation op,const unsigned int level = 0,unsigned int aNumberOfBlocks = 10){
        //
        //  Bevan Cheeseman 2017
        //
        //  Takes two particle data sets and adds them, and puts it in the first one
        //
        //  See std::transform for examples of BinaryOperation
        //
        //  Returns the result to another particle dataset
        //

        output.data.resize(data.size());

        APRIterator<T> apr_iterator(apr);

        size_t particle_number_start;
        size_t particle_number_stop;
        size_t total_particles_to_iterate;

        if(level==0){
            particle_number_start=0;
            particle_number_stop = total_number_particles();

        } else {
            particle_number_start = apr_iterator.particles_level_begin(level);
            particle_number_stop = apr_iterator.particles_level_end(level);
        }

        total_particles_to_iterate = particle_number_stop - particle_number_start;

        //determine if openMP should be used.
        if(total_particles_to_iterate < parallel_particle_number_threshold){
            aNumberOfBlocks = 1;
        }

        const size_t numOfElementsPerBlock = total_particles_to_iterate/aNumberOfBlocks;

        unsigned int blockNum;
#pragma omp parallel for private(blockNum) schedule(static)
        for (blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
            size_t offsetBegin = particle_number_start + blockNum * numOfElementsPerBlock;
            size_t offsetEnd = particle_number_start + offsetBegin + numOfElementsPerBlock;
            if (blockNum == aNumberOfBlocks - 1) {
                // Handle tailing elements if number of blocks does not divide.
                offsetEnd = particle_number_stop;
            }

            //Operation to be performed on the chunk
            std::transform(data.begin() + offsetBegin,data.begin() + offsetEnd, parts2.data.begin() + offsetBegin, output.data.begin() + offsetBegin, op);
        }

    }



    template<typename T,typename U,class UnaryOperator>
    void map(APR<T>& apr,ExtraParticleData<U>& output,UnaryOperator op,const unsigned int level = 0,unsigned int aNumberOfBlocks = 10){
        //
        //  Bevan Cheeseman 2018
        //
        //  Performs a unary operator on a particle dataset in parrallel and returns a new dataset with the result
        //
        //  See std::transform for examples of different operators to use
        //
        //

        output.data.resize(data.size());

        APRIterator<T> apr_iterator(apr);

        size_t particle_number_start;
        size_t particle_number_stop;
        size_t total_particles_to_iterate;

        if(level==0){
            particle_number_start=0;
            particle_number_stop = total_number_particles();

        } else {
            particle_number_start = apr_iterator.particles_level_begin(level);
            particle_number_stop = apr_iterator.particles_level_end(level);
        }

        total_particles_to_iterate = particle_number_stop - particle_number_start;

        //determine if openMP should be used.
        if(total_particles_to_iterate < parallel_particle_number_threshold){
            aNumberOfBlocks = 1;
        }

        const size_t numOfElementsPerBlock = total_particles_to_iterate/aNumberOfBlocks;

        unsigned int blockNum;
#pragma omp parallel for schedule(static) private(blockNum)
        for (blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
            size_t offsetBegin = particle_number_start + blockNum * numOfElementsPerBlock;
            size_t offsetEnd = particle_number_start + offsetBegin + numOfElementsPerBlock;
            if (blockNum == aNumberOfBlocks - 1) {
                // Handle tailing elements if number of blocks does not divide.
                offsetEnd = particle_number_stop;
            }

            //Operation to be performed on the chunk
            std::transform(data.begin() + offsetBegin,data.begin() + offsetEnd, output.data.begin() + offsetBegin, op);
        }

    }

    template<class UnaryOperator,typename T>
    void map_inplace(APR<T>& apr,UnaryOperator op,const unsigned int level = 0,unsigned int aNumberOfBlocks = 10){
        //
        //  Bevan Cheeseman 2018
        //
        //  Performs a unary operator on a particle dataset in parrallel and returns a new dataset with the result
        //
        //  See std::transform for examples of different operators to use
        //

        APRIterator<T> apr_iterator(apr);

        size_t particle_number_start;
        size_t particle_number_stop;
        size_t total_particles_to_iterate;

        if(level==0){
            particle_number_start=0;
            particle_number_stop = total_number_particles();

        } else {
            particle_number_start = apr_iterator.particles_level_begin(level);
            particle_number_stop = apr_iterator.particles_level_end(level);
        }

        total_particles_to_iterate = particle_number_stop - particle_number_start;

        //determine if openMP should be used.
        if(total_particles_to_iterate < parallel_particle_number_threshold){
            aNumberOfBlocks = 1;
        }

        const size_t numOfElementsPerBlock = total_particles_to_iterate/aNumberOfBlocks;

        unsigned int blockNum;
#pragma omp parallel for private(blockNum) schedule(static)
        for (blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
            size_t offsetBegin = particle_number_start + blockNum * numOfElementsPerBlock;
            size_t offsetEnd = particle_number_start + offsetBegin + numOfElementsPerBlock;
            if (blockNum == aNumberOfBlocks - 1) {
                // Handle tailing elements if number of blocks does not divide.
                offsetEnd = particle_number_stop;
            }

            //Operation to be performed on the chunk
            std::transform(data.begin() + offsetBegin,data.begin() + offsetEnd, data.begin() + offsetBegin, op);
        }

    }

};


#endif //PARTPLAY_EXTRAPARTICLEDATA_HPP
