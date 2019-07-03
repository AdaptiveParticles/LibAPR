//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_EXTRAPARTICLEDATA_HPP
#define PARTPLAY_EXTRAPARTICLEDATA_HPP

#include "../iterators/APRIterator.hpp"
#include "data_structures/Mesh/PixelData.hpp"

#include "GenData.hpp"

#include "../APR.hpp"
#include "numerics/APRNumerics.hpp"

#include <algorithm>
#include <vector>

template<typename DataType>
class ParticleData {

    static const uint64_t parallel_particle_number_threshold = 5000000l;

public:
    std::vector<DataType> data;

    APRCompress compressor;

    ParticleData() {};
    ParticleData(uint64_t aTotalNumberOfParticles) { init(aTotalNumberOfParticles); }

    void init(uint64_t aTotalNumberOfParticles) { data.resize(aTotalNumberOfParticles); }
    /*
     * Init dataset with enough particles up to level
     */
    void init(APR& apr,unsigned int level)  {
        auto it = apr.iterator();
        if(level==0){
            level = it.level_max();
        }
        data.resize(it.total_number_particles(level),0);
    }
    /*
     * Init dataset with enough particles up to level for tree
     */
    void init_tree(APR& apr,unsigned int level)  {
        auto it = apr.tree_iterator();
        data.resize(it.total_number_particles(level),0);
    }

    /*
     * Init dataset with enough for all particles
     */
    void init(APR& apr)  {
        auto it = apr.iterator();
        data.resize(it.total_number_particles(it.level_max()),0);
    }
    /*
     * Init dataset with enough for all particles in tree
     */
    void init_tree(APR& apr)  {
        auto it = apr.tree_iterator();

        data.resize(it.total_number_particles(it.level_max()),0);
    }

    uint64_t size() const  { return data.size(); }
    inline DataType& operator[](uint64_t aGlobalIndex) { return data[aGlobalIndex]; }


    inline DataType& operator[](const LinearIterator& it)  {
        return data[it.global_index()];
    }

    void fill_with_levels(APR &apr){
        auto it = apr.iterator();
        APRNumerics::general_fill_level(apr,*this,it,false);
    }

    void fill_with_levels_tree(APR &apr){
        auto it = apr.tree_iterator();
        APRNumerics::general_fill_level(apr,*this,it,true);
    }

    template<typename S>
    void copy_parts(APR &apr, const ParticleData<S> &particlesToCopy, uint64_t level = 0, unsigned int aNumberOfBlocks = 10);
    template<typename V,class BinaryOperation>
    void zip_inplace(APR &apr, const ParticleData<V> &parts2, BinaryOperation op, uint64_t level = 0, unsigned int aNumberOfBlocks = 10);
    template<typename V,class BinaryOperation>
    void zip(APR& apr, const ParticleData<V> &parts2, ParticleData<V>& output, BinaryOperation op, uint64_t level = 0, unsigned int aNumberOfBlocks = 10);
    template<class UnaryOperator>
    void map_inplace(APR& apr,UnaryOperator op,const uint64_t level = 0,unsigned int aNumberOfBlocks = 10);
    template<typename U,class UnaryOperator>
    inline void map(APR& apr,ParticleData<U>& output,UnaryOperator op,const uint64_t level = 0,unsigned int aNumberOfBlocks = 10);

    void set_to_zero()  {
        std::fill(data.begin(),data.end(),0);
    }

    template<typename imageType>
    void sample_parts_from_img_downsampled(APR& apr,PixelData<imageType>& img){
        APRNumerics::sample_parts_from_img_downsampled(apr,*this,img);
    }

};



/**
 * Copy's the data from one particle dataset to another
 */
template<typename DataType> template<typename S>
inline void ParticleData<DataType>::copy_parts(APR &apr, const ParticleData<S> &particlesToCopy, uint64_t level, unsigned int aNumberOfBlocks) {
    const uint64_t total_number_of_particles = particlesToCopy.data.size();

    //checking if its the right size, if it is, this should do nothing.
    data.resize(total_number_of_particles);

    auto apr_iterator = apr.iterator();

    size_t particle_number_start;
    size_t particle_number_stop;
    if (level==0){
        particle_number_start = 0;
        particle_number_stop = total_number_of_particles;

    } else {
        particle_number_start = apr_iterator.particles_level_begin(level);
        particle_number_stop = apr_iterator.particles_level_end(level);
    }

    //determine if openMP should be used.
    size_t total_particles_to_iterate = particle_number_stop - particle_number_start;
    if (total_particles_to_iterate < parallel_particle_number_threshold) {
        aNumberOfBlocks = 1;
    }

    const size_t numOfElementsPerBlock = total_particles_to_iterate/aNumberOfBlocks;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (unsigned int blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
        size_t offsetBegin = particle_number_start + blockNum * numOfElementsPerBlock;
        size_t offsetEnd =  offsetBegin + numOfElementsPerBlock;
        if (blockNum == aNumberOfBlocks - 1) {
            // Handle tailing elements if number of blocks does not divide.
            offsetEnd = particle_number_stop;
        }

        //Operation to be performed on the chunk
        std::copy(particlesToCopy.data.begin() + offsetBegin, particlesToCopy.data.begin() + offsetEnd, data.begin() + offsetBegin);
    }
}

/**
 * Takes two particle data sets and adds them, and puts it in the first one
 * Bevan Cheeseman 2017
 * TODO: zip and zip_inplace are doing technicaly same thing - merge them
 */
template<typename DataType> template<typename V,class BinaryOperation>
inline void ParticleData<DataType>::zip_inplace(APR &apr, const ParticleData<V> &parts2, BinaryOperation op, uint64_t level, unsigned int aNumberOfBlocks) {
    auto apr_iterator = apr.iterator();

    size_t particle_number_start;
    size_t particle_number_stop;
    if (level==0) {
        particle_number_start = 0;
        particle_number_stop = size();
    } else {
        particle_number_start = apr_iterator.particles_level_begin(level);
        particle_number_stop = apr_iterator.particles_level_end(level);
    }

    //determine if openMP should be used.
    size_t total_particles_to_iterate = particle_number_stop - particle_number_start;
    if (total_particles_to_iterate < parallel_particle_number_threshold) {
        aNumberOfBlocks = 1;
    }

    const size_t numOfElementsPerBlock = total_particles_to_iterate/aNumberOfBlocks;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (unsigned int blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
        size_t offsetBegin = particle_number_start + blockNum * numOfElementsPerBlock;
        size_t offsetEnd =  offsetBegin + numOfElementsPerBlock;
        if (blockNum == aNumberOfBlocks - 1) {
            // Handle tailing elements if number of blocks does not divide.
            offsetEnd = particle_number_stop;
        }

        //Operation to be performed on the chunk
        std::transform(data.begin() + offsetBegin, data.begin() + offsetEnd, parts2.data.begin() + offsetBegin, data.begin() + offsetBegin, op);
    }
}

/**
 * Takes two particle data sets and adds them, and puts it in the output
 * Bevan Cheeseman 2017
 */
template<typename DataType> template<typename V,class BinaryOperation>
inline void ParticleData<DataType>::zip(APR& apr, const ParticleData<V> &parts2, ParticleData<V>& output, BinaryOperation op, uint64_t level, unsigned int aNumberOfBlocks) {
    output.data.resize(data.size());

    auto apr_iterator = apr.iterator();

    size_t particle_number_start;
    size_t particle_number_stop;
    if (level==0) {
        particle_number_start = 0;
        particle_number_stop = size();
    } else {
        particle_number_start = apr_iterator.particles_level_begin(level);
        particle_number_stop = apr_iterator.particles_level_end(level);
    }

    //determine if openMP should be used.
    size_t total_particles_to_iterate = particle_number_stop - particle_number_start;
    if (total_particles_to_iterate < parallel_particle_number_threshold) {
        aNumberOfBlocks = 1;
    }

    const size_t numOfElementsPerBlock = total_particles_to_iterate/aNumberOfBlocks;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (unsigned int blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
        size_t offsetBegin = particle_number_start + blockNum * numOfElementsPerBlock;
        size_t offsetEnd =  offsetBegin + numOfElementsPerBlock;
        if (blockNum == aNumberOfBlocks - 1) {
            // Handle tailing elements if number of blocks does not divide.
            offsetEnd = particle_number_stop;
        }

        //Operation to be performed on the chunk
        std::transform(data.begin() + offsetBegin, data.begin() + offsetEnd, parts2.data.begin() + offsetBegin, output.data.begin() + offsetBegin, op);
    }
}

/**
 * Performs a unary operator on a particle dataset inplace in parrallel
 * Bevan Cheeseman 2018
 */
template<typename DataType> template<class UnaryOperator>
inline void ParticleData<DataType>::map_inplace(APR& apr,UnaryOperator op,const uint64_t level, unsigned int aNumberOfBlocks){
    auto apr_iterator = apr.iterator();

    size_t particle_number_start;
    size_t particle_number_stop;
    if (level==0) {
        particle_number_start=0;
        particle_number_stop = size();
    } else {
        particle_number_start = apr_iterator.particles_level_begin(level);
        particle_number_stop = apr_iterator.particles_level_end(level);
    }

    //determine if openMP should be used.
    size_t total_particles_to_iterate = particle_number_stop - particle_number_start;
    if (total_particles_to_iterate < parallel_particle_number_threshold){
        aNumberOfBlocks = 1;
    }

    const size_t numOfElementsPerBlock = total_particles_to_iterate/aNumberOfBlocks;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (unsigned int blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
        size_t offsetBegin = particle_number_start + blockNum * numOfElementsPerBlock;
        size_t offsetEnd =  offsetBegin + numOfElementsPerBlock;
        if (blockNum == aNumberOfBlocks - 1) {
            // Handle tailing elements if number of blocks does not divide.
            offsetEnd = particle_number_stop;
        }

        //Operation to be performed on the chunk
        std::transform(data.begin() + offsetBegin, data.begin() + offsetEnd, data.begin() + offsetBegin, op);
    }
}

/**
 * Performs a unary operator on a particle dataset in parrallel and returns it in output
 * Bevan Cheeseman 2018
 * TODO: map and map_inplace are doing technicaly same thing - merge them
 */
template<typename DataType> template <typename U,class UnaryOperator>
inline void ParticleData<DataType>::map(APR& apr,ParticleData<U>& output,UnaryOperator op,const uint64_t level,unsigned int aNumberOfBlocks) {
    output.data.resize(data.size());

    auto apr_iterator = apr.iterator();

    size_t particle_number_start;
    size_t particle_number_stop;
    if (level==0) {
        particle_number_start=0;
        particle_number_stop = size();
    } else {
        particle_number_start = apr_iterator.particles_level_begin(level);
        particle_number_stop = apr_iterator.particles_level_end(level);
    }

    //determine if openMP should be used.
    size_t total_particles_to_iterate = particle_number_stop - particle_number_start;
    if (total_particles_to_iterate < parallel_particle_number_threshold) {
        aNumberOfBlocks = 1;
    }

    const size_t numOfElementsPerBlock = total_particles_to_iterate/aNumberOfBlocks;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (unsigned int blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
        size_t offsetBegin = particle_number_start + blockNum * numOfElementsPerBlock;
        size_t offsetEnd =  offsetBegin + numOfElementsPerBlock;
        if (blockNum == aNumberOfBlocks - 1) {
            // Handle tailing elements if number of blocks does not divide.
            offsetEnd = particle_number_stop;
        }

        //Operation to be performed on the chunk
        std::transform(data.begin() + offsetBegin,data.begin() + offsetEnd, output.data.begin() + offsetBegin, op);
    }
}


#endif //PARTPLAY_EXTRAPARTICLEDATA_HPP
