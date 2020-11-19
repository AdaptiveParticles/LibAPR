//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_EXTRAPARTICLEDATA_HPP
#define PARTPLAY_EXTRAPARTICLEDATA_HPP

#include "data_structures/APR/iterators/APRIterator.hpp"
#include "data_structures/APR/iterators/LinearIterator.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/Mesh/ImagePatch.hpp"
#include "io/TiffUtils.hpp"

#include "GenData.hpp"

#include "../APR.hpp"
#include "numerics/APRNumerics.hpp"

#include <algorithm>
#include <vector>

#ifdef APR_USE_CUDA
#include "ParticleDataGpu.hpp"
#endif


template<typename DataType>
class ParticleData {

    static const uint64_t parallel_particle_number_threshold = 5000000l;

public:

#ifdef APR_USE_CUDA
    ParticleDataGpu<DataType> gpu_data;
#endif

    VectorData<DataType> data;

    APRCompress compressor;

    ParticleData() {};
    ParticleData(uint64_t aTotalNumberOfParticles) { init(aTotalNumberOfParticles); }

    ParticleData(const ParticleData& parts2Copy){
        data.copy(parts2Copy.data);
        compressor = parts2Copy.compressor;
    }

    void swap(ParticleData& aObj) {
        std::swap(compressor, aObj.compressor);
        data.swap(aObj.data);
    }

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
    inline const DataType& operator[](uint64_t aGlobalIndex) const { return data[aGlobalIndex]; }

    inline DataType& operator[](const LinearIterator& it)  {
        return data[it.global_index()];
    }

    void fill_with_levels(APR &apr){
        general_fill_level(apr,*this,false);
    }

    void fill_with_levels_tree(APR &apr){

        general_fill_level(apr,*this,true);
    }

    template<typename imageType>
    void sample_parts_from_img_downsampled(APR& apr,PixelData<imageType>& img){
        sample_parts_from_img_downsampled_gen(apr,*this,img);
    }

    void sample_parts_from_img_blocked(APR& apr, const std::string& aFileName, const int z_block_size = 256, const int ghost_z = 32) {
        sample_parts_from_img_blocked_gen(apr, *this, aFileName, z_block_size, ghost_z);
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
        data.fill(0);
    }

    void fill(DataType val) {
        data.fill(val);
    }


};


template<typename ImageType,typename ParticleDataType>
void sample_parts_from_img_downsampled_gen(APR& apr,ParticleDataType& parts,PixelData<ImageType>& input_image) {

    std::vector<PixelData<ImageType>> downsampled_img;
    //Down-sample the image for particle intensity estimation
    downsamplePyrmaid(input_image, downsampled_img, apr.level_max(), apr.level_min());

    //aAPR.get_parts_from_img_alt(input_image,aAPR.particles_intensities);
    sample_parts_from_img_downsampled_gen(apr,parts,downsampled_img);

    std::swap(input_image, downsampled_img.back());
}

/**
* Samples particles from an image using an image tree (img_by_level is a vector of images)
*/
template<typename ImageType,typename ParticleDataType>
void sample_parts_from_img_downsampled_gen(APR& apr,ParticleDataType& parts,std::vector<PixelData<ImageType>>& img_by_level){
    auto it = apr.iterator();
    parts.init(apr);

    for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for (int z = 0; z < it.z_num(level); ++z) {
            for (int x = 0; x < it.x_num(level); ++x) {
                for (it.begin(level, z, x);it <it.end();it++) {

                    parts[it] = img_by_level[level].at(it.y(),x,z);
                }
            }
        }
    }
}


template<typename ParticleDataType>
void sample_parts_from_img_blocked_gen(APR& apr, ParticleDataType& parts, const std::string& aFileName, const int z_block_size = 256, const int ghost_z = 32) {

    TiffUtils::TiffInfo inputTiff(aFileName);
    if (!inputTiff.isFileOpened()) {
        std::cerr << "ParticleData::sample_parts_from_img_blocked failed to open TIFF file " << aFileName << std::endl;
        return;
    }

    if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT8) {
        return sample_parts_from_img_blocked_gen<uint8_t, ParticleDataType>(apr, parts, inputTiff, z_block_size, ghost_z);
    } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_FLOAT) {
        return sample_parts_from_img_blocked_gen<float, ParticleDataType>(apr, parts, inputTiff, z_block_size, ghost_z);
    } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT16) {
        return sample_parts_from_img_blocked_gen<uint16_t, ParticleDataType>(apr, parts, inputTiff, z_block_size, ghost_z);
    } else {
        std::cerr << "ParticleData::sample_parts_from_img_blocked: unsupported file type. Input image must be a TIFF of data type uint8, uint16 or float32" << std::endl;
        return;
    }
}


/**
* Samples particles from an image using an image tree (img_by_level is a vector of images)
*/
template<typename ImageType,typename ParticleDataType>
void sample_parts_from_img_blocked_gen(APR& apr, ParticleDataType& parts, const TiffUtils::TiffInfo &aTiffFile, const int z_block_size, const int ghost_z){

    parts.init(apr);

    const int y_num = apr.org_dims(0);
    const int x_num = apr.org_dims(1);
    const int z_num = apr.org_dims(2);

    if(aTiffFile.iImgWidth != y_num || aTiffFile.iImgHeight != x_num || aTiffFile.iNumberOfDirectories != z_num) {
        std::cerr << "Warning: ParticleData::sample_parts_from_img_blocked - input image dimensions do not match APR dimensions" << std::endl;
    }

    const int number_z_blocks = std::max(z_num / z_block_size, 1);

    for(int block = 0; block < number_z_blocks; ++block) {

        int z_0 = block * z_block_size;
        int z_f = (block == (number_z_blocks - 1)) ? z_num : (block + 1) * z_block_size;

        int z_ghost_l = std::min(z_0, ghost_z);
        int z_ghost_r = std::min(z_num - z_f, ghost_z);

        ImagePatch patch;
        initPatchGlobal(patch, z_0 - z_ghost_l, z_f + z_ghost_r, 0, x_num, 0, y_num);

        patch.z_ghost_l = z_ghost_l;
        patch.z_ghost_r = z_ghost_r;
        patch.z_offset = z_0 - z_ghost_l;

        PixelData<ImageType> patchImage = TiffUtils::getMesh<ImageType>(aTiffFile, patch.z_begin_global, patch.z_end_global);
        sample_parts_from_img_downsampled_patch(apr, parts, patchImage, patch);
    }
}

template<typename ImageType,typename ParticleDataType>
void sample_parts_from_img_downsampled_patch(APR& apr, ParticleDataType& parts, PixelData<ImageType>& input_image, ImagePatch& patch) {

    auto it = apr.iterator();
    //Down-sample the image for particle intensity estimation at coarser resolutions
    std::vector<PixelData<ImageType>> img_by_level;
    downsamplePyrmaid(input_image, img_by_level, apr.level_max(), apr.level_min());

    for(int level = it.level_min(); level <= it.level_max(); ++level) {
//        const int level_factor = std::pow(2,(int)it.level_max()-level);
        const int level_factor = apr.level_size(level);

        const size_t z_ghost_l = patch.z_ghost_l / level_factor;
        const size_t z_ghost_r = patch.z_ghost_r / level_factor;

        const size_t x_ghost_l = patch.x_ghost_l / level_factor;
        const size_t x_ghost_r = patch.x_ghost_r / level_factor;

        const size_t y_ghost_l = patch.y_ghost_l / level_factor;
        const size_t y_ghost_r = patch.y_ghost_r / level_factor;

        const size_t offset_x = (patch.x_offset + patch.x_ghost_l) / level_factor - x_ghost_l;
        const size_t offset_y = (patch.y_offset + patch.y_ghost_l) / level_factor - y_ghost_l;
        const size_t offset_z = (patch.z_offset + patch.z_ghost_l) / level_factor - z_ghost_l;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for (int z = z_ghost_l; z < img_by_level[level].z_num - z_ghost_r; ++z) {
            for (int x = x_ghost_l; x < img_by_level[level].x_num - x_ghost_r; ++x) {
                it.begin(level, z+offset_z, x+offset_x);
                while(it.y() < (offset_y + y_ghost_l) && it < it.end()){
                    it++;
                }
                while(it.y() < (offset_y + img_by_level[level].y_num - y_ghost_r) && it < it.end()) {
                    parts[it] = img_by_level[level].at(it.y()-offset_y, x, z);
                    it++;
                }
            }
        }
    }

    std::swap(input_image, img_by_level.back()); // revert swap made by downsamplePyramid
}


template<typename ParticleDataType>
void general_fill_level(APR &apr,ParticleDataType& parts,bool tree){

    LinearIterator it;

    if(tree){
        it = apr.tree_iterator();
        parts.init_tree(apr);
    } else {
        it = apr.iterator();
        parts.init(apr);
    }

    for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(it)
#endif
        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {
                for (it.begin(level, z, x); it < it.end();
                     it++) {

                    parts[it] = level;

                }
            }
        }
    }

}

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
