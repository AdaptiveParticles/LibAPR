//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_EXTRAPARTICLEDATA_HPP
#define PARTPLAY_EXTRAPARTICLEDATA_HPP


#include <algorithm>


template<typename V> class APR;
template<typename V> class APRTree;
template<typename V> class APRIterator;

template<typename DataType>
class ExtraParticleData {

private:

    static const uint64_t parallel_particle_number_threshold = 5000000l;

public:

    std::vector<DataType> data;

    ExtraParticleData() {};
    template<typename S>
    ExtraParticleData(const APR<S> &apr) { init(apr); }

    template<typename S>
    ExtraParticleData(const APRTree<S> &apr_tree) { init_tree(apr_tree); }

    template<typename S>
    void init(const APR<S> &apr){
        data.resize(apr.total_number_particles());
    }

    template<typename S>
    void init_tree(const APRTree<S> &apr_tree){
        //initialization when using with APRTree class
        data.resize(apr_tree.total_number_parent_cells(),0);
    }

    uint64_t total_number_particles() const {
        return data.size();
    }

    /**
     * Access particle via iterator
     * @param apr_iterator
     * @return reference to stored particle
     */
    template<typename S>
    DataType& operator[](const APRIterator<S>& apr_iterator) {
        return data[apr_iterator.global_index()];
    }

    template<typename S>
    DataType get_particle(const APRIterator<S>& apr_iterator) const {
        return data[apr_iterator.global_index()];
    }

    template<typename S>
    void set_particle(const APRIterator<S>& apr_iterator, DataType set_val) {
        data[apr_iterator.global_index()] = set_val;
    }

    /**
     * Copy's the data from one particle dataset to another
     */
    template<typename S,typename T>
    void copy_parts(APR<T> &apr, const ExtraParticleData<S> &particlesToCopy, uint64_t level = 0, unsigned int aNumberOfBlocks = 10) {
        const uint64_t total_number_of_particles = particlesToCopy.data.size();

        //checking if its the right size, if it is, this should do nothing.
        data.resize(total_number_of_particles);

        APRIterator<T> apr_iterator(apr);

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
    template<typename V,class BinaryOperation,typename T>
    void zip_inplace(APR<T> &apr, const ExtraParticleData<V> &parts2, BinaryOperation op, uint64_t level = 0, unsigned int aNumberOfBlocks = 10) {
        APRIterator<T> apr_iterator(apr);

        size_t particle_number_start;
        size_t particle_number_stop;
        if (level==0) {
            particle_number_start = 0;
            particle_number_stop = total_number_particles();
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
    template<typename V,class BinaryOperation,typename T>
    void zip(APR<T>& apr, const ExtraParticleData<V> &parts2, ExtraParticleData<V>& output, BinaryOperation op, uint64_t level = 0, unsigned int aNumberOfBlocks = 10) {
        output.data.resize(data.size());

        APRIterator<T> apr_iterator(apr);

        size_t particle_number_start;
        size_t particle_number_stop;
        if (level==0) {
            particle_number_start = 0;
            particle_number_stop = total_number_particles();
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
     * Performs a unary operator on a particle dataset in parrallel and returns it in output
     * Bevan Cheeseman 2018
     * TODO: map and map_inplace are doing technicaly same thing - merge them
     */
    template<typename T,typename U,class UnaryOperator>
    void map(APR<T>& apr,ExtraParticleData<U>& output,UnaryOperator op,const uint64_t level = 0,unsigned int aNumberOfBlocks = 10){
        output.data.resize(data.size());

        APRIterator<T> apr_iterator(apr);

        size_t particle_number_start;
        size_t particle_number_stop;
        if (level==0) {
            particle_number_start=0;
            particle_number_stop = total_number_particles();
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

    /**
     * Performs a unary operator on a particle dataset inplace in parrallel
     * Bevan Cheeseman 2018
     */
    template<class UnaryOperator,typename T>
    void map_inplace(APR<T>& apr,UnaryOperator op,const uint64_t level = 0,unsigned int aNumberOfBlocks = 10){
        APRIterator<T> apr_iterator(apr);

        size_t particle_number_start;
        size_t particle_number_stop;
        if (level==0) {
            particle_number_start=0;
            particle_number_stop = total_number_particles();
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
};


#endif //PARTPLAY_EXTRAPARTICLEDATA_HPP
