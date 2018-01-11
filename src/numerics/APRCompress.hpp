///////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Lossy Compression scheme for the APR intensities, implimentation/(extention of ideas to APR) of (Balázs et al. 2017, A real-time compression library for
/// microscopy images)
///
///////////////////////////

#ifndef PARTPLAY_COMPRESSAPR_HPP
#define PARTPLAY_COMPRESSAPR_HPP

#include "src/data_structures/APR/APR.hpp"

class APRCompress {

public:

    template<typename U>
    void compress(APR<U>& apr) {


        ExtraPartCellData<float> predict_input(apr);
        ExtraPartCellData<float> predict_output(apr);

        std::vector<unsigned int> predict_directions = {1,3,5};

        unsigned int level = apr.level_max();

        unsigned int num_blocks = 10;

        predict_particles_by_level(apr,level,predict_output,predict_input,predict_directions,num_blocks);

    }



private:

    float e = 1.6;
    float background = 1;
    float cnv = 65636/30000;
    float q = .5;

    template<typename T,typename S>
    T variance_stabilitzation(S input);

    template<typename T,typename S>
    T inverse_variance_stabilitzation(S input);

    template<typename T,typename S>
    T calculate_symbols(S input);

    template<typename T,typename S>
    T inverse_calculate_symbols(S input);

    template<typename T,typename S,typename U>
    void predict_particles_by_level(APR<U>& apr,unsigned int level,ExtraPartCellData<T>& predict_input,ExtraPartCellData<S>& predict_output,std::vector<unsigned int>& predict_directions,unsigned int num_z_blocks);

};

template<typename T,typename S>
T APRCompress::variance_stabilitzation(S input){

    return (2*sqrt(std::max((T) (input-background),(T)0)/cnv + pow(e,2)) - 2*e)/q;

};

template<typename T,typename S>
T APRCompress::inverse_variance_stabilitzation(S input){

    float D = q*input + 2*e;

    if(D >= 2*e){
        D = (pow(D,2)/4.0 - pow(e,2))*cnv + background;
        return ((T) D);
    } else {
        return ((T) background);
    }

};

template<typename T,typename S>
T APRCompress::calculate_symbols(S input){

    int16_t val = input;
    return 2*(abs(val)) + (val >> 15);
};


template<typename T,typename S>
T APRCompress::inverse_calculate_symbols(S input){

    int16_t negative = input % 2;

    return  (1 - 2 * negative) * ((input + negative) / 2);
};

template<typename T,typename S,typename U>
void APRCompress::predict_particles_by_level(APR<U>& apr,unsigned int level,ExtraPartCellData<T>& predict_input,ExtraPartCellData<S>& predict_output,std::vector<unsigned int>& predict_directions,unsigned int num_z_blocks){
    //
    //  Performs prediction step using the predict directions in chunks of the dataset, given by z_index slice.
    //
    //  This allows parallelization of a recursive prediction process.
    //


    APR_iterator<U> apr_iterator(apr);
    APR_iterator<U> neighbour_iterator(apr);

    // Compute the z-slice blocks that are to be computed over.
    std::vector<unsigned int> z_block_begin;
    std::vector<unsigned int> z_block_end;

    unsigned int z_num = apr.spatial_index_z_max(level);

    num_z_blocks = std::min(z_num,num_z_blocks);

    z_block_begin.resize(num_z_blocks);
    z_block_end.resize(num_z_blocks);

    unsigned int size_of_block = floor(z_num/num_z_blocks);

    unsigned int csum = 0; //cumulative sum

    for (int i = 0; i < num_z_blocks; ++i) {
        z_block_begin[i] = csum;
        z_block_end[i] = csum + size_of_block;

        csum+=size_of_block;

    }

    z_block_end[num_z_blocks-1] = z_num; //fill in the extras;

    unsigned int z_block;
    uint64_t counter = 0;

#pragma omp parallel for schedule(static) private(z_block) firstprivate(apr_iterator,neighbour_iterator) reduction(+:counter)

    for (z_block = 0; z_block < num_z_blocks; ++z_block) {

        for (unsigned int z = z_block_begin[z_block]; z < z_block_end[z_block]; ++z) {

            uint64_t begin = apr_iterator.particles_z_begin(level, z);
            uint64_t end = apr_iterator.particles_z_end(level, z);

            for (uint64_t particle_number = apr_iterator.particles_z_begin(level, z);
                 particle_number < apr_iterator.particles_z_end(level, z); ++particle_number) {

                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                counter++;

                if (apr_iterator.z() == z) {

                } else {
                    std::cout << "broken" << std::endl;
                }
            }
        }
    }
}


#endif //PARTPLAY_COMPRESSAPR_HPP
