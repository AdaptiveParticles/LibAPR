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

    APRCompress(){

    };

    template<typename U>
    void compress(APR<U>& apr) {


        ExtraPartCellData<float> predict_input(apr);
        ExtraPartCellData<float> predict_output(apr);

        std::vector<unsigned int> predict_directions = {1,3,5};

        unsigned int level = apr.level_max();

        unsigned int num_blocks = 10;

        predict_input.copy_parts(apr.particles_int);

        //predict_input.map_inplace([this](const float a){return variance_stabilitzation<float>(a);},apr.level_max());

        //predict_input.map_inplace([this](const float a){return calculate_symbols<float>(a);},apr.level_max());


        //predict_particles_by_level(apr,level,predict_output,predict_input,predict_directions,num_blocks,0);

        //predict_particles_by_level(apr,level,predict_output,predict_input,predict_directions,num_blocks,1);

    }



private:

    float e = 1.6;
    float background = 1;
    float cnv = 65636/30000;
    float q = .5;

    template<typename S>
    S variance_stabilitzation(const S input);

    template<typename T,typename S>
    T inverse_variance_stabilitzation(S input);

    template<typename T,typename S>
    T calculate_symbols(S input);

    template<typename T,typename S>
    T inverse_calculate_symbols(S input);

    template<typename T,typename S,typename U>
    void predict_particles_by_level(APR<U>& apr,const unsigned int level,ExtraPartCellData<T>& predict_input,ExtraPartCellData<S>& predict_output,std::vector<unsigned int>& predict_directions,unsigned int num_z_blocks,const int decode_encode_flag);

};

template<typename S>
S APRCompress::variance_stabilitzation(const S input){

    return (2*sqrt(std::max((S) (input-background),(S)0)/cnv + pow(e,2)) - 2*e)/q;

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
void APRCompress::predict_particles_by_level(APR<U>& apr,const unsigned int level,ExtraPartCellData<T>& predict_input,ExtraPartCellData<S>& predict_output,std::vector<unsigned int>& predict_directions,unsigned int num_z_blocks,const int decode_encode_flag){
    //
    //  Performs prediction step using the predict directions in chunks of the dataset, given by z_index slice.
    //
    //  This allows parallelization of a recursive prediction process.
    //
    //  The decode and encode flag is used if it is predicting or reconstructing
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


    APR_timer timer;
    timer.verbose_flag = true;

    timer.start_timer("loop");

#pragma omp parallel for schedule(dynamic) private(z_block) firstprivate(apr_iterator,neighbour_iterator,z_block_begin,z_block_end)
    for (z_block = 0; z_block < num_z_blocks; ++z_block) {

        for (unsigned int z = z_block_begin[z_block]; z < z_block_end[z_block]; ++z) {

            for (uint64_t particle_number = apr_iterator.particles_z_begin(level, z);
                 particle_number < apr_iterator.particles_z_end(level, z); ++particle_number) {

                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                apr_iterator.update_all_neighbours();

                float count_neighbours = 0;
                float temp = 0;

                //Handle the z_blocking, neighbours shoudl not be used on the zblock begin
                if(z != z_block_begin[z_block]) {

                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int f = 0; f < predict_directions.size(); ++f) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

                        unsigned int face = predict_directions[f];
                        apr_iterator.update_direction_neighbours(f);

                        for (int index = 0; index < apr_iterator.number_neighbours_in_direction(face); ++index) {
                            // on each face, there can be 0-4 neighbours accessed by index
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, face, index)) {
                                //will return true if there is a neighbour defined

                                if (neighbour_iterator.depth() <= apr_iterator.depth()) {
                                    if(decode_encode_flag == 0) {
                                        //Encode
                                        temp += neighbour_iterator(predict_input);
                                    } else if (decode_encode_flag == 1) {
                                        //Decode
                                        temp += neighbour_iterator(predict_output);
                                    }
                                    count_neighbours++;
                                }
                            }
                        }
                    }
                }

                if(decode_encode_flag == 0) {
                    //Encode
                    if (count_neighbours > 0) {
                        apr_iterator(predict_output) =  apr_iterator(predict_input) - temp/count_neighbours;
                    } else {
                        apr_iterator(predict_output) = apr_iterator(predict_input);
                    }
                } else if (decode_encode_flag == 1) {
                    //Decode
                    if(count_neighbours > 0){
                        apr_iterator(predict_output) = apr(predict_input) + temp/count_neighbours;
                     } else {
                        apr_iterator(predict_output) = apr(predict_input);
                    }

                }


            }
        }
    }

    timer.stop_timer();

}


//        //now we only update the neighbours, and directly access them through a neighbour iterator
//        apr.update_all_neighbours();
//
//        float counter = 0;
//        float temp = 0;
//
//        //loop over all the neighbours and set the neighbour iterator to it
//        for (int f = 0; f < dir.size(); ++f) {
//            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//            unsigned int face = dir[f];
//
//            for (int index = 0; index < apr.number_neighbours_in_direction(face); ++index) {
//                // on each face, there can be 0-4 neighbours accessed by index
//                if(neigh_it.set_neighbour_iterator(apr, face, index)){
//                    //will return true if there is a neighbour defined
//                    if(neigh_it.depth() <= apr.depth()) {
//
//                        temp += neigh_it(var_scaled);
//                        counter++;
//                    }
//
//                }
//            }
//        }
//
//        if(counter > 0){
//            apr(prediction) = apr(var_scaled) - temp/counter;
//        } else {
//            apr(prediction) = apr(var_scaled);
//        }


#endif //PARTPLAY_COMPRESSAPR_HPP
