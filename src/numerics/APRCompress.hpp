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

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/ParticleData.hpp"
#include "misc/APRTimer.hpp"
#include <cmath>
#include <vector>


//template<typename ImageType>
class APRCompress {

public:

    void set_background(float bkgrd){
        background = bkgrd;
    }

    void set_compression_type(int type){
        compress_type = type;
    }

    int get_compression_type(){
        return compress_type;
    }

    void set_e_factor(float e_){
        e = e_;
    }

    template<typename ImageType>
    void  compress(APR &apr, ParticleData<ImageType> &symbols) {
        APRTimer timer;
        timer.verbose_flag = false;

        APRTimer timer_total;
        timer_total.verbose_flag = true;

        timer_total.start_timer("total compress");

        timer.start_timer("allocation");

        timer.stop_timer();

        if(this->background==1) {
            this->background = apr.parameters.background_intensity_estimate - 2 * apr.parameters.noise_sd_estimate;
        }

        std::cout << background << std::endl;

        timer.start_timer("copy");

        ///////////////////////////
        ///
        /// Only perform variance stabilization on the highest level particles
        ///
        ///////////////////////////


        if (compress_type == 1){
            std::cout << "Variance Stabalization Only" << std::endl;
            //variance stabilization only, no subsequent prediction step. (all particles)

            int i = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
            for (i = 0; i < symbols.data.size(); ++i) {
                //symbols.data[i] = calculate_symbols<ImageType,float>(variance_stabilitzation<float>(symbols.data[i]));
                symbols.data[i] = (ImageType) variance_stabilitzation<float>(symbols.data[i]);
            }


        } else if (compress_type == 2){
            //variance stabilization and prediction step.
            std::cout << "Variance Stabalization followed by (x,y,z) prediction" << std::endl;

            ParticleData<float> predict_input(apr.total_number_particles());
            ParticleData<float> predict_output(apr.total_number_particles());

            predict_input.copy_parts(apr,symbols);
            timer.stop_timer();

            predict_input.map_inplace(apr,[this](const float a) { return variance_stabilitzation<float>(a); });

            for (unsigned int level = apr.level_min(); level <= apr.level_max(); ++level) {

                predict_particles_by_level(apr, level, predict_input, predict_output, predict_directions,
                                           num_blocks, 0,false);
            }

            timer.start_timer("predict symbols");
            //compute the symbols
            predict_output.map(apr,symbols,[this](const float a){return calculate_symbols<ImageType,float>(a);});
            timer.stop_timer();

        }


        timer_total.stop_timer();
    }



    template<typename U>
    void decompress(APR& apr,ParticleData<U>& symbols,uint64_t start=0){

        APRTimer timer;
        timer.verbose_flag = false;
        timer.start_timer("total decompress");


        if(this->background==1) {
            this->background = apr.parameters.background_intensity_estimate - 2 * apr.parameters.noise_sd_estimate;
        }

       if (compress_type == 1){
            int i = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
           for (i = start; i < symbols.data.size(); ++i) {
               //symbols.data[i] = (ImageType) inverse_variance_stabilitzation<float>(inverse_calculate_symbols<float,ImageType>(symbols.data[i]));
               symbols.data[i] = (U) inverse_variance_stabilitzation<float>(symbols.data[i]);
           }

            //invert the stabilization
            //predict_output.map_inplace(apr,[this](const float a) { return inverse_variance_stabilitzation<float>(a); });


        } else if (compress_type ==2){

           ParticleData<float> predict_input(apr.total_number_particles());
           ParticleData<float> predict_output(apr.total_number_particles());
           //turn symbols back to floats.
           symbols.map(apr,predict_input,[this](const U a){return inverse_calculate_symbols<float,U>(a);});

            for (unsigned int level = apr.level_min(); level <= apr.level_max(); ++level) {
                //predict_output.copy_parts(apr,predict_input,level);
                predict_particles_by_level(apr, level, predict_input, predict_output, predict_directions,
                                           num_blocks, 1,false);
            }

            //decode predict
            //invert the stabilization
            predict_output.map_inplace(apr,[this](const float a) { return inverse_variance_stabilitzation<float>(a); });

           //now truncate and copy to uint16t
           symbols.copy_parts(apr,predict_output);

        }

        timer.stop_timer();

    }

    void set_quantization_factor(float q_){
        q = q_;
    }

    float get_quantization_factor(){
        return q;
    }

private:

    unsigned int num_blocks = 4;

    float e = 1.6;
    float background = 1;
    float cnv = 65636/30000;
    float q = .5;
    int compress_type = 0;

    std::vector<unsigned int> predict_directions = {1,3,5};

    template<typename S>
    S variance_stabilitzation(const S input);

    template<typename S>
    S inverse_variance_stabilitzation(const S input);

    template<typename T,typename S>
    T calculate_symbols(S input);

    template<typename T,typename S>
    T inverse_calculate_symbols(S input);

    template<typename T,typename S>
    void predict_particles_by_level(APR& apr,const unsigned int level,ParticleData<T>& predict_input,ParticleData<S>& predict_output,std::vector<unsigned int>& predict_directions,unsigned int num_z_blocks,const int decode_encode_flag,const bool rounding = true);
};

 template<typename S>
S APRCompress::variance_stabilitzation(const S input){

    return (2.0f*sqrt(std::max((S) (input-background),(S)0)/cnv + pow(e,2)) - 2.0f*e)/q;

}

 template<typename S>
S APRCompress::inverse_variance_stabilitzation(const S input){

    float D = q*input + 2*e;

    if(D >= 2*e){
        D = (pow(D,2)/4.0 - pow(e,2))*cnv + background;
        return ((S) D);
    } else {
        return ((S) background);
    }

}

 template<typename T,typename S>
T APRCompress::calculate_symbols(S input){

    int16_t val = round(input);
    return 2*(abs(val)) + (val >> 15);
}


 template<typename T,typename S>
T APRCompress::inverse_calculate_symbols(S input){

    int16_t negative = ((uint16_t) input) % 2;

    return  (1 - 2 * negative) * ((input + negative) / 2);
}

 template<typename T,typename S>
void APRCompress::predict_particles_by_level(APR& apr,const unsigned int level,ParticleData<T>& predict_input,ParticleData<S>& predict_output,std::vector<unsigned int>& predict_directions,unsigned int num_z_blocks,const int decode_encode_flag,const bool rounding){
    //
    //  Performs prediction step using the predict directions in chunks of the dataset, given by z_index slice.
    //
    //  This allows parallelization of a recursive prediction process.
    //
    //  The decode and encode flag is used if it is predicting or reconstructing
    //

    APRTimer timer;
    timer.verbose_flag = false;

    timer.start_timer("iterator initialization");

    auto apr_iterator = apr.iterator();
    auto neighbour_iterator = apr.iterator();

    timer.stop_timer();

    // Compute the z-slice blocks that are to be computed over.
    std::vector<unsigned int> z_block_begin;
    std::vector<unsigned int> z_block_end;

    unsigned int z_num = apr.spatial_index_z_max(level);

    if(z_num > num_z_blocks*8) {

        num_z_blocks = std::min(z_num, num_z_blocks);

        z_block_begin.resize(num_z_blocks);
        z_block_end.resize(num_z_blocks);

        unsigned int size_of_block = floor(z_num / num_z_blocks);

        unsigned int csum = 0; //cumulative sum

        for (unsigned int i = 0; i < num_z_blocks; ++i) {
            z_block_begin[i] = csum;
            z_block_end[i] = csum + size_of_block;

            csum += size_of_block;
        }

        z_block_end[num_z_blocks - 1] = z_num; //fill in the extras;


    } else {
        num_z_blocks = 1;
        z_block_begin.resize(num_z_blocks);
        z_block_end.resize(num_z_blocks);

        z_block_begin[0] = 0;
        z_block_end[0] = z_num;
    }

    unsigned int z_block;

    int x = 0;

    timer.start_timer("loop");

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z_block) firstprivate(neighbour_iterator,apr_iterator,z_block_begin,z_block_end)
#endif
    for (z_block = 0; z_block < num_z_blocks; ++z_block) {

        for (unsigned int z = z_block_begin[z_block]; z < z_block_end[z_block]; ++z) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    float count_neighbours = 0;
                    float temp = 0;

                    //Handle the z_blocking, neighbours shoudl not be used on the zblock begin
                    if (z != z_block_begin[z_block]) {

                        //loop over all the neighbours and set the neighbour iterator to it
                        for (unsigned int f = 0; f < predict_directions.size(); ++f) {
                            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

                            unsigned int face = predict_directions[f];
                            apr_iterator.find_neighbours_in_direction(face);

                            for (int index = 0; index < apr_iterator.number_neighbours_in_direction(face); ++index) {
                                // on each face, there can be 0-4 neighbours accessed by index
                                if (neighbour_iterator.set_neighbour_iterator(apr_iterator, face, index)) {
                                    //will return true if there is a neighbour defined

                                    if (neighbour_iterator.level() <= apr_iterator.level()) {
                                        if (decode_encode_flag == 0) {
                                            //Encode
                                            temp += floor(predict_input[neighbour_iterator]);
                                        } else if (decode_encode_flag == 1) {
                                            //Decode
                                            temp += predict_output[neighbour_iterator];
                                        }
                                        count_neighbours++;
                                    }
                                }
                            }
                        }
                    }

                    if (rounding) {

                        if (decode_encode_flag == 0) {
                            //Encode
                            if (count_neighbours > 0) {
                                predict_output[apr_iterator] = round(
                                        predict_input[apr_iterator] - temp / count_neighbours);
                            } else {
                                predict_output[apr_iterator] = round(predict_input[apr_iterator]);
                            }
                        } else if (decode_encode_flag == 1) {
                            //Decode
                            if (count_neighbours > 0) {

                                predict_output[apr_iterator] = round(
                                        predict_input[apr_iterator] + temp / count_neighbours);

                            } else {

                                predict_output[apr_iterator] = round(predict_input[apr_iterator]);
                            }

                        }
                    } else {
                        if (decode_encode_flag == 0) {
                            //Encode
                            if (count_neighbours > 0) {
                                predict_output[apr_iterator] =
                                        floor(predict_input[apr_iterator]) - floor(temp / count_neighbours);
                            } else {
                                predict_output[apr_iterator] = floor(predict_input[apr_iterator]);
                            }
                        } else if (decode_encode_flag == 1) {
                            //Decode
                            if (count_neighbours > 0) {

                                predict_output[apr_iterator] =
                                        predict_input[apr_iterator] + floor(temp / count_neighbours);

                            } else {

                                predict_output[apr_iterator] = predict_input[apr_iterator];
                            }
                        }
                    }
                }
            }
        }
    }

    timer.stop_timer();
}


#endif //PARTPLAY_COMPRESSAPR_HPP
