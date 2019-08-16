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

#include "misc/APRTimer.hpp"
#include <cmath>
#include <vector>
#include "data_structures/Mesh/PixelData.hpp"

class APRCompress {

public:

    APRTimer timer;

    void set_background(float bkgrd){
        background = bkgrd;
    }

    float get_background(){
        return background;
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
    void  compress(VectorData<ImageType>& symbols) {


        timer.start_timer("total compress");

        if (compress_type == 1){
            std::cout << "Variance Stabalization Only" << std::endl;
            //variance stabilization only, no subsequent prediction step. (all particles)

            size_t i = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
            for (i = 0; i < symbols.size(); ++i) {
                symbols[i] = calculate_symbols<ImageType,float>(variance_stabilitzation<float>(symbols[i]));
            }
        }

        timer.stop_timer();
    }


    template<typename ImageType>
    void decompress(VectorData<ImageType>& symbols,uint64_t start=0){


        timer.start_timer("total decompress");

       if (compress_type == 1){
            size_t i = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
           for (i = start; i < symbols.size(); ++i) {
               symbols[i] = (ImageType) inverse_variance_stabilitzation<float>(inverse_calculate_symbols<float,ImageType>(symbols[i]));
           }

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

//    unsigned int num_blocks = 4;

    float e = 1.6;
    float background = 1;
    float cnv = 65636/30000;
    float q = .5;
    int compress_type = 0;

    //std::vector<unsigned int> predict_directions = {1,3,5};

public:

    template<typename S>
    S variance_stabilitzation(const S input);

    template<typename S>
    S inverse_variance_stabilitzation(const S input);

    template<typename T,typename S>
    T calculate_symbols(S input);

    template<typename T,typename S>
    T inverse_calculate_symbols(S input);

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




#endif //PARTPLAY_COMPRESSAPR_HPP
