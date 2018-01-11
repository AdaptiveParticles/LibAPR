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

class CompressAPR {

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
    void predict_particles_by_level(APR<U>& apr,unsigned int level,ExtraPartCellData<T>& predict_input,ExtraPartCellData<S>& predict_output,std::vector<unsigned int>& predict_directions);

};

template<typename T,typename S>
T CompressAPR::variance_stabilitzation(S input){

    return (2*sqrt(std::max((T) (input-background),(T)0)/cnv + pow(e,2)) - 2*e)/q;

};

template<typename T,typename S>
T CompressAPR::inverse_variance_stabilitzation(S input){

    float D = q*input + 2*e;

    if(D >= 2*e){
        D = (pow(D,2)/4.0 - pow(e,2))*cnv + background;
        return ((T) D);
    } else {
        return ((T) background);
    }

};

template<typename T,typename S>
T CompressAPR::calculate_symbols(S input){

    int16_t val = input;
    return 2*(abs(val)) + (val >> 15);
};


template<typename T,typename S>
T CompressAPR::inverse_calculate_symbols(S input){

    int16_t negative = input % 2;

    return  (1 - 2 * negative) * ((input + negative) / 2);
};

template<typename T,typename S,typename U>
void CompressAPR::predict_particles_by_level(APR<U>& apr,unsigned int level,ExtraPartCellData<T>& predict_input,ExtraPartCellData<S>& predict_output,std::vector<unsigned int>& predict_directions){

    for (apr.begin(); apr.end() != 0; apr.it_forward()) {

        //get the minus neighbours (1,3,5)

        //now we only update the neighbours, and directly access them through a neighbour iterator
        apr.update_all_neighbours();

        float counter = 0;
        float temp = 0;

        //loop over all the neighbours and set the neighbour iterator to it
        for (int f = 0; f < dir.size(); ++f) {
            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
            unsigned int face = dir[f];

            for (int index = 0; index < apr.number_neighbours_in_direction(face); ++index) {
                // on each face, there can be 0-4 neighbours accessed by index
                if(neigh_it.set_neighbour_iterator(apr, face, index)){
                    //will return true if there is a neighbour defined
                    if(neigh_it.depth() <= apr.depth()) {
                        temp += neigh_it(prediction_reverse);
                        counter++;
                    }

                }
            }
        }

        if(counter > 0){
            apr(prediction_reverse) = apr(unsymbol) + temp/counter;
        } else {
            apr(prediction_reverse) = apr(unsymbol);
        }

    }


};


#endif //PARTPLAY_COMPRESSAPR_HPP
