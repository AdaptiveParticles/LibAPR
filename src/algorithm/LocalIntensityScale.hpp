//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_LOCAL_INTENSITY_SCALE_HPP
#define PARTPLAY_LOCAL_INTENSITY_SCALE_HPP

#include "data_structures/Mesh/PixelData.hpp"
#include "APRParameters.hpp"

class LocalIntensityScale {

public:
    template<typename T>
    void calc_abs_diff(const PixelData<T> &input_image, PixelData<T> &var);

    template<typename T>
    void calc_sat_mean_z(PixelData<T> &input, const size_t offset);

    template<typename T>
    void calc_sat_mean_x(PixelData<T> &input, const size_t offset);

    template<typename T>
    void calc_sat_mean_y(PixelData<T> &input, const size_t offset);

    void get_window(float &var_rescale, std::vector<int> &var_win, const APRParameters &par);
    template<typename T>
    void rescale_var_and_threshold(PixelData<T>& var,const float var_rescale, const APRParameters& par);
};

template<typename T>
inline void LocalIntensityScale::rescale_var_and_threshold(PixelData<T> &var, const float var_rescale, const APRParameters &par) {
    const float max_th = 60000.0;

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared)
    #endif
    for (size_t i = 0; i < var.mesh.size(); ++i) {
        float rescaled = var.mesh[i] * var_rescale;
        if (rescaled < par.sigma_th) {
            rescaled = (rescaled < par.sigma_th_max) ? max_th : par.sigma_th;
        }
        var.mesh[i] = rescaled;
    }
}

template<typename T>
inline void LocalIntensityScale::calc_abs_diff(const PixelData<T> &input_image, PixelData<T> &var) {
    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared)
    #endif
    for (size_t i = 0; i < input_image.mesh.size(); ++i) {
        var.mesh[i] = std::abs(var.mesh[i] - input_image.mesh[i]);
    }
}

/**
 * Compute the window size and set the rescaling factor
 * @param var_rescale
 * @param var_win
 * @param par
 */
inline void LocalIntensityScale::get_window(float& var_rescale, std::vector<int>& var_win, const APRParameters& par){
    const double rescale_store[6][8][3] = {{{22.2589,11.1016,8.3869},{25.1582,11.8891,9.03},   {30.6167,14.1926,9.7998}, {37.9925,16.9623,11.0813}, {41.9572,19.7608,12.4187}, {49.4073,21.5938,14.3182}, {56.1431,25.5847,14.931},  {60.8832,26.7749,21.1417}},
                                           {{33.8526,13.7341,8.6388},{35.9641,14.3717,9.0377}, {37.7067,15.5675,9.4528}, {41.051,16.9566,10.4615},  {44.7464,18.5599,11.8842}, {52.9174,21.2077,12.5411}, {57.0255,25.5539,14.365},  {66.6008,25.9241,15.3422}},
                                           {{54.7417,20.8889,12.075},{56.2098,21.7017,12.4667},{60.7089,21.9547,13.3998},{60.8244,24.465,13.6899},  {66.4504,25.6705,14.6285}, {80.5723,27.8058,16.2839}, {81.11,30.8859,17.3954},   {99.2642,36.412,20.9048}},
                                           {{73.1848,26.6382,15.251},{74.7952,27.9826,15.195}, {80.2526,28.357,16.1006}, {83.1349,30.2439,16.6018}, {89.1941,32.2252,16.3549}, {92.1605,33.0083,18.7942}, {93.753,37.0762,22.1166},  {111.0464,40.2133,23.4709}},
                                           {{88.5594,32.4552,18.167},{90.4278,32.3794,18.0685},{90.3799,32.4452,17.9486},{94.4443,33.649,18.7664},  {97.5961,35.3576,19.6612}, {101.4413,37.1114,19.9882},{112.5807,41.2781,21.134}, {118.4092,43.2994,23.881}},
                                           {{96.115,36.6599,18.6618},{97.3314,34.5362,18.5979},{94.3752,34.9931,18.598}, {104.1173,34.8291,19.3875},{100.2122,37.0696,19.6981},{106.0002,37.6281,20.4704},{111.4407,40.5927,20.9159},{118.9118,43.3307,22.6826}} };
    const double rescale_z[6][6] ={ {1,0.88158,0.74164,0.98504,0.97722,1.2746},
                                    {1.0782,1,0.90355,1.1194,1.081,1.2665},
                                    {1.3003,1.1901,1,1.2192,1.1557,1.2899},
                                    {1.1005,0.9449,0.73203,1,0.94031,1.0668},
                                    {1.2724,1.1063,0.85546,1.0792,1,1.1893},
                                    {1.0594,0.90244,0.62593,0.91011,0.811,1} };

    int psf_ind = std::max((float)(round(par.psfx/par.dx) - 1), 0.0f);
    psf_ind = std::min(psf_ind,5);

    const int win_1[] = {1,1,1,2,2,3};
    const int win_2[] = {2,3,4,4,5,6};

    int psf_indz = std::max((float)(round(par.psfz/par.dz) - 1), 0.0f);
    psf_indz = std::min(psf_indz, 5);

    var_win.resize(6);
    var_win[0] = win_1[psf_ind];
    var_win[1] = win_1[psf_ind];
    var_win[2] = win_1[psf_indz];
    var_win[3] = win_2[psf_ind];
    var_win[4] = win_2[psf_ind];
    var_win[5] = win_2[psf_indz];

    int window_ind_1 =  win_1[psf_ind] - 1;
    int window_ind_2 =  win_2[psf_ind] - 1;

    var_rescale = (float)rescale_store[psf_ind][window_ind_2][window_ind_1]*(float)rescale_z[psf_indz][psf_ind];
}

/**
 * Calculates a O(1) recursive mean using SAT.
 * @tparam T
 * @param input
 * @param offset
 */
template<typename T>
inline void LocalIntensityScale::calc_sat_mean_y(PixelData<T>& input, const size_t offset){
    const size_t z_num = input.z_num;
    const size_t x_num = input.x_num;
    const size_t y_num = input.y_num;

    std::vector<T> temp_vec(y_num);
    float divisor = 2 * offset + 1;

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) firstprivate(temp_vec)
    #endif
    for(size_t j = 0; j < z_num; ++j) {
        for(size_t i = 0; i < x_num; ++i){
            size_t index = j * x_num*y_num + i * y_num;

            //first pass over and calculate cumsum
            float temp = 0;
            for (size_t k = 0; k < y_num; ++k) {
                temp += input.mesh[index + k];
                temp_vec[k] = temp;
            }

            //handling boundary conditions (LHS)
            for (size_t k = 0; k <= offset; ++k) {
                input.mesh[index + k] = 0;
            }

            //second pass calculate mean
            for (size_t k = offset + 1; k < y_num; ++k) {
                input.mesh[index + k] = -temp_vec[k - offset - 1]/divisor;
            }

            //second pass calculate mean
            for (size_t k = 0; k < (y_num-offset); ++k) {
                input.mesh[index + k] += temp_vec[k + offset]/divisor;
            }

            float counter = 0;
            //handling boundary conditions (RHS)
            for (size_t k = (y_num - offset); k < (y_num); ++k) {
                counter++;
                input.mesh[index + k]*= divisor;
                input.mesh[index + k]+= temp_vec[y_num-1];
                input.mesh[index + k]*= 1.0/(divisor - counter);
            }

            //handling boundary conditions (LHS), need to rehandle the boundary
            for (size_t k = 1; k <= offset; ++k) {
                input.mesh[index + k] *= divisor/(k + offset + 1.0);
            }

            //end point boundary condition
            input.mesh[index] *= divisor/(offset + 1.0);
        }
    }
}

template<typename T>
inline void LocalIntensityScale::calc_sat_mean_x(PixelData<T>& input, const size_t offset) {
    const size_t z_num = input.z_num;
    const size_t x_num = input.x_num;
    const size_t y_num = input.y_num;

    std::vector<T> temp_vec(y_num*(2*offset + 1),0);

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) firstprivate(temp_vec)
    #endif
    for(size_t j = 0; j < z_num; j++) {
        size_t jxnumynum = j * x_num * y_num;

        for(size_t k = 0; k < y_num ; k++){
            temp_vec[k] = input.mesh[jxnumynum + k];
        }

        for(size_t i = 1; i < 2 * offset + 1; i++) {
            for(size_t k = 0; k < y_num; k++) {
                temp_vec[i*y_num + k] = input.mesh[jxnumynum + i*y_num + k] + temp_vec[(i-1)*y_num + k];
            }
        }

        // LHS boundary
        for(size_t i = 0; i < offset + 1; i++){
            for(size_t k = 0; k < y_num; k++) {
                input.mesh[jxnumynum + i * y_num + k] = (temp_vec[(i + offset) * y_num + k]) / (i + offset + 1);
            }
        }

        // middle
        size_t current_index = offset + 1;
        size_t index_modulo = 0;
        for(size_t i = offset + 1; i < x_num - offset; i++){
            // the current cumsum
            index_modulo = (current_index + offset) % (2*offset + 1); // current_index - offset - 1
            size_t previous_modulo = (current_index + offset - 1) % (2*offset + 1); // the index of previous cumsum

            for(size_t k = 0; k < y_num; k++) {
                float temp = input.mesh[jxnumynum + (i + offset)*y_num + k] + temp_vec[previous_modulo*y_num + k];
                input.mesh[jxnumynum + i*y_num + k] = (temp - temp_vec[index_modulo*y_num + k]) /
                                                      (2*offset + 1);
                temp_vec[index_modulo*y_num + k] = temp;
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }

        // RHS boundary
        current_index = (current_index + offset) % (2*offset + 1);
        for(size_t i = x_num - offset; i < x_num; i++){
            for(size_t k = 0; k < y_num; k++){
                input.mesh[jxnumynum + i*y_num + k] = (temp_vec[index_modulo*y_num + k] -
                                                       temp_vec[current_index*y_num + k]) / (x_num - i + offset);
            }
            current_index = (current_index + 1) % (2*offset + 1);
        }
    }
}

template<typename T>
inline void LocalIntensityScale::calc_sat_mean_z(PixelData<T>& input,const size_t offset) {
    const size_t z_num = input.z_num;
    const size_t x_num = input.x_num;
    const size_t y_num = input.y_num;

    std::vector<T> temp_vec(y_num*(2*offset + 1),0);
    size_t xnumynum = x_num * y_num;

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) firstprivate(temp_vec)
    #endif
    for(size_t i = 0; i < x_num; i++) {

        size_t iynum = i * y_num;

        //prefetching
        for(size_t k = 0; k < y_num ; k++){
            temp_vec[k] = input.mesh[iynum + k];
        }

        for(size_t j = 1; j < 2 * offset + 1; j++) {
            for(size_t k = 0; k < y_num; k++) {
                temp_vec[j*y_num + k] = input.mesh[j * xnumynum + iynum + k] + temp_vec[(j-1)*y_num + k];
            }
        }

        // LHS boundary
        for(size_t j = 0; j < offset + 1; j++){
            for(size_t k = 0; k < y_num; k++) {
                input.mesh[j * xnumynum + iynum + k] = (temp_vec[(j + offset)*y_num + k]) / (j + offset + 1);
            }
        }

        // middle
        size_t current_index = offset + 1;
        size_t index_modulo = 0;
        for(size_t j = offset + 1; j < z_num - offset; j++){

            index_modulo = (current_index + offset) % (2*offset + 1); // current_index - offset - 1
            size_t previous_modulo = (current_index + offset - 1) % (2*offset + 1); // the index of previous cumsum

            for(size_t k = 0; k < y_num; k++) {
                // the current cumsum
                float temp = input.mesh[(j + offset) * xnumynum + iynum + k] + temp_vec[previous_modulo*y_num + k];
                input.mesh[j * xnumynum + iynum + k] = (temp - temp_vec[index_modulo*y_num + k]) /
                                                       (2*offset + 1);
                temp_vec[index_modulo*y_num + k] = temp;
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }

        // RHS boundary
        current_index = (current_index + offset) % (2*offset + 1);
        for(size_t j = z_num - offset; j < z_num; j++){
            for(size_t k = 0; k < y_num; k++){
                input.mesh[j * xnumynum + iynum + k] = (temp_vec[index_modulo*y_num + k] -
                                                        temp_vec[current_index*y_num + k]) / (z_num - j + offset);
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }
    }
}

#endif //PARTPLAY_LOCAL_INTENSITY_SCALE_HPP
