//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_LOCAL_INTENSITY_SCALE_HPP
#define PARTPLAY_LOCAL_INTENSITY_SCALE_HPP

#include "data_structures/Mesh/PixelData.hpp"
#include "APRParameters.hpp"

class LocalIntensityScale {

protected:

    bool active_y = true;
    bool active_x = true;
    bool active_z = true;

    int number_active_dimensions = 3;


public:

    APRTimer timer;

void get_local_intensity_scale(PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, const APRParameters &par) {
    //
    //  Calculate the Local Intensity Scale (You could replace this method with your own)
    //
    //  Input: full sized image.
    //
    //  Output: down-sampled Local Intensity Scale (h) (Due to the Equivalence Optimization we only need down-sampled values)
    //

    float var_rescale;
    std::vector<int> var_win;
    get_window_alt(var_rescale, var_win, par, local_scale_temp);

    int win_y = var_win[0];
    int win_x = var_win[1];
    int win_z = var_win[2];
    int win_y2 = var_win[3];
    int win_x2 = var_win[4];
    int win_z2 = var_win[5];

    bool constant_scale = false;

    if(par.constant_intensity_scale || (number_active_dimensions == 0)){
        constant_scale = true; //include the case where the local intensity scale doesn't make sense due to the image being to small. (This is for just edge cases and sanity checking)
    }

    if (!constant_scale) {

        timer.start_timer("copy_intensities_from_bsplines");
        //copy across the intensities

        int y_num_t = local_scale_temp.y_num;
        int x_num_t = local_scale_temp.x_num;
        int z_num_t = local_scale_temp.z_num;

        //Addded
        PixelData<float> temp_copy;
        temp_copy.init(y_num_t,x_num_t,z_num_t);

        if(par.reflect_bc_lis) {

            temp_copy.copyFromMesh(local_scale_temp);
            timer.stop_timer();

            paddPixels(temp_copy, local_scale_temp, std::max(win_y, win_y2), std::max(win_x, win_x2),
                       std::max(win_z, win_z2));

            paddPixels(temp_copy, local_scale_temp2, std::max(win_y, win_y2), std::max(win_x, win_x2),
                       std::max(win_z, win_z2));
        } else {

            local_scale_temp2.copyFromMesh(local_scale_temp);

        }


        if (active_y) {
            timer.start_timer("calc_sat_mean_y");
            calc_sat_mean_y(local_scale_temp, win_y);
            timer.stop_timer();
        }
        if (active_x) {
            timer.start_timer("calc_sat_mean_x");
            calc_sat_mean_x(local_scale_temp, win_x);
            timer.stop_timer();
        }
        if (active_z) {
            timer.start_timer("calc_sat_mean_z");
            calc_sat_mean_z(local_scale_temp, win_z);
            timer.stop_timer();
        }

        timer.start_timer("second_pass_and_rescale");
        //calculate abs and subtract from original
        calc_abs_diff(local_scale_temp2, local_scale_temp);
        //Second spatial average
        if (active_y) {
            calc_sat_mean_y(local_scale_temp, win_y2);
        }
        if (active_x) {
            calc_sat_mean_x(local_scale_temp, win_x2);
        }
        if (active_z) {
            calc_sat_mean_z(local_scale_temp, win_z2);
        }

        rescale_var(local_scale_temp, var_rescale);
        timer.stop_timer();

        if(par.reflect_bc_lis) {

            local_scale_temp.swap(local_scale_temp2);

            unpaddPixels(local_scale_temp2, local_scale_temp, y_num_t, x_num_t,
                         z_num_t);

            local_scale_temp2.initWithResize(y_num_t, x_num_t, z_num_t);
            local_scale_temp2.copyFromMesh(temp_copy);
        }

    } else {

        float min_val = 660000;
        double sum = 0;
        float tmp;

        for(size_t i=0; i<local_scale_temp.mesh.size(); ++i) {
            tmp = local_scale_temp.mesh[i];

            sum += tmp;

            if(tmp < min_val) {
                min_val = tmp;
            }
        }

        float numel = (float) (local_scale_temp.y_num * local_scale_temp.x_num * local_scale_temp.z_num);
        float scale_val = (float) (sum / numel - min_val);

        for(size_t i = 0; i<local_scale_temp.mesh.size(); ++i) {
            local_scale_temp.mesh[i] = scale_val;
        }
    }

}

    template<typename T>
    void calc_abs_diff(const PixelData<T> &input_image, PixelData<T> &var);

    template<typename T>
    void calc_sat_mean_z(PixelData<T> &input, const size_t offset, bool boundaryReflect = false);

    template<typename T>
    void calc_sat_mean_x(PixelData<T> &input, const size_t offset, bool boundaryReflect = false);

    template<typename T>
    void calc_sat_mean_y(PixelData<T> &input, const size_t offset);

    void get_window(float &var_rescale, std::vector<int> &var_win, const APRParameters &par);

    template<typename T>
    void get_window_alt(float& var_rescale, std::vector<int>& var_win, const APRParameters& par, const PixelData<T>& img);

    template<typename T>
    void rescale_var(PixelData<T>& var,const float var_rescale);
};

template<typename T>
inline void LocalIntensityScale::rescale_var(PixelData<T> &var, const float var_rescale) {

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared)
    #endif
    for (size_t i = 0; i < var.mesh.size(); ++i) {
        float rescaled = var.mesh[i] * var_rescale;

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
 * Compute the window size and set the rescaling factor. Rescaling factors recomputed by Joel Jonsson.
 * Assuming isotropy!
 *
 * @param var_rescale
 * @param var_win
 * @param par
 * @param temp_img (image already allocated to correct size to compute the local intensity scale)
 */
template<typename T>
inline void LocalIntensityScale::get_window_alt(float& var_rescale, std::vector<int>& var_win, const APRParameters& par,const PixelData<T>& temp_img){

    const double rescale_store_3D[6] = {12.8214, 26.1256, 40.2795, 23.3692, 36.2061, 27.0385};
    const double rescale_store_2D[6] = {13.2421, 28.7069, 52.0385, 24.4272, 34.9565, 21.1891};
    const double rescale_store_1D[6] = {13.9040, 31.2843, 57.4037, 30.3767, 45.9930, 29.0890};
    // rescale_store_1D[3] is bimodal with a second mode at 45.4407

    int psf_ind = std::max((float)(round(par.psfx/par.dx) - 1), 0.0f);
    psf_ind = std::min(psf_ind,5);

    const int win_1[] = {1,1,1,2,2,3};
    const int win_2[] = {2,3,4,4,5,6};

    int win_val = std::max(win_1[psf_ind],win_2[psf_ind]);

    var_win.resize(6,0);

    if ( (int) temp_img.y_num > win_val) {
        active_y = true;
        var_win[0] = win_1[psf_ind];

        var_win[3] = win_2[psf_ind];
    } else {
        active_y = false;
    }

    if ((int) temp_img.x_num > win_val) {
        active_x = true;
        var_win[1] = win_1[psf_ind];
        var_win[4] = win_2[psf_ind];
    } else {
        active_x = false;
    }

    if ((int) temp_img.z_num > win_val) {
        active_z = true;
        var_win[2] = win_1[psf_ind];
        var_win[5] = win_2[psf_ind];
    } else {
        active_z = false;
    }

    number_active_dimensions = active_y + active_x + active_z;


    if( number_active_dimensions == 3 ) {
        var_rescale = (float)rescale_store_3D[psf_ind];
    } else if( number_active_dimensions == 2 ) {
        var_rescale = (float)rescale_store_2D[psf_ind];
    } else {
        var_rescale = (float)rescale_store_1D[psf_ind];
    }
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
inline void LocalIntensityScale::calc_sat_mean_x(PixelData<T>& input, const size_t offset, bool boundaryReflect) {

    const size_t z_num = input.z_num;
    const size_t x_num = input.x_num;
    const size_t y_num = input.y_num;

    const size_t divisor = offset + 1 + offset;
    std::vector<T> circularBuffer(y_num * divisor, 0);
    std::vector<T> sum(y_num, 0);

    auto &mesh = input.mesh;
    const size_t dimLen = x_num;

    if (dimLen < offset) {
        throw std::runtime_error("offset cannot be bigger than processed dimension length!");
    }

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) firstprivate(circularBuffer, sum)
#endif
    for (size_t j = 0; j < z_num; j++) {
        size_t jxnumynum = j * x_num * y_num;

        size_t count = 0; // counts number of active elements in filter
        size_t currElementOffset = 0; // offset of element in processed dimension
        size_t nextElementOffset = 1;
        size_t saveElementOffset = 0; // offset used to finish RHS boundary

        // Clear buffers so they can be reused in next 'z_num' loop
        std::fill(sum.begin(), sum.end(), 0); // Clear 'sum; vector before next loop
        std::fill(circularBuffer.begin(), circularBuffer.end(), 0);

        // saturate circular buffer with #offset elements since it will allow to calculate first element value on LHS
        while (count <= offset) {
            for (size_t k = 0; k < y_num; ++k) {
                auto v = mesh[jxnumynum + currElementOffset * y_num + k];
                sum[k] += v;
                circularBuffer[count * y_num + k] = v;
                if (boundaryReflect && count > 0) { circularBuffer[(2 * offset - count + 1) * y_num + k] = v; sum[k] += v;}
            }

            currElementOffset += nextElementOffset;
            ++count;
        }

        if (boundaryReflect) {
            count += offset; // elements in above loop in range [1, offset] were summed twice
        }

        // Pointer in circular buffer
        int beginPtr = (offset + 1) % divisor;

        // main loop going through all elements in range [0, x_num - 1 - offset], so till last element that
        // does not need handling RHS for offset '^'
        // x x x x ... x x x x x x x
        //                 o o ^ o o
        //
        const size_t lastElement = x_num - 1 - offset;
        for (size_t x = 0; x <= lastElement; ++x) {
            // Calculate and save currently processed element and move to the new one
            for (size_t k = 0; k < y_num; ++k) {
                mesh[jxnumynum + saveElementOffset * y_num + k] = sum[k] / count;
            }
            saveElementOffset += nextElementOffset;

            // There is no more elements to process in that loop, all stuff left to be processed is already in 'circularBuffer' buffer
            if (x == lastElement) break;

            for (size_t k = 0; k < y_num; ++k) {
                // Read new element
                T v = mesh[jxnumynum + currElementOffset * y_num + k];

                // Update sum to cover [-offset, offset] of currently processed element
                sum[k] -= circularBuffer[beginPtr * y_num + k];
                sum[k] += v;

                // Store new element in circularBuffer
                circularBuffer[beginPtr * y_num + k] = v;
            }

            // Move to next elements to read and in circular buffer
            count  = std::min(count + 1, divisor);
            beginPtr = (beginPtr + 1) % divisor;
            currElementOffset += nextElementOffset;
        }

        // boundaryPtr is used only in boundaryReflect mode, adding (2*offset+1) makes it always non-negative value
        int boundaryPtr = (beginPtr - 1 - 1 + (2*offset+1)) % divisor;

        // Handle last #offset elements on RHS
        while(saveElementOffset < currElementOffset) {
            // If filter length is too big in comparison to processed dimension
            // do not decrease 'count' since 'sum' of filter elements contains all elements from
            // processed dimension:
            // dim elements:        xxxxxx
            // filter elements:   oooooo^ooooo   (o - offset elements, ^ - middle of the filter)
            bool removeElementFromFilter = dimLen - (currElementOffset - saveElementOffset)/nextElementOffset > offset;

            if (removeElementFromFilter) {
                if (!boundaryReflect) count = count - 1;
            }

            for (size_t k = 0; k < y_num; ++k) {
                if (removeElementFromFilter || boundaryReflect) {
                    sum[k] -= circularBuffer[beginPtr * y_num + k];
                }

                if (boundaryReflect) {
                    sum[k] += circularBuffer[boundaryPtr * y_num + k];
                }

                mesh[jxnumynum + saveElementOffset * y_num + k] = sum[k] / count;
            }

            boundaryPtr = (boundaryPtr - 1 + (2*offset+1)) % divisor;
            beginPtr = (beginPtr + 1) % divisor;
            saveElementOffset += nextElementOffset;
        }
    }
}

template<typename T>
inline void LocalIntensityScale::calc_sat_mean_z(PixelData<T>& input, const size_t offset, bool boundaryReflect) {

    const size_t z_num = input.z_num;
    const size_t x_num = input.x_num;
    const size_t y_num = input.y_num;

    const size_t divisor = offset + 1 + offset;
    std::vector<T> circularBuffer(y_num * divisor, 0);
    std::vector<T> sum(y_num, 0);

    auto &mesh = input.mesh;
    size_t dimLen = z_num;

    if (dimLen < offset) {
        throw std::runtime_error("offset cannot be bigger than processed dimension length!");
    }

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) firstprivate(circularBuffer, sum)
#endif
    for (size_t j = 0; j < x_num; j++) {
        size_t jxnumynum = j * y_num;

        size_t count = 0; // counts number of active elements in filter
        size_t currElementOffset = 0; // offset of element in processed dimension
        size_t nextElementOffset = x_num;
        size_t saveElementOffset = 0; // offset used to finish RHS boundary

        // Clear buffers so they can be reused in next 'x_num' loop
        std::fill(sum.begin(), sum.end(), 0); // Clear 'sum; vector before next loop
        std::fill(circularBuffer.begin(), circularBuffer.end(), 0);

        // saturate circular buffer with #offset elements since it will allow to calculate first element value on LHS
        while(count <= offset) {
            for (size_t k = 0; k < y_num; ++k) {
                auto v = mesh[jxnumynum + currElementOffset * y_num + k];
                sum[k] += v;
                circularBuffer[count * y_num + k] = v;
                if (boundaryReflect && count > 0) { circularBuffer[(2 * offset - count + 1) * y_num + k] = v; sum[k] += v;}
            }

            currElementOffset += nextElementOffset;
            ++count;
        }

        if (boundaryReflect) {
            count += offset; // elements in above loop in range [1, offset] were summed twice
        }

        // Pointer in circular buffer
        int beginPtr = (offset + 1) % divisor;

        // main loop going through all elements in range [0, z_num - 1 - offset], so till last element that
        // does not need handling RHS for offset '^'
        // x x x x ... x x x x x x x
        //                 o o ^ o o
        //
        const size_t lastElement = z_num - 1 - offset;
        for (size_t z = 0; z <= lastElement; ++z) {
            // Calculate and save currently processed element and move to the new one
            for (size_t k = 0; k < y_num; ++k) {
                mesh[jxnumynum + saveElementOffset * y_num + k] = sum[k] / count;
            }
            saveElementOffset += nextElementOffset;

            // There is no more elements to process in that loop, all stuff left to be processed is already in 'circularBuffer' buffer
            if (z == lastElement) break;

            for (size_t k = 0; k < y_num; ++k) {
                // Read new element
                T v = mesh[jxnumynum + currElementOffset * y_num + k];

                // Update sum to cover [-offset, offset] of currently processed element
                sum[k] -= circularBuffer[beginPtr * y_num + k];
                sum[k] += v;

                // Save new element
                circularBuffer[beginPtr * y_num + k] = v;
            }

            // Move to next elements to read and in circular buffer
            count  = std::min(count + 1, divisor);
            beginPtr = (beginPtr + 1) % divisor;
            currElementOffset += nextElementOffset;
        }

        // boundaryPtr is used only in boundaryReflect mode, adding (2*offset+1) makes it always non-negative value
        int boundaryPtr = (beginPtr - 1 - 1 + (2*offset+1)) % divisor;

        // Handle last #offset elements on RHS
        while(saveElementOffset < currElementOffset) {
            // If filter length is too big in comparison to processed dimension
            // do not decrease 'count' since 'sum' of filter elements contains all elements from
            // processed dimension:
            // dim elements:        xxxxxx
            // filter elements:   oooooo^ooooo   (o - offset elements, ^ - middle of the filter)
            bool removeElementFromFilter = dimLen - (currElementOffset - saveElementOffset)/nextElementOffset > offset;

            if (removeElementFromFilter) {
                if (!boundaryReflect) count = count - 1;
            }

            for (size_t k = 0; k < y_num; ++k) {
                if (removeElementFromFilter || boundaryReflect) {
                    sum[k] -= circularBuffer[beginPtr * y_num + k];
                }

                if (boundaryReflect) {
                    sum[k] += circularBuffer[boundaryPtr * y_num + k];
                }

                mesh[jxnumynum + saveElementOffset * y_num + k] = sum[k] / count;
            }

            boundaryPtr = (boundaryPtr - 1 + (2*offset+1)) % divisor;
            beginPtr = (beginPtr + 1) % divisor;
            saveElementOffset += nextElementOffset;
        }
    }
}

#endif
