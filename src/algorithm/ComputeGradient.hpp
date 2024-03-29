//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_GRADIENT_HPP
#define PARTPLAY_GRADIENT_HPP

#include "data_structures/Mesh/PixelData.hpp"
#include "io/TiffUtils.hpp"

#ifdef HAVE_OPENMP
	#include "omp.h"
#endif
#include "algorithm/APRParameters.hpp"
#include "misc/APRTimer.hpp"

class ComputeGradient {

public:

    APRTimer timer;

    template<typename ImageType,typename tempType>
    inline void get_gradient(PixelData<ImageType> &image_temp, PixelData<ImageType> &grad_temp, PixelData<tempType> &local_scale_temp, const APRParameters &par);


    template<typename T>
    void get_smooth_bspline_3D(PixelData<T> &input, float lambda);

// Calculate inverse B-Spline Transform

    template<typename T>
    void calc_inv_bspline_y(PixelData<T> &input);

    template<typename T>
    void calc_inv_bspline_x(PixelData<T> &input);

    template<typename T>
    void calc_inv_bspline_z(PixelData<T> &input);

    struct three_temps {
        float temp_1, temp_2, temp_3;
    };

// Gradient computation

    template<typename S>
    void
    calc_bspline_fd_ds_mag(const PixelData<S> &input, PixelData<S> &grad, const float hx, const float hy, const float hz);

    template<typename T>
    void mask_gradient(PixelData<T>& grad_ds,const APRParameters& par);

    template<typename T,typename S>
    void threshold_gradient(PixelData<T> &grad, const PixelData<S> &img, const float Ip_th);

    template<typename T>
    void bspline_filt_rec_y(PixelData<T> &image, float lambda, float tol, int k0Len = -1);
    template<typename T>
    void bspline_filt_rec_x(PixelData<T> &image, float lambda, float tol, int k0Len = -1);
    template<typename T>
    void bspline_filt_rec_z(PixelData<T> &image, float lambda, float tol, int k0Len = -1);

    inline float impulse_resp(float k, float rho, float omg);

    inline float impulse_resp_back(float k, float rho, float omg, float gamma, float c0);

};

template<typename ImageType,typename tempType>
inline void ComputeGradient::get_gradient(PixelData<ImageType> &image_temp, PixelData<ImageType> &grad_temp, PixelData<tempType> &local_scale_temp, const APRParameters &par) {
    //  Bevan Cheeseman 2018
    //  Calculate the gradient from the input image. (You could replace this method with your own)
    //  Input: full sized image.
    //  Output: down-sampled by 2 gradient magnitude (Note, the gradient is calculated at pixel level then maximum down sampled within the loops below)

    timer.start_timer("smooth_bspline");
    if(par.lambda > 0) {
        get_smooth_bspline_3D(image_temp, par.lambda);
    }
    timer.stop_timer();


#ifdef HAVE_LIBTIFF
    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "smooth_bsplines.tif", image_temp);
    }
#endif

    timer.start_timer("calc_bspline_fd_mag_ds");
    calc_bspline_fd_ds_mag(image_temp,grad_temp,par.dx,par.dy,par.dz);
    timer.stop_timer();


    timer.start_timer("down-sample_b-spline");
    downsample(image_temp, local_scale_temp,
               [](const float &x, const float &y) -> float { return x + y; },
               [](const float &x) -> float { return x / 8.0; });
    timer.stop_timer();

    if(par.lambda > 0){
        if(image_temp.y_num > 2) {
            timer.start_timer("calc_inv_bspline_y");
            calc_inv_bspline_y(local_scale_temp);
            timer.stop_timer();
        }
        if(image_temp.x_num > 2) {
            timer.start_timer("calc_inv_bspline_x");
            calc_inv_bspline_x(local_scale_temp);
            timer.stop_timer();
        }
        if(image_temp.z_num > 2) {
            timer.start_timer("calc_inv_bspline_z");
            calc_inv_bspline_z(local_scale_temp);
            timer.stop_timer();
        }
    }

}



template<typename T>
void ComputeGradient::mask_gradient(PixelData<T>& grad_ds, const APRParameters& par){
    //
    //  Bevan Cheeseman 2018
    //
    //  Loads in a tiff image file and masks the gradient according to it
    //
    //

    PixelData<uint16_t> temp_mask;
    PixelData<uint16_t> temp_ds;

    std::string file_name = par.input_dir + par.mask_file;
#ifdef HAVE_LIBTIFF

    //TiffUtils::getMesh(file_name, temp_mask);

    temp_mask = TiffUtils::getMesh<uint16_t>(file_name);


    downsample(temp_mask, temp_ds,
               [](const T &x, const T &y) -> T { return std::max(x, y); },
               [](const T &x) -> T { return x; },
               true);

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared)
    #endif
    for (size_t i = 0; i < grad_ds.mesh.size(); ++i) {
        if (temp_ds.mesh[i]==0) {
            grad_ds.mesh[i] = 0;
        }
    }
#endif //HAVE_LIBTIFF
}

template<typename T,typename S>
void ComputeGradient::threshold_gradient(PixelData<T> &grad, const PixelData<S> &img, const float Ip_th){
    //
    //  Bevan Cheeseman 2016
    //

    #ifdef HAVE_OPENMP
	#pragma omp parallel for
    #endif
    for (size_t i = 0; i < img.mesh.size(); ++i) {
        if (img.mesh[i] <= Ip_th) { grad.mesh[i] = 0; }
    }
}

template<typename T>
void ComputeGradient::get_smooth_bspline_3D(PixelData<T>& input, float lambda) {
    //
    //  Gets smoothing bspline co-efficients for 3D
    //
    //

    APRTimer spline_timer;
    spline_timer.verbose_flag = false;

    float tol = 0.0001;

    if(input.y_num > 2) {
        //Y direction bspline
        spline_timer.start_timer("bspline_filt_rec_y");
        bspline_filt_rec_y(input, lambda, tol);
        spline_timer.stop_timer();
    }

    if(input.x_num > 2) {
        //X direction bspline
        spline_timer.start_timer("bspline_filt_rec_x");
        bspline_filt_rec_x(input, lambda, tol);
        spline_timer.stop_timer();
    }

    if(input.z_num > 2) {
//    //Z direction bspline
        spline_timer.start_timer("bspline_filt_rec_z");
        bspline_filt_rec_z(input, lambda, tol);
        spline_timer.stop_timer();
    }
}


inline float ComputeGradient::impulse_resp(float k,float rho,float omg){
    //  Impulse Response Function

    return (pow(rho,(std::abs(k)))*sin((std::abs(k) + 1)*omg)) / sin(omg);

}

inline float ComputeGradient::impulse_resp_back(float k,float rho,float omg,float gamma,float c0){
    //  Impulse Response Function (nominator eq. 4.8, denominator from eq. 4.7)

    return c0*pow(rho,std::abs(k))*(cos(omg*std::abs(k)) + gamma*sin(omg*std::abs(k)))*(1.0/(pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2)));
}


/**
 * floating point output -> no rounding or under-/overflow check
 */
template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, T>
round(float val, size_t &errCount) {
    return val;
}

/**
 * integer output -> check for under-/overflow and round
 */
template<typename T>
std::enable_if_t<!std::is_floating_point<T>::value, T>
round(float val, size_t &errCount) {

    val = std::round(val);

    if(val < std::numeric_limits<T>::min() || val > std::numeric_limits<T>::max()) {
        errCount++;
    }
    return val;
}



template<typename T>
void ComputeGradient::bspline_filt_rec_y(PixelData<T>& image,float lambda,float tol, int k0Len) {
    //
    //  Bevan Cheeseman 2016
    //
    // Recursive Filter Implimentation for Smoothing BSplines
    // B-Spline Signal Processing: Part 11-Efficient Design and Applications, Unser 1993

    float xi = 1 - 96*lambda + 24*lambda*sqrt(3 + 144*lambda); // eq 4.6
    float rho = (24*lambda - 1 - sqrt(xi))/(24*lambda)*sqrt((1/xi)*(48*lambda + 24*lambda*sqrt(3 + 144*lambda))); // eq 4.5
    float omg = atan(sqrt((1/xi)*(144*lambda - 1))); // eq 4.6

    float c0 = (1+ pow(rho,2))/(1-pow(rho,2)) * (1 - 2*rho*cos(omg) + pow(rho,2))/(1 + 2*rho*cos(omg) + pow(rho,2)); // eq 4.8
    float gamma = (1-pow(rho,2))/(1+pow(rho,2)) * (1/tan(omg)); // eq 4.8

    const float b1 = 2*rho*cos(omg);
    const float b2 = -pow(rho,2.0);

    const size_t z_num = image.z_num;
    const size_t x_num = image.x_num;
    const size_t y_num = image.y_num;
//    const size_t minLen = y_num;
    const size_t minLen = k0Len > 0 ? k0Len : std::min((size_t)(ceil(std::abs(log(tol)/log(rho)))),y_num);

    const size_t k0 = k0Len > 0 ? k0Len : (size_t)(ceil(std::abs(log(tol)/log(rho))));


    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);
//    std::cout << "CPUy xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " b1=" << b1 << " b2=" << b2 << " k0=" << k0 << " norm_factor=" << norm_factor << std::endl;
    // for boundaries
    std::vector<float> impulse_resp_vec_f(k0+3);  //forward
    for (size_t k = 0; k < (k0+3); ++k) {
        impulse_resp_vec_f[k] = impulse_resp(k,rho,omg);
    }

    std::vector<float> impulse_resp_vec_b(k0+3);  //backward
    for (size_t k = 0; k < (k0+3); ++k) {
        impulse_resp_vec_b[k] = impulse_resp_back(k,rho,omg,gamma,c0);
    }

    std::vector<float> bc1_vec(k0, 0);  //forward
    //y(1) init
    bc1_vec[1] = impulse_resp_vec_f[0];
    for (size_t k = 0; k < k0; ++k) {
        bc1_vec[k] += impulse_resp_vec_f[k+1];
    }

    //assumes a constant value at the end of the filter when the required ghost is bigger then the image
    for(size_t k = (minLen); k < k0;k++){
        bc1_vec[minLen-1] += bc1_vec[k];
    }

    std::vector<float> bc2_vec(k0, 0);  //backward
    //y(0) init
    for (size_t k = 0; k < k0; ++k) {
        bc2_vec[k] = impulse_resp_vec_f[k];
    }

    for(size_t k = (minLen); k < k0;k++){
        bc2_vec[minLen-1] += bc2_vec[k];
    }

    std::vector<float> bc3_vec(k0, 0);  //forward
    //y(N-1) init
    bc3_vec[0] = impulse_resp_vec_b[1];
    for (size_t k = 0; k < (k0-1); ++k) {
        bc3_vec[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    }

    for(size_t k = (minLen); k < k0;k++){
        bc3_vec[minLen-1] += bc3_vec[k];
    }

    std::vector<float> bc4_vec(k0, 0);  //backward
    //y(N) init
    bc4_vec[0] = impulse_resp_vec_b[0];
    for (size_t k = 1; k < k0; ++k) {
        bc4_vec[k] += 2*impulse_resp_vec_b[k];
    }

    for(size_t k = (minLen); k < k0;k++){
        bc4_vec[minLen-1] += bc4_vec[k];
    }

    APRTimer btime;
    btime.verbose_flag = false;

    size_t error_count = 0;     // count overflow errors

    //forwards direction
    btime.start_timer("forward_loop_y");
    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) reduction(+: error_count)
    #endif
    for (size_t z = 0; z < z_num; ++z) {
        const size_t jxnumynum = z * x_num * y_num;

        for (size_t x = 0; x < x_num; ++x) {
            float temp1 = 0;
            float temp2 = 0;
            float temp3 = 0;
            float temp4 = 0;
            const size_t iynum = x * y_num;

            //boundary conditions
            for (size_t k = 0; k < minLen; ++k) {
                temp1 += bc1_vec[k]*image.mesh[jxnumynum + iynum + k];
                temp2 += bc2_vec[k]*image.mesh[jxnumynum + iynum + k];
            }

            //boundary conditions
            for (size_t k = 0; k < minLen; ++k) {
                temp3 += bc3_vec[k]*image.mesh[jxnumynum + iynum + y_num - 1 - k];
                temp4 += bc4_vec[k]*image.mesh[jxnumynum + iynum + y_num - 1 - k];
            }

            //initialize the sequence
            image.mesh[jxnumynum + iynum + 0] = temp2;
            image.mesh[jxnumynum + iynum + 1] = temp1;

            for (auto it = (image.mesh.begin()+jxnumynum + iynum + 2); it !=  (image.mesh.begin()+jxnumynum + iynum + y_num); ++it) {
                float  temp = temp1*b1 + temp2*b2 + *it;
                *it = round<T>(temp, error_count);
                temp2 = temp1;
                temp1 = temp;
            }

            image.mesh[jxnumynum + iynum + y_num - 2] = round<T>(temp3*norm_factor, error_count);
            image.mesh[jxnumynum + iynum + y_num - 1] = round<T>(temp4*norm_factor, error_count);


        }
    }
    btime.stop_timer();


    btime.start_timer("backward_loop_y");
    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) reduction(+: error_count)
    #endif
    for (int64_t j = z_num - 1; j >= 0; --j) {
        const size_t jxnumynum = j * x_num * y_num;

        for (int64_t i = x_num - 1; i >= 0; --i) {
            const size_t iynum = i * y_num;

            float temp2 = image.mesh[jxnumynum + iynum + y_num - 1]/norm_factor;
            float temp1 = image.mesh[jxnumynum + iynum + y_num - 2]/norm_factor;

            for (auto it = (image.mesh.begin()+jxnumynum + iynum + y_num-3); it !=  (image.mesh.begin()+jxnumynum + iynum-1); --it) {
                float temp = temp1*b1 + temp2*b2 + *it;

                *it = round<T>(temp*norm_factor, error_count);

                temp2 = temp1;
                temp1 = temp;
            }
        }
    }
    btime.stop_timer();

    if(error_count > 0) {
        throw std::invalid_argument("integer under-/overflow encountered in ComputeGradient::bspline_filt_rec_y - try "
                                    "squashing the input image to a narrower range or use APRConverter<float>");
    }
}

template<typename T>
void ComputeGradient::bspline_filt_rec_z(PixelData<T>& image,float lambda,float tol, int k0Len){
    //
    //  Bevan Cheeseman 2016
    //
    //  Recursive Filter Implimentation for Smoothing BSplines

    float xi = 1 - 96*lambda + 24*lambda*sqrt(3 + 144*lambda);
    float rho = (24*lambda - 1 - sqrt(xi))/(24*lambda)*sqrt((1/xi)*(48*lambda + 24*lambda*sqrt(3 + 144*lambda)));
    float omg = atan(sqrt((1/xi)*(144*lambda - 1)));
    float c0 = (1+ pow(rho,2))/(1-pow(rho,2)) * (1 - 2*rho*cos(omg) + pow(rho,2))/(1 + 2*rho*cos(omg) + pow(rho,2));
    float gamma = (1-pow(rho,2))/(1+pow(rho,2)) * (1/tan(omg));

    const float b1 = 2*rho*cos(omg);
    const float b2 = -pow(rho,2.0);

    const size_t z_num = image.z_num;
    const size_t x_num = image.x_num;
    const size_t y_num = image.y_num;
    //const size_t minLen = std::min(z_num, std::min(x_num, y_num));
    //const size_t minLen = z_num;

    const size_t minLen = k0Len > 0 ? k0Len : std::min((size_t)(ceil(std::abs(log(tol)/log(rho)))), z_num);

    const size_t k0 = k0Len > 0 ? k0Len :(size_t)(ceil(std::abs(log(tol)/log(rho))));

    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);
//    std::cout << "CPUz xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " b1=" << b1 << " b2=" << b2 << " k0=" << k0 << " norm_factor=" << norm_factor << std::endl;

    //////////////////////////////////////////////////////////////
    //
    //  Setting up boundary conditions
    //
    //////////////////////////////////////////////////////////////

    std::vector<float> impulse_resp_vec_f(k0+3);  //forward
    for (size_t k = 0; k < (k0+3);k++){
        impulse_resp_vec_f[k] = impulse_resp(k,rho,omg);
    }

    std::vector<float> impulse_resp_vec_b(k0+3);  //backward
    for (size_t k = 0; k < (k0+3);k++){
        impulse_resp_vec_b[k] = impulse_resp_back(k,rho,omg,gamma,c0);
    }

    std::vector<float> bc1_vec(k0, 0);  //forward
    //y(1) init
    bc1_vec[1] = impulse_resp_vec_f[0];
    for(size_t k = 0; k < k0; k++){
        bc1_vec[k] += impulse_resp_vec_f[k+1];
    }

    //assumes a constant value at the end of the filter when the required ghost is bigger then the image
    for(size_t k = (minLen); k < k0;k++){
        bc1_vec[minLen-1] += bc1_vec[k];
    }


    std::vector<float> bc2_vec(k0, 0);  //backward
    //y(0) init
    for(size_t k = 0; k < k0; k++){
        bc2_vec[k] = impulse_resp_vec_f[k];
    }

    for(size_t k = (minLen); k < k0;k++){
        bc2_vec[minLen-1] += bc2_vec[k];
    }

    std::vector<float> bc3_vec(k0, 0);  //forward
    //y(N-1) init
    bc3_vec[0] = impulse_resp_vec_b[1];
    for(size_t k = 0; k < (k0-1); k++){
        bc3_vec[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    }

    for(size_t k = (minLen); k < k0;k++){
        bc3_vec[minLen-1] += bc3_vec[k];
    }

    std::vector<float> bc4_vec(k0, 0);  //backward
    //y(N) init
    bc4_vec[0] = impulse_resp_vec_b[0];
    for(size_t k = 1; k < k0; k++){
        bc4_vec[k] += 2*impulse_resp_vec_b[k];
    }

    for(size_t k = (minLen); k < k0;k++){
        bc4_vec[minLen-1] += bc4_vec[k];
    }

    //forwards direction
    std::vector<float> temp_vec1(y_num,0);
    std::vector<float> temp_vec2(y_num,0);
    std::vector<float> temp_vec3(y_num,0);
    std::vector<float> temp_vec4(y_num,0);

    size_t error_count = 0;     // count overflow errors

    //Initialization and boundary conditions
    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) firstprivate(temp_vec1, temp_vec2, temp_vec3, temp_vec4) reduction(+: error_count)
    #endif
    for (size_t i = 0; i < x_num; ++i) {

        std::fill(temp_vec1.begin(), temp_vec1.end(), 0);
        std::fill(temp_vec2.begin(), temp_vec2.end(), 0);
        std::fill(temp_vec3.begin(), temp_vec3.end(), 0);
        std::fill(temp_vec4.begin(), temp_vec4.end(), 0);

        size_t iynum = i * y_num;

        for (size_t j = 0; j < minLen; ++j) {
            size_t index = j * x_num * y_num + iynum;
            #ifdef HAVE_OPENMP
	        #pragma omp simd
            #endif
            for (int64_t k = y_num - 1; k >= 0; k--) {
                //forwards boundary condition
                temp_vec1[k] += bc1_vec[j] * image.mesh[index + k];
                temp_vec2[k] += bc2_vec[j] * image.mesh[index + k];
                //backwards boundary condition
                temp_vec3[k] += bc3_vec[j] * image.mesh[(z_num - 1 - j)*x_num*y_num + iynum + k];
                temp_vec4[k] += bc4_vec[j] * image.mesh[(z_num - 1 - j)*x_num*y_num + iynum + k];
            }
        }

        // ------  Causal Filter Loop
        //initialization
        for (size_t k = 0; k < y_num; ++k) {
            //z(0)
            image.mesh[iynum + k] = round<T>(temp_vec2[k], error_count);
        }

        for (size_t k = 0; k < y_num; ++k) {
            //y(1)
            image.mesh[x_num*y_num  + iynum + k] = round<T>(temp_vec1[k], error_count);
        }

        for (size_t j = 2; j < z_num; ++j) {
            size_t index = j * x_num * y_num + iynum;

            #ifdef HAVE_OPENMP
	        #pragma omp simd
            #endif
            for (size_t k = 0; k < y_num; ++k) {
                temp_vec2[k] = round<T>(1.0f*image.mesh[index + k] + b1*temp_vec1[k] + b2*temp_vec2[k], error_count);
            }

            std::swap(temp_vec1, temp_vec2);
            std::copy(temp_vec1.begin(), temp_vec1.begin()+ y_num, image.mesh.begin() + index);
        }

        // ------ Anti-Causal Filter Loop
        //initialization
        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(N)
            image.mesh[(z_num - 1)*x_num*y_num  + iynum + k] = round<T>(temp_vec4[k]*norm_factor, error_count);
        }

        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(N-1)
            image.mesh[(z_num - 2)*x_num*y_num  + iynum + k] = round<T>(temp_vec3[k]*norm_factor, error_count);
        }

        //main loop
        for (int64_t j = z_num - 3; j >= 0; --j) {
            size_t index = j * x_num * y_num + i * y_num;

            #ifdef HAVE_OPENMP
	        #pragma omp simd
            #endif
            for (int64_t k = y_num - 1; k >= 0; --k) {
                float temp = (image.mesh[index + k] +  b1*temp_vec3[k] + b2*temp_vec4[k]);
                image.mesh[index + k] = round<T>(temp*norm_factor, error_count);
                temp_vec4[k] = temp_vec3[k];
                temp_vec3[k] = temp;
            }
        }
    }

    if(error_count > 0) {
        throw std::invalid_argument("integer under-/overflow encountered in ComputeGradient::bspline_filt_rec_z - try "
                                    "squashing the input image to a narrower range or use APRConverter<float>");
    }
}

template<typename T>
void ComputeGradient::bspline_filt_rec_x(PixelData<T>& image,float lambda,float tol, int k0Len) {
    //
    //  Bevan Cheeseman 2016
    //
    //  Recursive Filter Implimentation for Smoothing BSplines

    float xi = 1 - 96*lambda + 24*lambda*sqrt(3 + 144*lambda);
    float rho = (24*lambda - 1 - sqrt(xi))/(24*lambda)*sqrt((1/xi)*(48*lambda + 24*lambda*sqrt(3 + 144*lambda)));
    float omg = atan(sqrt((1/xi)*(144*lambda - 1)));
    float c0 = (1+ pow(rho,2))/(1-pow(rho,2)) * (1 - 2*rho*cos(omg) + pow(rho,2))/(1 + 2*rho*cos(omg) + pow(rho,2));
    float gamma = (1-pow(rho,2))/(1+pow(rho,2)) * (1/tan(omg));

    const float b1 = 2*rho*cos(omg);
    const float b2 = -pow(rho,2.0);

    const size_t z_num = image.z_num;
    const size_t x_num = image.x_num;
    const size_t y_num = image.y_num;

//    const size_t minLen = x_num;
    const size_t minLen = k0Len > 0 ? k0Len : std::min((size_t)(ceil(std::abs(log(tol)/log(rho)))), x_num);
    const size_t k0 = k0Len > 0 ? k0Len : ((size_t)(ceil(std::abs(log(tol)/log(rho)))));
    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);

//    std::cout << "CPUx xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " b1=" << b1 << " b2=" << b2 << " k0=" << k0 << " norm_factor=" << norm_factor << std::endl;

    //////////////////////////////////////////////////////////////
    //
    //  Setting up boundary conditions
    //
    //////////////////////////////////////////////////////////////

    std::vector<float> impulse_resp_vec_f(k0+3);  //forward
    for (size_t k = 0; k < (k0+3);k++){
        impulse_resp_vec_f[k] = impulse_resp(k,rho,omg);
    }

    std::vector<float> impulse_resp_vec_b(k0+3);  //backward
    for (size_t k = 0; k < (k0+3);k++){
        impulse_resp_vec_b[k] = impulse_resp_back(k,rho,omg,gamma,c0);
    }

    std::vector<float> bc1_vec(k0, 0);  //forward
    //y(1) init
    bc1_vec[1] = impulse_resp_vec_f[0];
    for(size_t k = 0; k < k0;k++){
        bc1_vec[k] += impulse_resp_vec_f[k+1];
    }

    //assumes a constant value at the end of the filter when the required ghost is bigger then the image
    for(size_t k = (minLen); k < k0;k++){
        bc1_vec[minLen-1] += bc1_vec[k];
    }

    std::vector<float> bc2_vec(k0, 0);  //backward
    //y(0) init
    for(size_t k = 0; k < k0;k++){
        bc2_vec[k] = impulse_resp_vec_f[k];
    }

    for(size_t k = (minLen); k < k0;k++){
        bc2_vec[minLen-1] += bc2_vec[k];
    }

    std::vector<float> bc3_vec(k0, 0);  //forward
    //y(N-1) init
    bc3_vec[0] = impulse_resp_vec_b[1];
    for(size_t k = 0; k < (k0-1);k++){
        bc3_vec[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    }

    for(size_t k = (minLen); k < k0;k++){
        bc3_vec[minLen-1] += bc3_vec[k];
    }

    std::vector<float> bc4_vec(k0, 0);  //backward
    //y(N) init
    bc4_vec[0] = impulse_resp_vec_b[0];
    for(size_t k = 1; k < k0;k++){
        bc4_vec[k] += 2*impulse_resp_vec_b[k];
    }

    for(size_t k = (minLen); k < k0;k++){
        bc4_vec[minLen-1] += bc4_vec[k];
    }

    //forwards direction

    std::vector<float> temp_vec1(y_num,0);
    std::vector<float> temp_vec2(y_num,0);
    std::vector<float> temp_vec3(y_num,0);
    std::vector<float> temp_vec4(y_num,0);

    size_t error_count = 0;     // count overflow errors

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) firstprivate(temp_vec1, temp_vec2, temp_vec3, temp_vec4) reduction(+: error_count)
    #endif
    for (size_t j = 0;j < z_num; ++j) {
        std::fill(temp_vec1.begin(), temp_vec1.end(), 0);
        std::fill(temp_vec2.begin(), temp_vec2.end(), 0);
        std::fill(temp_vec3.begin(), temp_vec3.end(), 0);
        std::fill(temp_vec4.begin(), temp_vec4.end(), 0);

        size_t jxnumynum = j * y_num * x_num;

        for (size_t i = 0; i < minLen; ++i) {

            for (size_t k = 0; k < y_num; ++k) {
                //forwards boundary condition
                temp_vec1[k] += bc1_vec[i]*image.mesh[jxnumynum + i*y_num + k];
                temp_vec2[k] += bc2_vec[i]*image.mesh[jxnumynum + i*y_num + k];
                //backwards boundary condition
                temp_vec3[k] += bc3_vec[i]*image.mesh[jxnumynum + (x_num - 1 - i)*y_num + k];
                temp_vec4[k] += bc4_vec[i]*image.mesh[jxnumynum + (x_num - 1 - i)*y_num + k];
            }
        }

        //initialization
        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(0)
            image.mesh[jxnumynum  + k] = round<T>(temp_vec2[k], error_count);
        }

        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(1)
            image.mesh[jxnumynum  + y_num + k] = round<T>(temp_vec1[k], error_count);
        }

        for (size_t i = 2;i < x_num; ++i) {
            size_t index = i * y_num + jxnumynum;

            #ifdef HAVE_OPENMP
            #pragma omp simd
            #endif
            for (int64_t k = y_num - 1; k >= 0; k--) {
                temp_vec2[k] = round<T>(image.mesh[index + k] + b1*temp_vec1[k] + b2*temp_vec2[k], error_count);
            }

            std::swap(temp_vec1, temp_vec2);
            std::copy(temp_vec1.begin(), temp_vec1.begin() + y_num, image.mesh.begin() + index);
        }


        //Anti-Causal Filter Loop

        //initialization
        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(N)
            image.mesh[jxnumynum  + (x_num - 1)*y_num + k] = round<T>(temp_vec4[k]*norm_factor, error_count);
        }

        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(N-1)
            image.mesh[jxnumynum  + (x_num - 2)*y_num + k] = round<T>(temp_vec3[k]*norm_factor, error_count);
        }

        //main loop
        for (int64_t i = x_num - 3; i >= 0; --i){
            size_t index = jxnumynum + i*y_num;

            #ifdef HAVE_OPENMP
            #pragma omp simd
            #endif
            for (int64_t k = y_num - 1; k >= 0; k--){
                float temp = (image.mesh[index + k] + b1*temp_vec3[ k]+  b2*temp_vec4[ k]);
                image.mesh[index + k] = round<T>(temp*norm_factor, error_count);
                temp_vec4[k] = temp_vec3[k];
                temp_vec3[k] = temp;
            }
        }
    }

    if(error_count > 0) {
        throw std::invalid_argument("integer under-/overflow encountered in ComputeGradient::bspline_filt_rec_x - try "
                                    "squashing the input image to a narrower range or use APRConverter<float>");
    }
}

/**
 * Caclulation of signal value from B-Spline co-efficients
 */
template<typename T>
void ComputeGradient::calc_inv_bspline_y(PixelData<T>& input){
    //  Bevan Cheeseman 2016
    //
    //  Inverse cubic bspline inverse filter in y direciton (Memory direction)

    const int64_t z_num = input.z_num;
    const int64_t x_num = input.x_num;
    const int64_t y_num = input.y_num;

    const float a1 = 1.0/6.0;
    const float a2 = 4.0/6.0;
    const float a3 = 1.0/6.0;

    std::vector<float> temp_vec(y_num, 0);

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) firstprivate(temp_vec)
#endif
    for (int64_t j = 0; j < z_num; ++j) {
        for (int64_t i = 0;i < x_num; ++i) {

            const int64_t idx = j * x_num * y_num + i * y_num;

#ifdef HAVE_OPENMP
#pragma omp simd
#endif
            for (int64_t k = 0; k < y_num; ++k) {
                temp_vec[k] = input.mesh[idx+k];
            }

            //LHS boundary condition
            input.mesh[j*x_num*y_num + i*y_num] = a2*temp_vec[0];
            input.mesh[j*x_num*y_num + i*y_num] += (a1+a3)*temp_vec[1];

            for (int64_t k = 1; k < (y_num-1);k++){
                const int64_t idx = j * x_num * y_num + i * y_num + k;
                input.mesh[idx] = a1*temp_vec[k-1] + a2*temp_vec[k] + a3*temp_vec[k+1];
            }

            //RHS boundary condition
            input.mesh[j*x_num*y_num + i*y_num + y_num - 1] = (a1+a3)*temp_vec[y_num - 2];
            input.mesh[j*x_num*y_num + i*y_num + y_num - 1] += a2*temp_vec[y_num - 1];
        }
    }
}

template<typename T>
void ComputeGradient::calc_inv_bspline_z(PixelData<T>& input){
    //  Bevan Cheeseman 2016
    //
    //  Inverse cubic bspline inverse filter in x direciton (Off-stride direction)

    int64_t z_num = input.z_num;
    int64_t x_num = input.x_num;
    int64_t y_num = input.y_num;

    const float a1 = 1.0 / 6.0; // gaussian for sigma 0.60056
    const float a2 = 4.0 / 6.0;
    const float a3 = 1.0 / 6.0;

    std::vector<three_temps> temp_vec(y_num);

    int64_t xnumynum = x_num * y_num;

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) firstprivate(temp_vec)
    #endif
    for (int64_t i = 0; i < x_num; ++i) {
        int64_t iynum = i * y_num;

        //initialize the loop
        for (int64_t k = y_num - 1; k >= 0; k--) {
            temp_vec[k].temp_1 = input.mesh[xnumynum + iynum + k]; //second column in z_dir
            temp_vec[k].temp_2 = input.mesh[iynum + k]; // first column in z-dir
        }

        for (int64_t j = 0; j < z_num - 1; ++j) {
            const int64_t jxnumynum = j * xnumynum;

            const int64_t idx = jxnumynum + xnumynum + iynum;

            //initialize the loop
            #ifdef HAVE_OPENMP
	        #pragma omp simd
            #endif
            for (int64_t k = 0; k < (y_num); ++k) {
                temp_vec[k].temp_3 = input.mesh[idx + k]; // (j+1)th column in z dir
            }

            #ifdef HAVE_OPENMP
	        #pragma omp simd
            #endif
            for (int64_t k = 0; k < (y_num); ++k) {
                input.mesh[jxnumynum + iynum + k] = a1 * temp_vec[k].temp_1 + a2 * temp_vec[k].temp_2 + a3 * temp_vec[k].temp_3;
            }

            #ifdef HAVE_OPENMP
            #pragma omp simd
            #endif
            // TODO: use three separete vectors and swap them instead of one vector of triple floats
            for (int64_t k = 0; k < (y_num);k++){
                temp_vec[k].temp_1 = temp_vec[k].temp_2;
                temp_vec[k].temp_2 = temp_vec[k].temp_3;
            }
        }

        //then do the last boundary point (RHS)
        for (int64_t k = 0; k < (y_num);k++){
            input.mesh[(z_num - 1) * xnumynum + iynum + k] = (a1 + a3) * temp_vec[k].temp_1;
            input.mesh[(z_num - 1) * xnumynum + iynum + k] += a2 * temp_vec[k].temp_2;
        }
    }
}


template<typename T>
void ComputeGradient::calc_inv_bspline_x(PixelData<T>& input) {
    //  Bevan Cheeseman 2016
    //
    //  Inverse cubic bspline inverse filter in x direciton (Off-stride direction)

    int64_t z_num = input.z_num;
    int64_t x_num = input.x_num;
    int64_t y_num = input.y_num;

    const float a1 = 1.0/6.0;
    const float a2 = 4.0/6.0;
    const float a3 = 1.0/6.0;

    std::vector<three_temps> temp_vec(y_num);
    int64_t xnumynum = x_num * y_num;

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) firstprivate(temp_vec)
    #endif
    for(int64_t j = 0; j < z_num; ++j) {
        const int64_t jxnumynum = j * xnumynum;


        //initialize the loop
        for (int64_t k = y_num - 1; k >= 0; --k) {
            temp_vec[k].temp_1 = input.mesh[jxnumynum + y_num + k]; // second column in the XY plane
            temp_vec[k].temp_2 = input.mesh[jxnumynum + k];   // first column in the XY plane
        }

        //LHS boundary condition is accounted for with this initialization
        for (int64_t i = 0; i < x_num-1; ++i) {
            const int64_t iynum = i * y_num;

            const int64_t idx = jxnumynum + iynum + y_num;

            //initialize the loop
            #ifdef HAVE_OPENMP
	        #pragma omp simd
            #endif
            for (int64_t k = 0; k < y_num; ++k) {
                temp_vec[k].temp_3 = input.mesh[idx + k]; // get (i+1)th column
            }

            #ifdef HAVE_OPENMP
	        #pragma omp simd
            #endif
            for (int64_t k = 0; k < (y_num); k++) {
                input.mesh[jxnumynum + iynum + k] = a1 * temp_vec[k].temp_1 + a2 * temp_vec[k].temp_2 + a3 * temp_vec[k].temp_3;
            }

            // move two first y-columns to the right
            // TODO: instead of temp_vec of triple-floats we could use 3 separate vectors and switch them instead of copying data
            for (int64_t k = 0; k < (y_num); k++) {
                temp_vec[k].temp_1 = temp_vec[k].temp_2;
                temp_vec[k].temp_2 = temp_vec[k].temp_3;
            }
        }

        //then do the last boundary point (RHS)
        for (int64_t k = y_num - 1; k >= 0; k--) {
            input.mesh[jxnumynum + xnumynum - y_num + k] = (a1+a3) * temp_vec[k].temp_1 + a2 * temp_vec[k].temp_2;
        }
    }
}


/**
 * Calculates downsampled gradient (maximum magnitude) with 'replicate' boundary approach (nearest border value)
 * @param input - input mesh
 * @param grad - output gradient (must be initialized)
 * @param hx - step in x dir
 * @param hy - step in y dir
 * @param hz - step in z dir
 */
template<typename S>
void ComputeGradient::calc_bspline_fd_ds_mag(const PixelData<S> &input, PixelData<S> &grad, const float hx, const float hy,const float hz) {
    const size_t z_num = input.z_num;
    const size_t x_num = input.x_num;
    const size_t y_num = input.y_num;

    const size_t x_num_ds = grad.x_num;
    const size_t y_num_ds = grad.y_num;
    const size_t z_num_ds = grad.z_num;

    std::vector<S> temp(y_num, 0);
    const size_t xnumynum = x_num * y_num;

    size_t z_d = 0;
    #ifdef HAVE_OPENMP
    #pragma omp parallel for private(z_d) default(shared) firstprivate(temp)
    #endif
    for (z_d = 0; z_d < z_num_ds; ++z_d) {
        for (size_t z = 2*z_d; z <= std::min(2*z_d + 1,z_num - 1); ++z) {
            //double loop strategy required to make the OpenMP threadsafe, with the parent access, as there can now never be a race condition. (BC)

            // Belows pointers up, down... are forming stencil in X (left <-> right) and Z ( up <-> down) direction and
            // are pointing to whole Y column. If out of bounds then 'replicate' (nearest array border value) approach is used.
            //
            //                 up
            //   ...   left  center  right ...
            //                down

            const S *left = input.mesh.begin() + z * xnumynum + 0 * y_num; // boundary value is chosen
            const S *center = input.mesh.begin() + z * xnumynum + 0 * y_num;

            //LHS boundary condition is accounted for wiht this initialization
            const size_t zMinus = z > 0 ? z - 1 : 0 /* boundary */;
            const size_t zPlus = std::min(z + 1, z_num - 1 /* boundary */);

            for (size_t x = 0; x < x_num; ++x) {
                const S *up = input.mesh.begin() + zMinus * xnumynum + x * y_num;
                const S *down = input.mesh.begin() + zPlus * xnumynum + x * y_num;
                const size_t xPlus = std::min(x + 1, x_num - 1 /* boundary */);
                const S *right = input.mesh.begin() + z * xnumynum + xPlus * y_num;

                //compute the boundary values
                if (y_num >= 2) {
                    temp[0] = sqrt(pow((right[0] - left[0]) / (2 * hx), 2.0) + pow((down[0] - up[0]) / (2 * hz), 2.0) +
                                   pow((center[1] - center[0 /* boundary */]) / (2 * hy), 2.0));
                    temp[y_num - 1] = sqrt(pow((right[y_num - 1] - left[y_num - 1]) / (2 * hx), 2.0) +
                                           pow((down[y_num - 1] - up[y_num - 1]) / (2 * hz), 2.0) +
                                           pow((center[y_num - 1 /* boundary */] - center[y_num - 2]) / (2 * hy), 2.0));
                } else {
                    temp[0] = 0; // same values minus same values in x/y/z
                }

                //do the y gradient in range 1..y_num-2
#ifdef HAVE_OPENMP
#pragma omp simd
#endif
                for (size_t y = 1; y < y_num - 1; ++y) {
                    temp[y] = sqrt(pow((right[y] - left[y]) / (2 * hx), 2.0) + pow((down[y] - up[y]) / (2 * hz), 2.0) +
                                   pow((center[y + 1] - center[y - 1]) / (2 * hy), 2.0));
                }

                // Set as a downsampled gradient maximum from 2x2x2 gradient cubes
                int64_t z_2 = z / 2;
                int64_t x_2 = x / 2;
                for (size_t k = 0; k < y_num_ds; ++k) {
                    size_t k_s = std::min(2 * k + 1, y_num - 1);
                    const size_t idx = z_2 * x_num_ds * y_num_ds + x_2 * y_num_ds + k;
                    grad.mesh[idx] = std::max(temp[2 * k], std::max(temp[k_s], grad.mesh[idx]));
                }

                // move left, center to current center, right (both +1 to right)
                std::swap(left, center);
                std::swap(center, right);
            }
        }
    }
}


#endif //PARTPLAY_GRADIENT_HPP
