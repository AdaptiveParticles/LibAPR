//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_GRADIENT_HPP
#define PARTPLAY_GRADIENT_HPP

#include "src/data_structures/Mesh/MeshData.hpp"
#ifdef HAVE_OPENMP
	#include "omp.h"
#endif
#include "src/algorithm/APRParameters.hpp"
#include "src/misc/APRTimer.hpp"

class ComputeGradient {

public:
/*
 * Declerations
 */

    template<typename T>
    void bspline_filt_rec_y(MeshData<T> &image, float lambda, float tol);

    template<typename T>
    void bspline_filt_rec_x(MeshData<T> &image, float lambda, float tol);

    template<typename T>
    void bspline_filt_rec_z(MeshData<T> &image, float lambda, float tol);

    inline float impulse_resp(float k, float rho, float omg);

    inline float impulse_resp_back(float k, float rho, float omg, float gamma, float c0);

    template<typename T>
    void get_smooth_bspline_3D(MeshData<T> &input, APRParameters &pars);

// Calculate inverse B-Spline Transform

    template<typename T>
    void calc_inv_bspline_y(MeshData<T> &input);

    template<typename T>
    void calc_inv_bspline_x(MeshData<T> &input);

    template<typename T>
    void calc_inv_bspline_z(MeshData<T> &input);

    struct three_temps {
        float temp_1, temp_2, temp_3;
    };

// Gradient computation

    template<typename T>
    void calc_bspline_fd_y(MeshData<T> &input);

    template<typename T>
    void calc_bspline_fd_x(MeshData<T> &input);

    template<typename T>
    void calc_bspline_fd_z(MeshData<T> &input);

    template<typename T, typename S>
    void calc_bspline_fd_x_y_alt(MeshData<T> &input, MeshData<S> &grad, const float hx, const float hy);

    template<typename T, typename S>
    void calc_bspline_fd_z_alt(MeshData<T> &input, MeshData<S> &grad, const float h);

    template<typename T, typename S>
    void
    calc_bspline_fd_ds_mag(MeshData<T> &input, MeshData<S> &grad, const float hx, const float hy, const float hz);

    template<typename T,typename S>
    void mask_gradient(MeshData<T>& grad_ds,MeshData<S>& temp_ds,MeshData<T>& temp_full,APRParameters& par);

    template<typename T,typename S>
    void threshold_gradient(MeshData<T> &grad, const MeshData<S> &img, const float Ip_th);
};
/*
 * Implimentations
 */

template<typename T,typename S>
void ComputeGradient::mask_gradient(MeshData<T>& grad_ds,MeshData<S>& temp_ds,MeshData<T>& temp_full,APRParameters& par){
    //
    //  Bevan Cheeseman 2018
    //
    //  Loads in a tiff image file and masks the gradient according to it
    //
    //

    //location
    std::string file_name = par.input_dir + par.mask_file;

    TiffUtils::getMesh(file_name, temp_full);

    down_sample(temp_ds,temp_full,
                [](const T &x, const T &y) -> T { return std::max(x,y); },
                [](const T &x) -> T { return x; });

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared)
    #endif
    for (size_t i = 0; i < grad_ds.mesh.size(); ++i) {
        if (temp_ds.mesh[i]==0) {
            grad_ds.mesh[i] = 0;
        }
    }
}

template<typename T,typename S>
void ComputeGradient::threshold_gradient(MeshData<T> &grad, const MeshData<S> &img, const float Ip_th){
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
void ComputeGradient::bspline_filt_rec_y(MeshData<T>& image,float lambda,float tol){
    //
    //  Bevan Cheeseman 2016
    //
    //  Recursive Filter Implimentation for Smoothing BSplines (Unser 199*?)

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

    const size_t k0 = std::max(std::min((size_t)(ceil(std::abs(log(tol)/log(rho)))),z_num),(size_t)2);
    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);

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

    std::vector<float> bc2_vec(k0, 0);  //backward
    //y(0) init
    for (size_t k = 0; k < k0; ++k) {
        bc2_vec[k] = impulse_resp_vec_f[k];
    }

    std::vector<float> bc3_vec(k0, 0);  //forward
    //y(N-1) init
    bc3_vec[0] = impulse_resp_vec_b[1];
    for (size_t k = 0; k < (k0-1); ++k) {
        bc3_vec[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    }

    std::vector<float> bc4_vec(k0, 0);  //backward
    //y(N) init
    bc4_vec[0] = impulse_resp_vec_b[0];
    for (size_t k = 1; k < k0; ++k) {
        bc4_vec[k] += 2*impulse_resp_vec_b[k];
    }

    APRTimer btime;
    btime.verbose_flag = false;

    //forwards direction
    btime.start_timer("forward_loop_y");
    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared)
    #endif
    for (size_t z = 0; z < z_num; ++z) {
        const size_t jxnumynum = z * x_num * y_num;

        for (size_t x = 0; x < x_num; ++x) {
            float temp1 = 0;
            float temp2 = 0;
            float temp3 = 0;
            float temp4 = 0;
            const size_t iynum = x * y_num;

            for (size_t k = 0; k < k0; ++k) {
                temp1 += bc1_vec[k]*image.mesh[jxnumynum + iynum + k];
                temp2 += bc2_vec[k]*image.mesh[jxnumynum + iynum + k];
                temp3 += bc3_vec[k]*image.mesh[jxnumynum + iynum + y_num - 1 - k];
                temp4 += bc4_vec[k]*image.mesh[jxnumynum + iynum + y_num - 1 - k];
            }

            //initialize the sequence
            image.mesh[jxnumynum + iynum + 0] = temp2;
            image.mesh[jxnumynum + iynum + 1] = temp1;

            for (auto it = (image.mesh.begin()+jxnumynum + iynum + 2); it !=  (image.mesh.begin()+jxnumynum + iynum + y_num); ++it) {
                float  temp = temp1*b1 + temp2*b2 + *it;
                *it = temp;
                temp2 = temp1;
                temp1 = temp;
            }

            image.mesh[jxnumynum + iynum + y_num - 2] = temp3;
            image.mesh[jxnumynum + iynum + y_num - 1] = temp4;
        }
    }
    btime.stop_timer();


    btime.start_timer("backward_loop_y");
    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared)
    #endif
    for (int64_t j = z_num - 1; j >= 0; --j) {
        const size_t jxnumynum = j * x_num * y_num;

        for (int64_t i = x_num - 1; i >= 0; --i) {
            const size_t iynum = i * y_num;

            float temp2 = image.mesh[jxnumynum + iynum + y_num - 1];
            float temp1 = image.mesh[jxnumynum + iynum + y_num - 2];

            image.mesh[jxnumynum + iynum + y_num - 1]*=norm_factor;
            image.mesh[jxnumynum + iynum + y_num - 2]*=norm_factor;

            for (auto it = (image.mesh.begin()+jxnumynum + iynum + y_num-3); it !=  (image.mesh.begin()+jxnumynum + iynum-1); --it) {
                float temp = temp1*b1 + temp2*b2 + *it;
                *it = temp*norm_factor;
                temp2 = temp1;
                temp1 = temp;
            }
        }
    }
    btime.stop_timer();
}

template<typename T>
void ComputeGradient::bspline_filt_rec_z(MeshData<T>& image,float lambda,float tol){
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

    const size_t k0 = std::min((size_t)(ceil(std::abs(log(tol)/log(rho)))),z_num);
    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);

    //////////////////////////////////////////////////////////////
    //
    //  Setting up boundary conditions
    //
    //////////////////////////////////////////////////////////////

    std::vector<float> impulse_resp_vec_f(k0+3);  //forward
    for (int64_t k = 0; k < (k0+3);k++){
        impulse_resp_vec_f[k] = impulse_resp(k,rho,omg);
    }

    std::vector<float> impulse_resp_vec_b(k0+3);  //backward
    for (int64_t k = 0; k < (k0+3);k++){
        impulse_resp_vec_b[k] = impulse_resp_back(k,rho,omg,gamma,c0);
    }

    std::vector<float> bc1_vec(k0, 0);  //forward
    //y(1) init
    bc1_vec[1] = impulse_resp_vec_f[0];
    for( int64_t k = 0; k < k0; k++){
        bc1_vec[k] += impulse_resp_vec_f[k+1];
    }

    std::vector<float> bc2_vec(k0, 0);  //backward
    //y(0) init
    for( int64_t k = 0; k < k0; k++){
        bc2_vec[k] = impulse_resp_vec_f[k];
    }

    std::vector<float> bc3_vec(k0, 0);  //forward
    //y(N-1) init
    bc3_vec[0] = impulse_resp_vec_b[1];
    for( int64_t k = 0; k < (k0-1); k++){
        bc3_vec[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    }

    std::vector<float> bc4_vec(k0, 0);  //backward
    //y(N) init
    bc4_vec[0] = impulse_resp_vec_b[0];
    for( int64_t k = 1; k < k0; k++){
        bc4_vec[k] += 2*impulse_resp_vec_b[k];
    }

    //forwards direction
    std::vector<float> temp_vec1(y_num,0);
    std::vector<float> temp_vec2(y_num,0);
    std::vector<float> temp_vec3(y_num,0);
    std::vector<float> temp_vec4(y_num,0);

    //Initialization and boundary conditions
    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) firstprivate(temp_vec1, temp_vec2, temp_vec3, temp_vec4)
    #endif
    for (size_t i = 0; i < x_num; ++i) {

        std::fill(temp_vec1.begin(), temp_vec1.end(), 0);
        std::fill(temp_vec2.begin(), temp_vec2.end(), 0);
        std::fill(temp_vec3.begin(), temp_vec3.end(), 0);
        std::fill(temp_vec4.begin(), temp_vec4.end(), 0);

        size_t iynum = i * y_num;

        for (size_t j = 0; j < k0; ++j) {
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
            image.mesh[iynum + k] = temp_vec2[k];
        }

        for (size_t k = 0; k < y_num; ++k) {
            //y(1)
            image.mesh[x_num*y_num  + iynum + k] = temp_vec1[k];
        }

        for (size_t j = 2; j < z_num; ++j) {
            size_t index = j * x_num * y_num + iynum;

            #ifdef HAVE_OPENMP
	        #pragma omp simd
            #endif
            for (size_t k = 0; k < y_num; ++k) {
                temp_vec2[k] = 1.0*image.mesh[index + k] + b1*temp_vec1[k]+  b2*temp_vec2[k];
            }

            std::swap(temp_vec1, temp_vec2);
            std::copy(temp_vec1.begin(), temp_vec1.begin()+ y_num, image.mesh.begin() + index);
        }

        // ------ Anti-Causal Filter Loop
        //initialization
        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(N)
            image.mesh[(z_num - 1)*x_num*y_num  + iynum + k] = temp_vec4[k]*norm_factor;
        }

        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(N-1)
            image.mesh[(z_num - 2)*x_num*y_num  + iynum + k] = temp_vec3[k]*norm_factor;
        }

        //main loop
        for (int64_t j = z_num - 3; j >= 0; --j) {
            size_t index = j * x_num * y_num + i * y_num;

            #ifdef HAVE_OPENMP
	        #pragma omp simd
            #endif
            for (int64_t k = y_num - 1; k >= 0; --k) {
                float temp = (image.mesh[index + k] +  b1*temp_vec3[k]+  b2*temp_vec4[k]);
                image.mesh[index + k] = temp*norm_factor;
                temp_vec4[k] = temp_vec3[k];
                temp_vec3[k] = temp;
            }
        }
    }
}

template<typename T>
void ComputeGradient::bspline_filt_rec_x(MeshData<T>& image,float lambda,float tol){
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

    const size_t k0 = std::min((size_t)(ceil(std::abs(log(tol)/log(rho)))),z_num);
    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);

    //////////////////////////////////////////////////////////////
    //
    //  Setting up boundary conditions
    //
    //////////////////////////////////////////////////////////////

    std::vector<float> impulse_resp_vec_f(k0+3);  //forward
    for (int64_t k = 0; k < (k0+3);k++){
        impulse_resp_vec_f[k] = impulse_resp(k,rho,omg);
    }

    std::vector<float> impulse_resp_vec_b(k0+3);  //backward
    for (int64_t k = 0; k < (k0+3);k++){
        impulse_resp_vec_b[k] = impulse_resp_back(k,rho,omg,gamma,c0);
    }

    std::vector<float> bc1_vec(k0, 0);  //forward
    //y(1) init
    bc1_vec[1] = impulse_resp_vec_f[0];
    for( int64_t k = 0; k < k0;k++){
        bc1_vec[k] += impulse_resp_vec_f[k+1];
    }

    std::vector<float> bc2_vec(k0, 0);  //backward
    //y(0) init
    for( int64_t k = 0; k < k0;k++){
        bc2_vec[k] = impulse_resp_vec_f[k];
    }

    std::vector<float> bc3_vec(k0, 0);  //forward
    //y(N-1) init
    bc3_vec[0] = impulse_resp_vec_b[1];
    for( int64_t k = 0; k < (k0-1);k++){
        bc3_vec[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    }

    std::vector<float> bc4_vec(k0, 0);  //backward
    //y(N) init
    bc4_vec[0] = impulse_resp_vec_b[0];
    for( int64_t k = 1; k < k0;k++){
        bc4_vec[k] += 2*impulse_resp_vec_b[k];
    }

    //forwards direction

    std::vector<float> temp_vec1(y_num,0);
    std::vector<float> temp_vec2(y_num,0);
    std::vector<float> temp_vec3(y_num,0);
    std::vector<float> temp_vec4(y_num,0);

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) firstprivate(temp_vec1, temp_vec2, temp_vec3, temp_vec4)
    #endif
    for (size_t j = 0;j < z_num; ++j) {
        std::fill(temp_vec1.begin(), temp_vec1.end(), 0);
        std::fill(temp_vec2.begin(), temp_vec2.end(), 0);
        std::fill(temp_vec3.begin(), temp_vec3.end(), 0);
        std::fill(temp_vec4.begin(), temp_vec4.end(), 0);

        size_t jxnumynum = j * y_num * x_num;

        for (size_t i = 0; i < k0; ++i) {

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
            image.mesh[jxnumynum  + k] = temp_vec2[k];
        }

        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(1)
            image.mesh[jxnumynum  + y_num + k] = temp_vec1[k];
        }

        for (size_t i = 2;i < x_num; ++i) {
            size_t index = i * y_num + jxnumynum;

            #ifdef HAVE_OPENMP
            #pragma omp simd
            #endif
            for (int64_t k = y_num - 1; k >= 0; k--) {
                temp_vec2[k] = image.mesh[index + k] + b1*temp_vec1[k]+  b2*temp_vec2[k];
            }

            std::swap(temp_vec1, temp_vec2);
            std::copy(temp_vec2.begin(), temp_vec2.begin() + y_num, image.mesh.begin() + index);
        }


        //Anti-Causal Filter Loop

        //initialization
        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(N)
            image.mesh[jxnumynum  + (x_num - 1)*y_num + k] = temp_vec4[k]*norm_factor;
        }

        for (int64_t k = y_num - 1; k >= 0; --k) {
            //y(N-1)
            image.mesh[jxnumynum  + (x_num - 2)*y_num + k] = temp_vec3[k]*norm_factor;
        }

        //main loop
        for (int64_t i = x_num - 3; i >= 0; --i){
            size_t index = jxnumynum + i*y_num;

            #ifdef HAVE_OPENMP
            #pragma omp simd
            #endif
            for (int64_t k = y_num - 1; k >= 0; k--){
                float temp = (image.mesh[index + k] + b1*temp_vec3[ k]+  b2*temp_vec4[ k]);
                image.mesh[index + k] = temp*norm_factor;
                temp_vec4[k] = temp_vec3[k];
                temp_vec3[k] = temp;
            }
        }
    }
}

template<typename T>
void ComputeGradient::calc_inv_bspline_z(MeshData<T>& input){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Inverse cubic bspline inverse filter in x direciton (Off-stride direction)
    //
    //

    int64_t z_num = input.z_num;
    int64_t x_num = input.x_num;
    int64_t y_num = input.y_num;

    const float a1 = 1.0 / 6.0;
    const float a2 = 4.0 / 6.0;
    const float a3 = 1.0 / 6.0;

    std::vector<three_temps> temp_vec(y_num);

    int64_t xnumynum = x_num * y_num;
    int64_t j, k, iynum, jxnumynum;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(j, k, iynum, jxnumynum) \
                         firstprivate(temp_vec)
#endif
    for (int64_t i = 0; i < x_num; i++) {

        iynum = i * y_num;

        //initialize the loop
        for (k = y_num - 1; k >= 0; k--) {
            temp_vec[k].temp_1 = input.mesh[xnumynum + iynum + k];
            temp_vec[k].temp_2 = input.mesh[iynum + k];
        }

        for (j = 0; j < z_num - 1; j++) {

            jxnumynum = j * xnumynum;

            //initialize the loop
#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 0; k < (y_num);k++){
                temp_vec[k].temp_3 = input.mesh[jxnumynum + xnumynum + iynum + k];
            }

#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 0; k < (y_num);k++){
                input.mesh[jxnumynum + iynum + k] = a1 * temp_vec[k].temp_1 + a2 * temp_vec[k].temp_2 + a3 * temp_vec[k].temp_3;
            }

#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 0; k < (y_num);k++){
                temp_vec[k].temp_1 = temp_vec[k].temp_2;
                temp_vec[k].temp_2 = temp_vec[k].temp_3;
            }

        }

        //then do the last boundary point (RHS)
        for (k = 0; k < (y_num);k++){
            input.mesh[(z_num - 1) * xnumynum + iynum + k] = (a1 + a3) * temp_vec[k].temp_1;
            input.mesh[(z_num - 1) * xnumynum + iynum + k] += a2 * temp_vec[k].temp_2;
        }
    }
}


template<typename T>
void ComputeGradient::calc_inv_bspline_x(MeshData<T>& input){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Inverse cubic bspline inverse filter in x direciton (Off-stride direction)
    //
    //

    int64_t z_num = input.z_num;
    int64_t x_num = input.x_num;
    int64_t y_num = input.y_num;

    const float a1 = 1.0/6.0;
    const float a2 = 4.0/6.0;
    const float a3 = 1.0/6.0;

    std::vector<three_temps> temp_vec;
    temp_vec.resize(y_num);

    int64_t xnumynum = x_num * y_num;

    int64_t i, k, jxnumynum, iynum;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i, k, iynum, jxnumynum) \
                         firstprivate(temp_vec)
#endif
    for(int64_t j = 0; j < z_num; j++){

        jxnumynum = j * x_num * y_num;

        //initialize the loop
        for (k = y_num - 1; k >= 0; k--) {
            temp_vec[k].temp_1 = input.mesh[jxnumynum + y_num + k];
            temp_vec[k].temp_2 = input.mesh[jxnumynum + k];
        }

        //LHS boundary condition is accounted for with this initialization

        for(i = 0; i < x_num-1; i++){

            iynum = i * y_num;

            //initialize the loop
#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 0; k < (y_num); k++) {
                temp_vec[k].temp_3 = input.mesh[jxnumynum + iynum + y_num+ k];
            }

#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 0; k < (y_num); k++) {
                input.mesh[jxnumynum + iynum + k] = a1 * temp_vec[k].temp_1 + a2 * temp_vec[k].temp_2 + a3 * temp_vec[k].temp_3;
            }

            for (k = 0; k < (y_num); k++) {
                temp_vec[k].temp_1 = temp_vec[k].temp_2;
                temp_vec[k].temp_2 = temp_vec[k].temp_3;
            }

        }

        //then do the last boundary point (RHS)
        for (k = y_num - 1; k >= 0; k--) {
            input.mesh[jxnumynum + xnumynum - y_num + k] = (a1+a3) * temp_vec[k].temp_1 + a2 * temp_vec[k].temp_2;
        }

    }


}

template<typename T>
void ComputeGradient::get_smooth_bspline_3D(MeshData<T>& input,APRParameters& pars){
    //
    //  Gets smoothing bspline co-efficients for 3D
    //
    //

    APRTimer spline_timer;
    spline_timer.verbose_flag = true;

    float tol = 0.0001;
    float lambda = pars.lambda;

    //Y direction bspline
    spline_timer.start_timer("bspline_filt_rec_y");
    bspline_filt_rec_y(input,lambda,tol);
    spline_timer.stop_timer();

    //X direction bspline
    spline_timer.start_timer("bspline_filt_rec_x");
    bspline_filt_rec_x(input,lambda,tol);
    spline_timer.stop_timer();

    //Z direction bspline
    spline_timer.start_timer("bspline_filt_rec_z");
    bspline_filt_rec_z(input,lambda,tol);
    spline_timer.stop_timer();
}

/*
 *  Calculation of gradient from B-Splines
 */

template<typename T>
void ComputeGradient::calc_bspline_fd_y(MeshData<T>& input){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculate fd filt, for ygrad with bsplines
    //
    //

    const int64_t z_num = input.z_num;
    const int64_t x_num = input.x_num;
    const int64_t y_num = input.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<float> temp_vec;
    temp_vec.resize(y_num,0);

    int64_t i,k;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i, k) firstprivate(temp_vec)
#endif
    for(int64_t j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){


            for (k = 0; k < (y_num);k++){
                temp_vec[k] = input.mesh[j*x_num*y_num + i*y_num + k];
            }

            //LHS boundary condition
            input.mesh[j*x_num*y_num + i*y_num] = 0;

            for (k = 1; k < (y_num-1);k++){
                input.mesh[j*x_num*y_num + i*y_num + k] = a1*temp_vec[k-1];
                input.mesh[j*x_num*y_num + i*y_num + k] += a3*temp_vec[k+1];
            }

            //RHS boundary condition
            input.mesh[j*x_num*y_num + i*y_num + y_num - 1] = 0;

        }
    }


}
template<typename T>
void ComputeGradient::calc_bspline_fd_x(MeshData<T>& input){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculate fd filt, for xgrad with bsplines
    //
    //

    const int64_t z_num = input.z_num;
    const int64_t x_num = input.x_num;
    const int64_t y_num = input.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<float> temp_vec_1;
    temp_vec_1.resize(y_num,0);

    std::vector<float> temp_vec_2;
    temp_vec_2.resize(y_num,0);

    std::vector<float> temp_vec_3;
    temp_vec_3.resize(y_num,0);

    int64_t i,k;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i, k) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3)
#endif
    for(int64_t j = 0;j < z_num;j++){

        //initialize the loop
        for (k = 0; k < (y_num);k++){
            temp_vec_1[k] = input.mesh[j*x_num*y_num + 1*y_num + k];
            temp_vec_2[k] = input.mesh[j*x_num*y_num + (0)*y_num + k];
        }

        //LHS boundary condition is accounted for wiht this initialization


        for(i = 0;i < x_num-1;i++){

            //initialize the loop
            for (k = 0; k < (y_num);k++){
                temp_vec_3[k] = input.mesh[j*x_num*y_num + (i+1)*y_num + k];
            }


            for (k = 0; k < (y_num);k++){
                input.mesh[j*x_num*y_num + i*y_num + k] = a1*temp_vec_1[k];
                input.mesh[j*x_num*y_num + i*y_num + k] += a3*temp_vec_3[k];
            }

            for (k = 0; k < (y_num);k++){
                temp_vec_1[k] = temp_vec_2[k];
                temp_vec_2[k] = temp_vec_3[k];
            }

        }

        //then do the last boundary point (RHS)
        for (k = 0; k < (y_num);k++){
            input.mesh[j*x_num*y_num + (x_num - 1)*y_num + k] = (a1+a3)*temp_vec_1[k];
        }

    }



}


template<typename T,typename S>
void ComputeGradient::calc_bspline_fd_ds_mag(MeshData<T> &input, MeshData<S> &grad, const float hx, const float hy,const float hz){
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculate fd filt, for xgrad with bsplines

    const size_t z_num = input.z_num;
    const size_t x_num = input.x_num;
    const size_t y_num = input.y_num;

    const size_t z_num_ds = grad.z_num;
    const size_t x_num_ds = grad.x_num;
    const size_t y_num_ds = grad.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<S> temp(y_num,0);
    const size_t xnumynum = x_num * y_num;

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared)firstprivate(temp)
    #endif
    for (size_t j = 0;j < z_num; ++j) {
        S *temp_vec_1 = input.mesh.begin() + j*x_num*y_num + 1*y_num;
        S *temp_vec_2 = input.mesh.begin() + j*x_num*y_num;

        //LHS boundary condition is accounted for wiht this initialization
        const size_t j_m = std::max((size_t)0,j-1);
        const size_t j_p = std::min(z_num-1, j+1);

        for (size_t i = 0; i < x_num-1; ++i) {
            S *temp_vec_4 = input.mesh.begin() + j_m*xnumynum + i * y_num;
            S *temp_vec_5 = input.mesh.begin() + j_p*xnumynum + i * y_num;
            S *temp_vec_3 = input.mesh.begin() + j*x_num*y_num + (i+1)*y_num;

            //compute the boundary values
            temp[0] = sqrt(pow((a1*temp_vec_1[0] + a3*temp_vec_3[0])/hx,2.0)  + pow((a1*temp_vec_4[0] + a3*temp_vec_5[0])/hz,2.0));

            //do the y gradient
            #ifdef HAVE_OPENMP
	        #pragma omp simd
            #endif
            for (size_t k = 1; k < (y_num-1); ++k) {
                temp[k] =  sqrt(pow((a1*temp_vec_4[k] + a3*temp_vec_5[k])/hz,2.0) +  pow((a1*temp_vec_2[k-1] + a3*temp_vec_2[k+1])/hy,2.0) + pow((a1*temp_vec_1[k] + a3*temp_vec_3[k])/hx,2.0));
            }

            temp[y_num - 1] = sqrt(pow((a1*temp_vec_1[(y_num-1)] + a3*temp_vec_3[(y_num-1)])/hx,2.0)  + pow((a1*temp_vec_4[(y_num-1)] + a3*temp_vec_5[(y_num-1)])/hz,2.0));

            int64_t j_2 = j/2;
            int64_t i_2 = i/2;

            #ifdef HAVE_OPENMP
            #pragma omp simd
            #endif
            for (size_t k = 0; k < (y_num_ds); ++k) {
                size_t k_s = std::min(2*k+1, y_num-1);
                grad.mesh[j_2*x_num_ds*y_num_ds + i_2*y_num_ds + k] = std::max(temp[2*k],grad.mesh[j_2*x_num_ds*y_num_ds + i_2*y_num_ds + k]);
                grad.mesh[j_2*x_num_ds*y_num_ds + i_2*y_num_ds + k]= std::max(temp[k_s],grad.mesh[j_2*x_num_ds*y_num_ds + i_2*y_num_ds + k]);
            }

            std::swap(temp_vec_1, temp_vec_2);
            std::swap(temp_vec_2, temp_vec_3);
        }
    }
}

template<typename T,typename S>
void ComputeGradient::calc_bspline_fd_x_y_alt(MeshData<T>& input,MeshData<S>& grad,const float hx,const float hy){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculate fd filt, for xgrad with bsplines
    //
    //

    const int64_t z_num = input.z_num;
    const int64_t x_num = input.x_num;
    const int64_t y_num = input.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<float> temp_vec_1;
    temp_vec_1.resize(y_num,0);

    std::vector<float> temp_vec_2;
    temp_vec_2.resize(y_num,0);

    std::vector<float> temp_vec_3;
    temp_vec_3.resize(y_num,0);

    int64_t i,k,j;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(k,i,j) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3)
#endif
    for(j = 0;j < z_num;j++){

        //initialize the loop
        for (k = 0; k < (y_num);k++){
            temp_vec_1[k] = input.mesh[j*x_num*y_num + 1*y_num + k];
            temp_vec_2[k] = input.mesh[j*x_num*y_num + (0)*y_num + k];
        }

        //LHS boundary condition is accounted for wiht this initialization


        for(i = 0;i < x_num-1;i++){

            //initialize the loop
#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 0; k < (y_num);k++){
                temp_vec_3[k] = input.mesh[j*x_num*y_num + (i+1)*y_num + k];
            }

            //do the y gradient
#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 1; k < (y_num-1);k++){
                grad.mesh[j*x_num*y_num + i*y_num + k] = pow((a1*temp_vec_2[k-1] + a3*temp_vec_2[k+1])/hy,2.0);
            }

#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            //calc teh x gradient
            for (k = 0; k < (y_num);k++){
                grad.mesh[j*x_num*y_num + i*y_num + k] += pow((a1*temp_vec_1[k] + a3*temp_vec_3[k])/hx,2.0);

            }

            std::swap(temp_vec_1, temp_vec_2);
            std::swap(temp_vec_2, temp_vec_3);

        }


    }



}

template<typename T,typename S>
void ComputeGradient::calc_bspline_fd_z_alt(MeshData<T>& input,MeshData<S>& grad,const float h){

    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculate fd filt, for xgrad with bsplines
    //
    //

    const int64_t z_num = input.z_num;
    const int64_t x_num = input.x_num;
    const int64_t y_num = input.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<float> temp_vec_1;
    temp_vec_1.resize(y_num,0);

    std::vector<float> temp_vec_2;
    temp_vec_2.resize(y_num,0);

    std::vector<float> temp_vec_3;
    temp_vec_3.resize(y_num,0);

    int64_t k,j,index,i;
    int64_t xnumynum = x_num * y_num;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(k,j,index,i) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3)
#endif
    for(i = 0;i < x_num;i++){

        //initialize the loop
        for (k = 0; k < (y_num);k++){
            temp_vec_1[k] = input.mesh[xnumynum + i*y_num + k];
            temp_vec_2[k] = input.mesh[i*y_num + k];
        }

        for(j = 0;j < z_num-1;j++){

            index = j * x_num * y_num + i*y_num;

            //initialize the loop
#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 0; k < (y_num);k++){
                temp_vec_3[k] = input.mesh[index + xnumynum +  k];
            }

#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 0; k < (y_num);k++){
                grad.mesh[index + k] = sqrt(grad.mesh[index + k] + pow((a1*temp_vec_1[k] + a3*temp_vec_3[k])/h,2.0f));
            }

            std::swap(temp_vec_1, temp_vec_2);
            std::swap(temp_vec_2, temp_vec_3);

        }
        //account for last position
        for (k = 0; k < (y_num);k++){
            grad.mesh[(z_num-1)*x_num*y_num + i*y_num + k] = sqrt(grad.mesh[(z_num-1)*x_num*y_num + i*y_num + k]);
        }

    }

}

template<typename T>
void ComputeGradient::calc_bspline_fd_z(MeshData<T>& input){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculate fd filt, for xgrad with bsplines
    //
    //

    const int64_t z_num = input.z_num;
    const int64_t x_num = input.x_num;
    const int64_t y_num = input.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<float> temp_vec_1;
    temp_vec_1.resize(y_num,0);

    std::vector<float> temp_vec_2;
    temp_vec_2.resize(y_num,0);

    std::vector<float> temp_vec_3;
    temp_vec_3.resize(y_num,0);

    int64_t j,k;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(j, k) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3)
#endif
    for(int64_t i = 0;i < x_num;i++){

        //initialize the loop
        for (k = 0; k < (y_num);k++){
            temp_vec_1[k] = input.mesh[1*x_num*y_num + i*y_num + k];
            temp_vec_2[k] = input.mesh[0*x_num*y_num + i*y_num + k];
        }

        for(j = 0;j < z_num-1;j++){

            //initialize the loop
            for (k = 0; k < (y_num);k++){
                temp_vec_3[k] = input.mesh[(j+1)*x_num*y_num + (i)*y_num + k];
            }

            for (k = 0; k < (y_num);k++){
                input.mesh[j*x_num*y_num + i*y_num + k] = a1*temp_vec_1[k];
                input.mesh[j*x_num*y_num + i*y_num + k] += a3*temp_vec_3[k];
            }

            for (k = 0; k < (y_num);k++){
                temp_vec_1[k] = temp_vec_2[k];
                temp_vec_2[k] = temp_vec_3[k];
            }

        }

        //then do the last boundary point (RHS)
        for (k = 0; k < (y_num);k++){
            input.mesh[(z_num - 1)*x_num*y_num + i*y_num + k] = (a1+a3)*temp_vec_1[k];
        }

    }


}


/*
 * Caclulation of signal value from B-Spline co-efficients
 */

template<typename T>
void ComputeGradient::calc_inv_bspline_y(MeshData<T>& input){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Inverse cubic bspline inverse filter in y direciton (Memory direction)
    //
    //

    const int64_t z_num = input.z_num;
    const int64_t x_num = input.x_num;
    const int64_t y_num = input.y_num;

    const float a1 = 1.0/6.0;
    const float a2 = 4.0/6.0;
    const float a3 = 1.0/6.0;

    std::vector<float> temp_vec;
    temp_vec.resize(y_num,0);

    //loop unrolling

    int64_t i, k, j;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i, k, j) firstprivate(temp_vec)
#endif
    for(j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){

#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 0; k < (y_num);k++){
                temp_vec[k] = input.mesh[j*x_num*y_num + i*y_num + k];
            }

            //LHS boundary condition
            input.mesh[j*x_num*y_num + i*y_num] = a2*temp_vec[0];
            input.mesh[j*x_num*y_num + i*y_num] += (a1+a3)*temp_vec[1];

#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
            for (k = 1; k < (y_num-1);k++){
                input.mesh[j*x_num*y_num + i*y_num + k] = a1*temp_vec[k-1];
                input.mesh[j*x_num*y_num + i*y_num + k] += a2*temp_vec[k];
                input.mesh[j*x_num*y_num + i*y_num + k] += a3*temp_vec[k+1];
            }

            //RHS boundary condition
            input.mesh[j*x_num*y_num + i*y_num + y_num - 1] = (a1+a3)*temp_vec[y_num - 2];
            input.mesh[j*x_num*y_num + i*y_num + y_num - 1] += a2*temp_vec[y_num - 1];


        }
    }


}





inline float ComputeGradient::impulse_resp(float k,float rho,float omg){
    //
    //  Impulse Response Function
    //
    //


    return (pow(rho,(std::abs(k)))*sin((std::abs(k) + 1)*omg))/(sin(omg));

}
inline float ComputeGradient::impulse_resp_back(float k,float rho,float omg,float gamma,float c0){
    //
    //  Impulse Response Function
    //
    //


    return c0*pow(rho,std::abs(k))*(cos(omg*std::abs(k)) + gamma*sin(omg*std::abs(k)))*(1.0/(pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2)));

}


#endif //PARTPLAY_GRADIENT_HPP
