//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_GRADIENT_HPP
#define PARTPLAY_GRADIENT_HPP

#include "src/data_structures/Mesh/MeshData.hpp"
#include "omp.h"
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
    void threshold_gradient(MeshData<T>& grad,MeshData<S>& img,const float Ip_th);


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
                [](float x, float y) { return std::max(x,y); },
                [](float x) { return x; });

    uint64_t i = 0;

#pragma omp parallel for default(shared) private(i)
    for ( i = 0; i < grad_ds.mesh.size(); ++i) {

        if(temp_ds.mesh[i]==0){
            grad_ds.mesh[i] = 0;
        }
    }

}

template<typename T,typename S>
void ComputeGradient::threshold_gradient(MeshData<T>& grad,MeshData<S>& img,const float Ip_th){
    //
    //  Bevan Cheeseman 2016
    //
    //

    const int z_num = img.z_num;
    const int x_num = img.x_num;
    const int y_num = img.y_num;


    int i,k;
    float rescaled;

#pragma omp parallel for default(shared) private(i,k,rescaled)
    for(int j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){

            for (k = 0; k < (y_num);k++){
                if(img.mesh[j*x_num*y_num + i*y_num + k] <= Ip_th){
                    grad.mesh[j*x_num*y_num + i*y_num + k] = 0;
                }
            }
        }
    }

}




template<typename T>
void ComputeGradient::bspline_filt_rec_y(MeshData<T>& image,float lambda,float tol){
    //
    //  Bevan Cheeseman 2016
    //
    //  Recursive Filter Implimentation for Smoothing BSplines (Unser 199*?)
    //
    //

    float rho;
    float xi;
    float omg;
    float c0;
    float gamma;


    xi = 1 - 96*lambda + 24*lambda*sqrt(3 + 144*lambda);
    rho = (24*lambda - 1 - sqrt(xi))/(24*lambda)*sqrt((1/xi)*(48*lambda + 24*lambda*sqrt(3 + 144*lambda)));
    omg = atan(sqrt((1/xi)*(144*lambda - 1)));

    c0 = (1+ pow(rho,2))/(1-pow(rho,2)) * (1 - 2*rho*cos(omg) + pow(rho,2))/(1 + 2*rho*cos(omg) + pow(rho,2));

    gamma = (1-pow(rho,2))/(1+pow(rho,2)) * (1/tan(omg));

    const float b1 = 2*rho*cos(omg);
    const float b2 = -pow(rho,2.0);

    const int z_num = image.z_num;
    const int x_num = image.x_num;
    const int y_num = image.y_num;

    const int k0 = std::min((int)(ceil(std::abs(log(tol)/log(rho)))),z_num);

    float temp = 0;
    float temp1 = 0;
    float temp2 = 0;

    float temp3;
    float temp4;

    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);


    std::vector<float> impulse_resp_vec_f;  //forward
    std::vector<float> impulse_resp_vec_b;  //backward


    // for boundaries

    for (int k = 0; k < (k0+3);k++){
        impulse_resp_vec_f.push_back(impulse_resp(k,rho,omg));
    }


    for (int k = 0; k < (k0+3);k++){
        impulse_resp_vec_b.push_back(impulse_resp_back(k,rho,omg,gamma,c0));
    }

    //initialize

    std::vector<float> bc1_vec;  //forward
    std::vector<float> bc2_vec;  //backward
    std::vector<float> bc3_vec;  //forward
    std::vector<float> bc4_vec;  //backward

    bc1_vec.resize(k0,0);
    bc2_vec.resize(k0,0);
    bc3_vec.resize(k0,0);
    bc4_vec.resize(k0,0);

    //y(1) init

    bc1_vec[1] = impulse_resp_vec_f[0];

    for( int k = 0; k < k0;k++){
        bc1_vec[k] += impulse_resp_vec_f[k+1];
    }

    //y(0) init

    for( int k = 0; k < k0;k++){
        bc2_vec[k] = impulse_resp_vec_f[k];
    }


    //y(N-1) init
    bc3_vec[0] = impulse_resp_vec_b[1];

    for( int k = 0; k < (k0-1);k++){
        bc3_vec[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    }

    //y(N) init

    bc4_vec[0] = impulse_resp_vec_b[0];

    for( int k = 1; k < k0;k++){
        bc4_vec[k] += 2*impulse_resp_vec_b[k];
    }


    Part_timer btime;

    btime.verbose_flag = false;

    btime.start_timer("forward_loop_y");

    //forwards direction

    int i, k, jxnumynum, iynum;

#pragma omp parallel for default(shared) private(i, k, iynum, jxnumynum, temp1, temp2, temp3, temp4, temp)
    for(int j = 0;j < z_num;j++){

        jxnumynum = j * x_num * y_num;

        for(i = 0;i < x_num;i++){
            temp1 = 0;
            temp2 = 0;
            temp3 = 0;
            temp4 = 0;
            temp = 0;
            iynum = i * y_num;

            for (k = 0; k < k0; k++) {
                temp1 = temp1 + bc1_vec[k]*image.mesh[jxnumynum + iynum + k];
                temp2 = temp2 + bc2_vec[k]*image.mesh[jxnumynum + iynum + k];
            }

            for (k = 0; k < k0; k++) {
                temp3 = temp3 + bc3_vec[k]*image.mesh[jxnumynum + iynum + y_num - 1 - k];
                temp4 = temp4 + bc4_vec[k]*image.mesh[jxnumynum + iynum + y_num - 1 - k];

            }


            //initialize the sequence
            image.mesh[jxnumynum + iynum + 0] = temp2;
            image.mesh[jxnumynum + iynum + 1] = temp1;

            for (k = 2; k < y_num; k++){
                temp = temp1*b1 + temp2*b2 + image.mesh[jxnumynum + iynum + k];
                image.mesh[jxnumynum + iynum + k] = temp;
                temp2 = temp1;
                temp1 = temp;
            }

            image.mesh[jxnumynum + iynum + y_num - 1] = temp4;
            image.mesh[jxnumynum + iynum + y_num - 2] = temp3;

            //then replace the values for the backwards recursion

        }
    }

    btime.stop_timer();


    btime.start_timer("backward_loop_y");

#pragma omp parallel for default(shared) private(i, k, iynum, jxnumynum, temp1, temp2, temp)
    for(int j = z_num - 1; j >= 0; j--){

        jxnumynum = j * x_num * y_num;

        for(i = x_num - 1; i >= 0; i--){

            iynum = i * y_num;

            temp2 = image.mesh[jxnumynum + iynum + y_num - 1];
            temp1 = image.mesh[jxnumynum + iynum + y_num - 2];
            temp = 0;

            image.mesh[jxnumynum + iynum + y_num - 1]*=norm_factor;
            image.mesh[jxnumynum + iynum + y_num - 2]*=norm_factor;

            for (k = y_num-3; k >= 0; k--){
                temp = (temp1*b1 + temp2*b2 + image.mesh[jxnumynum + iynum + k]);
                image.mesh[jxnumynum + iynum + k] = temp*norm_factor;
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
    //
    //


    float rho;
    float xi;
    float omg;
    float c0;
    float gamma;


    xi = 1 - 96*lambda + 24*lambda*sqrt(3 + 144*lambda);
    rho = (24*lambda - 1 - sqrt(xi))/(24*lambda)*sqrt((1/xi)*(48*lambda + 24*lambda*sqrt(3 + 144*lambda)));
    omg = atan(sqrt((1/xi)*(144*lambda - 1)));

    c0 = (1+ pow(rho,2))/(1-pow(rho,2)) * (1 - 2*rho*cos(omg) + pow(rho,2))/(1 + 2*rho*cos(omg) + pow(rho,2));

    gamma = (1-pow(rho,2))/(1+pow(rho,2)) * (1/tan(omg));

    const float b1 = 2*rho*cos(omg);
    const float b2 = -pow(rho,2.0);

    const int z_num = image.z_num;
    const int x_num = image.x_num;
    const int y_num = image.y_num;

    const int k0 = std::min((int)(ceil(std::abs(log(tol)/log(rho)))),z_num);

    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);

    //////////////////////////////////////////////////////////////
    //
    //  Setting up boundary conditions
    //
    //////////////////////////////////////////////////////////////

    std::vector<float> impulse_resp_vec_f;  //forward
    std::vector<float> impulse_resp_vec_b;  //backward

    for (int k = 0; k < (k0+3);k++){
        impulse_resp_vec_f.push_back(impulse_resp(k,rho,omg));
    }


    for (int k = 0; k < (k0+3);k++){
        impulse_resp_vec_b.push_back(impulse_resp_back(k,rho,omg,gamma,c0));
    }

    std::vector<float> bc1_vec;  //forward
    std::vector<float> bc2_vec;  //backward
    std::vector<float> bc3_vec;  //forward
    std::vector<float> bc4_vec;  //backward

    bc1_vec.resize(k0,0);
    bc2_vec.resize(k0,0);
    bc3_vec.resize(k0,0);
    bc4_vec.resize(k0,0);

    //y(1) init

    bc1_vec[1] = impulse_resp_vec_f[0];

    for( int k = 0; k < k0; k++){
        bc1_vec[k] += impulse_resp_vec_f[k+1];
    }

    //y(0) init

    for( int k = 0; k < k0; k++){
        bc2_vec[k] = impulse_resp_vec_f[k];
    }


    //y(N-1) init
    bc3_vec[0] = impulse_resp_vec_b[1];

    for( int k = 0; k < (k0-1); k++){
        bc3_vec[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    }

    //y(N) init

    bc4_vec[0] = impulse_resp_vec_b[0];

    for( int k = 1; k < k0; k++){
        bc4_vec[k] += 2*impulse_resp_vec_b[k];
    }

    //forwards direction

    std::vector<float> temp_vec1;
    temp_vec1.resize(y_num,0);

    std::vector<float> temp_vec2;
    temp_vec2.resize(y_num,0);

    std::vector<float> temp_vec3;
    temp_vec3.resize(y_num,0);

    std::vector<float> temp_vec4;
    temp_vec4.resize(y_num,0);

    //Initialization and boundary conditions

    int index, iynum, j, k;

#pragma omp parallel for default(shared) private(j, k, iynum, index) \
        firstprivate(temp_vec1, temp_vec2, temp_vec3, temp_vec4)
    for(int i = 0; i < x_num; i++){

        std::fill(temp_vec1.begin(), temp_vec1.end(), 0);
        std::fill(temp_vec2.begin(), temp_vec2.end(), 0);
        std::fill(temp_vec3.begin(), temp_vec3.end(), 0);
        std::fill(temp_vec4.begin(), temp_vec4.end(), 0);

        iynum = i * y_num;

        for(j = 0; j < k0; j++){

            index = j * x_num * y_num + iynum;

#pragma omp simd
            //forwards boundary condition loop
            for (k = y_num - 1; k >= 0; k--){

                temp_vec1[k] += bc1_vec[j] * image.mesh[index + k];

                temp_vec2[k] += bc2_vec[j] * image.mesh[index + k];

            }

#pragma omp simd
            //backwards boundary condition loop
            for (k = y_num - 1; k >= 0; k--){

                temp_vec3[k] += bc3_vec[j] * image.mesh[(z_num - 1 - j)*x_num*y_num + iynum + k];

                temp_vec4[k] += bc4_vec[j] * image.mesh[(z_num - 1 - j)*x_num*y_num + iynum + k];

            }


        }


        //Causal Filter Loop


        //initialization
        for (k = 0; k < y_num; k++){
            //z(0)
            image.mesh[iynum + k] = temp_vec2[k];
        }


        for (k = 0; k < y_num; k++){
            //y(1)
            image.mesh[x_num*y_num  + iynum + k] = temp_vec1[k];
        }

        for(j = 2; j < z_num; j++){

            index = j * x_num * y_num + iynum;

#pragma omp simd
            for (k = 0; k < y_num; k++){
                image.mesh[index + k] = image.mesh[index + k] + b1*temp_vec1[k]+  b2*temp_vec2[k];
            }

            std::swap(temp_vec1, temp_vec2);
            std::copy(image.mesh.begin() + index, image.mesh.begin() + index + y_num, temp_vec1.begin());


        }

        //Anti-Causal Filter Loop


        //initialization
        for (k = y_num - 1; k >= 0; k--){
            //y(N)
            image.mesh[(z_num - 1)*x_num*y_num  + iynum + k] = temp_vec4[k]*norm_factor;
        }


        for (k = y_num - 1; k >= 0; k--){
            //y(N-1)
            image.mesh[(z_num - 2)*x_num*y_num  + iynum + k] = temp_vec3[k]*norm_factor;
        }

        //main loop
        for(j = z_num - 3; j >= 0; j--){

            index = j * x_num * y_num + i * y_num;

#pragma omp simd
            for (k = y_num - 1; k >= 0; k--){
                image.mesh[index + k] = (image.mesh[index + k] +  b1*temp_vec3[k]+  b2*temp_vec4[k])*norm_factor;
                temp_vec4[k] = temp_vec3[k];
                temp_vec3[k] = image.mesh[index + k]*(1/norm_factor);
                //image.mesh[index + k] *= norm_factor;
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
    //
    //


    float rho;
    float xi;
    float omg;
    float c0;
    float gamma;


    xi = 1 - 96*lambda + 24*lambda*sqrt(3 + 144*lambda);
    rho = (24*lambda - 1 - sqrt(xi))/(24*lambda)*sqrt((1/xi)*(48*lambda + 24*lambda*sqrt(3 + 144*lambda)));
    omg = atan(sqrt((1/xi)*(144*lambda - 1)));

    c0 = (1+ pow(rho,2))/(1-pow(rho,2)) * (1 - 2*rho*cos(omg) + pow(rho,2))/(1 + 2*rho*cos(omg) + pow(rho,2));

    gamma = (1-pow(rho,2))/(1+pow(rho,2)) * (1/tan(omg));

    const float b1 = 2*rho*cos(omg);
    const float b2 = -pow(rho,2.0);

    const int z_num = image.z_num;
    const int x_num = image.x_num;
    const int y_num = image.y_num;

    const int k0 = std::min((int)(ceil(std::abs(log(tol)/log(rho)))),z_num);

    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);

    //////////////////////////////////////////////////////////////
    //
    //  Setting up boundary conditions
    //
    //////////////////////////////////////////////////////////////

    std::vector<float> impulse_resp_vec_f;  //forward
    std::vector<float> impulse_resp_vec_b;  //backward

    for (int k = 0; k < (k0+3);k++){
        impulse_resp_vec_f.push_back(impulse_resp(k,rho,omg));
    }


    for (int k = 0; k < (k0+3);k++){
        impulse_resp_vec_b.push_back(impulse_resp_back(k,rho,omg,gamma,c0));
    }

    std::vector<float> bc1_vec;  //forward
    std::vector<float> bc2_vec;  //backward
    std::vector<float> bc3_vec;  //forward
    std::vector<float> bc4_vec;  //backward

    bc1_vec.resize(k0,0);
    bc2_vec.resize(k0,0);
    bc3_vec.resize(k0,0);
    bc4_vec.resize(k0,0);

    //y(1) init

    bc1_vec[1] = impulse_resp_vec_f[0];

    for( int k = 0; k < k0;k++){
        bc1_vec[k] += impulse_resp_vec_f[k+1];
    }

    //y(0) init

    for( int k = 0; k < k0;k++){
        bc2_vec[k] = impulse_resp_vec_f[k];
    }


    //y(N-1) init
    bc3_vec[0] = impulse_resp_vec_b[1];

    for( int k = 0; k < (k0-1);k++){
        bc3_vec[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    }

    //y(N) init

    bc4_vec[0] = impulse_resp_vec_b[0];

    for( int k = 1; k < k0;k++){
        bc4_vec[k] += 2*impulse_resp_vec_b[k];
    }

    //forwards direction

    std::vector<float> temp_vec1;
    temp_vec1.resize(y_num,0);

    std::vector<float> temp_vec2;
    temp_vec2.resize(y_num,0);

    std::vector<float> temp_vec3;
    temp_vec3.resize(y_num,0);

    std::vector<float> temp_vec4;
    temp_vec4.resize(y_num,0);

    //Initialization and boundary conditions

    int k, i, jxnumynum, index;

#pragma omp parallel for default(shared) private(i, k, jxnumynum, index) \
        firstprivate(temp_vec1, temp_vec2, temp_vec3, temp_vec4)
    for(int j = 0;j < z_num; j++){

        std::fill(temp_vec1.begin(), temp_vec1.end(), 0);
        std::fill(temp_vec2.begin(), temp_vec2.end(), 0);
        std::fill(temp_vec3.begin(), temp_vec3.end(), 0);
        std::fill(temp_vec4.begin(), temp_vec4.end(), 0);

        jxnumynum = j * y_num * x_num;

        for(i = 0; i < k0; i++){

            //forwards boundary condition loop
            for (k = 0; k < y_num; k++){


                temp_vec1[k] += bc1_vec[i]*image.mesh[jxnumynum + i*y_num + k];

                temp_vec2[k] += bc2_vec[i]*image.mesh[jxnumynum + i*y_num + k];

            }

            //backwards boundary condition loop
            for (k = 0; k < y_num;k++){

                temp_vec3[k] += bc3_vec[i]*image.mesh[jxnumynum + (x_num - 1 - i)*y_num + k];

                temp_vec4[k] += bc4_vec[i]*image.mesh[jxnumynum + (x_num - 1 - i)*y_num + k];

            }


        }




        jxnumynum = j * x_num * y_num;

        //initialization
        for (k = y_num - 1; k >= 0; k--) {
            //y(0)
            image.mesh[jxnumynum  + k] = temp_vec2[k];
        }


        for (k = y_num - 1; k >= 0; k--) {
            //y(1)
            image.mesh[jxnumynum  + y_num + k] = temp_vec1[k];
        }

        for(i = 2;i < x_num;i++){

            index = i * y_num + jxnumynum;

#pragma omp simd
            for (k = y_num - 1; k >= 0; k--) {
                image.mesh[index + k] = image.mesh[index + k] + b1*temp_vec1[k]+  b2*temp_vec2[k];
            }

            std::swap(temp_vec1, temp_vec2);
            std::copy(image.mesh.begin() + index, image.mesh.begin() + index + y_num, temp_vec1.begin());

        }


        //Anti-Causal Filter Loop


        //initialization
        for (k = y_num - 1; k >= 0; k--){
            //y(N)
            image.mesh[jxnumynum  + (x_num - 1)*y_num + k] = temp_vec4[k]*norm_factor;
        }


        for (k = y_num - 1; k >= 0; k--){
            //y(N-1)
            image.mesh[jxnumynum  + (x_num - 2)*y_num + k] = temp_vec3[k]*norm_factor;
        }

        //main loop
        for(i = x_num - 3; i >= 0; i--){

            index = jxnumynum + i*y_num;

#pragma omp simd
            for (k = y_num - 1; k >= 0; k--){
                image.mesh[index + k] = (image.mesh[index + k] + b1*temp_vec3[ k]+  b2*temp_vec4[ k])*norm_factor;
                temp_vec4[k] = temp_vec3[k];
                temp_vec3[k] = image.mesh[index + k]*(1/norm_factor);
                //image.mesh[index + k] *= norm_factor;
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

    int z_num = input.z_num;
    int x_num = input.x_num;
    int y_num = input.y_num;

    const float a1 = 1.0 / 6.0;
    const float a2 = 4.0 / 6.0;
    const float a3 = 1.0 / 6.0;

    std::vector<three_temps> temp_vec;
    temp_vec.resize(y_num);

    int xnumynum = x_num * y_num;

    int j, k, iynum, jxnumynum;

#pragma omp parallel for default(shared) private(j, k, iynum, jxnumynum) \
                         firstprivate(temp_vec)
    for (int i = 0; i < x_num; i++) {

        iynum = i * y_num;

        //initialize the loop
        for (k = y_num - 1; k >= 0; k--) {
            temp_vec[k].temp_1 = input.mesh[xnumynum + iynum + k];
            temp_vec[k].temp_2 = input.mesh[iynum + k];
        }

        for (j = 0; j < z_num - 1; j++) {

            jxnumynum = j * xnumynum;

            //initialize the loop
#pragma omp simd
            for (k = 0; k < (y_num);k++){
                temp_vec[k].temp_3 = input.mesh[jxnumynum + xnumynum + iynum + k];
            }

#pragma omp simd
            for (k = 0; k < (y_num);k++){
                input.mesh[jxnumynum + iynum + k] = a1 * temp_vec[k].temp_1 + a2 * temp_vec[k].temp_2 + a3 * temp_vec[k].temp_3;
            }

#pragma omp simd
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

    int z_num = input.z_num;
    int x_num = input.x_num;
    int y_num = input.y_num;

    const float a1 = 1.0/6.0;
    const float a2 = 4.0/6.0;
    const float a3 = 1.0/6.0;

    std::vector<three_temps> temp_vec;
    temp_vec.resize(y_num);

    int xnumynum = x_num * y_num;

    int i, k, jxnumynum, iynum;

#pragma omp parallel for default(shared) private(i, k, iynum, jxnumynum) \
                         firstprivate(temp_vec)
    for(int j = 0; j < z_num; j++){

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
#pragma omp simd
            for (k = 0; k < (y_num); k++) {
                temp_vec[k].temp_3 = input.mesh[jxnumynum + iynum + y_num+ k];
            }

#pragma omp simd
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

    spline_timer.verbose_flag = false;

    float tol = 0.0001;
    float lambda = pars.lambda;

    //Y direction bspline
    bspline_filt_rec_y(input,lambda,tol);

    spline_timer.start_timer("bspline_filt_rec_z");

    //Z direction bspline
    bspline_filt_rec_z(input,lambda,tol);

    spline_timer.stop_timer();

    spline_timer.start_timer("bspline_filt_rec_x");

    //X direction bspline
    bspline_filt_rec_x(input,lambda,tol);

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

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<float> temp_vec;
    temp_vec.resize(y_num,0);

    int i,k;

#pragma omp parallel for default(shared) private(i, k) firstprivate(temp_vec)
    for(int j = 0;j < z_num;j++){

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

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<float> temp_vec_1;
    temp_vec_1.resize(y_num,0);

    std::vector<float> temp_vec_2;
    temp_vec_2.resize(y_num,0);

    std::vector<float> temp_vec_3;
    temp_vec_3.resize(y_num,0);

    int i,k;

#pragma omp parallel for default(shared) private(i, k) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3)
    for(int j = 0;j < z_num;j++){

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
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculate fd filt, for xgrad with bsplines
    //
    //


    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    const int z_num_ds = grad.z_num;
    const int x_num_ds = grad.x_num;
    const int y_num_ds = grad.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<S> temp_vec_1;
    temp_vec_1.resize(y_num,0);

    std::vector<S> temp_vec_2;
    temp_vec_2.resize(y_num,0);

    std::vector<S> temp_vec_3;
    temp_vec_3.resize(y_num,0);

    std::vector<S> temp_vec_4;
    temp_vec_4.resize(y_num,0);

    std::vector<S> temp_vec_5;
    temp_vec_5.resize(y_num,0);

    std::vector<S> temp_vec_6;
    temp_vec_6.resize(y_num,0);

    int i,k,j;

    int xnumynum = x_num * y_num;


#pragma omp parallel for default(shared) private(k,i,j) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3,temp_vec_4,temp_vec_5,temp_vec_6)
    for(j = 0;j < z_num;j++){

        //initialize the loop
        for (k = 0; k < (y_num);k++){
            temp_vec_1[k] = input.mesh[j*x_num*y_num + 1*y_num + k];
            temp_vec_2[k] = input.mesh[j*x_num*y_num + (0)*y_num + k];
        }

        //LHS boundary condition is accounted for wiht this initialization

        const int j_m = std::max(0,j-1);
        const int j_p = std::min(z_num-1,j+1);

        for(i = 0;i < x_num-1;i++){

                //initialize the z loop
#pragma omp simd
            for (k = 0; k < (y_num); k++) {
                temp_vec_4[k] = input.mesh[j_m*xnumynum + i * y_num + k];
                temp_vec_5[k] = input.mesh[j_p*xnumynum +  i * y_num + k];
            }

            //initialize the loop
#pragma omp simd
            for (k = 0; k < (y_num);k++){
                temp_vec_3[k] = input.mesh[j*x_num*y_num + (i+1)*y_num + k];
            }


            //compute the boundary values
            temp_vec_6[0] = sqrt(pow((a1*temp_vec_1[0] + a3*temp_vec_3[0])/hx,2.0)  + pow((a1*temp_vec_4[0] + a3*temp_vec_5[0])/hz,2.0));

            //do the y gradient
#pragma omp simd
            for (k = 1; k < (y_num-1);k++){
                temp_vec_6[k] =  sqrt(pow((a1*temp_vec_4[k] + a3*temp_vec_5[k])/hz,2.0) +  pow((a1*temp_vec_2[k-1] + a3*temp_vec_2[k+1])/hy,2.0) + pow((a1*temp_vec_1[k] + a3*temp_vec_3[k])/hx,2.0));
            }

            temp_vec_6[y_num - 1] = sqrt(pow((a1*temp_vec_1[(y_num-1)] + a3*temp_vec_3[(y_num-1)])/hx,2.0)  + pow((a1*temp_vec_4[(y_num-1)] + a3*temp_vec_5[(y_num-1)])/hz,2.0));

            int j_2 = j/2;
            int i_2 = i/2;

#pragma omp simd
            for (k = 0; k < (y_num_ds);k++) {
                grad.mesh[j_2*x_num_ds*y_num_ds + i_2*y_num_ds + k] = std::max(temp_vec_6[2*k],grad.mesh[j_2*x_num_ds*y_num_ds + i_2*y_num_ds + k]);
            }

            int k_s;

#pragma omp simd
            for (k = 0; k < (y_num_ds);k++) {
                k_s = std::min(2*k+1,y_num-1);
                grad.mesh[j_2*x_num_ds*y_num_ds + i_2*y_num_ds + k] = std::max(temp_vec_6[2*k+1],grad.mesh[j_2*x_num_ds*y_num_ds + i_2*y_num_ds + k]);
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

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<float> temp_vec_1;
    temp_vec_1.resize(y_num,0);

    std::vector<float> temp_vec_2;
    temp_vec_2.resize(y_num,0);

    std::vector<float> temp_vec_3;
    temp_vec_3.resize(y_num,0);

    int i,k,j;

#pragma omp parallel for default(shared) private(k,i,j) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3)
    for(j = 0;j < z_num;j++){

        //initialize the loop
        for (k = 0; k < (y_num);k++){
            temp_vec_1[k] = input.mesh[j*x_num*y_num + 1*y_num + k];
            temp_vec_2[k] = input.mesh[j*x_num*y_num + (0)*y_num + k];
        }

        //LHS boundary condition is accounted for wiht this initialization


        for(i = 0;i < x_num-1;i++){

            //initialize the loop
#pragma omp simd
            for (k = 0; k < (y_num);k++){
                temp_vec_3[k] = input.mesh[j*x_num*y_num + (i+1)*y_num + k];
            }

            //do the y gradient
#pragma omp simd
            for (k = 1; k < (y_num-1);k++){
                grad.mesh[j*x_num*y_num + i*y_num + k] = pow((a1*temp_vec_2[k-1] + a3*temp_vec_2[k+1])/hy,2.0);
            }

#pragma omp simd
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

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<float> temp_vec_1;
    temp_vec_1.resize(y_num,0);

    std::vector<float> temp_vec_2;
    temp_vec_2.resize(y_num,0);

    std::vector<float> temp_vec_3;
    temp_vec_3.resize(y_num,0);

    int k,j,index,i;
    int xnumynum = x_num * y_num;

#pragma omp parallel for default(shared) private(k,j,index,i) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3)
    for(i = 0;i < x_num;i++){

        //initialize the loop
        for (k = 0; k < (y_num);k++){
            temp_vec_1[k] = input.mesh[xnumynum + i*y_num + k];
            temp_vec_2[k] = input.mesh[i*y_num + k];
        }

        for(j = 0;j < z_num-1;j++){

            index = j * x_num * y_num + i*y_num;

            //initialize the loop
#pragma omp simd
            for (k = 0; k < (y_num);k++){
                temp_vec_3[k] = input.mesh[index + xnumynum +  k];
            }

#pragma omp simd
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

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    const float a1 = -1.0/2.0;
    const float a3 = 1.0/2.0;

    std::vector<float> temp_vec_1;
    temp_vec_1.resize(y_num,0);

    std::vector<float> temp_vec_2;
    temp_vec_2.resize(y_num,0);

    std::vector<float> temp_vec_3;
    temp_vec_3.resize(y_num,0);

    int j,k;

#pragma omp parallel for default(shared) private(j, k) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3)
    for(int i = 0;i < x_num;i++){

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

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    const float a1 = 1.0/6.0;
    const float a2 = 4.0/6.0;
    const float a3 = 1.0/6.0;

    std::vector<float> temp_vec;
    temp_vec.resize(y_num,0);

    //loop unrolling

    int i, k, j;

#pragma omp parallel for default(shared) private(i, k, j) firstprivate(temp_vec)
    for(j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){

#pragma omp simd
            for (k = 0; k < (y_num);k++){
                temp_vec[k] = input.mesh[j*x_num*y_num + i*y_num + k];
            }

            //LHS boundary condition
            input.mesh[j*x_num*y_num + i*y_num] = a2*temp_vec[0];
            input.mesh[j*x_num*y_num + i*y_num] += (a1+a3)*temp_vec[1];

#pragma omp simd
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
