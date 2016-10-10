#ifndef PARTPLAY_GRADIENT_H
#define PARTPLAY_GRADIENT_H

// Mateusz Susik, 2016

#include "../data_structures/meshclass.h"
#include "../data_structures/structure_parts.h"

struct three_temps{
    float temp_1, temp_2, temp_3;
};

template<typename T>
void calc_inv_bspline_y(Mesh_data<T>& input){
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


template<typename T>
void calc_inv_bspline_x(Mesh_data<T>& input){
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
void calc_inv_bspline_z(Mesh_data<T>& input){
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
void calc_bspline_fd_y(Mesh_data<T>& input){
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
void calc_bspline_fd_x(Mesh_data<T>& input){
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

template<typename T>
void calc_bspline_fd_x_y_alt(Mesh_data<T>& input,Mesh_data<T>& grad,const float hx,const float hy){
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

#pragma omp parallel for default(shared) private(k,i) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3)
    for(int j = 0;j < z_num;j++){

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

template<typename T>
void calc_bspline_fd_z_alt(Mesh_data<T>& input,Mesh_data<T>& grad,const float h){

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

    int k,j,index;
    int xnumynum = x_num * y_num;

#pragma omp parallel for default(shared) private(k,j,index) firstprivate(temp_vec_1, temp_vec_2, temp_vec_3)
    for(int i = 0;i < x_num;i++){

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
void calc_bspline_fd_z(Mesh_data<T>& input){
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



inline float impulse_resp(float k,float rho,float omg){
    //
    //  Impulse Response Function
    //
    //


    return (pow(rho,(std::abs(k)))*sin((std::abs(k) + 1)*omg))/(sin(omg));

}
inline float impulse_resp_back(float k,float rho,float omg,float gamma,float c0){
    //
    //  Impulse Response Function
    //
    //


    return c0*pow(rho,std::abs(k))*(cos(omg*std::abs(k)) + gamma*sin(omg*std::abs(k)))*(1.0/(pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2)));

}
template<typename T>
void bspline_filt_rec_y(Mesh_data<T>& image,float lambda,float tol){
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

    const int k0 = (ceil(std::abs(log(tol)/log(rho))));

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
void bspline_filt_rec_x(Mesh_data<T>& image,float lambda,float tol){
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

    const int k0 = (ceil(std::abs(log(tol)/log(rho))));

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
                image.mesh[index + k] = image.mesh[index + k] + b1*temp_vec3[ k]+  b2*temp_vec4[ k];
                temp_vec4[k] = temp_vec3[k];
                temp_vec3[k] = image.mesh[index + k];
                image.mesh[index + k] *= norm_factor;
            }

        }
    }


}
template<typename T>
void bspline_filt_rec_z(Mesh_data<T>& image,float lambda,float tol){
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

    const int k0 = (ceil(std::abs(log(tol)/log(rho))));

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
                image.mesh[index + k] = image.mesh[index + k] +  b1*temp_vec3[k]+  b2*temp_vec4[k];
                temp_vec4[k] = temp_vec3[k];
                temp_vec3[k] = image.mesh[index + k];
                image.mesh[index + k] *= norm_factor;
            }

        }
    }


}
template<typename T>
void get_smooth_bspline_3D(Mesh_data<T>& input,Part_rep& p_rep){
    //
    //  Gets smoothing bspline co-efficients for 3D
    //
    //

    Part_timer spline_timer;

    spline_timer.verbose_flag = false;

    float tol = 0.0001;
    float lambda = p_rep.pars.lambda;


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


template< typename T>
void get_gradient_3D(Part_rep &p_rep, Mesh_data<T> &input_image, Mesh_data<T> &grad_image){
    //
    //  Bevan Cheeseman 2016
    //
    //  CPU pipeline for calculating gradient magnitude and bspline fit using recursive filters
    //


    Part_timer ptime;
    ptime.verbose_flag = false;

    //fit the splines using recursive filters
    get_smooth_bspline_3D(input_image,p_rep);

    ptime.start_timer("calc_bspline_fd_x_y_alt");

    calc_bspline_fd_x_y_alt(input_image,grad_image,p_rep.pars.dx,p_rep.pars.dy);

    ptime.stop_timer();
    ptime.start_timer("calc_spline_fd_zalt");

    calc_bspline_fd_z_alt(input_image,grad_image,p_rep.pars.dy);

    ptime.stop_timer();

    // Getting the BSPLINE fit for the VAR calculation

    ptime.start_timer("calc_inv_bspline_y");

    calc_inv_bspline_y(input_image);
    ptime.stop_timer();
    ptime.start_timer("calc_inv_bspline_x");
    calc_inv_bspline_x(input_image);
    ptime.stop_timer();
    ptime.start_timer("calc_inv_bspline_z");
    calc_inv_bspline_z(input_image);
    ptime.stop_timer();


}



#endif //PARTPLAY_GRADIENT_H