//////////////////////////////////////////////////////////////
//
//
//  ImageGen 2016 Bevan Cheeseman
//
//  Meshdata class for storing the image/mesh data
//
//
//
//
///////////////////////////////////////////////////////////////

#ifndef PARTPLAY_MESHCLASS_H
#define PARTPLAY_MESHCLASS_H

#include <vector>
#include <cmath>

#include "structure_parts.h"

struct coords3d {
    int x,y,z;

    coords3d operator *(uint16_t multiplier)
    {
        coords3d new_;
        new_.y = y * multiplier;
        new_.x = x * multiplier;
        new_.z = z * multiplier;
        return new_;
    }

    coords3d operator -(uint16_t diff)
    {
        coords3d new_;
        new_.y = this->y - diff;
        new_.x = this->x - diff;
        new_.z = this->z - diff;
        return new_;
    }

    friend bool operator <=(coords3d within, coords3d boundaries)
    {
        return within.y <= boundaries.y && within.x <= boundaries.x && within.z <= boundaries.z;
    }

    friend bool operator <(coords3d within, coords3d boundaries)
    {
        return within.y < boundaries.y && within.x < boundaries.x && within.z < boundaries.z;
    }
    
    friend bool operator ==(coords3d within, coords3d boundaries)
    {
        return within.y == boundaries.y && within.x == boundaries.x && within.z == boundaries.z;
    }

    friend std::ostream& operator<<(std::ostream& os, const coords3d& coords)
    {
        return std::cout << coords.y << " " << coords.x << " " << coords.z;
    }

    bool contains(coords3d neighbour, uint8_t multiplier)
    {
        return abs(this->x - neighbour.x) <= multiplier &&
               abs(this->y - neighbour.y) <= multiplier &&
               abs(this->z - neighbour.z) <= multiplier;
    }

};



template <class T>
class Mesh_data{
    //Defines what a particle is and what characteristics it has
public :

    int y_num;
    int x_num;
    int z_num;

    //data on local
    std::vector<T> mesh;

    



    Mesh_data()
            :y_num(0),x_num(0),z_num(0)
    {}

    Mesh_data(int y_num,int x_num,int z_num)
            :y_num(y_num),x_num(x_num),z_num(z_num)
    {
        mesh.resize(y_num*x_num*z_num);
        //mesh.resize(y_num,std::vector<std::vector<T> >(x_num,std::vector<T>(z_num)));
    }

    template <class U> Mesh_data<U> to_type(){
        Mesh_data<U> new_value(y_num, x_num, z_num);
        std::copy(mesh.begin(), mesh.end(), new_value.mesh.begin());
        return new_value;
    }

    T& operator ()(int i, int j,int k){
        j = std::min(j,x_num-1);
        i = std::min(i,y_num-1);
        k = std::min(k,z_num-1);

        return mesh[y_num*(j) + i + (k)*x_num*y_num];
        //return mesh[i][j][k];
    }

    size_t index(coords3d coords) const{
       return coords.z * (size_t)x_num * y_num + coords.x * y_num + coords.y;
    }


    void set_size(int y_num_,int x_num_,int z_num_){

        y_num = y_num_;
        x_num = x_num_;
        z_num = z_num_;
    }

    void initialize(T val)
    {
        mesh.resize(y_num*x_num*z_num,val);
        //mesh.insert(mesh.begin(),y_num*x_num*z_num,val);
        //mesh.resize(y_num,std::vector<std::vector<T> >(x_num,std::vector<T>(z_num)));
    }

    void initialize(int y_num_,int x_num_,int z_num_,T val)
    {
        y_num = y_num_;
        x_num = x_num_;
        z_num = z_num_;

        mesh.resize(y_num*x_num*z_num,val);
        //mesh.insert(mesh.begin(),y_num*x_num*z_num,val);
        //mesh.resize(y_num,std::vector<std::vector<T> >(x_num,std::vector<T>(z_num)));
    }

    void preallocate(int y_num_,int x_num_,int z_num_,T val)
    {


        const int z_num_ds = ceil(1.0*z_num_/2.0);
        const int x_num_ds = ceil(1.0*x_num_/2.0);
        const int y_num_ds = ceil(1.0*y_num_/2.0);

        initialize(y_num_ds, x_num_ds, z_num_ds, val);
    }



    void zero()
    {

        std::vector<T>().swap(mesh);
    }




    void setzero()
    {

        std::fill(mesh.begin(), mesh.end(), 0);
    }

    void setones()
    {

        std::fill(mesh.begin(), mesh.end(), 1.0);
    }

    void transpose(){

        std::vector<T> v2;
        std::swap(mesh, v2);

        for( unsigned int k = 0; k < z_num;k++){
            for (unsigned int i = 0; i < y_num; i++) {
                for (unsigned int j = 0; j < x_num; j++) {
                    mesh.push_back(v2[k*x_num*y_num + j * y_num + i]);
                }
            }
        }

        y_num = x_num;
        x_num = y_num;

    }



};


template<typename T>
void const_upsample_img(Mesh_data<T>& input_us,Mesh_data<T>& input,std::vector<unsigned int>& max_dims){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Creates a constant upsampling of an image
    //
    //
    
    Part_timer timer;
    
    timer.verbose_flag = false;
    
    //restrict the domain to be only as big as possibly needed
    
    int y_size_max = ceil(max_dims[0]/2.0)*2;
    int x_size_max = ceil(max_dims[1]/2.0)*2;
    int z_size_max = ceil(max_dims[2]/2.0)*2;
    
    const int z_num = std::min(input.z_num*2,z_size_max);
    const int x_num = std::min(input.x_num*2,x_size_max);
    const int y_num = std::min(input.y_num*2,y_size_max);
    
    const int z_num_ds_l = z_num/2;
    const int x_num_ds_l = x_num/2;
    const int y_num_ds_l = y_num/2;
    
    const int x_num_ds = input.x_num;
    const int y_num_ds = input.y_num;
    
    input_us.y_num = y_num;
    input_us.x_num = x_num;
    input_us.z_num = z_num;
    
    timer.start_timer("resize");
    
    //input_us.initialize(y_num, x_num,z_num,0);
    //input_us.mesh.resize(y_num*x_num*z_num);
    
    timer.stop_timer();
    
    std::vector<T> temp_vec;
    temp_vec.resize(y_num_ds,0);
    
    
    
    timer.start_timer("up_sample_const");
    
    unsigned int j, i, k;
    
#pragma omp parallel for default(shared) private(j,i,k) firstprivate(temp_vec) if(z_num_ds_l*x_num_ds_l > 100)
    for(j = 0;j < z_num_ds_l;j++){
        
        for(i = 0;i < x_num_ds_l;i++){
            
//            //four passes
//            
//            unsigned int offset = j*x_num_ds*y_num_ds + i*y_num_ds;
//            //first take into cache
//            for (k = 0; k < y_num_ds_l;k++){
//                temp_vec[k] = input.mesh[offset + k];
//            }
//            
//            //(0,0)
//            
//            offset = 2*j*x_num*y_num + 2*i*y_num;
//            //then do the operations two by two
//            for (k = 0; k < y_num_ds_l;k++){
//                input_us.mesh[offset + 2*k] = temp_vec[k];
//                input_us.mesh[offset + 2*k + 1] = temp_vec[k];
//            }
//            
//            //(0,1)
//            offset = (2*j+1)*x_num*y_num + 2*i*y_num;
//            //then do the operations two by two
//            for (k = 0; k < y_num_ds_l;k++){
//                input_us.mesh[offset + 2*k] = temp_vec[k];
//                input_us.mesh[offset + 2*k + 1] = temp_vec[k];
//            }
//            
//            offset = 2*j*x_num*y_num + (2*i+1)*y_num;
//            //(1,0)
//            //then do the operations two by two
//            for (k = 0; k < y_num_ds_l;k++){
//                input_us.mesh[offset + 2*k] = temp_vec[k];
//                input_us.mesh[offset + 2*k + 1] = temp_vec[k];
//            }
//            
//            offset = (2*j+1)*x_num*y_num + (2*i+1)*y_num;
//            //(1,1)
//            //then do the operations two by two
//            for (k = 0; k < y_num_ds_l;k++){
//                input_us.mesh[offset + 2*k] = temp_vec[k];
//                input_us.mesh[offset + 2*k + 1] = temp_vec[k];
//            }
            //first take into cache
            for (k = 0; k < y_num_ds_l;k++){
                temp_vec[k] = input.mesh[j*x_num_ds*y_num_ds + i*y_num_ds + k];
            }
            
            //(0,0)
            
            //then do the operations two by two
            for (k = 0; k < y_num_ds_l;k++){
                input_us.mesh[2*j*x_num*y_num + 2*i*y_num + 2*k] = temp_vec[k];
                input_us.mesh[2*j*x_num*y_num + 2*i*y_num + 2*k + 1] = temp_vec[k];
            }
            
            //(0,1)
            
            //then do the operations two by two
            for (k = 0; k < y_num_ds_l;k++){
                input_us.mesh[(2*j+1)*x_num*y_num + 2*i*y_num + 2*k] = temp_vec[k];
                input_us.mesh[(2*j+1)*x_num*y_num + 2*i*y_num + 2*k + 1] = temp_vec[k];
            }
            
            //(1,0)
            //then do the operations two by two
            for (k = 0; k < y_num_ds_l;k++){
                input_us.mesh[2*j*x_num*y_num + (2*i+1)*y_num + 2*k] = temp_vec[k];
                input_us.mesh[2*j*x_num*y_num + (2*i+1)*y_num + 2*k + 1] = temp_vec[k];
            }
            
            //(1,1)
            //then do the operations two by two
            for (k = 0; k < y_num_ds_l;k++){
                input_us.mesh[(2*j+1)*x_num*y_num + (2*i+1)*y_num + 2*k] = temp_vec[k];
                input_us.mesh[(2*j+1)*x_num*y_num + (2*i+1)*y_num + 2*k + 1] = temp_vec[k];
            }
            
            
        }
    }
    
    timer.stop_timer();
    
    
    
    
}

template<typename T, typename L1, typename L2>
void down_sample(Mesh_data<T>& test_a, Mesh_data<T>& test_a_ds, L1 reduce, L2 constant_operator,
                 bool with_allocation = false){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Downsampling
    //
    //

    const int z_num = test_a.z_num;
    const int x_num = test_a.x_num;
    const int y_num = test_a.y_num;

    const int z_num_ds = (int) ceil(1.0*z_num/2.0);
    const int x_num_ds = (int) ceil(1.0*x_num/2.0);
    const int y_num_ds = (int) ceil(1.0*y_num/2.0);

    Part_timer timer;
    //timer.verbose_flag = true;

    if(with_allocation) {
        timer.start_timer("downsample_initalize");

        test_a_ds.initialize(y_num_ds, x_num_ds, z_num_ds, 0);

        timer.stop_timer();
    }

    timer.start_timer("downsample_loop");
    std::vector<T> temp_vec;
    temp_vec.resize(y_num,0);


    int i, k, si_, sj_, sk_;

#pragma omp parallel for default(shared) private(i,k,si_,sj_,sk_) firstprivate(temp_vec)
    for(int j = 0;j < z_num_ds; j++) {


        for (i = 0; i < x_num_ds; i++) {

            si_ = std::min(2 * i + 1, x_num - 1);
            sj_ = std::min(2 * j + 1, z_num - 1);

            //four passes

            //first take into cache
            for (k = 0; k < y_num; k++) {
                temp_vec[k] = test_a.mesh[2 * j * x_num * y_num + 2 * i * y_num + k];
            }

            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] = temp_vec[2 * k];
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[sk_]);
            }

            //first take into cache
            for (k = 0; k < y_num; k++) {
                temp_vec[k] = test_a.mesh[2 * j * x_num * y_num + si_ * y_num + k];
            }


            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[2 * k]);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[sk_]);
            }


            //first take into cache
            for (k = 0; k < y_num; k++) {
                temp_vec[k] = test_a.mesh[sj_ * x_num * y_num + 2 * i * y_num + k];
            }


            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[2 * k]);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[sk_]);
            }

            //first take into cache
            for (k = 0; k < y_num; k++) {
                temp_vec[k] = test_a.mesh[sj_ * x_num * y_num + si_ * y_num + k];
            }


            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[2 * k]);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[sk_]);
                //final operaions
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        constant_operator(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k]);
            }

        }

    }

    timer.stop_timer();

}

#endif //PARTPLAY_MESHCLASS_H
