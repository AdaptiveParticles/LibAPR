
#ifndef PARTPLAY_PARTICLE_MAP_HPP
#define PARTPLAY_PARTICLE_MAP_HPP

#include "meshclass.h"
#include "structure_parts.h"

//#include "APR/APR.hpp"
#include <cassert>

#define EMPTY 0
#define TAKENSTATUS 1
#define SLOPESTATUS 3
#define NEIGHBOURSTATUS 2
#define ASCENDANT 8
#define SEEDASCENDANT 9
#define PROPOGATE 15
#define ASCENDANTNEIGHBOUR 16

#define MOORE 1
#define VONNEUMANN 0


#define NEIGHBOURLOOP(jn,in,kn, boundaries) \
for(jn = boundaries[0][0]; jn < boundaries[0][1]; jn++) \
    for(in = boundaries[1][0]; in < boundaries[1][1]; in++) \
        for(kn = boundaries[2][0]; kn < boundaries[2][1]; kn++)


#define CHILDRENLOOP(jn,in,kn, children_boundaries) \
for(jn = j * 2; jn < j * 2 + children_boundaries[0]; jn++) \
    for(in = i * 2; in < i * 2 + children_boundaries[1]; in++) \
        for(kn = k * 2; kn < k * 2 + children_boundaries[2]; kn++)

// don't try to optimize check boundaries - every check is needed due to parallelism

#define CHECKBOUNDARIES(axis,var,limit,boundaries) \
    if (var == 0) { \
        boundaries[axis][0] = 0;\
    } else {\
        boundaries[axis][0] = -1;\
    }\
    if (var == limit) {\
        boundaries[axis][1] = 1;\
    } else {\
        boundaries[axis][1] = 2;\
    }

template <typename T>
class Particle_map {
    //Defines what a particle is and what characteristics it has
private:

    std::vector<unsigned int> dims;

    bool neigh_type = MOORE;
    
    void check_boundaries(short axis, int var, int limit, short (&boundaries)[3][2])
    {

        if (var == 0) {
            boundaries[axis][0] = 0;
            boundaries[axis][1] = 2;
        } else if (var == 1) {
            boundaries[axis][0] = -1;
        }
        if (var == limit) {
            boundaries[axis][1] = 1;
        }
    }

    void fill_neighbours(int level)
    {
        const int x_num = layers[level].x_num;
        const int y_num = layers[level].y_num;
        const int z_num = layers[level].z_num;

        int j,i,k,neighbour_index,jn,in,kn,index,parts = 0;
        uint8_t status;

        short boundaries[3][2] = {{0,2},{0,2},{0,2}};

        // loop unrolling in order to avoid concurrent write
        for(int out = 0; out < std::min(3,z_num); out ++) {


#pragma omp parallel for default(shared) private(j,i,k,neighbour_index,jn,in,kn,status,index) firstprivate(boundaries) \
        reduction(+:parts) if(z_num * x_num * y_num > 100000)
            for (int j = out; j < z_num; j += 3) {

                CHECKBOUNDARIES(0, j, z_num - 1, boundaries);

                for (i = 0; i < x_num; i++) {

                    CHECKBOUNDARIES(1, i, x_num - 1, boundaries);
                    index = j * x_num * y_num + i * y_num;

                    for (k = 0; k < y_num; k++) {

                        CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                        status = layers[level].mesh[index + k];

                        if (status == TAKENSTATUS || status == PROPOGATE) {

                            NEIGHBOURLOOP(jn, in, kn, boundaries) {
                                        neighbour_index = index + jn * x_num * y_num + in * y_num + kn + k;

                                        if (layers[level].mesh[neighbour_index] == EMPTY) {
                                            layers[level].mesh[neighbour_index] = NEIGHBOURSTATUS;
                                            parts++;
                                        }
                                    }
                            parts += 8;

                            fill_parent(j, i, k, x_num, y_num, level - 1);
                        } else if (status == ASCENDANT) {
                            fill_parent(j, i, k, x_num, y_num, level - 1);
                        }
                    }
                }
            }
        }

        all_parts += parts;
    }



    void fill_neighbours_alt(int level)
    {
        const int x_num = layers[level].x_num;
        const int y_num = layers[level].y_num;
        const int z_num = layers[level].z_num;

        int j,i,k,neighbour_index,jn,in,kn,index,parts = 0;
        uint8_t status;

        short boundaries[3][2] = {{0,2},{0,2},{0,2}};

        // loop unrolling in order to avoid concurrent write
        for(int out = 0; out < std::min(3,z_num); out ++) {


#pragma omp parallel for default(shared) private(j,i,k,neighbour_index,jn,in,kn,status,index) firstprivate(boundaries) \
        reduction(+:parts) if(z_num * x_num * y_num > 100000)
            for (int j = out; j < z_num; j += 3) {

                CHECKBOUNDARIES(0, j, z_num - 1, boundaries);

                for (i = 0; i < x_num; i++) {

                    CHECKBOUNDARIES(1, i, x_num - 1, boundaries);
                    index = j * x_num * y_num + i * y_num;

                    for (k = 0; k < y_num; k++) {

                        CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                        status = layers[level].mesh[index + k];

                        if (status == TAKENSTATUS) {

                            NEIGHBOURLOOP(jn, in, kn, boundaries) {
                                        neighbour_index = index + jn * x_num * y_num + in * y_num + kn + k;

                                        if (layers[level].mesh[neighbour_index] == EMPTY) {
                                            layers[level].mesh[neighbour_index] = NEIGHBOURSTATUS;
                                            parts++;
                                            fill_parent(jn, in, kn, x_num, y_num, level - 1);
                                        }
                                    }
                            parts += 8;

                            fill_parent_seed(j, i, k, x_num, y_num, level - 1);
                        } else if (status == ASCENDANT) {
                            fill_parent_seed(j, i, k, x_num, y_num, level - 1);
                        }
                    }
                }
            }
        }

        all_parts += parts;
    }






    void set_ascendant_neighbours(int level)
    {

        const int x_num = layers[level].x_num;
        const int y_num = layers[level].y_num;
        const int z_num = layers[level].z_num;

        short boundaries[3][2] = {{0,2},{0,2},{0,2}};

        int i,k,jn,in,kn,neighbour_index,index;

        uint8_t status;

        // loop unrolling in order to avoid concurrent write
        for(int out = 0; out < std::min(3,z_num); out ++) {

#pragma omp parallel for default(shared) private(i,k,neighbour_index,jn,in,kn,status,index) firstprivate(boundaries) \
        if(z_num * x_num * y_num > 100000) schedule(static)
            for (int j = out; j < z_num; j += 3) {

                CHECKBOUNDARIES(0, j, z_num - 1, boundaries);

                for (i = 0; i < x_num; i++) {
                    CHECKBOUNDARIES(1, i, x_num - 1, boundaries);
                    index = j * x_num * y_num + i * y_num;

                    for (k = 0; k < y_num; k++) {

                        CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                        status = layers[level].mesh[index + k];

                        if (status == ASCENDANT) {
                            NEIGHBOURLOOP(jn, in, kn, boundaries) {

                                        neighbour_index = index + jn * x_num * y_num + in * y_num + kn + k;

                                        if (layers[level].mesh[neighbour_index] == EMPTY) {
                                            // EMPTY or TAKENSTATUS
                                            layers[level].mesh[neighbour_index] = ASCENDANTNEIGHBOUR;
                                        }

                                        if (layers[level].mesh[neighbour_index] == TAKENSTATUS) {
                                            // EMPTY or TAKENSTATUS
                                            layers[level].mesh[neighbour_index] = ASCENDANTNEIGHBOUR;

                                        }
                                    }
                        }
                    }
                }
            }
        }
    }

    void set_ascendant_neighbours_1(int level)
    {

        const int x_num = layers[level].x_num;
        const int y_num = layers[level].y_num;
        const int z_num = layers[level].z_num;

        short boundaries[3][2] = {{0,2},{0,2},{0,2}};

        int i,k,jn,in,kn,neighbour_index,index;

        uint8_t status;

        // loop unrolling in order to avoid concurrent write
        for(int out = 0; out < std::min(3,z_num); out ++) {

#pragma omp parallel for default(shared) private(i,k,neighbour_index,jn,in,kn,status,index) firstprivate(boundaries) \
        if(z_num * x_num * y_num > 100000) schedule(static)
            for (int j = out; j < z_num; j += 3) {

                CHECKBOUNDARIES(0, j, z_num - 1, boundaries);

                for (i = 0; i < x_num; i++) {
                    CHECKBOUNDARIES(1, i, x_num - 1, boundaries);
                    index = j * x_num * y_num + i * y_num;

                    for (k = 0; k < y_num; k++) {

                        CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                        status = layers[level].mesh[index + k];

                        if (status == ASCENDANT) {
                            NEIGHBOURLOOP(jn, in, kn, boundaries) {

                                        neighbour_index = index + jn * x_num * y_num + in * y_num + kn + k;

                                        if (layers[level].mesh[neighbour_index] == EMPTY) {
                                            // EMPTY or TAKENSTATUS
                                            layers[level].mesh[neighbour_index] = ASCENDANTNEIGHBOUR;
                                        }

                                        if (layers[level].mesh[neighbour_index] == TAKENSTATUS) {
                                            // EMPTY or TAKENSTATUS

                                            layers[level].mesh[neighbour_index] = PROPOGATE;
                                        }
                                    }
                        }
                    }
                }
            }
        }
    }



    void set_ascendant_filler(int level)
    {

        const int x_num = layers[level].x_num;
        const int y_num = layers[level].y_num;
        const int z_num = layers[level].z_num;

        short boundaries[3][2] = {{0,2},{0,2},{0,2}};

        int i,k,jn,in,kn,neighbour_index,index;

        uint8_t status;

        // loop unrolling in order to avoid concurrent write
        for(int out = 0; out < std::min(3,z_num); out ++) {

#pragma omp parallel for default(shared) private(i,k,neighbour_index,jn,in,kn,status,index) firstprivate(boundaries) \
        if(z_num * x_num * y_num > 100000) schedule(static)
            for (int j = out; j < z_num; j += 3) {

                CHECKBOUNDARIES(0, j, z_num - 1, boundaries);

                for (i = 0; i < x_num; i++) {
                    CHECKBOUNDARIES(1, i, x_num - 1, boundaries);
                    index = j * x_num * y_num + i * y_num;

                    for (k = 0; k < y_num; k++) {

                        CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                        status = layers[level].mesh[index + k];

                        if (status == ASCENDANT || status == SEEDASCENDANT) {
                            NEIGHBOURLOOP(jn, in, kn, boundaries) {

                                        neighbour_index = index + jn * x_num * y_num + in * y_num + kn + k;

                                        if (layers[level].mesh[neighbour_index] == EMPTY) {
                                            // EMPTY or TAKENSTATUS
                                            layers[level].mesh[neighbour_index] = SLOPESTATUS;
                                        }

                                    }
                        }
                    }
                }
            }
        }
    }


    void set_slope(int level)
    {
        short children_boundaries[3] = {2,2,2};
        const int x_num = layers[level].x_num;
        const int y_num = layers[level].y_num;
        const int z_num = layers[level].z_num;

        int prev_x_num = layers[level + 1].x_num;
        int prev_y_num = layers[level + 1].y_num;
        int prev_z_num = layers[level + 1].z_num;

        int i, k, jn, in, kn, children_index, index, parts=0;
        uint8_t children_status, status;

#pragma omp parallel for default(shared) \
        private(i,k,children_index,jn,in,kn,children_status,status,index) \
        if(z_num * x_num * y_num > 10000) reduction(+:parts) firstprivate(level, children_boundaries)
        for(int j = 0; j < z_num; j++) {

            if( j == z_num - 1 && prev_z_num % 2 ) {
                children_boundaries[0] = 1;
            }

            for ( i = 0; i < x_num; i++) {

                if( i == x_num - 1 && prev_x_num % 2 ) {
                    children_boundaries[1] = 1;
                } else if( i == 0 ){
                    children_boundaries[1] = 2;
                }

                index = j * x_num * y_num + i * y_num;


                for ( k = 0; k < y_num; k++) {

                    if( k == y_num - 1 && prev_y_num % 2 ) {
                        children_boundaries[2] = 1;
                    } else if( k == 0 ){
                        children_boundaries[2] = 2;
                    }

                    status = layers[level].mesh[index + k];

                    if(status == ASCENDANTNEIGHBOUR || status == PROPOGATE) {

                        // go down, and set empty children to SLOPESTATUS
                        CHILDRENLOOP(jn, in, kn, children_boundaries) {

                                    children_index = jn * prev_x_num * prev_y_num + in * prev_y_num + kn;
                                    children_status = layers[level + 1].mesh[children_index];
                                    if(children_status == EMPTY) {
                                        layers[level + 1].mesh[children_index] = SLOPESTATUS;
                                        parts++;
                                    }
                                }

                    }

                }
            }
        }

        all_parts += parts;
    }




    void set_slope_new(int level)
    {
        short children_boundaries[3] = {2,2,2};
        const int x_num = layers[level].x_num;
        const int y_num = layers[level].y_num;
        const int z_num = layers[level].z_num;

        int prev_x_num = layers[level + 1].x_num;
        int prev_y_num = layers[level + 1].y_num;
        int prev_z_num = layers[level + 1].z_num;

        int i, k, jn, in, kn, children_index, index, parts=0;
        uint8_t children_status, status;

#pragma omp parallel for default(shared) \
        private(i,k,children_index,jn,in,kn,children_status,status,index) \
        if(z_num * x_num * y_num > 10000) reduction(+:parts) firstprivate(level, children_boundaries)
        for(int j = 0; j < z_num; j++) {

            if( j == z_num - 1 && prev_z_num % 2 ) {
                children_boundaries[0] = 1;
            }

            for ( i = 0; i < x_num; i++) {

                if( i == x_num - 1 && prev_x_num % 2 ) {
                    children_boundaries[1] = 1;
                } else if( i == 0 ){
                    children_boundaries[1] = 2;
                }

                index = j * x_num * y_num + i * y_num;


                for ( k = 0; k < y_num; k++) {

                    if( k == y_num - 1 && prev_y_num % 2 ) {
                        children_boundaries[2] = 1;
                    } else if( k == 0 ){
                        children_boundaries[2] = 2;
                    }

                    status = layers[level].mesh[index + k];

                    if(status == ASCENDANT) {

                        // go down, and set empty children to SLOPESTATUS
                        CHILDRENLOOP(jn, in, kn, children_boundaries) {

                                    children_index = jn * prev_x_num * prev_y_num + in * prev_y_num + kn;
                                    children_status = layers[level + 1].mesh[children_index];
                                    if(children_status == EMPTY) {
                                        layers[level + 1].mesh[children_index] = SLOPESTATUS;
                                        parts++;
                                    }
                                }

                    }

                }
            }
        }

        all_parts += parts;
    }


    void fill_parent(int j, int i, int k, int x_num, int y_num, int new_level)
    {

        if(new_level >= k_min) {
            int new_x_num = ((x_num + 1) / 2);
            int new_y_num = ((y_num + 1) / 2);
            int new_index = (j / 2) * new_x_num * new_y_num + (i / 2) * new_y_num + (k / 2);

            if (layers[new_level].mesh[new_index] != TAKENSTATUS)
                layers[new_level].mesh[new_index] = ASCENDANT;
        }
    }

    void fill_parent_seed(int j, int i, int k, int x_num, int y_num, int new_level)
    {

        if(new_level >= k_min) {
            int new_x_num = ((x_num + 1) / 2);
            int new_y_num = ((y_num + 1) / 2);
            int new_index = (j / 2) * new_x_num * new_y_num + (i / 2) * new_y_num + (k / 2);

            layers[new_level].mesh[new_index] = SEEDASCENDANT;

        }
    }

    void fill_parent_neigh_filler(int j, int i, int k, int x_num, int y_num, int new_level)
    {

        if(new_level >= k_min) {
            int new_x_num = ((x_num + 1) / 2);
            int new_y_num = ((y_num + 1) / 2);
            int new_index = (j / 2) * new_x_num * new_y_num + (i / 2) * new_y_num + (k / 2);

            if (layers[new_level].mesh[new_index] != SEEDASCENDANT)
                layers[new_level].mesh[new_index] = ASCENDANT;

        }
    }



public :

    std::vector<Mesh_data<uint8_t>> layers;
    std::vector<Mesh_data<T>> downsampled;

    /*
    std::vector<std::vector<uint32_t>> intensity_pointers;
    std::vector<uint16_t> intensities;
     */

    int all_parts = 0;
    float k_min;
    float k_max;

    Particle_map() {}

    Particle_map(Part_rep p_rep)
    {
        dims = p_rep.org_dims;
        k_min = p_rep.pl_map.k_min;
        k_max = p_rep.pl_map.k_max;
    }

    Particle_map(std::vector<unsigned int> org_dims,unsigned int depth_min,unsigned int depth_max)
    {
        dims = org_dims;
        k_min = depth_min;
        k_max = depth_max - 1;
    }

    void initialize(std::vector<unsigned int> dims)
    {
        //make so you can reference the array as k
        layers.resize(k_max + 1);
        for(int k_ = k_min; k_ < (k_max + 1) ;k_ ++){
            layers[k_].initialize(ceil(1.0*dims[0]/pow(2.0,1.0*k_max - k_ + 1)),
                                  ceil(1.0*dims[1]/pow(2.0,1.0*k_max - k_ + 1)),
                                  ceil(1.0*dims[2]/pow(2.0,1.0*k_max - k_ + 1)),
                                  EMPTY);
        }
    }

    void pushing_scheme(Part_rep& p_rep)
    {
        //
        //  Bevan Cheeseman 2016
        //
        //  New scheme for calculating the valid resolution mapping, from D_l. Should provide the same results as lifting_scheme_part_list
        //
        //


        int type = p_rep.pars.pull_scheme;

        if(type==1) {

            //loop over all levels of k
            for (int level = p_rep.pl_map.k_max; p_rep.pl_map.k_min <= level; level--) {

                // firstly push up

                // SPREAD DIRECT NEIGHBOURSTATUS

                if (level != p_rep.pl_map.k_max) {

                    set_ascendant_neighbours(level);

                    set_slope(level);

                }

                fill_neighbours(level);

            }

        } else if(type==2){

            //loop over all levels of k
            for (int level = p_rep.pl_map.k_max; p_rep.pl_map.k_min <= level; level--) {

                // firstly push up

                // SPREAD DIRECT NEIGHBOURSTATUS

                if (level != p_rep.pl_map.k_max) {

                    set_ascendant_neighbours_1(level);

                    set_slope(level);

                }

                fill_neighbours(level);

            }


        } else if (type==3){

            //loop over all levels of k
            for (int level = p_rep.pl_map.k_max; p_rep.pl_map.k_min <= level; level--) {

                // firstly push up

                // SPREAD DIRECT NEIGHBOURSTATUS

                if (level != p_rep.pl_map.k_max) {

                    set_slope_new(level);

                }

                fill_neighbours_alt(level);


                if (level != p_rep.pl_map.k_max) {

                    set_ascendant_filler(level);

                }

            }



        }





    }

    void downsample(Mesh_data<T> &original_image)
    {
        downsampled.resize(k_max+2);
        downsampled.back() = std::move(original_image);

        auto sum = [](T x, T y) { return x+y; };
        auto divide_by_8 = [](T x) { return x * (1.0/8.0); };

        for (int level = k_max+1; level > k_min; level--) {
            down_sample(downsampled[level], downsampled[level - 1], sum, divide_by_8, true);
        }
    }

    void closest_pixel(Mesh_data<T> &original_image){
        //
        //  Scheme for calculating at best estimate f(x_p) = f_p for the noise free analysis, requried since particles of higher resolution levels sit on off pixel locations
        //

        downsample(original_image);

        float x,y,z;


        for (int level = k_max; level > k_min; level--) {

            float step_size = pow(2,k_max + 1 - level);

            for (int q = 0; q < downsampled[level].z_num; ++q) {
                for (int k = 0; k < downsampled[level].x_num; ++k) {
                    for (int i = 0; i < downsampled[level].y_num; ++i) {

                        x = round((k+0.5)*step_size)-1;
                        y = round((i+0.5)*step_size)-1;
                        z = round((q+0.5)*step_size)-1;

                        float temp = 0;

                        int max_x = std::min((int)(x+1),downsampled[k_max+1].x_num-1);
                        int max_y = std::min((int)(y+1),downsampled[k_max+1].y_num-1);
                        int max_z = std::min((int)(z+1),downsampled[k_max+1].z_num-1);

                        float counter = 0;

                        //ydirection
                        for (int l_z = z; l_z <= max_z; ++l_z) {
                            for (int l_x = x; l_x <= max_x; ++l_x) {
                                for (int l_y = y; l_y <= max_y; ++l_y) {

                                    temp += downsampled[k_max+1].mesh[l_y + (l_x) * downsampled[k_max+1].y_num + l_z * downsampled[k_max+1].y_num * downsampled[k_max+1].x_num];
                                    counter++;
                                }
                            }
                        }



                        downsampled[level].mesh[i + (k) * downsampled[level].y_num + q * downsampled[level].y_num * downsampled[level].x_num] = temp/counter;

                    }
                }
            }


        }



    }


    void fill(float k, Mesh_data<T> &input)
    {
        //
        //  Bevan Cheeseman 2016
        //
        //  Updates the hash table from the down sampled images
        //

        const int z_num = input.z_num;
        const int x_num = input.x_num;
        const int y_num = input.y_num;

        int temp;
        int i,q;

        std::vector<uint8_t> &topvec = layers[k_max].mesh;

        if (k == k_max){
            // k_max loop, has to include
#pragma omp parallel for default(shared) private(i,q,temp)
            for(int j = 0;j < z_num;j++){
                for(i = 0;i < x_num;i++){
                    for (q = 0; q < (y_num);q++){

                        temp = input.mesh[j*x_num*y_num + i*y_num + q] >= k;

                        if ( temp ) {
                            topvec[j*x_num*y_num + i*y_num + q] = TAKENSTATUS;
                        }
                    }
                }
            }



        } else if (k == k_min){
        // k_max loop, has to include
#pragma omp parallel for default(shared) private(i,q,temp) if(z_num*x_num*y_num > 100000)
            for(int j = 0;j < z_num;j++){
                for(i = 0;i < x_num;i++){
                    for (q = 0; q < (y_num);q++){
                        
                        temp = input.mesh[j*x_num*y_num + i*y_num + q] <= k;
                        
                        if ( temp ) {
                            layers[k_min].mesh[j*x_num*y_num + i*y_num + q] = TAKENSTATUS;
                        }
                    }
                }
            }
            
            
            
        } else{
            // other k's

#pragma omp parallel for default(shared) private(i,q,temp) if(z_num*x_num*y_num > 100000)
            for(int j = 0;j < z_num;j++){
                for(i = 0;i < x_num;i++){
#ifndef _MSC_VER
#pragma omp simd
#endif
                    for (q = 0; q < y_num;q++){

                        temp = input.mesh[j*x_num*y_num + i*y_num + q] == k;

                        if (temp) {
                            layers[k].mesh[j*x_num*y_num + i*y_num + q] = TAKENSTATUS;
                        }
                    }
                }
            }

        }
    }

};

template <typename T>
void preallocate(std::vector<Mesh_data<T>> &to_prealocate, const int y_num, const int x_num,
                 const int z_num, const Part_rep &p_rep)
{

    int z_num_ds = (int) ceil(1.0*z_num/2.0);
    int x_num_ds = (int) ceil(1.0*x_num/2.0);
    int y_num_ds = (int) ceil(1.0*y_num/2.0);

    to_prealocate.resize(p_rep.pl_map.k_max + 1);

    for(int i = p_rep.pl_map.k_max; i >= p_rep.pl_map.k_min; i--)
    {
        to_prealocate[i].initialize(y_num_ds,x_num_ds,z_num_ds,0);
        z_num_ds = (int) ceil(1.0*z_num_ds/2.0);
        x_num_ds = (int) ceil(1.0*x_num_ds/2.0);
        y_num_ds = (int) ceil(1.0*y_num_ds/2.0);
    }

}

template <typename T>
void preallocate(std::vector<Mesh_data<T>> &to_prealocate, const int y_num, const int x_num,
                 const int z_num, const unsigned int k_max,const unsigned int k_min)
{

    int z_num_ds = (int) ceil(1.0*z_num/2.0);
    int x_num_ds = (int) ceil(1.0*x_num/2.0);
    int y_num_ds = (int) ceil(1.0*y_num/2.0);

    to_prealocate.resize(k_max + 1);

    for(int i = k_max; i >= k_min; i--)
    {
        to_prealocate[i].initialize(y_num_ds,x_num_ds,z_num_ds,0);
        z_num_ds = (int) ceil(1.0*z_num_ds/2.0);
        x_num_ds = (int) ceil(1.0*x_num_ds/2.0);
        y_num_ds = (int) ceil(1.0*y_num_ds/2.0);
    }

}


#endif //PARTPLAY_PARTICLE_MAP_HPP
