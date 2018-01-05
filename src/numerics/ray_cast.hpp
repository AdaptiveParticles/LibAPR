#ifndef _ray_cast_num_h
#define _ray_cast_num_h
//////////////////////////////////////////////////
//
//
//  Bevan Cheeseman 2016
//
//  Ray casting numerics
//
//
//////////////////////////////////////////////////

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>

#include "src/data_structures/APR/ExtraPartCellData.hpp"

#include "../data_structures/APR/APR.hpp"

#include "../../src/vis/Camera.h"
#include "../../src/vis/RaytracedObject.h"


const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

struct proj_par{
    int proj_type = 0;
    int status_th = 10;
    float Ip_th = 0;
    int start_th = 5;
    int direction = 0;
    bool avg_flag = true;

    //new parameters
    float height = 0.5;
    float radius_factor = 1.5;
    float theta_0 = 0;
    float theta_final = .1;
    float theta_delta = 0.01;

    float scale_y = 1.0;
    float scale_x = 1.0;
    float scale_z = 1.0;

    std::string name = "raycast";

};

template<typename U,typename S,typename V,class BinaryOperation>
void apr_raycast(APR<U>& apr,ExtraPartCellData<S>& particle_data,proj_par& pars,Mesh_data<V>& cast_views,BinaryOperation op) {

    //
    //  Bevan Cheeseman 2018
    //
    //  Simple ray case example, multi ray, accumulating, parralell projection
    //
    //


    unsigned int imageWidth = apr.pc_data.org_dims[1];
    unsigned int imageHeight = apr.pc_data.org_dims[0];

    ////////////////////////////////
    //  Set up the projection stuff
    ///////////////////////////////

    float height = pars.height;

    float radius = pars.radius_factor * apr.pc_data.org_dims[0];

    ///////////////////////////////////////////
    //
    //  Set up Perspective
    //
    ////////////////////////////////////////////

    float x0 = height * apr.pc_data.org_dims[1] * pars.scale_x;
    float y0 = apr.pc_data.org_dims[0] * .5 * pars.scale_y;
    float z0 = apr.pc_data.org_dims[2] * .5 * pars.scale_z;

    float x0f = height * apr.pc_data.org_dims[1]* pars.scale_x;
    float y0f = apr.pc_data.org_dims[0] * .5 * pars.scale_y;
    float z0f = apr.pc_data.org_dims[2] * .5 * pars.scale_z;

    float theta_0 = pars.theta_0;
    float theta_f = pars.theta_final;
    float theta_delta = pars.theta_delta;

    int num_views = floor((theta_f - theta_0)/theta_delta) + 1;


    cast_views.initialize(imageHeight,imageWidth,num_views,0);

    unsigned int view_count = 0;

    float init_val;

    init_val = 0;

    Part_timer timer;

    timer.verbose_flag = true;

    timer.start_timer("ray cast parts");

    for (float theta = theta_0; theta <= theta_f; theta += theta_delta) {

        //////////////////////////////
        ///
        /// Create an projected image at each depth
        ///
        //////////////////////////////

        std::vector<Mesh_data<S>> depth_slice;

        depth_slice.resize(apr.pc_data.depth_max + 1);

        depth_slice[apr.pc_data.depth_max].initialize(imageHeight,imageWidth,1,init_val);

        std::vector<int> depth_vec;
        depth_vec.resize(apr.pc_data.depth_max + 1);

        for(int i = apr.pc_data.depth_min;i < apr.pc_data.depth_max;i++){
            float d = pow(2,apr.pc_data.depth_max - i);
            depth_slice[i].initialize(ceil(depth_slice[apr.pc_data.depth_max].y_num/d),ceil(depth_slice[apr.pc_data.depth_max].x_num/d),1,init_val);
            depth_vec[i] = d;
        }

        //////////////////////////////
        ///
        /// Set up the new camera views
        ///
        //////////////////////////////


        Camera cam = Camera(glm::vec3(x0, y0 + radius * sin(theta), z0 + radius * cos(theta)),
                            glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));
        cam.setTargeted(glm::vec3(x0f, y0f, z0f));

        cam.setPerspectiveCamera((float) imageWidth / (float) imageHeight, (float) (60.0f / 180.0f * M_PI), 0.5f,
                                 70.0f);

        // ray traced object, sitting on the origin, with no rotation applied
        RaytracedObject o = RaytracedObject(glm::vec3(0.0f, 0.0f, 0.0f), glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));

        glm::mat4 inverse_projection = glm::inverse(*cam.getProjection());
        glm::mat4 inverse_modelview = glm::inverse((*cam.getView()) * (*o.getModel()));

        const glm::mat4 mvp = (*cam.getProjection()) * (*cam.getView());

        const float x_camera = x0;
        const float y_camera = y0 + radius * sin(theta);
        const float z_camera = z0 + radius * cos(theta);

        ///////////////////////////////////////////
        ///
        /// Perform ray cast over APR
        ///
        /////////////////////////////////////////

        //  Set up the APR parallel iterators (these are required for the parallel iteration)

        APR_iterator<float> apr_it;
        uint64_t part;
        apr.init_by_part_iteration(apr_it);

#pragma omp parallel for schedule(static) private(part) firstprivate(apr_it,mvp)
        for (part = 0; part < apr.num_parts_total; ++part) {
            apr_it.set_part(part);

            //get apr info
            const int y = apr_it.y();

            const float y_actual = apr_it.y_global()*pars.scale_y;
            const float x_actual = apr_it.x_global()*pars.scale_x;
            const float z_actual = apr_it.z_global()*pars.scale_z;

            const int depth = apr_it.depth();

            //set up the ray position
            glm::vec2 pos = o.worldToScreen(mvp, glm::vec3(x_actual ,y_actual, z_actual), depth_slice[depth].x_num, depth_slice[depth].y_num);

            const int dim1 = round(-pos.y);
            const int dim2 = round(-pos.x);

            if (dim1 > 0 & dim2 > 0 & (dim1 < depth_slice[depth].y_num) &
                (dim2 < depth_slice[depth].x_num)) {
                //get the particle value
                S temp_int = apr_it(particle_data);

                depth_slice[depth].mesh[dim1 + (dim2) * depth_slice[depth].y_num] = op(temp_int, depth_slice[depth].mesh[dim1 + (dim2) * depth_slice[depth].y_num]);
            }
        }

        //////////////////////////////////////////////
        ///
        /// Now merge the ray-casts between the different resolutions
        ///
        ////////////////////////////////////////////////

        uint64_t depth;

        uint64_t depth_min = apr.pc_data.depth_min;

        unsigned int y_,z_,x_,j_,i,k;

        for (depth = (depth_min); depth < apr.pc_data.depth_max; depth++) {

            const int step_size = pow(2, apr.pc_data.depth_max - depth);
#pragma omp parallel for default(shared) private(z_,x_,j_,i,k) schedule(guided) if (depth > 8)
            for (x_ = 0; x_ < depth_slice[depth].x_num; x_++) {
                //both z and x are explicitly accessed in the structure

                for (y_ = 0; y_ < depth_slice[depth].y_num; y_++) {

                    const float curr_int = depth_slice[depth].mesh[y_ + (x_) * depth_slice[depth].y_num];

                    const int dim1 = y_ * step_size;
                    const int dim2 = x_ * step_size;

                    //add to all the required rays
                    const int offset_max_dim1 = std::min((int) depth_slice[apr.pc_data.depth_max].y_num,
                                                         (int) (dim1 + step_size));
                    const int offset_max_dim2 = std::min((int) depth_slice[apr.pc_data.depth_max].x_num,
                                                         (int) (dim2 + step_size));

                    if (curr_int > 0) {

                        for (k = dim2; k < offset_max_dim2; ++k) {
                            for (i = dim1; i < offset_max_dim1; ++i) {
                                depth_slice[apr.pc_data.depth_max].mesh[i +
                                                                  (k) * depth_slice[apr.pc_data.depth_max].y_num] = op(
                                        curr_int, depth_slice[apr.pc_data.depth_max].mesh[i + (k) *
                                                                                        depth_slice[apr.pc_data.depth_max].y_num]);

                            }
                        }
                    }

                }
            }
        }


        //copy data across
        std::copy(depth_slice[apr.pc_data.depth_max].mesh.begin(),depth_slice[apr.pc_data.depth_max].mesh.end(),cast_views.mesh.begin() + view_count*imageHeight*imageWidth);

        view_count++;
    }

    timer.stop_timer();

};

template<typename S,typename U>
float perpsective_mesh_raycast(PartCellStructure<U,uint64_t>& pc_struct,proj_par& pars,Mesh_data<S>& image) {
    //
    //  Bevan Cheeseman 2017
    //
    //  Max Ray Cast Proposective Projection
    //
    //



    unsigned int imageWidth = image.x_num;
    unsigned int imageHeight = image.y_num;

    float height = pars.height;

    float radius = pars.radius_factor * image.y_num;

    ///////////////////////////////////////////
    //
    //  Set up Perspective
    //
    ////////////////////////////////////////////

    float x0 = height * image.x_num * pars.scale_x;
    float y0 = image.y_num * .5 * pars.scale_y;
    float z0 = image.z_num * .5 * pars.scale_z;

    float x0f = height * image.x_num * pars.scale_x;
    float y0f = image.y_num * .5 * pars.scale_y;
    float z0f = image.z_num * .5 * pars.scale_z;

    float theta_0 = pars.theta_0;
    float theta_f = pars.theta_final;
    float theta_delta = pars.theta_delta;

    int num_views = floor((theta_f - theta_0)/theta_delta) + 1;

    Mesh_data<S> cast_views;

    cast_views.initialize(imageHeight,imageWidth,num_views,0);

    Part_timer timer;

    timer.verbose_flag = false;

    uint64_t view_count = 0;

    timer.start_timer("ray cast mesh prospective");

    for (float theta = theta_0; theta <= theta_f; theta += theta_delta) {



        Camera cam = Camera(glm::vec3(x0, y0 + radius * sin(theta), z0 + radius * cos(theta)),
                            glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));
        cam.setTargeted(glm::vec3(x0f, y0f, z0f));

        cam.setPerspectiveCamera((float) imageWidth / (float) imageHeight, (float) (60.0f / 180.0f * M_PI), 0.5f,
                                 70.0f);

//    cam.setOrthographicCamera(imageWidth, imageHeight, 1.0f, 200.0f);
        // ray traced object, sitting on the origin, with no rotation applied
        RaytracedObject o = RaytracedObject(glm::vec3(0.0f, 0.0f, 0.0f), glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));

        glm::mat4 inverse_projection = glm::inverse(*cam.getProjection());
        glm::mat4 inverse_modelview = glm::inverse((*cam.getView()) * (*o.getModel()));

        const glm::mat4 mvp = (*cam.getProjection()) * (*cam.getView());


        Mesh_data<S> proj_img;
        proj_img.initialize(imageHeight, imageWidth, 1, 0);

        //Need to add here a parameters here


        bool end_domain = false;

        //choose random direction to propogate along


        int counter = 0;


        const int dir = pars.direction;

        int z_, x_, j_, y_, i, k;

        //loop over the resolutions of the structure
        const unsigned int x_num_ = image.x_num;
        const unsigned int z_num_ = image.z_num;
        const float step_size = 1;
        const unsigned int y_num_ = image.y_num;

#pragma omp parallel for default(shared) private(z_,x_,j_,i,k) firstprivate(mvp)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_ * z_ + x_;

                for (j_ = 0; j_ < y_num_; j_++) {


                    glm::vec2 pos = o.worldToScreen(mvp, glm::vec3((float) x_*pars.scale_x, (float) j_*pars.scale_y, (float) z_*pars.scale_z), imageWidth,
                                                    imageHeight);

                    const S temp_int = image.mesh[j_ + x_ * image.y_num + z_ * image.x_num * image.y_num];

                    const int dim1 = round(-pos.y);
                    const int dim2 = round(-pos.x);

                    if (dim1 > 0 & dim2 > 0 & (dim1 < proj_img.y_num) & (dim2 < proj_img.x_num)) {

                        proj_img.mesh[dim1 + (dim2) * proj_img.y_num] = std::max(temp_int, proj_img.mesh[dim1 + (dim2) *
                                                                                                                proj_img.y_num]);
                    }
                }
            }
        }

        std::copy(proj_img.mesh.begin(),proj_img.mesh.end(),cast_views.mesh.begin() + view_count*imageHeight*imageWidth);

        view_count++;


    }

    timer.stop_timer();



    debug_write(cast_views, pars.name + "perspective_mesh_projection");

    return (timer.t2 - timer.t1);


}

template<typename S,class UnaryOperator,typename U>
void apr_perspective_raycast_depth(ExtraPartCellData<uint16_t>& y_vec,ExtraPartCellData<S>& particle_data,ExtraPartCellData<U>& particle_data_2,proj_par& pars,UnaryOperator op,const bool depth_scale = false){
    //
    //  Bevan Cheeseman 2017
    //
    //  Simple ray case example, multi ray, accumulating, parralell projection
    //
    //

    //////////////////////////////
    //
    //  This creates data sets where each particle is a cell.
    //
    //  This same code can be used where there are multiple particles per cell as in original pc_struct, however, the particles have to be accessed in a different way.
    //
    //////////////////////////////



    //Need to add here a parameters here

    unsigned int imageWidth = y_vec.org_dims[1];
    unsigned int imageHeight = y_vec.org_dims[0];

    //
    //  Set up the projection stuff
    //
    //
    //

    float height = pars.height;

    float radius = pars.radius_factor * y_vec.org_dims[0];

    ///////////////////////////////////////////
    //
    //  Set up Perspective
    //
    ////////////////////////////////////////////

    float x0 = height * y_vec.org_dims[1] * pars.scale_x;
    float y0 = y_vec.org_dims[0] * .5 * pars.scale_y;
    float z0 = y_vec.org_dims[2] * .5 * pars.scale_z;

    float x0f = height * y_vec.org_dims[1]* pars.scale_x;
    float y0f = y_vec.org_dims[0] * .5 * pars.scale_y;
    float z0f = y_vec.org_dims[2] * .5 * pars.scale_z;

    float theta_0 = pars.theta_0;
    float theta_f = pars.theta_final;
    float theta_delta = pars.theta_delta;

    int num_views = floor((theta_f - theta_0)/theta_delta) + 1;

    Mesh_data<S> cast_views;

    cast_views.initialize(imageHeight,imageWidth,num_views,0);

    unsigned int view_count = 0;

    float init_val;

    if(depth_scale){

        init_val = 64000;

    } else {
        init_val = 0;
    }


    Part_timer timer;

    timer.verbose_flag = true;


    timer.start_timer("ray cast parts");


    for (float theta = theta_0; theta <= theta_f; theta += theta_delta) {

        std::vector<Mesh_data<float>> depth_slice;

        depth_slice.resize(y_vec.depth_max + 1);

        depth_slice[y_vec.depth_max].initialize(imageHeight,imageWidth,1,init_val);

        std::vector<int> depth_vec;
        depth_vec.resize(y_vec.depth_max + 1);

        for(int i = y_vec.depth_min;i < y_vec.depth_max;i++){
            float d = pow(2,y_vec.depth_max - i);
            depth_slice[i].initialize(ceil(depth_slice[y_vec.depth_max].y_num/d),ceil(depth_slice[y_vec.depth_max].x_num/d),1,init_val);
            depth_vec[i] = d;
        }

        std::vector<Mesh_data<S>> depth_slice_2;

        depth_slice_2.resize(y_vec.depth_max + 1);

        depth_slice_2[y_vec.depth_max].initialize(imageHeight,imageWidth,1,0);

        for(int i = y_vec.depth_min;i < y_vec.depth_max;i++){
            float d = pow(2,y_vec.depth_max - i);
            depth_slice_2[i].initialize(ceil(depth_slice[y_vec.depth_max].y_num/d),ceil(depth_slice[y_vec.depth_max].x_num/d),1,0);

        }



        Camera cam = Camera(glm::vec3(x0, y0 + radius * sin(theta), z0 + radius * cos(theta)),
                            glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));
        cam.setTargeted(glm::vec3(x0f, y0f, z0f));

        cam.setPerspectiveCamera((float) imageWidth / (float) imageHeight, (float) (60.0f / 180.0f * M_PI), 0.5f,
                                 70.0f);

//    cam.setOrthographicCamera(imageWidth, imageHeight, 1.0f, 200.0f);
        // ray traced object, sitting on the origin, with no rotation applied
        RaytracedObject o = RaytracedObject(glm::vec3(0.0f, 0.0f, 0.0f), glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));

        glm::mat4 inverse_projection = glm::inverse(*cam.getProjection());
        glm::mat4 inverse_modelview = glm::inverse((*cam.getView()) * (*o.getModel()));

        const glm::mat4 mvp = (*cam.getProjection()) * (*cam.getView());


        const float x_camera = x0;
        const float y_camera = y0 + radius * sin(theta);
        const float z_camera = z0 + radius * cos(theta);

        int z_, x_, j_, y_, i, k;

        for (uint64_t depth = (y_vec.depth_min); depth <= y_vec.depth_max; depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = y_vec.x_num[depth];
            const unsigned int z_num_ = y_vec.z_num[depth];

            const unsigned int y_size = depth_slice[depth].y_num;
            const unsigned int x_size = depth_slice[depth].x_num;

            const float step_size_x = pow(2, y_vec.depth_max - depth)* pars.scale_x;
            const float step_size_y = pow(2, y_vec.depth_max - depth)* pars.scale_y;
            const float step_size_z = pow(2, y_vec.depth_max - depth)* pars.scale_z;


#pragma omp parallel for default(shared) private(z_,x_,j_,i,k) firstprivate(mvp)  schedule(guided) if(z_num_*x_num_ > 1000)
            for (z_ = 0; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = 0; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_ * z_ + x_;

                    for (j_ = 0; j_ < y_vec.data[depth][pc_offset].size(); j_++) {


                        const int y = y_vec.data[depth][pc_offset][j_];

                        const float y_actual = (y+0.5) * step_size_y;
                        const float x_actual = (x_+0.5) * step_size_x;
                        const float z_actual = (z_+0.5) * step_size_z;


                        glm::vec2 pos = o.worldToScreen(mvp, glm::vec3(x_actual ,y_actual, z_actual), x_size, y_size);

                        const int dim1 = round(-pos.y);
                        const int dim2 = round(-pos.x);



                        if (dim1 > 0 & dim2 > 0 & (dim1 < depth_slice[depth].y_num) &
                            (dim2 < depth_slice[depth].x_num)) {

                                float distance = sqrt(pow(y_camera-y_actual,2)+pow(x_camera-x_actual,2)+pow(z_camera-z_actual,2));

                                const float temp_depth = (particle_data.data[depth][pc_offset][j_] > 0)*distance;

                                if(temp_depth > 0) {

                                    const float curr_depth = depth_slice[depth].mesh[dim1 + (dim2) * depth_slice[depth].y_num];

                                    if(temp_depth < curr_depth){
                                        depth_slice[depth].mesh[dim1 + (dim2) * depth_slice[depth].y_num]  = temp_depth;

                                        const S val = particle_data_2.data[depth][pc_offset][j_] ;

                                        depth_slice_2[depth].mesh[dim1 + (dim2) * depth_slice[depth].y_num]  = val;
                                    }

                                }

                        }
                    }
                }
            }
        }


        uint64_t depth;

        uint64_t depth_min = y_vec.depth_min;


        for (depth = (depth_min); depth < y_vec.depth_max; depth++) {

            const int step_size = pow(2, y_vec.depth_max - depth);
//#pragma omp parallel for default(shared) private(z_,x_,j_,i,k) schedule(guided) if (depth > 8)
            for (x_ = 0; x_ < depth_slice[depth].x_num; x_++) {
                //both z and x are explicitly accessed in the structure

                for (y_ = 0; y_ < depth_slice[depth].y_num; y_++) {

                    const float curr_int = depth_slice[depth].mesh[y_ + (x_) * depth_slice[depth].y_num];
                    const float curr_int_2 = depth_slice_2[depth].mesh[y_ + (x_) * depth_slice[depth].y_num];

                    const int dim1 = y_ * step_size;
                    const int dim2 = x_ * step_size;

                    //add to all the required rays
                    const int offset_max_dim1 = std::min((int) depth_slice[y_vec.depth_max].y_num,
                                                         (int) (dim1 + step_size));
                    const int offset_max_dim2 = std::min((int) depth_slice[y_vec.depth_max].x_num,
                                                         (int) (dim2 + step_size));

                    if (curr_int < 64000) {

                        for (k = dim2; k < offset_max_dim2; ++k) {
                            for (i = dim1; i < offset_max_dim1; ++i) {

                                const float q = depth_slice[y_vec.depth_max].mesh[i + (k) *
                                                                                           depth_slice[y_vec.depth_max].y_num];

                                if(curr_int < q){
                                    depth_slice[y_vec.depth_max].mesh[i +
                                                                      (k) * depth_slice[y_vec.depth_max].y_num]  = curr_int;

                                    depth_slice_2[y_vec.depth_max].mesh[i +
                                                                      (k) * depth_slice[y_vec.depth_max].y_num]  = curr_int_2;
                                }



                            }
                        }
                    }

                }
            }
        }


        //copy data across
        std::copy(depth_slice_2[y_vec.depth_max].mesh.begin(),depth_slice_2[y_vec.depth_max].mesh.end(),cast_views.mesh.begin() + view_count*imageHeight*imageWidth);

        view_count++;
    }

    timer.stop_timer();


    debug_write(cast_views, pars.name + "_perspective_part_projection_dual");

}




#endif
