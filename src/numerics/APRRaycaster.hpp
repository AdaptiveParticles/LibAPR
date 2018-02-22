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

#include "../data_structures/APR/ExtraPartCellData.hpp"
#include "../data_structures/APR/APR.hpp"
#include "../../src/vis/Camera.h"
#include "../../src/vis/RaytracedObject.h"

class APRRaycaster {

public:

    int direction = 0;

    float height = 0.5;
    float radius_factor = 1.5;
    float theta_0 = 0;
    float theta_final = .1;
    float theta_delta = 0.01;

    float scale_y = 1.0;
    float scale_x = 1.0;
    float scale_z = 1.0;

    float jitter_factor = 0.5;

    bool jitter = false;

    std::string name = "raycast";

    APRRaycaster(){
    }

    template<typename U,typename S,typename V,class BinaryOperation>
    void perform_raycast(APR<U>& apr,ExtraParticleData<S>& particle_data,MeshData<V>& cast_views,BinaryOperation op);

    template<typename S,typename U>
    float perpsective_mesh_raycast(MeshData<S>& image,MeshData<U>& cast_views);
};


template<typename U,typename S,typename V,class BinaryOperation>
void APRRaycaster::perform_raycast(APR<U>& apr,ExtraParticleData<S>& particle_data,MeshData<V>& cast_views,BinaryOperation op) {

    //
    //  Bevan Cheeseman 2018
    //
    //  Simple ray case example, multi ray, accumulating, parralell projection
    //
    //


    uint64_t imageWidth = apr.orginal_dimensions(1);
    uint64_t imageHeight = apr.orginal_dimensions(0);

    ////////////////////////////////
    //  Set up the projection stuff
    ///////////////////////////////

    float height = this->height;

    float radius = this->radius_factor * apr.orginal_dimensions(0);

    ///////////////////////////////////////////
    //
    //  Set up Perspective
    //
    ////////////////////////////////////////////

    float x0 = height * apr.orginal_dimensions(1) * this->scale_x;
    float y0 = apr.orginal_dimensions(0) * .5 * this->scale_y;
    float z0 = apr.orginal_dimensions(2) * .5 * this->scale_z;

    float x0f = height * apr.orginal_dimensions(1)* this->scale_x;
    float y0f = apr.orginal_dimensions(0) * .5 * this->scale_y;
    float z0f = apr.orginal_dimensions(2) * .5 * this->scale_z;

    float theta_0 = this->theta_0;
    float theta_f = this->theta_final;
    float theta_delta = this->theta_delta;

    uint64_t num_views = floor((theta_f - theta_0)/theta_delta) ;

    cast_views.init(imageHeight, imageWidth, num_views, 0);

    uint64_t view_count = 0;
    float init_val=0;


    APRTimer timer;

    timer.verbose_flag = true;

    timer.start_timer("Compute APR maximum projection raycast");

    /////////////////////////////////////
    ///
    ///  Initialization of loop variables
    ///
    /////////////////////////////////////

    std::vector<MeshData<S>> depth_slice;
    depth_slice.resize(apr.level_max() + 1);

    std::vector<float> depth_vec;
    depth_vec.resize(apr.level_max() + 1);
    depth_slice[apr.level_max()].init(imageHeight,imageWidth,1,init_val);


    for(size_t i = apr.level_min();i < apr.level_max();i++){
        float d = pow(2,apr.level_max() - i);
        depth_slice[i].init(ceil(depth_slice[apr.level_max()].y_num/d),ceil(depth_slice[apr.level_max()].x_num/d),1,init_val);
        depth_vec[i] = d;
    }

    depth_vec[apr.level_max()] = 1;

    //jitter the parts to remove ray cast artifacts
    const bool jitter = this->jitter;
    const float jitter_factor = this->jitter_factor;

    ExtraParticleData<float> jitter_x;
    ExtraParticleData<float> jitter_y;
    ExtraParticleData<float> jitter_z;

    //initialize the iterator
    APRIterator<U> apr_iterator(apr);
    uint64_t particle_number;

    if(jitter){

        jitter_x.init(apr);
        jitter_y.init(apr);
        jitter_z.init(apr);

        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            jitter_x[apr_iterator] = jitter_factor*((std::rand()-500)%1000)/1000.0f;
            jitter_y[apr_iterator] = jitter_factor*((std::rand()-500)%1000)/1000.0f;
            jitter_z[apr_iterator] = jitter_factor*((std::rand()-500)%1000)/1000.0f;
        }
    }

    //main loop changing the view angle
    for (float theta = theta_0; theta <= theta_f; theta += theta_delta) {

        //////////////////////////////
        ///
        /// Create an projected image at each depth
        ///
        //////////////////////////////

        for(size_t i = apr.level_min();i <= apr.level_max();i++){
            std::fill(depth_slice[i].mesh.begin(),depth_slice[i].mesh.end(),init_val);
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

        const glm::mat4 mvp = (*cam.getProjection()) * (*cam.getView());

        ///////////////////////////////////////////
        ///
        /// Perform ray cast over APR
        ///
        /////////////////////////////////////////

        //  Set up the APR parallel iterators (these are required for the parallel iteration)

        float y_actual,x_actual,z_actual;

#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number,y_actual,x_actual,z_actual) firstprivate(apr_iterator,mvp)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            //get apr info

            if(jitter){
                 y_actual = (apr_iterator.y() + 0.5f + jitter_y[apr_iterator])*this->scale_y*depth_vec[apr_iterator.level()];
                 x_actual = (apr_iterator.x() + 0.5f + jitter_x[apr_iterator])*this->scale_x*depth_vec[apr_iterator.level()];
                 z_actual = (apr_iterator.z() + 0.5f + jitter_z[apr_iterator])*this->scale_z*depth_vec[apr_iterator.level()];
            } else{
                 y_actual = apr_iterator.y_global()*this->scale_y;
                 x_actual = apr_iterator.x_global()*this->scale_x;
                 z_actual = apr_iterator.z_global()*this->scale_z;
            }

            const int level = apr_iterator.level();

            //set up the ray position
            glm::vec2 pos = o.worldToScreen(mvp, glm::vec3(x_actual ,y_actual, z_actual), depth_slice[level].x_num, depth_slice[level].y_num);

            const int dim1 = round(-pos.y);
            const int dim2 = round(-pos.x);

            if ((dim1 > 0) & (dim2 > 0) & (dim1 < (int64_t)depth_slice[level].y_num) & (dim2 < (int64_t)depth_slice[level].x_num)) {
                //get the particle value
                S temp_int = particle_data[apr_iterator];

                depth_slice[level].mesh[dim1 + (dim2) * depth_slice[level].y_num] = op(temp_int, depth_slice[level].mesh[dim1 + (dim2) * depth_slice[level].y_num]);
            }
        }

        //////////////////////////////////////////////
        ///
        /// Now merge the ray-casts between the different resolutions
        ///
        ////////////////////////////////////////////////

        uint64_t level;

        uint64_t level_min = apr.level_min();

        unsigned int y_,x_,i,k;

        for (level = (level_min); level < apr.level_max(); level++) {

            const unsigned int step_size = pow(2, apr.level_max() - level);
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(x_,i,k) schedule(guided) if (level > 8)
#endif
            for (x_ = 0; x_ < depth_slice[level].x_num; x_++) {
                //both z and x are explicitly accessed in the structure

                for (y_ = 0; y_ < depth_slice[level].y_num; y_++) {

                    const float curr_int = depth_slice[level].mesh[y_ + (x_) * depth_slice[level].y_num];

                    const unsigned int dim1 = y_ * step_size;
                    const unsigned int dim2 = x_ * step_size;

                    //add to all the required rays
                    const unsigned int offset_max_dim1 = std::min((int) depth_slice[apr.level_max()].y_num,
                                                         (int) (dim1 + step_size));
                    const unsigned int offset_max_dim2 = std::min((int) depth_slice[apr.level_max()].x_num,
                                                         (int) (dim2 + step_size));

                    if (curr_int > 0) {

                        for (k = dim2; k < offset_max_dim2; ++k) {
                            for (i = dim1; i < offset_max_dim1; ++i) {
                                depth_slice[apr.level_max()].mesh[i +
                                                                  (k) * depth_slice[apr.level_max()].y_num] = op(
                                        curr_int, depth_slice[apr.level_max()].mesh[i + (k) *
                                                                                        depth_slice[apr.level_max()].y_num]);

                            }
                        }
                    }

                }
            }
        }


        //copy data across
        std::copy(depth_slice[apr.level_max()].mesh.begin(),depth_slice[apr.level_max()].mesh.end(),cast_views.mesh.begin() + view_count*imageHeight*imageWidth);

        view_count++;


        if(view_count >= num_views){
            break;
        }
    }

    timer.stop_timer();
    float elapsed_seconds = timer.t2 - timer.t1;

    std::cout << elapsed_seconds/(view_count*1.0) <<  " seconds per view" << std::endl;


}

template<typename S,typename U>
float APRRaycaster::perpsective_mesh_raycast(MeshData<S>& image,MeshData<U>& cast_views) {
    //
    //  Bevan Cheeseman 2017
    //
    //  Max Ray Cast Proposective Projection on mesh
    //
    //

    uint64_t imageWidth = image.x_num;
    uint64_t imageHeight = image.y_num;

    float height = this->height;

    float radius = this->radius_factor * image.y_num;

    ///////////////////////////////////////////
    //
    //  Set up Perspective
    //
    ////////////////////////////////////////////

    float x0 = height * image.x_num * this->scale_x;
    float y0 = image.y_num * .5f * this->scale_y;
    float z0 = image.z_num * .5f * this->scale_z;

    float x0f = height * image.x_num * this->scale_x;
    float y0f = image.y_num * .5f * this->scale_y;
    float z0f = image.z_num * .5f * this->scale_z;

    float theta_0 = this->theta_0;
    float theta_f = this->theta_final;
    float theta_delta = this->theta_delta;

    int num_views = (int) floor((theta_f - theta_0)/theta_delta);


    cast_views.init(imageHeight, imageWidth, num_views, 0);

    APRTimer timer;

    timer.verbose_flag = true;

    int64_t view_count = 0;

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

        const glm::mat4 mvp = (*cam.getProjection()) * (*cam.getView());


        MeshData<S> proj_img;
        proj_img.init(imageHeight, imageWidth, 1, 0);

        unsigned int z_, x_, j_;

        //loop over the resolutions of the structure
        const unsigned int x_num_ = image.x_num;
        const unsigned int z_num_ = image.z_num;
        const unsigned int y_num_ = image.y_num;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(mvp)
#endif
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                for (j_ = 0; j_ < y_num_; j_++) {


                    glm::vec2 pos = o.worldToScreen(mvp, glm::vec3((float) x_*this->scale_x, (float) j_*this->scale_y, (float) z_*this->scale_z), imageWidth,
                                                    imageHeight);

                    const S temp_int = image.mesh[j_ + x_ * image.y_num + z_ * image.x_num * image.y_num];

                    const int dim1 = (int) round(-pos.y);
                    const int dim2 = (int) round(-pos.x);

                    if ((dim1 > 0) & (dim2 > 0) & (dim1 < (int64_t)proj_img.y_num) & (dim2 < (int64_t)proj_img.x_num)) {

                        proj_img.mesh[dim1 + (dim2) * proj_img.y_num] = std::max(temp_int, proj_img.mesh[dim1 + (dim2) *
                                                                                                                proj_img.y_num]);
                    }
                }
            }
        }

        std::copy(proj_img.mesh.begin(),proj_img.mesh.end(),cast_views.mesh.begin() + view_count*imageHeight*imageWidth);

        view_count++;

        if(view_count == num_views){
            break;
        }


    }

    timer.stop_timer();
    float elapsed_seconds = (float)(timer.t2 - timer.t1);

    return elapsed_seconds;
}





#endif
