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
#include <cmath>

#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/APR/particles/PartCellData.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"

#include "data_structures/APR/APR.hpp"

#include "numerics/APRReconstruction.hpp"

#include "vis/Camera.h"
#include "vis/Object.h"
#include "vis/RaytracedObject.h"
#include "vis/RaycastUtils.h"

class APRRaycaster {

public:

    int direction = 0;

    float height = 0.5;
    float radius_factor = 1.5;
    float theta_0 = 0;
    float theta_final = .1;
    float theta_delta = 0.01;

    float phi = 0;
    float phi_s = 0;

    float scale_y = 1.0;
    float scale_x = 1.0;
    float scale_z = 1.0;

    float jitter_factor = 0.5;

    bool jitter = false;

    float scale_down = 1.0;

    std::string name = "raycast";

    template<typename S, typename V, class BinaryOperation>
            void perform_raycast(APR &apr, ParticleData<S> &particle_data, PixelData<V> &cast_views, BinaryOperation op);

    template<typename S, typename V, typename T, class BinaryOperation>
    void perform_raycast_patch(APR &apr, ParticleData<S> &particle_data,
                               ParticleData<T> &treeData, PixelData<V> &cast_views, ReconPatch rp,
                               BinaryOperation op);

    template<typename S, typename U>
    float perpsective_mesh_raycast(PixelData<S> &image, PixelData<U> &cast_views);

    // Stuff below is hiding implementation so eventually glm header files do not
    // have to be delivered with libAPR
    GlmObjectsContainer *glmObjects;

private:

    std::vector<PixelData<uint16_t>> depth_slice;


};

template<typename S, typename V, typename T, class BinaryOperation>
void APRRaycaster::perform_raycast_patch(APR &apr, ParticleData<S> &particle_data,
                                         ParticleData<T> &treeData, PixelData<V> &cast_views,
                                         ReconPatch reconPatch, BinaryOperation op) {

    //
    //  Bevan Cheeseman 2018
    //
    //  Simple ray case example, multi ray, accumulating, parralell projection
    //
    //


    int max_level = apr.level_max() + reconPatch.level_delta;

    int max_img_y = ceil(apr.org_dims(0) * pow(2.0, reconPatch.level_delta));
    int max_img_x = ceil(apr.org_dims(1) * pow(2.0, reconPatch.level_delta));
    int max_img_z = ceil(apr.org_dims(2) * pow(2.0, reconPatch.level_delta));


    if (reconPatch.y_end == -1) {
        reconPatch.y_begin = 0;
        reconPatch.y_end = max_img_y;
    } else {
        reconPatch.y_begin = std::max(0, reconPatch.y_begin);
        reconPatch.y_end = std::min(max_img_y, reconPatch.y_end);
    }

    if (reconPatch.x_end == -1) {
        reconPatch.x_begin = 0;
        reconPatch.x_end = max_img_x;
    } else {
        reconPatch.x_begin = std::max(0, reconPatch.x_begin);
        reconPatch.x_end = std::min(max_img_x, reconPatch.x_end);
    }

    if (reconPatch.z_end == -1) {
        reconPatch.z_begin = 0;
        reconPatch.z_end = max_img_z;
    } else {
        reconPatch.z_begin = std::max(0, reconPatch.z_begin);
        reconPatch.z_end = std::min(max_img_z, reconPatch.z_end);
    }


    const int x_begin = reconPatch.x_begin;
    const int x_end = reconPatch.x_end;

    const int z_begin = reconPatch.z_begin;
    const int z_end = reconPatch.z_end;

    const int y_begin = reconPatch.y_begin;
    const int y_end = reconPatch.y_end;

    if (y_begin > y_end || x_begin > x_end || z_begin > z_end) {
        std::cout << "Invalid Patch Size: Exiting" << std::endl;
        return;
    }

    int imageWidth = apr.org_dims(1);
    int imageHeight = apr.org_dims(0);

    if(scale_down != 1){
        imageWidth = scale_down*imageWidth;
        imageHeight = scale_down*imageHeight;

    }


    ////////////////////////////////
    //  Set up the projection stuff
    ///////////////////////////////

    float height = this->height;

    float radius = this->radius_factor * apr.org_dims(0);

    ///////////////////////////////////////////
    //
    //  Set up Perspective
    //
    ////////////////////////////////////////////

    float x0 = height * apr.org_dims(1)* this->scale_x ;
    float y0 = apr.org_dims(0) * .5 * this->scale_y;
    float z0 = apr.org_dims(2) * .5 * this->scale_z;

    float x0f = height * apr.org_dims(1) * this->scale_x;
    float y0f = apr.org_dims(0) * .5 * this->scale_y;
    float z0f = apr.org_dims(2) * .5 * this->scale_z;

    float theta_0 = this->theta_0;
//    float theta_f = this->theta_final;
//    float theta_delta = this->theta_delta;

    float phi = this->phi;

    int num_views = 1;

    cast_views.initWithResize(imageHeight, imageWidth, num_views);

    int view_count = 0;
    float init_val = 0;


    APRTimer timer;

    timer.verbose_flag = true;

    timer.start_timer("Compute APR maximum projection raycast");

    /////////////////////////////////////
    ///
    ///  Initialization of loop variables
    ///
    /////////////////////////////////////


    depth_slice.resize(max_level + 1);

    VectorData<float> depth_vec;
    depth_vec.resize(max_level + 1);

    int max_threads = 1;

    depth_slice[max_level].initWithResize(imageHeight, imageWidth, max_threads);
    depth_slice[max_level].fill(init_val);

    for (int i = apr.level_min(); i < max_level; i++) {
        float d = pow(2, max_level - i);
        depth_slice[i].initWithResize(ceil(depth_slice[max_level].y_num / d),
                                     ceil(depth_slice[max_level].x_num / d), max_threads);
        depth_slice[i].fill(init_val);
        depth_vec[i] = d;
    }

    depth_slice[max_level].swap(cast_views);

    depth_vec[max_level] = 1;

    float theta = theta_0;

    //jitter the parts to remove ray cast artifacts
    const bool jitter = this->jitter;
    const float jitter_factor = this->jitter_factor;

    ParticleData<float> jitter_x;
    ParticleData<float> jitter_y;
    ParticleData<float> jitter_z;

    //initialize the iterator
    auto apr_iterator = apr.iterator();

    if (jitter) {

        jitter_x.init(apr.total_number_particles());
        jitter_y.init(apr.total_number_particles());
        jitter_z.init(apr.total_number_particles());

        for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.z_num(level); z++) {
                for (x = 0; x < apr_iterator.x_num(level); ++x) {
                    for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                         apr_iterator++) {
                        jitter_x[apr_iterator] = jitter_factor * ((std::rand() - 500) % 1000) / 1000.0f;
                        jitter_y[apr_iterator] = jitter_factor * ((std::rand() - 500) % 1000) / 1000.0f;
                        jitter_z[apr_iterator] = jitter_factor * ((std::rand() - 500) % 1000) / 1000.0f;
                    }
                }
            }
        }
    }


    //main loop changing the view angle
    //for (float theta = theta_0; theta <= theta_f; theta += theta_delta) {

        //////////////////////////////
        ///
        /// Create an projected image at each depth
        ///
        //////////////////////////////

        for (int i = apr.level_min(); i <= max_level; i++) {
            std::fill(depth_slice[i].mesh.begin(), depth_slice[i].mesh.end(), init_val);
        }

        //////////////////////////////
        ///
        /// Set up the new camera views
        ///
        //////////////////////////////
        initObjects(glmObjects, imageWidth, imageHeight, radius, theta, x0, y0, z0, x0f, y0f, z0f, phi, phi_s);

        ///////////////////////////////////////////
        ///
        /// Perform ray cast over APR
        ///
        /////////////////////////////////////////

        //  Set up the APR parallel iterators (these are required for the parallel iteration)


        //note the use of the dynamic OpenMP schedule.
        for (int level = std::min(max_level, apr.level_max());
             level >= apr_iterator.level_min(); --level) {

            const float step_size = pow(2, max_level - level);

            int x_begin_l = (int) floor(x_begin / step_size);
            int x_end_l = std::min((int) ceil(x_end / step_size), apr_iterator.x_num(level));

            int z_begin_l = (int) floor(z_begin / step_size);
            int z_end_l = std::min((int) ceil(z_end / step_size), apr_iterator.z_num(level));

            int y_begin_l = (int) floor(y_begin / step_size);
            int y_end_l = std::min((int) ceil(y_end / step_size), apr_iterator.y_num(level));

            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(auto) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = z_begin_l; z < z_end_l; z++) {
                for (x = x_begin_l; x < x_end_l; ++x) {

                    float y_actual, x_actual, z_actual;

                    for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                         apr_iterator++) {

                        if ((apr_iterator.y() >= y_begin_l) && (apr_iterator.y() < y_end_l)) {

                            //get apr info

                            if (jitter) {
                                y_actual = (apr_iterator.y() + 0.5f + jitter_y[apr_iterator]) * this->scale_y *
                                           depth_vec[level];
                                x_actual = (x + 0.5f + jitter_x[apr_iterator]) * this->scale_x *
                                           depth_vec[level];
                                z_actual = (z + 0.5f + jitter_z[apr_iterator]) * this->scale_z *
                                           depth_vec[level];
                            } else {
                                y_actual = apr_iterator.y_global(level,apr_iterator.y()) * this->scale_y;
                                x_actual = apr_iterator.x_global(level,x) * this->scale_x;
                                z_actual = apr_iterator.z_global(level,z) * this->scale_z;
                            }


                            int dim1 = 0;
                            int dim2 = 0;
                            getPos(glmObjects,dim1, dim2, x_actual, y_actual, z_actual, depth_slice[level].x_num,
                                   depth_slice[level].y_num);

                            //dim1 = dim1*scale_down;
                            //dim2 = dim2*scale_down;

                            if ((dim1 > 0) & (dim2 > 0) &
                                (dim1 < depth_slice[level].y_num) &
                                (dim2 < depth_slice[level].x_num)) {
                                //get the particle value
                                const S temp_int = particle_data[apr_iterator];

                                depth_slice[level].at(dim1,dim2,0) = op(temp_int,
                                                                          depth_slice[level].at(dim1,dim2,0));

//                                depth_slice[level].mesh[dim1 + (dim2) * depth_slice[level].y_num] = op(temp_int,
//                                                                                                       depth_slice[level].mesh[
//                                                                                                               dim1 +
//                                                                                                               (dim2) *
//                                                                                                               depth_slice[level].y_num]);
                            }
                        } else {
                            if ((apr_iterator.y() >= y_end_l)) {
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (max_level < apr_iterator.level_max()) {


            auto tree_it = apr.tree_iterator();


            int level = max_level;

            const float step_size = pow(2, max_level - level);

            int x_begin_l = (int) floor(x_begin / step_size);
            int x_end_l = std::min((int) ceil(x_end / step_size), apr_iterator.x_num(level));

            int z_begin_l = (int) floor(z_begin / step_size);
            int z_end_l = std::min((int) ceil(z_end / step_size), apr_iterator.z_num(level));

            int y_begin_l = (int) floor(y_begin / step_size);
            int y_end_l = std::min((int) ceil(y_end / step_size), apr_iterator.y_num(level));

            int z = 0;
            int x = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x, z) firstprivate(tree_it)
#endif
            for (z = z_begin_l; z < z_end_l; z++) {

                for (x = x_begin_l; x < x_end_l; ++x) {

                    float y_actual, x_actual, z_actual;

                    for (tree_it.begin(level, z, x);
                         tree_it <
                         tree_it.end(); tree_it++) {


                        if ((tree_it.y() >= y_begin_l) && (tree_it.y() < y_end_l)) {
                            //get apr info


                            y_actual = (tree_it.y() + 0.5) * pow(2, apr.level_max() - level) * this->scale_y;
                            x_actual = (x + 0.5) * pow(2, apr.level_max() - level) * this->scale_x;
                            z_actual = (z + 0.5) * pow(2, apr.level_max() - level) * this->scale_z;


                            int dim1 = 0;
                            int dim2 = 0;
                            getPos(glmObjects, dim1, dim2, x_actual, y_actual, z_actual, depth_slice[level].x_num,
                                   depth_slice[level].y_num);

                           // dim1 = dim1*scale_down;
                            //dim2 = dim2*scale_down;

                            if ((dim1 > 0) & (dim2 > 0) &
                                (dim1 < depth_slice[level].y_num) &
                                (dim2 < depth_slice[level].x_num)) {
                                //get the particle value
                                const S temp_int = treeData[tree_it];

                                depth_slice[level].mesh[dim1 + (dim2) * depth_slice[level].y_num] = op(temp_int,
                                                                                                       depth_slice[level].mesh[
                                                                                                               dim1 +
                                                                                                               (dim2) *
                                                                                                               depth_slice[level].y_num]);
                            } else {
                                if ((tree_it.y() >= y_end_l)) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }


        killObjects(glmObjects);


        //////////////////////////////////////////////
        ///
        /// Now merge the ray-casts between the different resolutions
        ///
        ////////////////////////////////////////////////

        int level;
        int level_min = apr.level_min();

        int x_, i, k;

        for (level = (level_min); level < max_level; level++) {

            const int step_size = pow(2, max_level - level);
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(x_, i, k) schedule(guided) if (level > 8)
#endif
            for (x_ = 0; x_ < depth_slice[level].x_num; x_++) {
                //both z and x are explicitly accessed in the structure

                for (int y_ = 0; y_ < depth_slice[level].y_num; y_++) {

                    const float curr_int = depth_slice[level].at(y_,x_,0);

                    const int dim1 = y_ * step_size;
                    const int dim2 = x_ * step_size;

                    //add to all the required rays
                    const int offset_max_dim1 = std::min(depth_slice[max_level].y_num, (dim1 + step_size));
                    const int offset_max_dim2 = std::min(depth_slice[max_level].x_num, (dim2 + step_size));

                    if (curr_int > 0) {

                        for (k = dim2; k < offset_max_dim2; ++k) {
                            for (i = dim1; i < offset_max_dim1; ++i) {
                                depth_slice[max_level].mesh[i +
                                                                  (k) * depth_slice[max_level].y_num] = op(
                                        curr_int, depth_slice[max_level].mesh[i + (k) *
                                                                                        depth_slice[max_level].y_num]);

                            }
                        }
                    }

                }
            }
        }



        //copy data across
        //std::copy(depth_slice[max_level].mesh.begin(), depth_slice[max_level].mesh.end(),
          //        cast_views.mesh.begin() + view_count * imageHeight * imageWidth);

        depth_slice[max_level].swap(cast_views);

        view_count++;

//        if (view_count >= num_views) {
//            break;
//        }
    //}

    timer.stop_timer();
//    float elapsed_seconds = timer.t2 - timer.t1;

//    std::cout << elapsed_seconds / (view_count * 1.0) << " seconds per view" << std::endl;

}

template<typename S, typename V, class BinaryOperation>
void APRRaycaster::perform_raycast(APR &apr, ParticleData<S> &particle_data, PixelData<V> &cast_views,
                                   BinaryOperation op) {

    //
    //  Bevan Cheeseman 2018
    //
    //  Simple ray case example, multi ray, accumulating, parralell projection
    //
    //


    int imageWidth = apr.org_dims(1);
    int imageHeight = apr.org_dims(0);

    ////////////////////////////////
    //  Set up the projection stuff
    ///////////////////////////////

    float height = this->height;

    float radius = this->radius_factor * apr.org_dims(0);

    ///////////////////////////////////////////
    //
    //  Set up Perspective
    //
    ////////////////////////////////////////////

    float x0 = height * apr.org_dims(1) * this->scale_x;
    float y0 = apr.org_dims(0) * .5 * this->scale_y;
    float z0 = apr.org_dims(2) * .5 * this->scale_z;

    float x0f = height * apr.org_dims(1) * this->scale_x;
    float y0f = apr.org_dims(0) * .5 * this->scale_y;
    float z0f = apr.org_dims(2) * .5 * this->scale_z;

    float theta_0 = this->theta_0;
    float theta_f = this->theta_final;
    float theta_delta = this->theta_delta;

    int num_views = floor((theta_f - theta_0) / theta_delta);

    cast_views.initWithResize(imageHeight, imageWidth, num_views);

    int view_count = 0;
    float init_val = 0;

    APRTimer timer;

    timer.verbose_flag = true;

    timer.start_timer("Compute APR maximum projection raycast");

    /////////////////////////////////////
    ///
    ///  Initialization of loop variables
    ///
    /////////////////////////////////////

    std::vector<PixelData<S>> depth_slice;
    depth_slice.resize(apr.level_max() + 1);

    std::vector<float> depth_vec;
    depth_vec.resize(apr.level_max() + 1);
    depth_slice[apr.level_max()].initWithValue(imageHeight, imageWidth, 1, init_val);


    for (int i = apr.level_min(); i < apr.level_max(); i++) {
        float d = pow(2, apr.level_max() - i);
        depth_slice[i].initWithValue(ceil(depth_slice[apr.level_max()].y_num / d),
                                     ceil(depth_slice[apr.level_max()].x_num / d), 1, init_val);
        depth_vec[i] = d;
    }

    depth_vec[apr.level_max()] = 1;

    //jitter the parts to remove ray cast artifacts
    const bool jitter = this->jitter;
    const float jitter_factor = this->jitter_factor;

    ParticleData<float> jitter_x;
    ParticleData<float> jitter_y;
    ParticleData<float> jitter_z;

    //initialize the iterator
    auto apr_iterator = apr.iterator();

    if (jitter) {

        jitter_x.init(apr.total_number_particles());
        jitter_y.init(apr.total_number_particles());
        jitter_z.init(apr.total_number_particles());

        for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.z_num(level); z++) {
                for (x = 0; x < apr_iterator.x_num(level); ++x) {
                    for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                         apr_iterator++) {
                        jitter_x[apr_iterator] = jitter_factor * ((std::rand() - 500) % 1000) / 1000.0f;
                        jitter_y[apr_iterator] = jitter_factor * ((std::rand() - 500) % 1000) / 1000.0f;
                        jitter_z[apr_iterator] = jitter_factor * ((std::rand() - 500) % 1000) / 1000.0f;
                    }
                }
            }
        }
    }

    //main loop changing the view angle
    for (float theta = theta_0; theta <= theta_f; theta += theta_delta) {

        //////////////////////////////
        ///
        /// Create an projected image at each depth
        ///
        //////////////////////////////

        for (int i = apr.level_min(); i <= apr.level_max(); i++) {
            std::fill(depth_slice[i].mesh.begin(), depth_slice[i].mesh.end(), init_val);
        }

        //////////////////////////////
        ///
        /// Set up the new camera views
        ///
        //////////////////////////////
        initObjects(glmObjects, imageWidth, imageHeight, radius, theta, x0, y0, z0, x0f, y0f, z0f, phi, phi_s);

        ///////////////////////////////////////////
        ///
        /// Perform ray cast over APR
        ///
        /////////////////////////////////////////

        //  Set up the APR parallel iterators (these are required for the parallel iteration)

        float y_actual, x_actual, z_actual;

        for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x, y_actual, x_actual, z_actual) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.z_num(level); z++) {
                for (x = 0; x < apr_iterator.x_num(level); ++x) {
                    for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                         apr_iterator++) {

                        //get apr info

                        if (jitter) {
                            y_actual = (apr_iterator.y() + 0.5f + jitter_y[apr_iterator]) * this->scale_y *
                                       depth_vec[level];
                            x_actual = (x + 0.5f + jitter_x[apr_iterator]) * this->scale_x *
                                       depth_vec[level];
                            z_actual = (z + 0.5f + jitter_z[apr_iterator]) * this->scale_z *
                                       depth_vec[level];
                        } else {
                            y_actual = apr_iterator.y_global(level,apr_iterator.y()) * this->scale_y;
                            x_actual = apr_iterator.x_global(level,x) * this->scale_x;
                            z_actual = apr_iterator.z_global(level,z) * this->scale_z;
                        }


                        int dim1 = 0;
                        int dim2 = 0;
                        getPos(dim1, dim2, x_actual, y_actual, z_actual, depth_slice[level].x_num,
                               depth_slice[level].y_num);

                        if ((dim1 > 0) & (dim2 > 0) &
                            (dim1 < depth_slice[level].y_num) &
                            (dim2 < depth_slice[level].x_num)) {
                            //get the particle value
                            S temp_int = particle_data[apr_iterator];

                            depth_slice[level].mesh[dim1 + (dim2) * depth_slice[level].y_num] = op(temp_int,
                                                                                                   depth_slice[level].mesh[
                                                                                                           dim1 +
                                                                                                           (dim2) *
                                                                                                           depth_slice[level].y_num]);
                        }
                    }
                }
            }
        }
        killObjects(glmObjects);

        //////////////////////////////////////////////
        ///
        /// Now merge the ray-casts between the different resolutions
        ///
        ////////////////////////////////////////////////

        int level;

        int level_min = apr.level_min();

        int y_, x_, i, k;

        for (level = level_min; level < apr.level_max(); level++) {

            const int step_size = pow(2, apr.level_max() - level);
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(x_, i, k) schedule(guided) if (level > 8)
#endif
            for (x_ = 0; x_ < depth_slice[level].x_num; x_++) {
                //both z and x are explicitly accessed in the structure

                for (y_ = 0; y_ < depth_slice[level].y_num; y_++) {

                    const float curr_int = depth_slice[level].mesh[y_ + (x_) * depth_slice[level].y_num];

                    const int dim1 = y_ * step_size;
                    const int dim2 = x_ * step_size;

                    //add to all the required rays
                    const int offset_max_dim1 = std::min(depth_slice[apr.level_max()].y_num, (dim1 + step_size));
                    const int offset_max_dim2 = std::min(depth_slice[apr.level_max()].x_num, (dim2 + step_size));

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
        std::copy(depth_slice[apr.level_max()].mesh.begin(), depth_slice[apr.level_max()].mesh.end(),
                  cast_views.mesh.begin() + view_count * imageHeight * imageWidth);

        view_count++;


        if (view_count >= num_views) {
            break;
        }
    }

    timer.stop_timer();
    float elapsed_seconds = timer.t2 - timer.t1;

    std::cout << elapsed_seconds / (view_count * 1.0) << " seconds per view" << std::endl;


}

template<typename S, typename U>
float APRRaycaster::perpsective_mesh_raycast(PixelData<S> &image, PixelData<U> &cast_views) {
    //
    //  Bevan Cheeseman 2017
    //
    //  Max Ray Cast Proposective Projection on mesh
    //
    //

    int imageWidth = image.x_num;
    int imageHeight = image.y_num;

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

    int num_views = (int) floor((theta_f - theta_0) / theta_delta);


    cast_views.initWithValue(imageHeight, imageWidth, num_views, 0);

    APRTimer timer;

    timer.verbose_flag = true;

    int view_count = 0;

    timer.start_timer("ray cast mesh prospective");

    for (float theta = theta_0; theta <= theta_f; theta += theta_delta) {
        initObjects(glmObjects, imageWidth, imageHeight, radius, theta, x0, y0, z0, x0f, y0f, z0f, phi, phi_s);

        PixelData<S> proj_img;
        proj_img.initWithValue(imageHeight, imageWidth, 1, 0);

        int z_, x_, j_;

        //loop over the resolutions of the structure
        const int x_num_ = image.x_num;
        const int z_num_ = image.z_num;
        const int y_num_ = image.y_num;

#ifdef HAVE_OPENMP
        //#pragma omp parallel for default(shared) private(z_,x_,j_)
#endif
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                for (j_ = 0; j_ < y_num_; j_++) {
                    const S temp_int = image.mesh[j_ + x_ * image.y_num + z_ * image.x_num * image.y_num];
                    int dim1 = 0;
                    int dim2 = 0;
                    getPos(glmObjects, dim1, dim2, (float) x_ * this->scale_x, (float) j_ * this->scale_y,
                           (float) z_ * this->scale_z, imageWidth, imageHeight);

                    if ((dim1 > 0) & (dim2 > 0) &
                        (dim1 < proj_img.y_num) &
                        (dim2 < proj_img.x_num)) {

                        proj_img.mesh[dim1 + (dim2) * proj_img.y_num] = std::max(temp_int, proj_img.mesh[dim1 + (dim2) *
                                                                                                                proj_img.y_num]);
                    }
                }
            }
        }
        killObjects(glmObjects);
        std::copy(proj_img.mesh.begin(), proj_img.mesh.end(),
                  cast_views.mesh.begin() + view_count * imageHeight * imageWidth);

        view_count++;

        if (view_count == num_views) {
            break;
        }
    }

    timer.stop_timer();
    float elapsed_seconds = (float) (timer.t2 - timer.t1);

    return elapsed_seconds;
}


#endif
