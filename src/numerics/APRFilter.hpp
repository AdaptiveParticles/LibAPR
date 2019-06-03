//
// Created by cheesema on 31.10.18.
//

#ifndef APR_TIME_APRFILTER_HPP
#define APR_TIME_APRFILTER_HPP
#include "data_structures/APR/APR.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRTreeNumerics.hpp"

#define ZERO_PAD 0
#define REFLECT_PAD 1

class APRFilter {

public:
    template< typename S, typename T,typename R>
    void convolve(APR &apr, std::vector<PixelData<T>>& stencils, ParticleData<S> &particle_input, ParticleData<R> &particle_output);

    bool boundary_cond = 1;

    bool nl_mult=false;

    template<typename T>
    void generate_smooth_stencil(std::vector<PixelData<T>>& stencils){

        stencils.resize(3);

       // std::vector<float> mid_val = {0.5,0.1,1.0f/27.0f};
        std::vector<float> mid_val = {1.0f/27.0f,1.0f/27.0f,1.0f/27.0f};

        for (int j = 0; j < stencils.size(); ++j) {
            stencils[j].init(3,3,3);
            float mid = mid_val[j];

            for (int i = 0; i < stencils[j].mesh.size(); ++i) {
                stencils[j].mesh[i] = (1.0f-mid)/(stencils[j].mesh.size()*1.0f - 1);

            }
            stencils[j].at(1,1,1) = mid;
        }


    }

    template<typename T>
    void generate_derivative_stencil(PixelData<T>& stencil){

    }

    template<typename T>
    inline void set_boundary_conditions_yx(const bool boundary_condition,const std::vector<int>& stencil_half,const std::vector<int> &stencil_shape,PixelData<T>& temp_vec,const uint64_t z){

        const size_t x_num = temp_vec.x_num;
        const size_t y_num = temp_vec.y_num;

        if(boundary_condition == REFLECT_PAD){
            const uint64_t z_off = z%stencil_shape[2];

            //first do the x reflection (0 -> stencil_half)
            for (int x = 0; x < stencil_half[1]; ++x) {
                for (int y = stencil_half[0]; y < (y_num - stencil_half[0]); ++y) {
                    temp_vec.at(y,stencil_half[1]-1-x,z_off) = temp_vec.at(y,stencil_half[1]+x,z_off);
                }
            }

            //first do the x reflection (x_num - 1 -> x_num - 1 - stencil_half)
            for (int x = 0; x < stencil_half[1]; ++x) {
                for (int y = stencil_half[0]; y < (y_num - stencil_half[0]); ++y) {
                    temp_vec.at(y,x_num -stencil_half[1] + x,z_off) = temp_vec.at(y,x_num -stencil_half[1] - 1 - x,z_off);
                }
            }

            // y reflection (0 -> stencil_half)
            for (int x = 0; x < x_num; ++x) {
                for (int y = 0; y < stencil_half[0]; ++y) {
                    temp_vec.at(stencil_half[0] - 1 - y,x,z_off) = temp_vec.at(stencil_half[0] + y,x,z_off);
                }
            }

            //y reflection
            for (int x = 0; x < x_num; ++x) {
                for (int y = 0; y < stencil_half[0]; ++y) {
                    temp_vec.at(y_num -stencil_half[0] + y,x,z_off) = temp_vec.at(y_num -stencil_half[0] - 1 - y,x,z_off);
                }
            }


        }


    }


    template<typename ImageType>
    inline void update_same_level(const uint64_t level,
                                  const uint64_t z,const uint64_t x,APRIterator &it,PixelData<float> &temp_vec,
                                  ParticleData<ImageType> &inputParticles,
                                  const std::vector<int> &stencil_shape,
                                  const std::vector<int> &stencil_half, const uint64_t y_num_m, const uint64_t x_num_m,const bool boundary_cond,const uint64_t mesh_offset){

        for (it.begin(level, z, x);it < it.end();it++) {

            temp_vec.mesh[it.y() + stencil_half[0] + mesh_offset] = inputParticles[it];
        }
    }


    template<typename R>
    void update_dense_array(const uint64_t level,
                            const uint64_t z,
                            APR &apr,
                            APRIterator &apr_it,
                            APRTreeIterator &tree_it,
                            ParticleData<float> &tree_data,
                            PixelData<float> &temp_vec,
                            ParticleData<R> &inputParticles,
                            const std::vector<int> &stencil_shape,
                            const std::vector<int> &stencil_half) {

        uint64_t x;

        const uint64_t x_num_m = temp_vec.x_num;
        const uint64_t y_num_m = temp_vec.y_num;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_it)
#endif
        for (x = 0; x < apr_it.x_num(level); ++x) {

            //
            //  This loop recreates particles at the current level, using a simple copy
            //

            uint64_t mesh_offset = (x + stencil_half[1]) * y_num_m + x_num_m * y_num_m * (z % stencil_shape[2]);

            update_same_level(level,z,x,apr_it,temp_vec,inputParticles,stencil_shape, stencil_half, y_num_m, x_num_m,boundary_cond,mesh_offset);

        }

        if (level > apr_it.level_min()) {
            const int y_num = apr_it.y_num(level);

            //
            //  This loop interpolates particles at a lower level (Larger Particle Cell or resolution), by simple uploading
            //

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_it)
#endif
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {

                for (apr_it.begin(level - 1, z / 2, x / 2);
                     apr_it < apr_it.end();
                     apr_it++) {

                    int y_m = std::min(2 * apr_it.y() + 1, y_num - 1);    // 2y+1+offset

                    temp_vec.at(2 * apr_it.y() + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = inputParticles[apr_it];//particleData[apr_iterator];
                    temp_vec.at(y_m + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = inputParticles[apr_it];//particleData[apr_iterator];

                }
            }
        }

        /******** start of using the tree iterator for downsampling ************/

        if (level < apr_it.level_max()) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(tree_it)
#endif
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (tree_it.begin(level, z, x);
                     tree_it < tree_it.end();
                     tree_it++) {

                    temp_vec.at(tree_it.y() + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = tree_data[tree_it];
                }
            }
        }

        set_boundary_conditions_yx(boundary_cond,stencil_half,stencil_shape,temp_vec,z);


    }



    template<typename T, typename S, typename R, typename C>
    void downsample_stencil(PixelData<T> &aInput, PixelData<S> &aOutput, R reduce, C constant_operator, bool aInitializeOutput = true) {

        const size_t z_num = aInput.z_num;
        const size_t x_num = aInput.x_num;
        const size_t y_num = aInput.y_num;

        // downsampled dimensions twice smaller (rounded up)
        const size_t z_num_ds = std::max((int)z_num-2, 1);
        const size_t x_num_ds = std::max((int)x_num-2, 1);
        const size_t y_num_ds = std::max((int)y_num-2, 1);

        if (aInitializeOutput) {
            aOutput.init(y_num_ds, x_num_ds, z_num_ds);
        }

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for (size_t z_ds = 0; z_ds < z_num_ds; ++z_ds) {
            for (size_t x_ds = 0; x_ds < x_num_ds; ++x_ds) {

                //const ArrayWrapper<T> &inMesh = aInput.mesh;
                //ArrayWrapper<S> &outMesh = aOutput.mesh;

                for (size_t y_ds = 0; y_ds < y_num_ds; ++y_ds) {

                    float outValue = 0;

                    for(size_t z = z_ds; z < std::min(z_num, z_ds+3); ++z) {
                        for(size_t x = x_ds; x<std::min(x_num, x_ds+3); ++x) {
                            for(size_t y = y_ds; y<std::min(y_num, y_ds+3); ++y) {
                                outValue = reduce(outValue, aInput.mesh[z*x_num*y_num + x*y_num + y]);
                            }
                        }
                    }

                    aOutput.mesh[z_ds*x_num_ds*y_num_ds + x_ds*y_num_ds + y_ds] = constant_operator(outValue);
                }
            }
        }
    }

/**
 * Downsample a stencil by level_delta levels in such a way that applying the downsampled stencil closely
 * corresponds to applying the original stencil to particles of level = original_level-level_delta.
 * @tparam T                    input data type
 * @tparam S                    output data type
 * @param aInput                input stencil  (PixelData<T>)
 * @param aOutput               output stencil (PixelData<S>)
 * @param level_delta           level difference between input and output
 * @param normalize             should the stencil be normalized (sum to unity)? (default false = no)
 * @param aInitializeOutput     should the output be initialized? (default true = yes)
 */

    template<typename T, typename S>
    void downsample_stencil_alt(const PixelData<T>& aInput, PixelData<S>& aOutput, int level_delta, bool normalize = false, bool aInitializeOutput = true) {

        const size_t z_num = aInput.z_num;
        const size_t x_num = aInput.x_num;
        const size_t y_num = aInput.y_num;

        const float size_factor = pow(2, level_delta);
        //const int ndim = (y_num>1) + (x_num > 1) + (z_num>1);

        int k = ceil(z_num / size_factor);
        const size_t z_num_ds = (k % 2 == 0) ? k+1 : k;

        k = ceil(x_num / size_factor);
        const size_t x_num_ds = (k % 2 == 0) ? k+1 : k;

        k = ceil(y_num / size_factor);
        const size_t y_num_ds = (k % 2 == 0) ? k+1 : k;

        if (aInitializeOutput) {
            aOutput.init(y_num_ds, x_num_ds, z_num_ds);
        }

        const float offsety = (size_factor*y_num_ds - y_num)/2.0f;
        const float offsetx = (size_factor*x_num_ds - x_num)/2.0f;
        const float offsetz = (size_factor*z_num_ds - z_num)/2.0f;

//#ifdef HAVE_OPENMP
//#pragma omp parallel for default(shared)
//#endif
        float sum = 0;
        for (size_t z_ds = 0; z_ds < z_num_ds; ++z_ds) {
            for (size_t x_ds = 0; x_ds < x_num_ds; ++x_ds) {
                for (size_t y_ds = 0; y_ds < y_num_ds; ++y_ds) {

                    float outValue = 0;

                    for(size_t z = 0; z < z_num; ++z) {
                        for(size_t x = 0; x < x_num; ++x) {
                            for(size_t y = 0; y < y_num; ++y) { // y < std::min((float)y_num, y_ds+size_factor+1) ?

                                float ybegin = y+offsety;
                                float xbegin = x+offsetx;
                                float zbegin = z+offsetz;

                                float overlapy = std::max(size_factor*y_ds, std::min(ybegin+1, size_factor*(y_ds+1))) - std::min(size_factor*(y_ds+1), std::max(ybegin, size_factor*y_ds));
                                float overlapx = std::max(size_factor*x_ds, std::min(xbegin+1, size_factor*(x_ds+1))) - std::min(size_factor*(x_ds+1), std::max(xbegin, size_factor*x_ds));
                                float overlapz = std::max(size_factor*z_ds, std::min(zbegin+1, size_factor*(z_ds+1))) - std::min(size_factor*(z_ds+1), std::max(zbegin, size_factor*z_ds));

                                float factor = overlapy * overlapx * overlapz;

                                outValue += factor * aInput.mesh[z*x_num*y_num + x*y_num + y];
                            }
                        }
                    }

                    aOutput.mesh[z_ds*x_num_ds*y_num_ds + x_ds*y_num_ds + y_ds] = outValue; // / pow(size_factor, ndim);
                    sum += outValue;
                }
            }
        }

        if(normalize) {
            float factor = 1.0f / sum;
            for (int i = 0; i < aOutput.mesh.size(); ++i) {
                aOutput.mesh[i] *= factor;
            }
        }

    }


    template<typename S,typename R>
    void create_test_particles_equiv(APR& apr,const std::vector<PixelData<float>> &stencil_vec,ParticleData<S>& input_particles
            ,ParticleData<R>& output_particles){

        ParticleData<float> part_tree;

        output_particles.init(apr.total_number_particles());

        APRTreeNumerics::fill_tree_mean(apr, input_particles, part_tree);

        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();

        int stencil_counter = 0;

        for (uint64_t level_local = apr_it.level_max(); level_local >= apr_it.level_min(); --level_local) {

            PixelData<float> by_level_recon;
            by_level_recon.init(apr_it.y_num(level_local),apr_it.x_num(level_local),apr_it.z_num(level_local),0);

            //for (uint64_t level = std::max((uint64_t)(level_local-1),(uint64_t)apr_iterator.level_min()); level <= level_local; ++level) {
            for (uint64_t level = apr_it.level_min(); level <= level_local; ++level) {
                int z = 0;
                int x = 0;
                const float step_size = pow(2, level_local - level);


#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_it)
#endif
                for (z = 0; z < apr_it.z_num(level); z++) {
                    for (x = 0; x < apr_it.x_num(level); ++x) {
                        for (apr_it.begin(level, z, x); apr_it < apr_it.end();
                             apr_it++) {

                            int dim1 = apr_it.y() * step_size;
                            int dim2 = apr_it.x() * step_size;
                            int dim3 = apr_it.z() * step_size;

                            float temp_int;
                            //add to all the required rays

                            temp_int = input_particles[apr_it];

                            const int offset_max_dim1 = std::min((int) by_level_recon.y_num, (int) (dim1 + step_size));
                            const int offset_max_dim2 = std::min((int) by_level_recon.x_num, (int) (dim2 + step_size));
                            const int offset_max_dim3 = std::min((int) by_level_recon.z_num, (int) (dim3 + step_size));

                            for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                                for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                                    for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                                        by_level_recon.mesh[i + (k) * by_level_recon.y_num + q * by_level_recon.y_num * by_level_recon.x_num] = temp_int;
                                    }
                                }
                            }
                        }
                    }
                }
            }


            if(level_local < apr_it.level_max()){

                uint64_t level = level_local;

                const float step_size = 1;

                int z = 0;
                int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(tree_it)
#endif
                for (z = 0; z < tree_it.z_num(level); z++) {
                    for (x = 0; x < tree_it.x_num(level); ++x) {
                        for (tree_it.set_new_lzx(level, z, x);
                             tree_it < tree_it.end();
                             tree_it++) {

                            int dim1 = tree_it.y() * step_size;
                            int dim2 = tree_it.x() * step_size;
                            int dim3 = tree_it.z() * step_size;

                            float temp_int;
                            //add to all the required rays

                            temp_int = part_tree[tree_it];

                            const int offset_max_dim1 = std::min((int) by_level_recon.y_num, (int) (dim1 + step_size));
                            const int offset_max_dim2 = std::min((int) by_level_recon.x_num, (int) (dim2 + step_size));
                            const int offset_max_dim3 = std::min((int) by_level_recon.z_num, (int) (dim3 + step_size));

                            for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                                for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                                    for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                                        by_level_recon.mesh[i + (k) * by_level_recon.y_num + q * by_level_recon.y_num * by_level_recon.x_num] = temp_int;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            int x = 0;
            int z = 0;
            uint64_t level = level_local;

            PixelData<float> stencil(stencil_vec[stencil_counter], true);

            //const PixelData<float> &stencil = stencil_vec[stencil_counter];
            std::vector<int> stencil_halves = {((int)stencil.y_num-1)/2, ((int)stencil.x_num-1)/2, ((int)stencil.z_num-1)/2};

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {
                //lastly loop over particle locations and compute filter.
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_it.set_new_lzx(level, z, x);
                         apr_it < apr_it.end();
                         apr_it++) {

                        double neigh_sum = 0;
                        int counter = 0;

                        const int k = apr_it.y(); // offset to allow for boundary padding
                        const int i = x;

                        for (int l = -stencil_halves[2]; l < stencil_halves[2]+1; ++l) {
                            for (int q = -stencil_halves[1]; q < stencil_halves[1]+1; ++q) {
                                for (int w = -stencil_halves[0]; w < stencil_halves[0]+1; ++w) {

                                    if((k+w)>=0 & (k+w) < (apr.spatial_index_y_max(level))){
                                        if((i+q)>=0 & (i+q) < (apr.spatial_index_x_max(level))){
                                            if((z+l)>=0 & (z+l) < (apr.spatial_index_z_max(level))){
                                                neigh_sum += stencil.mesh[counter] * by_level_recon.at(k + w, i + q, z+l);
                                            }
                                        }
                                    }
                                    counter++;
                                }
                            }
                        }

                        output_particles[apr_it] = neigh_sum;//std::roundf(neigh_sum/(1.0f*pow((float)2*stencil_halves[0]+1, apr.apr_access.number_dimensions)));
                    }
                }
            }

//            std::string image_file_name = apr.parameters.input_dir + std::to_string(level_local) + "_by_level.tif";
//            TiffUtils::saveMeshAsTiff(image_file_name, by_level_recon);

            stencil_counter = std::min(stencil_counter+1, (int)stencil_vec.size()-1);
        }

        PixelData<float> recon_standard;
        APRReconstruction::interp_img(apr,recon_standard, output_particles);

    }



};


template<typename S, typename T,typename R>
void APRFilter::convolve(APR &apr, std::vector<PixelData<T>>& stencils, ParticleData<S> &particle_input, ParticleData<R> &particle_output) {

    particle_output.init(particle_input.total_number_particles());

    const bool boundary = boundary_cond;

    /**** initialize and fill the apr tree ****/
    ParticleData<float> tree_data;

    apr.init_tree();

    APRTreeNumerics::fill_tree_mean(apr, particle_input, tree_data);

    /*** iterators for accessing apr data ***/
    auto apr_it = apr.iterator();
    auto tree_it = apr.tree_iterator();

    // assert stencil_shape compatible with apr org_dims?

    for (int level = apr.level_max(); level >= apr.level_min(); --level) {

        int stencil_num = std::min((int)stencils.size()-1,(int)( apr.level_max()-level));

        PixelData<T> stencil;
        stencil.init(stencils[stencil_num]);
        stencil.copyFromMesh(stencils[stencil_num]);

        const std::vector<int> stencil_shape = {(int) stencil.y_num,
                                                (int) stencil.x_num,
                                                (int) stencil.z_num};
        const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2, (stencil_shape[1] - 1) / 2,
                                               (stencil_shape[2] - 1) / 2};

        unsigned int z = 0;
        unsigned int x = 0;

        const uint64_t z_num = apr_it.z_num(level);

        const uint64_t y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_it.y_num(level) +
                                                                    stencil_shape[0] - 1 : 1;
        const uint64_t x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_it.x_num(level) +
                                                                    stencil_shape[1] - 1 : 1;

        PixelData<float> temp_vec;
        temp_vec.initWithValue(y_num_m,
                               x_num_m,
                               stencil_shape[2],(float) 0.0f); //zero padded boundaries

        if(boundary == REFLECT_PAD){
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array(level,
                                   padd,
                                   apr,
                                   apr_it,
                                   tree_it,
                                   tree_data,
                                   temp_vec,
                                   particle_input,
                                   stencil_shape,
                                   stencil_half);

                uint64_t index_in = temp_vec.x_num * temp_vec.y_num * ((padd) % stencil_shape[2]);
                uint64_t index_out = temp_vec.x_num * temp_vec.y_num * (stencil_shape[2]-padd-1);
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                for (x = 0; x < temp_vec.x_num; ++x) {
                    std::copy(temp_vec.mesh.begin() + index_in + (x + 0) * temp_vec.y_num,
                              temp_vec.mesh.begin() + index_in + (x + 1) * temp_vec.y_num, temp_vec.mesh.begin() + index_out + (x + 0) * temp_vec.y_num);
                }

            }


        } else {
            //initial condition
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array(level,
                                   padd,
                                   apr,
                                   apr_it,
                                   tree_it,
                                   tree_data,
                                   temp_vec,
                                   particle_input,
                                   stencil_shape,
                                   stencil_half);


            }
        }

        for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

            if (z < (z_num - stencil_half[2])) {
                //update the next z plane for the access
                update_dense_array(level, z + stencil_half[2], apr, apr_it, tree_it, tree_data,
                                   temp_vec, particle_input, stencil_shape, stencil_half);
            } else {
                //padding

                if(boundary == REFLECT_PAD){

                    uint64_t index_in = temp_vec.x_num * temp_vec.y_num * ((z) % stencil_shape[2]); //need to check this for larger stencils
                    uint64_t index_out = temp_vec.x_num * temp_vec.y_num * ((z+ stencil_half[2]) % stencil_shape[2]);
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::copy(temp_vec.mesh.begin() + index_in + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index_in + (x + 1) * temp_vec.y_num, temp_vec.mesh.begin() + index_out + (x + 0) * temp_vec.y_num);
                    }


                } else {
                    //zero padd
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                }
            }

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_it)
#endif
            for (x = 0; x < apr_it.x_num(level); ++x) {
                for (apr_it.begin(level, z, x);
                     apr_it < apr_it.end();
                     apr_it++) {

                    float neigh_sum = 0;
                    //int counter = 0;

                    const int k = apr_it.y() + stencil_half[0]; // offset to allow for boundary padding
                    const int i = x + stencil_half[1];

                    float factor = 1.0;

                    //compute the stencil
                    if(nl_mult){
                        for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                            for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                    neigh_sum += (
                                            stencil.at(w + stencil_half[0], q + stencil_half[1], l + stencil_half[2]) *
                                            std::floor(factor*log(temp_vec.at(k + w, i + q, (z + l) % stencil_shape[2]))))/factor;

                                }
                            }
                        }

                    } else {
                        for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                            for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                    neigh_sum += (
                                            stencil.at(w + stencil_half[0], q + stencil_half[1], l + stencil_half[2]) *
                                            temp_vec.at(k + w, i + q, (z + l) % stencil_shape[2]));

                                }
                            }
                        }
                    }

                    particle_output[apr_it] = neigh_sum;


                }//y, pixels/columns
            }//x , rows
        }//z
    }
}


#endif //APR_TIME_APRFILTER_HPP
