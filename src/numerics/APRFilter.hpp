//
// Created by cheesema on 31.10.18.
//

#ifndef APR_TIME_APRFILTER_HPP
#define APR_TIME_APRFILTER_HPP
#include "APRReconstruction.hpp"
#include "numerics/APRStencilFunctions.hpp"

#include<math.h>

#define ZERO_PAD 0
#define REFLECT_PAD 1

template<typename T>
class ImageBuffer {

public:
    uint64_t y_num;
    uint64_t x_num;
    uint64_t z_num;
    std::vector<T> mesh;

    ImageBuffer() {
        init(0, 0, 0);
    }

    ImageBuffer(uint64_t aSizeOfY, uint64_t aSizeOfX, uint64_t aSizeOfZ) {
        init(aSizeOfY, aSizeOfX, aSizeOfZ);
    }

    void init(uint64_t aSizeOfY, uint64_t aSizeOfX, uint64_t aSizeOfZ) {
        y_num = aSizeOfY;
        x_num = aSizeOfX;
        z_num = aSizeOfZ;
        mesh.resize(y_num * x_num * z_num);
    }

    T& at(size_t y, size_t x, size_t z) {
        size_t idx = z * x_num * y_num + x * y_num + y;
        return mesh[idx];
    }
};

class APRFilter {

public:
    template<typename ParticleDataTypeInput, typename T,typename ParticleDataTypeOutput>
    void convolve(APR &apr, std::vector<PixelData<T>>& stencils, ParticleDataTypeInput &particle_input, ParticleDataTypeOutput &particle_output);

    template<typename ParticleDataTypeInput, typename T,typename ParticleDataTypeOutput>
    void convolve_pencil(APR &apr, std::vector<PixelData<T>>& stencils, ParticleDataTypeInput &particle_input, ParticleDataTypeOutput &particle_output);

    template<typename ParticleDataTypeInput, typename T,typename ParticleDataTypeOutput>
    void richardson_lucy(APR &apr, ParticleDataTypeInput &particle_input, ParticleDataTypeOutput &particle_output,
                         std::vector<PixelData<T>>& psf_vec, std::vector<PixelData<T>>& psf_flipped_vec, int number_iterations);

    template<typename ParticleDataTypeInput, typename T,typename ParticleDataTypeOutput>
    void richardson_lucy(APR &apr, ParticleDataTypeInput &particle_input, ParticleDataTypeOutput &particle_output,
                         PixelData<T> &psf, int number_iterations, bool use_stencil_downsample=true, bool normalize=false);

    bool boundary_cond = ZERO_PAD;

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
    void apply_boundary_conditions_xy(const uint64_t z, ImageBuffer<T> &temp_vec, const bool boundary_condition,const std::vector<int>& stencil_half,const std::vector<int> &stencil_shape){

        const int x_num = (int)temp_vec.x_num;
        const int y_num = (int)temp_vec.y_num;
        const uint64_t base_offset = (z % stencil_shape[2]) * x_num * y_num;

        if(boundary_condition == REFLECT_PAD){

            //first do the x reflection (0 -> stencil_half)
            for (int x = 0; x < stencil_half[1]; ++x) {

                uint64_t index_in = (stencil_half[1]+x+1) * y_num + base_offset;
                uint64_t index_out = (stencil_half[1]-x-1) * y_num + base_offset;

                std::copy(temp_vec.mesh.begin() + index_in + stencil_half[0],
                          temp_vec.mesh.begin() + index_in + y_num - stencil_half[0],
                          temp_vec.mesh.begin() + index_out + stencil_half[0]);
            }

            //first do the x reflection (x_num - 1 -> x_num - 1 - stencil_half)
            for (int x = 0; x < stencil_half[1]; ++x) {

                uint64_t index_in = (x_num - stencil_half[1] - 2 - x) * y_num + base_offset;
                uint64_t index_out = (x_num - stencil_half[1] + x) * y_num + base_offset;

                std::copy(temp_vec.mesh.begin() + index_in + stencil_half[0],
                          temp_vec.mesh.begin() + index_in + y_num - stencil_half[0],
                          temp_vec.mesh.begin() + index_out + stencil_half[0]);
            }

            // y reflection (0 -> stencil_half)
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared)
#endif
            for (int x = 0; x < x_num; ++x) {

                uint64_t offset = stencil_half[0] + x * y_num + base_offset;

                for (int y = 0; y < stencil_half[0]; ++y) {

                    temp_vec.mesh[offset - 1 - y] = temp_vec.mesh[offset + y + 1];
                }
            }

            //y reflection (y_num - stencil_half -> y_num)
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared)
#endif
            for (int x = 0; x < x_num; ++x) {

                uint64_t offset = y_num - stencil_half[0] + x * y_num + base_offset;

                for (int y = 0; y < stencil_half[0]; ++y) {

                    temp_vec.mesh[offset + y] = temp_vec.mesh[offset - 2 - y];
                }
            }
        } else { // zero pad

            //first pad y (x = 0 -> stencil_half)
            for (int x = 0; x < stencil_half[1]; ++x) {
                uint64_t index_start = x * y_num + base_offset;
                std::fill(temp_vec.mesh.begin() + index_start, temp_vec.mesh.begin() + index_start + y_num, 0);
            }

            //first pad y (x = x_num - stencil_half -> x_num)
            for (int x = 0; x < stencil_half[1]; ++x) {
                uint64_t index_start = (x_num - stencil_half[1] + x) * y_num + base_offset;
                std::fill(temp_vec.mesh.begin() + index_start, temp_vec.mesh.begin() + index_start + y_num, 0);
            }

            //then pad x (y = 0 -> stencil_half)
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared)
#endif
            for(int x = 0; x < x_num; ++x) {
                uint64_t offset = x * y_num + base_offset;
                for(int y = 0; y < stencil_half[0]; ++y) {
                    temp_vec.mesh[offset + y] = 0;
                }
            }

            // then pad x (y = y_num - stencil_half -> ynum)
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared)
#endif
            for(int x = 0; x < x_num; ++x) {
                uint64_t offset = y_num - stencil_half[0] + x * y_num + base_offset;
                for(int y = 0; y < stencil_half[0]; ++y) {
                    temp_vec.mesh[offset + y] = 0;
                }
            }
        }
    }





    template<typename T>
    void apply_boundary_conditions_z(const int z, const int z_num, ImageBuffer<T> &temp_vec, const bool boundary_condition,
                                     const bool low_end, const std::vector<int>& stencil_half,const std::vector<int> &stencil_shape) {

        const int x_num = temp_vec.x_num;
        const int y_num = temp_vec.y_num;
        uint64_t out_offset = (z % stencil_shape[2]) * x_num * y_num;

        if(boundary_condition == REFLECT_PAD) { // fixme
            if(low_end) {

                uint64_t z_in = (stencil_half[2] + (stencil_half[2] - z)) % stencil_shape[2];
                uint64_t in_offset = z_in * x_num * y_num;

                // copy slice at z_in to z
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
                for(int x = 0; x < x_num; ++x) {
                    std::copy(temp_vec.mesh.begin() + in_offset + x * y_num,
                              temp_vec.mesh.begin() + in_offset + (x+1) * y_num,
                              temp_vec.mesh.begin() + out_offset + x *y_num);
                }

            } else {

                uint64_t r = z_num - 1 + stencil_half[2]; // z index of the boundary
                uint64_t z_in = (r - (z-r)) % stencil_shape[2];

                uint64_t in_offset = z_in * x_num * y_num;

                // copy slice at z_in to z
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
                for(int x = 0; x < x_num; ++x) {
                    std::copy(temp_vec.mesh.begin() + in_offset + x * y_num,
                              temp_vec.mesh.begin() + in_offset + (x+1) * y_num,
                              temp_vec.mesh.begin() + out_offset + x * y_num);
                }
            }
        } else { // zero padding

            // fill slice at z with zeroes
            T pad_value = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
            for(int x = 0; x < x_num; ++x) {
                std::fill(temp_vec.mesh.begin() + out_offset + x * y_num,
                          temp_vec.mesh.begin() + out_offset + (x+1) * y_num,
                          pad_value);
            }
        }
    }




    template<typename T>
    void apply_boundary_conditions_x(const uint64_t x, const uint64_t x_num, ImageBuffer<T> &temp_vec, const bool boundary_condition,
                                     const bool low_end, const std::vector<int>& stencil_half,const std::vector<int> &stencil_shape) {

        const uint64_t y_num = temp_vec.y_num;
        const uint64_t xy_num = temp_vec.x_num * y_num;

        if(boundary_condition == REFLECT_PAD) { // fixme
            if(low_end) {
                for(int z = 0; z < stencil_shape[2]; ++z) {
                    uint64_t x_in = (stencil_half[1] + (stencil_half[1] - x)) % stencil_shape[1];
                    uint64_t in_offset = z * xy_num + x_in * y_num;
                    uint64_t out_offset = z * xy_num + (x % stencil_shape[1]) * y_num;

                    std::copy(temp_vec.mesh.begin() + in_offset,
                              temp_vec.mesh.begin() + in_offset + y_num,
                              temp_vec.mesh.begin() + out_offset);
                }
            } else {
                for(int z = 0; z < stencil_shape[2]; ++z) {
                    uint64_t r = x_num - 1 + stencil_half[1]; // x index of the boundary
                    uint64_t x_in = (r - (x-r)) % stencil_shape[2];
                    uint64_t in_offset = z * xy_num + x_in * y_num;
                    uint64_t out_offset = z * xy_num + (x % stencil_shape[1]) * y_num;

                    std::copy(temp_vec.mesh.begin() + in_offset,
                              temp_vec.mesh.begin() + in_offset + y_num,
                              temp_vec.mesh.begin() + out_offset);
                }
            }
        } else { // zero padding

            for(int z = 0; z < stencil_shape[2]; ++z) {
                uint64_t out_offset = z * xy_num + (x % stencil_shape[1]) * y_num;

                std::fill(temp_vec.mesh.begin() + out_offset,
                          temp_vec.mesh.begin() + out_offset + y_num,
                          0);
            }
        }
    }



    template<typename T>
    inline void apply_boundary_conditions_y(const uint64_t z, const uint64_t x,ImageBuffer<T> &temp_vec, const bool boundary_condition,const std::vector<int> &stencil_half,const std::vector<int> &stencil_shape){

        const size_t y_num = temp_vec.y_num;
        const uint64_t base_offset = (z % stencil_shape[2]) * temp_vec.x_num * y_num + (x % stencil_shape[1]) * y_num;

        if(boundary_condition == REFLECT_PAD){
            for(int y = 0; y < stencil_half[0]; ++y) {
                temp_vec.mesh[base_offset + stencil_half[0] - 1 - y] = temp_vec.mesh[base_offset + stencil_half[0] + 1 + y];
            }

            for(int y = 0; y < stencil_half[0]; ++y) {
                const uint64_t r = y_num - 1 - stencil_half[0];

                temp_vec.mesh[base_offset + r + 1 + y] = temp_vec.mesh[base_offset + r - 1 - y];
            }

        } else { // zero pad
            for(int y = 0; y < stencil_half[0]; ++y) {
                temp_vec.mesh[base_offset + y] = 0;
            }

            for(int y = 0; y < stencil_half[0]; ++y) {
                temp_vec.mesh[base_offset + y_num - stencil_half[0] + y] = 0;
            }
        }
    }

    /**
     * Fills a pixel image with the particle values at a given level and depth (z), where the particles exactly match
     * the pixels.
     */
    template<typename T, typename ParticleDataType>
    void update_same_level(const int level,
                           const int z,
                           APR &apr,
                           ImageBuffer<T> &temp_vec,
                           ParticleDataType &inputParticles,
                           std::vector<int> stencil_half,
                           std::vector<int> stencil_shape){

        auto apr_it = apr.iterator();
        const int x_num_m = temp_vec.x_num;
        const int y_num_m = temp_vec.y_num;

        uint64_t base_offset = stencil_half[0] + x_num_m * y_num_m * ((z + stencil_half[2]) % stencil_shape[2]);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it)
#endif
        for (int x = 0; x < apr_it.x_num(level); ++x) {

            uint64_t mesh_offset = base_offset + (x + stencil_half[1]) * y_num_m;

            for (apr_it.begin(level, z, x); apr_it < apr_it.end(); apr_it++) {
                temp_vec.mesh[apr_it.y() + mesh_offset] = inputParticles[apr_it];
            }
        }
    }


    /**
     * Fills a pixel image with the particle values from one level below a given level and depth (z), that is, the
     * particles correspond to groups of 2^dim pixels.
     */
    template<typename T, typename ParticleDataType>
    void update_higher_level(const int level,
                             const int z,
                             APR &apr,
                             ImageBuffer<T> &temp_vec,
                             ParticleDataType &inputParticles,
                             const std::vector<int> &stencil_half,
                             const std::vector<int> &stencil_shape) {

        auto apr_it = apr.iterator();
        const int x_num_m = temp_vec.x_num;
        const int y_num_m = temp_vec.y_num;

        uint64_t base_offset = stencil_half[0] + x_num_m * y_num_m * ((z + stencil_half[2]) % stencil_shape[2]);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it)
#endif
        for (int x = 0; x < apr_it.x_num(level); ++x) {

            uint64_t mesh_offset = base_offset + (x + stencil_half[1]) * y_num_m;

            for (apr_it.begin(level-1, z/2, x/2); apr_it < apr_it.end(); ++apr_it) {
                int y_m = std::min(2 * apr_it.y() + 1, (int) apr_it.y_num(level) - 1);    // 2y+1+offset

                temp_vec.mesh[2*apr_it.y() + mesh_offset] = inputParticles[apr_it];
                temp_vec.mesh[y_m + mesh_offset] = inputParticles[apr_it];
            }
        }
    }


    template<typename T, typename ParticleDataType>
    void update_higher_level(const int level,
                             const int z,
                             APR &apr,
                             ImageBuffer<T> &temp_vec,
                             ParticleDataType &inputParticles,
                             const std::vector<int> &stencil_half,
                             const std::vector<int> &stencil_shape,
                             const int num_parent_levels) {

        auto apr_it = apr.iterator();
        const int x_num_m = temp_vec.x_num;
        const int y_num_m = temp_vec.y_num;

        uint64_t base_offset = stencil_half[0] + x_num_m * y_num_m * ((z + stencil_half[2]) % stencil_shape[2]);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it) collapse(2)
#endif
        for(int dlevel = 1; dlevel <= num_parent_levels; ++dlevel) {
            for (int x = 0; x < apr_it.x_num(level); ++x) {

                uint64_t mesh_offset = base_offset + (x + stencil_half[1]) * y_num_m;
                int step_size = std::pow(2, dlevel);

                for (apr_it.begin(level - dlevel, z / step_size, x / step_size); apr_it < apr_it.end(); ++apr_it) {
                    int y = step_size * apr_it.y();
                    int y_m = std::min(y + step_size, (int) apr_it.y_num(level));
                    if(y_m > y) {
                        std::fill(temp_vec.mesh.begin() + mesh_offset + y, temp_vec.mesh.begin() + mesh_offset + y_m,
                                  inputParticles[apr_it]);
                    } else {
                        std::cerr << "may be missing data at (l, z, x) = (" << level << ", " << z << ", " << x << ")" << std::endl;
                    }
                }
            }
        }
    }


    /**
     * Fills a pixel image with the particle values from one level above a given level and depth (z), that is, the
     * pixels correspond to groups of 2^dim particles. The values must be precomputed (e.g., through APRTreeNumerics::fill_tree_mean)
     * and passed to the function through tree_data
     */
    template<typename T, typename ParticleDataType>
    void update_lower_level(const int level,
                            const int z,
                            APR &apr,
                            ImageBuffer<T> &temp_vec,
                            ParticleDataType &tree_data,
                            const std::vector<int> &stencil_half,
                            const std::vector<int> &stencil_shape) {

        auto tree_it = apr.tree_iterator();
        const int x_num_m = temp_vec.x_num;
        const int y_num_m = temp_vec.y_num;

        uint64_t base_offset = stencil_half[0] + x_num_m * y_num_m * ((z + stencil_half[2]) % stencil_shape[2]);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(tree_it)
#endif
        for (int x = 0; x < tree_it.x_num(level); ++x) {

            uint64_t mesh_offset = base_offset + (x + stencil_half[1]) * y_num_m;

            for (tree_it.begin(level, z, x); tree_it < tree_it.end(); tree_it++) {
                temp_vec.mesh[tree_it.y() + mesh_offset] = tree_data[tree_it];
            }
        }
    }


    /**
     * Reconstruct isotropic neighborhoods around the particles at a given level and depth (z) in a pixel image.
     */
    template<typename T, typename ParticleDataType, typename ParticleTreeDataType>
    void update_dense_array(const int level,
                            const int z,
                            APR &apr,
                            ParticleDataType &tree_data,
                            ImageBuffer<T> &temp_vec,
                            ParticleTreeDataType &inputParticles,
                            const std::vector<int> &stencil_shape,
                            const std::vector<int> &stencil_half,
                            const bool boundary) {

        update_same_level(level, z, apr, temp_vec, inputParticles, stencil_half, stencil_shape);

        if (level > apr.level_min()) {
            update_higher_level(level, z, apr, temp_vec, inputParticles, stencil_half, stencil_shape);
        }

        if (level < apr.level_max()) {
            update_lower_level(level, z, apr, temp_vec, tree_data, stencil_half, stencil_shape);
        }

        apply_boundary_conditions_xy(z+stencil_half[2], temp_vec, boundary, stencil_half, stencil_shape);
    }


    /**
 * Reconstruct isotropic neighborhoods around the particles at a given level and depth (z) in a pixel image.
 */
    template<typename T, typename ParticleDataType, typename ParticleTreeDataType>
    void update_dense_array(const int level,
                            const int z,
                            APR &apr,
                            ParticleDataType &tree_data,
                            ImageBuffer<T> &temp_vec,
                            ParticleTreeDataType &inputParticles,
                            const std::vector<int> &stencil_shape,
                            const std::vector<int> &stencil_half,
                            const int num_parent_levels,
                            const bool boundary) {

        update_same_level(level, z, apr, temp_vec, inputParticles, stencil_half, stencil_shape);

        if (level > apr.level_min()) {
            update_higher_level(level, z, apr, temp_vec, inputParticles, stencil_half, stencil_shape, num_parent_levels);
        }

        if (level < apr.level_max()) {
            update_lower_level(level, z, apr, temp_vec, tree_data, stencil_half, stencil_shape);
        }

        apply_boundary_conditions_xy(z+stencil_half[2], temp_vec, boundary, stencil_half, stencil_shape);
    }


    template<typename T, typename ParticleDataType>
    void run_convolution(APR &apr, const int z, const int level, ImageBuffer<T> &temp_vec, PixelData<T> &stencil,
                         ParticleDataType &outputParticles, const std::vector<int> &stencil_half, const std::vector<int> &stencil_shape){

        auto apr_it = apr.iterator();
        const int x_num = temp_vec.x_num;
        const int y_num = temp_vec.y_num;
        int x;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_it)
#endif
        for (x = 0; x < apr_it.x_num(level); ++x) {
            for (apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                T val = 0;
                int y = apr_it.y();
                size_t counter = 0;

                // compute the value FIXME
                for(int iz = 0; iz < stencil_shape[2]; ++iz) {

                    uint64_t offset = ((z + iz) % stencil_shape[2]) * x_num * y_num + x * y_num + y;

                    for(int ix = 0; ix < stencil_shape[1]; ++ix) {
                        for(int iy = 0; iy < stencil_shape[0]; ++ iy) {
                            val += temp_vec.mesh[offset + ix * y_num + iy] * stencil.mesh[counter++];
                        }
                    }
                }
                outputParticles[apr_it] = val;
            }
        }
    }


    template<typename T, typename ParticleDataType>
    void run_convolution_pencil(APR &apr, const int level, const int z, const int x, ImageBuffer<T> &temp_vec, PixelData<T> &stencil,
                                ParticleDataType &outputParticles, const std::vector<int> &stencil_half, const std::vector<int> &stencil_shape){

        auto apr_it = apr.iterator();
        const int y_num = temp_vec.y_num;
        const int xy_num = temp_vec.x_num * y_num;

        for (apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

            T val = 0;
            int y = apr_it.y();
            size_t counter = 0;

            // compute the value FIXME
            for(int iz = 0; iz < stencil_shape[2]; ++iz) {

                uint64_t base_offset = ((z + iz) % stencil_shape[2]) * xy_num + y;

                for(int ix = 0; ix < stencil_shape[1]; ++ix) {

                    uint64_t offset = base_offset + ((x + ix) % stencil_shape[1]) * y_num;

                    for(int iy = 0; iy < stencil_shape[0]; ++ iy) {
                        val += temp_vec.mesh[offset + iy] * stencil.mesh[counter++];
                    }
                }
            }
            outputParticles[apr_it] = val;
        }
    }


    template<typename T>
    static void create_gaussian_filter(PixelData<T>& stencil, float sigma, int size) {

        stencil.init(size, size, size);

        float gauss[size];
        int c = 0;

        for(int i = -size/2; i <= size/2; ++i) {
            gauss[c++] = exp(-i*i / (2*sigma*sigma)) / (sigma * sqrtf(2*M_PI));
        }

        for(int i = 0; i < size; ++i) {
            for(int j = 0; j < size; ++j) {
                for(int k = 0; k < size; ++k) {
                    stencil.at(i, j, k) = gauss[i] * gauss[j] * gauss[k];
                }
            }
        }

    }


    template<typename ParticleDataTypeInput, typename T, typename ParticleDataTypeOutput>
    void create_test_particles_equiv(APR& apr, const std::vector<PixelData<T>> &stencil_vec,
                                     ParticleDataTypeInput& input_particles, ParticleDataTypeOutput& output_particles){


        output_particles.init(apr);

        ParticleData<T> part_tree;
        APRTreeNumerics::fill_tree_mean(apr, input_particles, part_tree);

        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();

        int stencil_counter = 0;

        for (int level_local = apr_it.level_max(); level_local >= apr_it.level_min(); --level_local) {

            ImageBuffer<T> by_level_recon;
            by_level_recon.init(apr_it.y_num(level_local),apr_it.x_num(level_local),apr_it.z_num(level_local));

            //for (uint64_t level = std::max((uint64_t)(level_local-1),(uint64_t)apr_iterator.level_min()); level <= level_local; ++level) {
            for (int level = apr_it.level_min(); level <= level_local; ++level) {
                int z = 0;
                int x = 0;
                const float step_size = pow(2, level_local - level);


#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_it)
#endif
                for (z = 0; z < apr_it.z_num(level); z++) {
                    for (x = 0; x < apr_it.x_num(level); ++x) {
                        for (apr_it.begin(level, z, x); apr_it < apr_it.end(); apr_it++) {

                            int dim1 = apr_it.y() * step_size;
                            int dim2 = x * step_size;
                            int dim3 = z * step_size;

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

                int level = level_local;

                const float step_size = 1;

                int z = 0;
                int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(tree_it)
#endif
                for (z = 0; z < tree_it.z_num(level); z++) {
                    for (x = 0; x < tree_it.x_num(level); ++x) {
                        for (tree_it.begin(level, z, x); tree_it < tree_it.end(); tree_it++) {

                            by_level_recon.mesh[z*by_level_recon.x_num*by_level_recon.y_num + x*by_level_recon.y_num + tree_it.y()] = part_tree[tree_it];
                        }
                    }
                }
            }

            int x = 0;
            int z = 0;
            int level = level_local;

            PixelData<T> stencil(stencil_vec[stencil_counter], true);

            std::vector<int> stencil_halves = {((int)stencil.y_num-1)/2, ((int)stencil.x_num-1)/2, ((int)stencil.z_num-1)/2};

            for (z = 0; z < apr.z_num(level); ++z) {
                //lastly loop over particle locations and compute filter.
                for (x = 0; x < apr.x_num(level); ++x) {
                    for (apr_it.begin(level, z, x); apr_it < apr_it.end(); apr_it++) {

                        T neigh_sum = 0;
                        size_t counter = 0;

                        const int k = apr_it.y(); // offset to allow for boundary padding
                        const int i = x;

                        if(boundary_cond) {
                            int iy, ix, iz;

                            for (int l = -stencil_halves[2]; l < stencil_halves[2]+1; ++l) {
                                for (int q = -stencil_halves[1]; q < stencil_halves[1]+1; ++q) {
                                    for (int w = -stencil_halves[0]; w < stencil_halves[0]+1; ++w) {

                                        iy = k+w;
                                        ix = i+q;
                                        iz = z+l;

                                        if(iy < 0) {
                                            iy = -iy;
                                        } else if(iy >= apr_it.y_num(level)) {
                                            iy = (apr_it.y_num(level) - 1) - (iy - (apr_it.y_num(level) - 1));
                                        }

                                        if(ix < 0) {
                                            ix = -ix;
                                        } else if(ix >= apr_it.x_num(level)) {
                                            ix = (apr_it.x_num(level) - 1) - (ix - (apr_it.x_num(level) - 1));
                                        }

                                        if(iz < 0) {
                                            iz = -iz;
                                        } else if(iz >= apr_it.z_num(level)) {
                                            iz = (apr_it.z_num(level) - 1) - (iz - (apr_it.z_num(level) - 1));
                                        }

                                        neigh_sum += stencil.mesh[counter++] * by_level_recon.at(iy, ix, iz);
                                    }
                                }
                            }
                        } else {
                            for (int l = -stencil_halves[2]; l < stencil_halves[2]+1; ++l) {
                                for (int q = -stencil_halves[1]; q < stencil_halves[1]+1; ++q) {
                                    for (int w = -stencil_halves[0]; w < stencil_halves[0]+1; ++w) {

                                        if(((k+w)>=0) & ((k+w) < (apr.y_num(level)))){
                                            if(((i+q)>=0) & ((i+q) < (apr.x_num(level)))){
                                                if(((z+l)>=0) & ((z+l) < (apr.z_num(level)))){
                                                    neigh_sum += stencil.mesh[counter] * by_level_recon.at(k + w, i + q, z+l);
                                                }
                                            }
                                        }
                                        counter++;
                                    }
                                }
                            }
                        }

                        output_particles[apr_it] = neigh_sum;
                    }
                }
            }

            stencil_counter = std::min(stencil_counter+1, (int)stencil_vec.size()-1);
        }
    }


    /**
     * Fills a pixel image with the particle values at a given level and depth (z), where the particles exactly match
     * the pixels.
     */
    template<typename T, typename ParticleDataType, typename APRIteratorType>
    inline void update_same_level(const int level,
                                  const int z,
                                  const int x,
                                  APRIteratorType &apr_it,
                                  ImageBuffer<T> &temp_vec,
                                  ParticleDataType &inputParticles,
                                  std::vector<int> stencil_half,
                                  std::vector<int> stencil_shape){

        uint64_t mesh_offset = ((z + stencil_half[2]) % stencil_shape[2]) * temp_vec.x_num * temp_vec.y_num +
                               ((x + stencil_half[1]) % stencil_shape[1]) * temp_vec.y_num +
                               stencil_half[0];

        for (apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
            temp_vec.mesh[apr_it.y() + mesh_offset] = inputParticles[apr_it];
        }

    }



    template<typename T, typename ParticleDataType, typename APRIteratorType>
    inline void update_higher_level(const int level,
                                    const int z,
                                    const int x,
                                    APRIteratorType &apr_it,
                                    ImageBuffer<T> &temp_vec,
                                    ParticleDataType &inputParticles,
                                    const std::vector<int> &stencil_half,
                                    const std::vector<int> &stencil_shape) {

        uint64_t mesh_offset = ((z + stencil_half[2]) % stencil_shape[2]) * temp_vec.x_num * temp_vec.y_num +
                               ((x + stencil_half[1]) % stencil_shape[1]) * temp_vec.y_num +
                               stencil_half[0];

        for (apr_it.begin(level-1, z/2, x/2); apr_it < apr_it.end(); ++apr_it) {
            int y_m = std::min(2 * apr_it.y() + 1, (int) apr_it.y_num(level) - 1);    // 2y+1+offset

            temp_vec.mesh[2*apr_it.y() + mesh_offset] = inputParticles[apr_it];
            temp_vec.mesh[y_m + mesh_offset] = inputParticles[apr_it];
        }
    }


/**
 * Fills a pixel image with the particle values from one level above a given level and depth (z), that is, the
 * pixels correspond to groups of 2^dim particles. The values must be precomputed (e.g., through APRTreeNumerics::fill_tree_mean)
 * and passed to the function through tree_data
 */
    template<typename T, typename ParticleDataType, typename APRTreeIteratorType>
    inline void update_lower_level(const int level,
                                   const int z,
                                   const int x,
                                   APRTreeIteratorType &tree_it,
                                   ImageBuffer<T> &temp_vec,
                                   ParticleDataType &tree_data,
                                   const std::vector<int> &stencil_half,
                                   const std::vector<int> &stencil_shape) {

        uint64_t mesh_offset = ((z + stencil_half[2]) % stencil_shape[2]) * temp_vec.x_num * temp_vec.y_num +
                               ((x + stencil_half[1]) % stencil_shape[1]) * temp_vec.y_num +
                               stencil_half[0];

        for (tree_it.begin(level, z, x); tree_it < tree_it.end(); ++tree_it) {
            temp_vec.mesh[tree_it.y() + mesh_offset] = tree_data[tree_it];
        }
    }


/**
 * Reconstruct isotropic neighborhoods around the particles at a given level and depth (z) in a pixel image.
 */
    template<typename T, typename ParticleDataType, typename ParticleTreeDataType>
    void update_dense_array(const int level,
                            const int z,
                            const int x,
                            APR &apr,
                            ParticleDataType &tree_data,
                            ImageBuffer<T> &temp_vec,
                            ParticleTreeDataType &inputParticles,
                            const std::vector<int> &stencil_shape,
                            const std::vector<int> &stencil_half,
                            const bool boundary) {

        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();

        update_same_level(level, z, x, apr_it, temp_vec, inputParticles, stencil_half, stencil_shape);

        if (level > apr.level_min()) {
            update_higher_level(level, z, x, apr_it, temp_vec, inputParticles, stencil_half, stencil_shape);
        }

        if (level < apr.level_max()) {
            update_lower_level(level, z, x, tree_it, temp_vec, tree_data, stencil_half, stencil_shape);
        }

        apply_boundary_conditions_y(z+stencil_half[2], x+stencil_half[1], temp_vec, boundary, stencil_half, stencil_shape);
    }


    template<typename inputType, typename outputType, typename stencilType>
    void convolve_pixel(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil) {

        output.init(input);

        int y_num = input.y_num;
        int x_num = input.x_num;
        int z_num = input.z_num;

        const std::vector<int> stencil_shape = {(int) stencil.y_num,
                                                (int) stencil.x_num,
                                                (int) stencil.z_num};

        const std::vector<int> stencil_half = {(stencil_shape[0] - 1)/2,
                                               (stencil_shape[1] - 1)/2,
                                               (stencil_shape[2] - 1)/2};

        int z;

#ifdef HAVE_OPENMP
#pragma omp parallel for private(z)
#endif
        for (z = 0; z < z_num; ++z) {
            for (int x = 0; x < x_num; ++x) {
                for (int y = 0; y < y_num; ++y) {

                    float neighbour_sum=0;
                    int counter = 0;

                    for (int l = -stencil_half[2]; l < stencil_half[2]+1; ++l) {
                        for (int q = -stencil_half[1]; q < stencil_half[1]+1; ++q) {
                            for (int w = -stencil_half[0]; w < stencil_half[0]+1; ++w) {

                                if( ((y+w)>=0) && ((y+w) < y_num) ){
                                    if( ((x+q)>=0) && ((x+q) < x_num) ) {
                                        if(((z+l)>=0) & ((z+l) < z_num) ) {
                                            neighbour_sum += stencil.mesh[counter] * input.at(y+w, x+q, z+l);
                                        }
                                    }
                                }
                                counter++;
                            }
                        }
                    }

                    output.at(y,x,z) = neighbour_sum;

                }
            }
        }
    }

};

// todo: make static
template<typename ParticleDataTypeInput, typename T,typename ParticleDataTypeOutput>
void APRFilter::convolve(APR &apr, std::vector<PixelData<T>>& stencils, ParticleDataTypeInput &particle_input, ParticleDataTypeOutput &particle_output) {

    particle_output.init(apr);

    //const bool boundary = boundary_cond;

    /**** initialize and fill the apr tree ****/
    ParticleData<T> tree_data;
    APRTreeNumerics::fill_tree_mean(apr, particle_input, tree_data);

    /// allocate image buffer with pad for isotropic patch reconstruction
    // this is reused for lower levels -- assumes that the stencils are not increasing in size !
    int y_num_m = (apr.org_dims(0) > 1) ? apr.y_num(apr.level_max()) + stencils[0].y_num - 1 : 1;
    int x_num_m = (apr.org_dims(1) > 1) ? apr.x_num(apr.level_max()) + stencils[0].x_num - 1 : 1;
    ImageBuffer<T> temp_vec(y_num_m, x_num_m, stencils[0].z_num);

    for (int level = apr.level_max(); level >= apr.level_min(); --level) {

        int stencil_num = std::min((int)stencils.size()-1,(int)(apr.level_max()-level));

        const std::vector<int> stencil_shape = {(int) stencils[stencil_num].y_num,
                                                (int) stencils[stencil_num].x_num,
                                                (int) stencils[stencil_num].z_num};
        const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2,
                                               (stencil_shape[1] - 1) / 2,
                                               (stencil_shape[2] - 1) / 2};

        int max_stencil_radius = *std::max_element(stencil_half.begin(), stencil_half.end());
        int max_num_parent_levels = std::ceil(std::log2(max_stencil_radius+2)-1);
        int num_parent_levels = std::min(max_num_parent_levels, level - (int)apr.level_min());

        const int z_num = apr.z_num(level);

        y_num_m = (apr.org_dims(0) > 1) ? apr.y_num(level) + stencil_shape[0] - 1 : 1;
        x_num_m = (apr.org_dims(1) > 1) ? apr.x_num(level) + stencil_shape[1] - 1 : 1;

        // modify temp_vec boundaries but leave the allocated memory intact
        temp_vec.y_num = y_num_m;
        temp_vec.x_num = x_num_m;
        temp_vec.z_num = stencil_shape[2];

        if(z_num > stencil_half[2]) {

            // Initial fill of temp_vec
            for (int iz = 0; iz <= stencil_half[2]; ++iz) {
                update_dense_array(level, iz, apr, tree_data, temp_vec, particle_input, stencil_shape, stencil_half,
                                   num_parent_levels, boundary_cond);
            }

            // Boundary condition in z direction (lower end)
            for (int iz = 0; iz < stencil_half[2]; ++iz) {
                apply_boundary_conditions_z(iz, z_num, temp_vec, boundary_cond, true, stencil_half, stencil_shape);
            }

            // first iteration out of loop to avoid extra if statements
            run_convolution(apr, 0, level, temp_vec, stencils[stencil_num], particle_output, stencil_half,
                            stencil_shape);

            int z;
            // "interior" iterations
            for (z = 1; z < (z_num - stencil_half[2]); ++z) {
                update_dense_array(level, z + stencil_half[2], apr, tree_data, temp_vec, particle_input, stencil_shape,
                                   stencil_half, num_parent_levels, boundary_cond);
                run_convolution(apr, z, level, temp_vec, stencils[stencil_num], particle_output, stencil_half, stencil_shape);
            }

            // the remaining iterations with boundary condition in z direction (upper end)
            for (z = (z_num - stencil_half[2]); z < z_num; ++z) {
                apply_boundary_conditions_z(z + 2 * stencil_half[2], z_num, temp_vec, boundary_cond, false,
                                            stencil_half, stencil_shape);
                run_convolution(apr, z, level, temp_vec, stencils[stencil_num], particle_output, stencil_half, stencil_shape);
            }

        } else {

            for (int iz = 0; iz < z_num; ++iz) {
                update_dense_array(level, iz, apr, tree_data, temp_vec, particle_input, stencil_shape, stencil_half,
                                   num_parent_levels, boundary_cond);
            }

            for(int iz = z_num; iz <= stencil_half[2]; ++iz) {
                apply_boundary_conditions_z(iz+stencil_half[2], z_num, temp_vec, boundary_cond, false, stencil_half, stencil_shape);
            }

            // Boundary condition in z direction (lower end)
            for (int iz = 0; iz < stencil_half[2]; ++iz) {
                apply_boundary_conditions_z(iz, z_num, temp_vec, boundary_cond, true, stencil_half, stencil_shape);
            }

            run_convolution(apr, 0, level, temp_vec, stencils[stencil_num], particle_output, stencil_half, stencil_shape);

            for(int z = 1; z < z_num; ++z) {
                apply_boundary_conditions_z(z+2*stencil_half[2], z_num, temp_vec, boundary_cond, false, stencil_half, stencil_shape);
                run_convolution(apr, z, level, temp_vec, stencils[stencil_num], particle_output, stencil_half, stencil_shape);
            }
        }
    }
}


template<typename ParticleDataTypeInput, typename T,typename ParticleDataTypeOutput>
void APRFilter::convolve_pencil(APR &apr, std::vector<PixelData<T>>& stencils, ParticleDataTypeInput &particle_input, ParticleDataTypeOutput &particle_output) {

    particle_output.init(apr);

    const bool boundary = boundary_cond;

    /**** initialize and fill the apr tree ****/
    ParticleData<T> tree_data;
    APRTreeNumerics::fill_tree_mean(apr, particle_input, tree_data);

    size_t y_num_m = (apr.org_dims(0) > 1) ? apr.y_num(apr.level_max()) + stencils[0].y_num - 1 : 1;

    int num_threads = 1;
#ifdef HAVE_OPENMP
    num_threads = omp_get_max_threads();
#endif

    std::vector<ImageBuffer<T>> temp_vecs(num_threads);

    for(int i = 0; i < num_threads; ++i) {
        temp_vecs[i].init(y_num_m, stencils[0].x_num, stencils[0].z_num);
    }

    for (int level = apr.level_max(); level >= apr.level_min(); --level) {

        int stencil_num = std::min((int)stencils.size()-1,(int)(apr.level_max()-level));

        const std::vector<int> stencil_shape = {(int) stencils[stencil_num].y_num,
                                                (int) stencils[stencil_num].x_num,
                                                (int) stencils[stencil_num].z_num};
        const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2,
                                               (stencil_shape[1] - 1) / 2,
                                               (stencil_shape[2] - 1) / 2};

        int z, x;

        const int z_num = apr.z_num(level);
        const int x_num = apr.x_num(level);

        y_num_m = (apr.org_dims(0) > 1) ? apr.y_num(level) + stencil_shape[0] - 1 : 1;

        // modify temp_vec boundaries but leave the allocated memory intact
        for(int i = 0; i < num_threads; ++i) {
            temp_vecs[i].y_num = y_num_m;
            temp_vecs[i].x_num = stencil_shape[1];
            temp_vecs[i].z_num = stencil_shape[2];
        }

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x)
#endif
        for (z = 0; z < z_num; ++z) {

            int thread_num = 0;
#ifdef HAVE_OPENMP
            thread_num = omp_get_thread_num();
#endif

            const int z_start = std::max((int)z - stencil_half[2], 0);
            const int z_end = std::min((int)z+stencil_half[2]+1, (int)z_num);

            /// initial fill of temp_vec
            for(int iz = z_start; iz < z_end; ++iz) {
                for (int ix = 0; ix <= stencil_half[1]; ++ix) {
                    update_dense_array(level, iz, ix, apr, tree_data, temp_vecs[thread_num], particle_input, stencil_shape, stencil_half, boundary);
                }
            }

            if (z < stencil_half[2]) {
                for(int iz = z; iz < stencil_half[2]; ++iz) {
                    apply_boundary_conditions_z(iz, z_num, temp_vecs[thread_num], boundary, true, stencil_half, stencil_shape);
                }
            } else if( z >= z_num - stencil_half[2]) {
                for(int iz = z_num - stencil_half[2]; iz <= z; ++iz) {
                    apply_boundary_conditions_z(iz + 2*stencil_half[2], z_num, temp_vecs[thread_num], boundary, false, stencil_half, stencil_shape);
                }
            } /// end of initial fill

            for (int ix = 0; ix < stencil_half[1]; ++ix) {
                apply_boundary_conditions_x(ix, x_num, temp_vecs[thread_num], boundary, true, stencil_half, stencil_shape);
            }

            run_convolution_pencil(apr, level, z, 0, temp_vecs[thread_num], stencils[stencil_num], particle_output, stencil_half, stencil_shape);

            for(x = 1; x < x_num; ++x) {

                if( x < x_num - stencil_half[1] ){
                    const int z_start = std::max((int)z - stencil_half[2], 0);
                    const int z_end = std::min((int)z+stencil_half[2]+1, (int)z_num);

                    for(int iz = z_start; iz < z_end; ++iz) {
                        update_dense_array(level, iz, x + stencil_half[1], apr, tree_data, temp_vecs[thread_num], particle_input, stencil_shape, stencil_half, boundary);
                    }

                    if(z < stencil_half[2]) {
                        for(int iz = z; iz < stencil_half[2]; ++iz) {
                            apply_boundary_conditions_z(iz, z_num, temp_vecs[thread_num], boundary, true, stencil_half, stencil_shape);
                        }
                    } else if( z >= z_num - stencil_half[2]) {
                        for(int iz = z_num - stencil_half[2]; iz <= z; ++iz) {
                            apply_boundary_conditions_z(iz + 2*stencil_half[2], z_num, temp_vecs[thread_num], boundary, false, stencil_half, stencil_shape);
                        }
                    }

                } else {
                    apply_boundary_conditions_x(x + 2*stencil_half[1], x_num, temp_vecs[thread_num], boundary, false, stencil_half, stencil_shape);
                }

                run_convolution_pencil(apr, level, z, x, temp_vecs[thread_num], stencils[stencil_num], particle_output, stencil_half, stencil_shape);
            }
        }
    }
}


template<typename Input1Type,typename Input2Type, typename OutputType>
inline void multiply(const Input1Type& in1, const Input2Type& in2, OutputType& out) {

    assert(in1.size() == in2.size());
    assert(in1.size() == out.size());

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(none) shared(in1, in2, out)
#endif
    for(uint64_t idx = 0; idx < in1.size(); ++idx) {
        out[idx] = in1[idx] * in2[idx];
    }
}

template<typename Input1Type,typename Input2Type, typename OutputType>
inline void divide(const Input1Type& in1, const Input2Type& in2, OutputType& out) {

    assert(in1.size() == in2.size());
    assert(in1.size() == out.size());

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(none) shared(in1, in2, out)
#endif
    for(uint64_t idx = 0; idx < in1.size(); ++idx) {
        out[idx] = in1[idx] / in2[idx];
    }
}


template<typename ParticleDataTypeInput, typename T,typename ParticleDataTypeOutput>
void APRFilter::richardson_lucy(APR &apr, ParticleDataTypeInput &particle_input, ParticleDataTypeOutput &particle_output,
                                std::vector<PixelData<T>>& psf_vec, std::vector<PixelData<T>>& psf_flipped_vec, int number_iterations) {

    particle_output.init(apr.total_number_tree_particles());
    ParticleData<T> relative_blur(apr.total_number_tree_particles());
    ParticleData<T> error_est(apr.total_number_tree_particles());

    // initialize output with 1s
    std::fill(particle_output.data.begin(), particle_output.data.end(), 1);

    for(int iter = 0; iter < number_iterations; ++iter) {
        convolve(apr, psf_flipped_vec, particle_output, relative_blur);
        divide(particle_input, relative_blur, relative_blur);
        convolve(apr, psf_vec, relative_blur, error_est);
        multiply(error_est, particle_output, particle_output);
    }
}


template<typename ParticleDataTypeInput, typename T,typename ParticleDataTypeOutput>
void APRFilter::richardson_lucy(APR &apr, ParticleDataTypeInput &particle_input, ParticleDataTypeOutput &particle_output,
                                PixelData<T> &psf, int number_iterations, bool use_stencil_downsample, bool normalize) {

    PixelData<T> psf_flipped(psf, false);
    for(int i = 0; i < psf.size(); ++i) {
        psf_flipped.mesh[i] = psf.mesh[psf.size()-1-i];
    }

    std::vector<PixelData<T>> psf_vec;
    std::vector<PixelData<T>> psf_flipped_vec;

    int nstencils = use_stencil_downsample ? apr.level_max() - apr.level_min() : 1;
    get_downsampled_stencils(psf, psf_vec, nstencils, normalize);
    get_downsampled_stencils(psf_flipped, psf_flipped_vec, nstencils, normalize);

    richardson_lucy(apr, particle_input, particle_output, psf_vec, psf_flipped_vec, number_iterations);
}


#endif //APR_TIME_APRFILTER_HPP
