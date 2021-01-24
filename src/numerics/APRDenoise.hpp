//
// Created by bevan on 29/11/2020.
//

#ifndef LIBAPR_APRDENOISE_H
#define LIBAPR_APRDENOISE_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>

#include <random>
#include <unordered_set>

#include "data_structures/Mesh/PixelData.hpp"
#include "io/hdf5functions_blosc.h"
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "numerics/APRReconstruction.hpp"

//Helper classes

template<typename stencilType>
class Stencil {
public:
    std::vector<stencilType> linear_coeffs;
    std::vector<stencilType> non_linear_coeffs;

    std::vector<int> stencil_dims = {0, 0, 0};

};


class StencilSetUp {

    bool non_linear_flag = true;

public:

    StencilSetUp(APR& apr) {
      auto it = apr.iterator();
      dim = it.number_dimensions();
    }

    size_t dim;
    std::vector<int64_t> index;
    std::vector<std::vector<int>> stencil_index_base;

    PixelData<double> stencil_index_1;
    PixelData<double> stencil_index_2;

    int stencil_span;

    std::vector<int> stencil_dims = {0, 0, 0};

    int center_index;

    std::vector<uint64_t> l_index_1;

    std::vector<uint64_t> nl_index_1;
    std::vector<uint64_t> nl_index_2;

    void setup_pairs_demo(int number_pairs) {

        nl_index_1.resize(number_pairs, 0);
        nl_index_2.resize(number_pairs, 0);

        int number_linear_pts = index.size();

        for (int i = 0; i < number_pairs; ++i) {
            //nl_index_1[i] = (uint64_t)(rand() % number_linear_pts);
            nl_index_1[i] = i;
            nl_index_2[i] = (uint64_t) (rand() % number_linear_pts);
        }

    }

    void stencil_l_index() {

        l_index_1.resize(stencil_index_base.size() - 1, 0);

        int counter = 0;

        for (int i = 0; i < l_index_1.size(); ++i) {
            if (i != center_index) {
                l_index_1[counter] = i;
                counter++;
            }
        }


    }


    void setup_standard(std::vector<int> &stencil_dims_in) {

        non_linear_flag = false;

        stencil_dims[0] = stencil_dims_in[0];
        stencil_dims[1] = stencil_dims_in[1];
        stencil_dims[2] = stencil_dims_in[2];

        std::vector<int> temp;
        temp.resize(3);

        uint64_t counter = 0;

        for (int i = -stencil_dims[2]; i < (stencil_dims[2] + 1); ++i) {
            for (int j = -stencil_dims[1]; j < (stencil_dims[1] + 1); ++j) {
                for (int k = -stencil_dims[0]; k < (stencil_dims[0] + 1); ++k) {

                    temp[0] = k;
                    temp[1] = j;
                    temp[2] = i;
                    stencil_index_base.push_back(temp);

                    if ((i == 0) && (j == 0) && (k == 0)) {
                        center_index = counter;
                    } else {
                        l_index_1.push_back(counter);
                    }

                    counter++;

                }
            }
        }

        stencil_span = std::max(stencil_dims[0], std::max(stencil_dims[1], stencil_dims[2]));

    }

    template<typename T>
    void calculate_global_index(PixelData<T> &img) {

        uint64_t x_num = img.x_num + 2 * stencil_span;
        uint64_t y_num = img.y_num + 2 * stencil_span;

        index.resize(stencil_index_base.size());

        for (int i = 0; i < stencil_index_base.size(); ++i) {
            index[i] = stencil_index_base[i][0] + stencil_index_base[i][1] * y_num +
                       stencil_index_base[i][2] * y_num * x_num;
        }


    }






};

class APRStencils {


public:
    std::vector<Stencil<float>> stencils;
    int dim;
    int number_levels;

    void init(int _number_levels,int _dim)  {
      number_levels = _number_levels;
      dim = _dim;
      stencils.resize(number_levels+1);
    }


  void read_stencil(const std::string& file_name) {

    hid_t fileId = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t base = H5Gopen2(fileId, "/", H5P_DEFAULT);

    int number_levels;
    int dim;
    //stored relative to maximum level to allow the read and write functions to be used with different sized APRs. (inherent assumption of only the image size changing not resolution).
    readAttr(H5T_NATIVE_INT, "number_levels", base, &number_levels);
    readAttr(H5T_NATIVE_INT, "dim", base, &dim);

    this->dim = dim;
    this->number_levels = number_levels;

    stencils.resize(number_levels + 1);

    for (int d_level = 0; d_level <= number_levels; ++d_level) {

      auto &stencil = stencils[d_level];

      std::string level_name = "_level_" + std::to_string(d_level);

      int num_pts_1;
      int num_pts_2;

      //get the number of points
      readAttr(H5T_NATIVE_INT, "num_pts_l" + level_name, base, &num_pts_1);
      readAttr(H5T_NATIVE_INT, "num_pts_nl" + level_name, base, &num_pts_2);

      //read in the full stencil
      std::vector<double> coeff_full;
      coeff_full.resize(num_pts_1 + num_pts_2);

      std::string coeff_n = "coeff" + level_name;

      hdf5_load_data_blosc(base, H5T_NATIVE_DOUBLE, coeff_full.data(), coeff_n.c_str());

      //now compute the linear stencil

      stencil.linear_coeffs.resize(num_pts_1, 0); //need to include the 0 center
      auto offset = 0;

      for (int k1 = 0; k1 < stencil.linear_coeffs.size(); ++k1) {

        stencil.linear_coeffs[k1] = coeff_full[k1 + offset];

      }

      stencil.non_linear_coeffs.resize(num_pts_2, 0);

      for (int k1 = 0; k1 < stencil.non_linear_coeffs.size(); ++k1) {

        stencil.non_linear_coeffs[k1] = coeff_full[k1 + num_pts_1];

      }

    }

    H5Fclose(fileId);

  }


  void write_stencil(std::string &file_name) {

    hid_t fileId = hdf5_create_file_blosc(file_name);

    //hid_t base = H5Gcreate2(fileId, "/a",  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if(this->number_levels==0){
      std::cerr << "Attempting to write non-intialized APR Stencils" << std::endl;
    }

    int _number_levels = this->number_levels;
    //stored relative to maximum level to allow the read and write functions to be used with different sized APRs. (inherent assumption of only the image size changing not resolution).
    writeAttr(H5T_NATIVE_INT, "number_levels", fileId, &_number_levels);

    writeAttr(H5T_NATIVE_INT, "dim", fileId, &(this->dim));

    for (int d_level = 0; d_level <= number_levels; ++d_level) {

      auto &stencil = stencils[d_level];

      std::string level_name = "_level_" + std::to_string(d_level);

      int num_pts_linear = stencil.linear_coeffs.size();
      int num_pts_non_linear = stencil.non_linear_coeffs.size();

      //get the number of points
      writeAttr(H5T_NATIVE_INT, "num_pts_l" + level_name, fileId, &num_pts_linear);
      writeAttr(H5T_NATIVE_INT, "num_pts_nl" + level_name, fileId, &num_pts_non_linear);

      int dim1 = stencil.stencil_dims[0];
      int dim2 = stencil.stencil_dims[1];
      int dim3 = stencil.stencil_dims[2];

      writeAttr(H5T_NATIVE_INT, "dim_1" + level_name, fileId, &dim1);
      writeAttr(H5T_NATIVE_INT, "dim_2" + level_name, fileId, &dim2);
      writeAttr(H5T_NATIVE_INT, "dim_3" + level_name, fileId, &dim3);

      //read in the full stencil
      std::vector<double> coeff_full;
      coeff_full.resize(num_pts_linear + num_pts_non_linear);

      auto offset = 0;

      for (int k1 = 0; k1 < stencil.linear_coeffs.size(); ++k1) {

        coeff_full[k1 + offset] = stencil.linear_coeffs[k1];

      }

      for (int k1 = 0; k1 < stencil.non_linear_coeffs.size(); ++k1) {

        coeff_full[k1 + num_pts_linear] = stencil.non_linear_coeffs[k1];

      }

      writeData(H5T_NATIVE_DOUBLE, "coeff" + level_name, coeff_full, fileId);

    }

    H5Fclose(fileId);

  }


    template<typename T>
    static void write_stencil(std::string &file_name, Stencil<T> &stencil) {

        hid_t fileId = hdf5_create_file_blosc(file_name);

        //hid_t base = H5Gcreate2(fileId, "/a",  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        double num_pts_linear = stencil.linear_coeffs.size();
        double num_pts_non_linear = stencil.non_linear_coeffs.size();

        //get the number of points
        writeAttr(H5T_NATIVE_DOUBLE, "num_pts_l", fileId, &num_pts_linear);
        writeAttr(H5T_NATIVE_DOUBLE, "num_pts_nl", fileId, &num_pts_non_linear);

        double dim1 = stencil.stencil_dims[0];
        double dim2 = stencil.stencil_dims[1];
        double dim3 = stencil.stencil_dims[2];

        writeAttr(H5T_NATIVE_DOUBLE, "dim_1", fileId, &dim1);
        writeAttr(H5T_NATIVE_DOUBLE, "dim_2", fileId, &dim2);
        writeAttr(H5T_NATIVE_DOUBLE, "dim_3", fileId, &dim3);

        //read in the full stencil
        std::vector<double> coeff_full;
        coeff_full.resize(num_pts_linear + num_pts_non_linear);

        auto offset = 0;

        for (int k1 = 0; k1 < stencil.linear_coeffs.size(); ++k1) {

            coeff_full[k1 + offset] = stencil.linear_coeffs[k1];

        }


        for (int k1 = 0; k1 < stencil.non_linear_coeffs.size(); ++k1) {

            coeff_full[k1 + num_pts_linear] = stencil.non_linear_coeffs[k1];

        }

        writeData(H5T_NATIVE_DOUBLE, "coeff", coeff_full, fileId);

        H5Fclose(fileId);

    }

private:


    static void readAttr(hid_t type, std::string name, hid_t aGroupId, void *aDest) {
        hid_t attr_id = H5Aopen(aGroupId, name.c_str(), H5P_DEFAULT);
        H5Aread(attr_id, type, aDest);
        H5Aclose(attr_id);
    }

    static void writeAttr(hid_t type, std::string name, hid_t aGroupId, const void *const aSrc) {
        hsize_t dims[] = {1};
        hdf5_write_attribute_blosc(aGroupId, type, name.c_str(), 1, dims, aSrc);
    }
    template<typename T>
    static void writeData(hid_t type, std::string name, T aContainer, hid_t location) {
        hsize_t dims[] = {aContainer.size()};
        const hsize_t rank = 1;
        hdf5_write_data_blosc(location, type, name.c_str(), rank, dims, aContainer.data(), BLOSC_ZSTD, 1l, 0);
    }



};

std::vector<int> sample_without_replacement(int k, int N, std::default_random_engine &gen) {
    // Sample k elements from the range [1, N] without replacement
    // k should be <= N

    // Create an unordered set to store the samples
    std::unordered_set<int> samples;

    // Sample and insert values into samples
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(1, r)(gen);
        if (!samples.insert(v).second) samples.insert(r);
    }

    // Copy samples into vector
    std::vector<int> result(samples.begin(), samples.end());

    // Shuffle vector
    std::shuffle(result.begin(), result.end(), gen);

    return result;
};

class APRDenoise {

public:

    APRTimer timer;

    int max_level = 4;
    int others_level = 1;

    int number_levels = 4;

    int iteration_max = 1000;
    int iteration_others = 600;

    int N_max = 1000;
    int N_ = 1000;

    int level_min = 0;

    bool estimate_center_flag = true;

    int train_factor = 3; // minimum number of points above the stencil size required for the stencil to train.

    float tolerance = 0.05;

    // Train APR

    template<typename S>
    void train_denoise(APR &apr, ParticleData <S> &parts, APRStencils &aprStencils) {

        train_denoise(apr, parts, parts, aprStencils);
    }


    template<typename S>
    void train_denoise(APR &apr, ParticleData <S> &parts, ParticleData <S> &parts_gt, APRStencils &aprStencils) {
        //
        //  Trains a level dependent de-noising model
        //

        bool verbose = true;

        auto it = apr.iterator();


        int viable_levels = 0;

        for(int level = it.level_max(); level >= it.level_min(); level--){
            uint64_t total_parts = it.particles_level_end(level) - it.particles_level_begin(level);

            //check if enough particles to train a kernel; (K*size of kernel?)

            int stencil_sz;
            if (level == apr.level_max()) {
              stencil_sz = max_level;

            } else {
              stencil_sz = others_level;

            }

            int pts = std::pow(2*stencil_sz+1,it.number_dimensions())*train_factor;

            if(total_parts >= pts){
                viable_levels++;

            }

        }

        if(viable_levels < this->number_levels) {
          std::cout << "Not enough particles at levels to train kernel of desired number of levels" << std::endl;
          std::cout << "Setting number of kernel levels to: " << viable_levels << std::endl;
          this->number_levels = viable_levels;
        }

        aprStencils.init(this->number_levels,it.number_dimensions());

        int stencil_level = aprStencils.number_levels;

        float tol_ = this->tolerance;

        APRTimer timer(true);

        timer.start_timer("train");

        for (int level = apr.level_max(); level >= apr.level_min(); --level) {

            StencilSetUp setUp(apr);

            int stencil_sz;

            int it;
            int N;

            if (level == apr.level_max()) {
                stencil_sz = max_level;
                it = iteration_max;
                N = N_max;
            } else {
                it = iteration_others;
                N = N_;
                stencil_sz = others_level;

            }

            //set the dimension of the stencils learned.
            std::vector<int> stencil_dim;
            stencil_dim.resize(3);

            if (aprStencils.dim == 3) {
                stencil_dim[0] = stencil_sz;
                stencil_dim[1] = stencil_sz;
                stencil_dim[2] = stencil_sz;

            } else if (aprStencils.dim == 2) {

                stencil_dim[0] = stencil_sz;
                stencil_dim[1] = stencil_sz;
                stencil_dim[2] = 0;

            } else if (aprStencils.dim == 1) {

                stencil_dim[0] = stencil_sz;
                stencil_dim[1] = 0;
                stencil_dim[2] = 0;

            }


            setUp.setup_standard(stencil_dim);

            //change the stencil size, the stencil is way to big for the lower levels.
            assemble_system_guided(apr, parts, parts_gt, setUp, aprStencils.stencils[stencil_level], N, it, level,
                                   tol_,verbose);

            tol_ = tol_ / 8; // check this is this necessary, and I don't think the decay makes much sense beyond the first two layers. (This was based on a noise reduction per level).

            if(stencil_level > 1){
              stencil_level--;
            } else {
              break;
            }

        }

        timer.stop_timer();


    }


// Apply model APR

    template<typename R,typename S>
    void apply_denoise(APR &apr, ParticleData <R> &parts_in, ParticleData <S> &parts_out, APRStencils &aprStencils) {

        APRTimer timer(true);

        int stencil_level = aprStencils.number_levels;

        timer.start_timer("apply");

        for (int level = apr.level_max(); level >= std::max((int) apr.level_min(), level_min); --level) {


            StencilSetUp setUp(apr);
            setUp.setup_standard(aprStencils.stencils[stencil_level].stencil_dims);

            apply_conv_guided(apr, parts_in, parts_out, setUp, aprStencils.stencils[stencil_level], level);

            if(stencil_level > 1){
                stencil_level--;
            }

        }
        timer.stop_timer();

    }


    template<typename T, typename R,typename S>
    float
    apply_conv_guided(APR &apr, ParticleData <T> &parts_in,ParticleData <R> &parts_out, StencilSetUp &stencilSetUp, Stencil<S> &stencil, int level) {

        timer.verbose_flag = true;

        PixelData<T> img; //change to be level dependent
        // apr.interp_img(img,parts);

        if(parts_out.size() != parts_in.size()) {
          parts_out.init(apr);
        }

        int delta = level - apr.level_max();

        APRReconstruction::interp_img_us_smooth(apr, img, parts_in, false, delta);

        stencilSetUp.calculate_global_index(img);

        PixelData<T> pad_img;

        timer.start_timer("pad_img");

        paddPixels(img, pad_img, stencilSetUp.stencil_span, stencilSetUp.stencil_span, stencilSetUp.stencil_span);

        timer.stop_timer();

        const uint64_t off_y = (uint64_t) stencilSetUp.stencil_span;
        const uint64_t off_x = (uint64_t) std::min((size_t) stencilSetUp.stencil_span, img.x_num - 1);
        const uint64_t off_z = (uint64_t) std::min((size_t) stencilSetUp.stencil_span, img.z_num - 1);

        const uint64_t x_num_p = img.x_num + 2 * off_x;
        const uint64_t y_num_p = img.y_num + 2 * off_y;

        std::vector<float> local_vec;

        local_vec.resize(stencilSetUp.index.size(), 0);

        const int stencil_size = stencilSetUp.index.size();
        const int unroll_size = 16;
        const int loop_sz = (stencil_size / unroll_size) * unroll_size;

        timer.start_timer("conv guided");


        auto it = apr.iterator();

        uint64_t z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(local_vec, it)
#endif
        for (z = 0; z < img.z_num; ++z) {
            for (uint64_t x = 0; x < img.x_num; ++x) {

                float temp_val = 0;

                const uint64_t global_off = off_y + (off_x + x) * y_num_p + (off_z + z) * x_num_p * y_num_p;

                for (it.begin(level, z, x); it < it.end(); it++) {

                    uint64_t y = it.y();

                    //Get the local stencil of points
                    const uint64_t global_off_l = y + global_off;

                    int i = 0;

                    for (; i < loop_sz; i += unroll_size) {
                        local_vec[i] = pad_img.mesh[global_off_l + stencilSetUp.index[i]];
                        local_vec[i + 1] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 1]];
                        local_vec[i + 2] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 2]];
                        local_vec[i + 3] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 3]];
                        local_vec[i + 4] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 4]];
                        local_vec[i + 5] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 5]];
                        local_vec[i + 6] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 6]];
                        local_vec[i + 7] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 7]];
                        local_vec[i + 8] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 8]];
                        local_vec[i + 9] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 9]];
                        local_vec[i + 10] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 10]];
                        local_vec[i + 11] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 11]];
                        local_vec[i + 12] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 12]];
                        local_vec[i + 13] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 13]];
                        local_vec[i + 14] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 14]];
                        local_vec[i + 15] = pad_img.mesh[global_off_l + stencilSetUp.index[i + 15]];
                    }

                    for (; i < stencil_size; i++) {
                        local_vec[i] = pad_img.mesh[global_off_l + stencilSetUp.index[i]];
                    }


                    parts_out[it] = std::inner_product(local_vec.begin(),
                                                 local_vec.end(),
                                                 stencil.linear_coeffs.begin(),
                                                 temp_val);

                }
            }
        }

        timer.stop_timer();

        float time = timer.timings.back();
        return time;

    }


    template<typename T, typename S>
    float apply_nl_conv(PixelData<T> &img, StencilSetUp &stencilSetUp, Stencil<S> &stencil) {

        timer.verbose_flag = true;

        PixelData<T> pad_img;

        timer.start_timer("pad_img");

        padd_boundary2(img, pad_img, stencilSetUp.stencil_span);

        timer.stop_timer();

        const uint64_t off_y = (uint64_t) stencilSetUp.stencil_span;
        const uint64_t off_x = (uint64_t) stencilSetUp.stencil_span;
        const uint64_t off_z = (uint64_t) std::min((size_t) stencilSetUp.stencil_span, img.z_num - 1);

        const uint64_t x_num_p = img.x_num + 2 * off_x;
        const uint64_t y_num_p = img.y_num + 2 * off_y;

        std::vector<float> local_vec;

        local_vec.resize(stencilSetUp.index.size(), 0);

        std::vector<float> nl_local_vec;
        nl_local_vec.resize(stencilSetUp.nl_index_1.size(), 0);


        timer.start_timer("conv");

        uint64_t z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(z) firstprivate(local_vec, nl_local_vec)
#endif
        for (z = 0; z < img.z_num; ++z) {
            for (uint64_t x = 0; x < img.x_num; ++x) {

                const uint64_t img_off = x * img.y_num + z * (img.y_num * img.z_num);

                double temp_val = 0;

                const uint64_t global_off = off_y + (off_x + x) * y_num_p + (off_z + z) * x_num_p * y_num_p;

                for (uint64_t y = 0; y < img.y_num; ++y) {

                    //Get the local stencil of points
                    const uint64_t global_off_l = y + global_off;

                    for (int i = 0; i < stencilSetUp.index.size(); ++i) {
                        local_vec[i] = pad_img.mesh[global_off_l + stencilSetUp.index[i]];
                    }

                    //Apply the linear kernel
                    temp_val = 0;

                    temp_val = std::inner_product(local_vec.begin(), local_vec.end(), stencil.linear_coeffs.begin(),
                                                  temp_val);

                    //Apply the non-linear kernel
                    for (int i = 0; i < stencilSetUp.nl_index_1.size(); ++i) {
                        nl_local_vec[i] = local_vec[stencilSetUp.nl_index_1[i]] * local_vec[stencilSetUp.nl_index_2[i]];
                    }

                    temp_val = std::inner_product(nl_local_vec.begin(), nl_local_vec.end(),
                                                  stencil.non_linear_coeffs.begin(), temp_val);

                    //store_val;
                    img.mesh[img_off + y] = temp_val;

                }
            }
        }

        timer.stop_timer();

        return timer.timings.back();


    }

    template<typename T, typename S>
    void
    assemble_system_guided(APR &apr, ParticleData <T> &parts, ParticleData <T> &parts_g, StencilSetUp &stencilSetUp,
                           Stencil<S> &stencil, int N, int num_rep, int level, float factor = 0.05,
                           bool verbose = false) {

        timer.verbose_flag = verbose;
        timer.verbose_flag = true;

        PixelData<T> img; //change to be level dependent
        // apr.interp_img(img,parts);
        int delta = (level - apr.level_max());

        auto level_ = level;

        auto apr_iterator = apr.iterator();

        uint64_t total_parts = apr_iterator.particles_level_end(level_) - apr_iterator.particles_level_begin(level_);

        APRReconstruction::interp_img_us_smooth(apr, img, parts, false, delta);
        stencilSetUp.calculate_global_index(img);

        if (total_parts < N) {
            N =  stencilSetUp.l_index_1.size()*this->train_factor;
        }

        PixelData<T> pad_img;

        timer.start_timer("pad_img");

        paddPixels(img, pad_img, stencilSetUp.stencil_span, stencilSetUp.stencil_span, stencilSetUp.stencil_span);

        timer.stop_timer();

        const uint64_t off_y = (uint64_t) stencilSetUp.stencil_span;
        const uint64_t off_x = (uint64_t) stencilSetUp.stencil_span;
        const uint64_t off_z = (uint64_t) std::min((size_t) stencilSetUp.stencil_span, img.z_num - 1);

        const uint64_t x_num_p = img.x_num + 2 * off_x;
        const uint64_t y_num_p = img.y_num + 2 * off_y;

        std::vector<T> local_vec;

        local_vec.resize(stencilSetUp.index.size(), 0);

        std::vector<float> nl_local_vec;
        nl_local_vec.resize(stencilSetUp.nl_index_1.size(), 0);

        auto n = (int) stencilSetUp.l_index_1.size() + stencilSetUp.nl_index_1.size();

        std::vector<uint64_t> random_index;

        random_index.resize(N);

        auto l_num = stencilSetUp.l_index_1.size();

        //PixelData<float> A_temp;
        //A_temp.init(n,N,1);

        std::vector<double> b_temp;
        b_temp.resize(N);

        std::vector<std::vector<double>> coeff_store;
        coeff_store.resize(num_rep);



        timer.start_timer("assemble");


        auto total_p = (img.x_num * img.z_num * img.y_num) / N;

        total_p = total_parts / N;

        std::vector<double> A_temp;
        A_temp.resize(n * N, 0);

        Eigen::Map<Eigen::MatrixXd> A(A_temp.data(), n, N);

        //Eigen::MatrixXd A(n,N);

        Eigen::VectorXd coeff_prev(n);

        std::vector<uint16_t> x_vec;
        std::vector<uint16_t> y_vec;
        std::vector<uint16_t> z_vec;

        std::vector<uint64_t> global_index;


        int z = 0;
        int x = 0;


        for (z = 0; z < apr_iterator.z_num(level_); z++) {
            for (x = 0; x < apr_iterator.x_num(level_); ++x) {

                for (apr_iterator.begin(level_, z, x); apr_iterator < apr_iterator.end(); apr_iterator++) {

                    x_vec.push_back(x);
                    z_vec.push_back(z);
                    y_vec.push_back(apr_iterator.y());

                    global_index.push_back(apr_iterator.global_index());

                }
            }
        }

        timer.stop_timer();

        timer.start_timer("solve loop");


        for (int l = 0; l < num_rep; ++l) {

            APRTimer timer_l(verbose);

            timer_l.start_timer("random");

            random_index[0] = std::rand() % total_p + 1;

            for (int j = 1; j < random_index.size(); ++j) {

                random_index[j] = random_index[j - 1] + std::rand() % total_p + 1;
            }

            timer_l.stop_timer();

            uint64_t k = 0;

            timer_l.start_timer("A");

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(k) firstprivate(local_vec)
#endif
            for (k = 0; k < N; ++k) {

                const uint64_t r_i = random_index[k];

                const uint64_t z = z_vec[r_i];
                const uint64_t x = x_vec[r_i];
                const uint64_t y = y_vec[r_i];

                const uint64_t global_off = off_y + (off_x + x) * y_num_p + (off_z + z) * x_num_p * y_num_p;

                //Get the local stencil of points
                const uint64_t global_off_l = y + global_off;

                for (int i = 0; i < stencilSetUp.index.size(); ++i) {
                    local_vec[i] = pad_img.mesh[global_off_l + stencilSetUp.index[i]];
                }

                //b_temp[k] = local_vec[stencilSetUp.center_index];
                b_temp[k] = parts_g[global_index[r_i]];

                for (int i = 0; i < stencilSetUp.l_index_1.size(); ++i) {
                    //A_temp(i, k, 0) = local_vec[stencilSetUp.l_index_1[i]];
                    A(i, k) = local_vec[stencilSetUp.l_index_1[i]];
                }

                //Apply the non-linear kernel
                for (int i = 0; i < stencilSetUp.nl_index_1.size(); ++i) {
                    //A_temp(i + l_num, k, 0) =
                    //      local_vec[stencilSetUp.nl_index_1[i]] * local_vec[stencilSetUp.nl_index_2[i]];
                    A(i + l_num, k) =
                            local_vec[stencilSetUp.nl_index_1[i]] * local_vec[stencilSetUp.nl_index_2[i]];
                }


            }

            timer_l.stop_timer();

            timer_l.start_timer("solve");

            std::vector<double> norm_c;
            norm_c.resize(n, 0);

            Eigen::Map<Eigen::VectorXd> norm_c_(norm_c.data(), n);


            std::vector<double> std;

            for (int i = 0; i < A.rows(); i++) {
                Eigen::ArrayXd vec = A.row(i);
                double std_dev = std::sqrt((vec - vec.mean()).square().sum() / (vec.size() - 1));
                //std_dev = 1;
                norm_c_(i) = std_dev;
                A.row(i) = A.row(i) / std_dev;
                std.push_back(std_dev);
            }


            Eigen::Map<Eigen::VectorXd> b(b_temp.data(), N);

            coeff_store[l].resize(n, 0);

            if (l == 0) {
                float eps = .01;
                for (int i = 0; i < n; ++i) {
                    coeff_store[l][i] = std::rand() * 2 * eps - eps;
                }


            }

            Eigen::Map<Eigen::VectorXd> coeff(coeff_store[l].data(), n);

            Eigen::LeastSquaresConjugateGradient<Eigen::MatrixXd, Eigen::IdentityPreconditioner> solver;

            //Need to compute desired tol
            float val = factor;
            Eigen::VectorXf ones_v = val * Eigen::VectorXf::Ones(N);

            float norm_b = b.norm();
            float norm_e = ones_v.norm();

            float tol = norm_e / norm_b;

            solver.setMaxIterations(800);
            solver.setTolerance(tol);
            solver.compute(A.transpose());

            for (int m = 0; m < coeff_prev.size(); ++m) {
                coeff_prev(m) = coeff_prev(m) * norm_c_(m);
            }

            if (l > 0) {
                solver.setMaxIterations(20);
                coeff = solver.solveWithGuess(b, coeff_prev);
            } else {
                tol *= 0.0001;
                solver.setTolerance(tol);
                coeff = solver.solve(b);

            }


            for (int i1 = 0; i1 < n; ++i1) {
                coeff(i1) = coeff(i1) / norm_c_(i1);
            }

            coeff_prev = coeff;

            if (verbose) {
                std::cout << "#iterations: " << solver.iterations() << std::endl;
            }

            timer_l.stop_timer();

        }

        timer.stop_timer();


        //now compute the linear stencil

        stencil.linear_coeffs.resize(stencilSetUp.index.size(), 0); //need to include the 0 center
        auto offset = 0;

        stencil.stencil_dims[0] = stencilSetUp.stencil_dims[0];
        stencil.stencil_dims[1] = stencilSetUp.stencil_dims[1];
        stencil.stencil_dims[2] = stencilSetUp.stencil_dims[2];

        int include = std::floor(0.5 * num_rep);

        for (int k1 = 0; k1 < stencil.linear_coeffs.size(); ++k1) {

            if (k1 == stencilSetUp.center_index) {
                offset = -1;
            } else {

                double sum = 0;
                double counter = 0;

                for (int i = include; i < num_rep; ++i) {
                    sum += coeff_store[i][k1 + offset];
                    counter++;
                }
                stencil.linear_coeffs[k1] = sum / (counter * 1.0);

            }

        }


        stencil.non_linear_coeffs.resize(stencilSetUp.nl_index_1.size(), 0);

        for (int k1 = 0; k1 < stencil.non_linear_coeffs.size(); ++k1) {


            double sum = 0;
            double counter = 0;

            for (int i = include; i < num_rep; ++i) {
                sum += coeff_store[i][k1 + l_num];
                counter++;
            }
            stencil.non_linear_coeffs[k1] = sum / (counter * 1.0);

        }



        if(this->estimate_center_flag) {
          // set the center pixel to the largest weight in the kernel, and then re-normalise the kernel

          float factor = 0;
          float sum = 0;

          for (int j = 0; j < stencil.linear_coeffs.size(); ++j) {
            factor = std::max(factor, std::abs(stencil.linear_coeffs[j]));
            sum += stencil.linear_coeffs[j];
          }

          stencil.linear_coeffs[stencilSetUp.center_index] = factor;

          for (int j = 0; j < stencil.linear_coeffs.size(); ++j) {
            stencil.linear_coeffs[j] = stencil.linear_coeffs[j]*(sum)/(sum+factor);
          }

        }


        //NAN check, this can occur if you have noise free data, and the system becomes ill-posed. (likely other reasons as well, hence the warning).

        bool valid_check = true;

        float c_sum = 0;

        for (int l1 = 0; l1 < stencil.linear_coeffs.size(); ++l1) {
          if(std::isnan(stencil.linear_coeffs[l1])){
            valid_check = false;
          }
            c_sum += stencil.linear_coeffs[l1];
        }

        // Does it sum close to 1? very liberal to detect agian non-convergence.
        float error_threshold = 0.2;
        if(std::abs(c_sum - 1.0f) > error_threshold){
            valid_check = false;
        }

        if(!valid_check){
            std::wcerr << "Inference hasn't converged to a solution, setting the kernel to identity. This could be due to std(signal) = 0, or the signal is noise free." << std::endl;

            std::fill(stencil.linear_coeffs.begin(),stencil.linear_coeffs.end(),0);
            stencil.linear_coeffs[stencilSetUp.center_index] = 1;
        }



    }



};

//template<typename ImageType>
//class CameraNoiseRemover {
//
//    PixelData<ImageType> mask_img;
//
//    PixelData<float> mask_correction;
//
//    PixelData<float> sum_img;
//
//public:
//
//    template<typename S>
//    void compute_correction_subsampling(PixelData<S>& input_img){
//
//        int N_s = 1;
//
//        std::default_random_engine gen;
//
//        PixelData<double> mask_temp;
//        mask_temp.initWithValue(input_img.y_num,input_img.x_num,1,0);
//
//        int it = 400;
//
//        for (int k = 0; k < it; ++k) {
//
//            std::vector<int> random_samples = sample_without_replacement(N_s, input_img.z_num - 1, gen);
//
//            PixelData<S> sub_sample;
//            sub_sample.init(input_img.y_num, input_img.x_num, N_s);
//
//            int sl_sz = input_img.x_num * input_img.z_num;
//
//            for (int i = 0; i < N_s; ++i) {
//                int offset_big = sl_sz * random_samples[i];
//                int offset_small = i * sl_sz;
//                std::copy(input_img.mesh.begin() + offset_big, input_img.mesh.begin() + offset_big + sl_sz,
//                          sub_sample.mesh.begin() + offset_small);
//
//            }
//
//            compute_sum(sub_sample);
//            compute_mask(sub_sample);
//            for (int j = 0; j < mask_correction.mesh.size(); ++j) {
//                mask_temp.mesh[j] += mask_correction.mesh[j];
//            }
//
//        }
//
//        for (int j = 0; j < mask_correction.mesh.size(); ++j) {
//            mask_correction.mesh[j] = (mask_temp.mesh[j]/(1.0*it));
//        }
//
//
//    }
//
//    template<typename S>
//    void compute_correction_apr_slice(PixelData<S>& input_img){
//
//        int N_s = 1;
//
//        std::default_random_engine gen;
//
//        PixelData<double> mask_temp;
//        mask_temp.initWithValue(input_img.y_num,input_img.x_num,1,0);
//
//        int it = 10;
//
//        APRConverter<S> aprConverter;
//        aprConverter.par.grad_th = 50;
//
//        for (int k = 0; k < it; ++k) {
//
//            std::vector<int> random_samples = sample_without_replacement(N_s, input_img.z_num - 1, gen);
//
//            PixelData<S> sub_sample;
//            sub_sample.init(input_img.y_num, input_img.x_num, N_s);
//
//            int sl_sz = input_img.x_num * input_img.z_num;
//
//            for (int i = 0; i < N_s; ++i) {
//                int offset_big = sl_sz * random_samples[i];
//                int offset_small = i * sl_sz;
//                std::copy(input_img.mesh.begin() + offset_big, input_img.mesh.begin() + offset_big + sl_sz,
//                          sub_sample.mesh.begin() + offset_small);
//
//            }
//
//            APR apr;
//            aprConverter.get_apr_method_time(apr,sub_sample,false);
//
//            compute_correction_apr(apr,sub_sample);
//
//            for (int j = 0; j < mask_correction.mesh.size(); ++j) {
//                mask_temp.mesh[j] += mask_correction.mesh[j];
//            }
//
//        }
//
//        for (int j = 0; j < mask_correction.mesh.size(); ++j) {
//            mask_correction.mesh[j] = (mask_temp.mesh[j]/(1.0*it));
//        }
//
//
//    }
//
//    template<typename T,typename S>
//    void compute_correction_apr(APR& apr,PixelData<S>& input_img,ParticleData<T>& parts){
//        //
//        //  Estimates a multiplicative camera noise distribution (x,y), assuming f(x,y,z) = gt(1+nu(x,y)) + eps(x,y,z)
//        //
//        //  Does this by smoothing f using (x,y) plane then comparing the sum of the original and smoothed over all slices.
//        //
//
//
//        //first compute the sum from the input image
//        compute_sum(input_img);
//
//        APRStencils aprStencils;
//
//        LearnDenoise learnDenoise;
//
//        aprStencils.dim = 2;
//
//        ParticleData<T> temp_parts;
//        temp_parts.data.resize(apr.total_number_particles(),0);
//
//        std::copy(parts.data.begin(),parts.data.end(),temp_parts.data.begin());
//
//        //temp_parts.copy_parts(apr,apr.particles_intensities);
//
//        learnDenoise.max_level = 4;
//        learnDenoise.others_level = 2;
//
//
//        learnDenoise.train_denoise(apr,temp_parts,aprStencils);
//
//        learnDenoise.apply_denoise(apr,temp_parts,aprStencils);
//
//        PixelData<float> s_img;
//        s_img.initWithValue(input_img.y_num,input_img.x_num,1,0);
//
//        //looping over APR and summing up the image to z.
//
//        auto apr_iterator = apr.iterator();
//
//        for (unsigned int level = apr_iterator.level_min(); level <= (apr_iterator.level_max()); ++level) {
//            int z = 0;
//            int x = 0;
//
//            const bool l_max = (level== apr_iterator.level_max());
//
//            const float step_size = pow(2, apr.level_max() - level);
//
//            for (z = 0; z < apr_iterator.z_num(level); z++) {
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
//#endif
//                for (int x = 0; x < apr_iterator.x_num(level); ++x) {
//
//                    for(apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); apr_iterator++){
//
//                        if(l_max){
//                            s_img.at(apr_iterator.y(),x,0) += temp_parts[apr_iterator];
//                        } else {
//                            int dim1 = apr_iterator.y() * step_size;
//                            int dim2 = x * step_size;
//                            int dim3 = z * step_size;
//
//                            float temp_int;
//                            //add to all the required rays
//
//                            temp_int = temp_parts[apr_iterator];
//
//                            const int offset_max_dim1 = std::min((int) input_img.y_num, (int) (dim1 + step_size));
//                            const int offset_max_dim2 = std::min((int) input_img.x_num, (int) (dim2 + step_size));
//                            const int offset_max_dim3 = std::min((int) input_img.z_num, (int) (dim3 + step_size));
//
//
//                            for (int64_t k = dim2; k < offset_max_dim2; ++k) {
//                                for (int64_t i = dim1; i < offset_max_dim1; ++i) {
//                                    s_img.at(i,k,0) += temp_int*(offset_max_dim3 - dim3);
//                                }
//                            }
//
//                        }
//
//                    }
//                }
//            }
//        }
//
//
//
//
//        std::string image_file_name = "/Volumes/BevanT5/Denoise/check.tif";
//        TiffUtils::saveMeshAsTiff(image_file_name, s_img,false);
//
//        StencilSetUp setUp;
//
//        std::vector<int> stencil_dims;
//        stencil_dims.resize(3,0);
//
//        int dim = 4;
//
//        stencil_dims[0] = dim;
//        stencil_dims[1] = dim;
//
//        setUp.setup_standard(stencil_dims);
//
//        setUp.calculate_global_index(s_img);
//
//        setUp.stencil_l_index();
//
//        Stencil<float> stencil;
//
//        learnDenoise.assemble_system(s_img,setUp,stencil,1000,600,.05,false);
//
//        //learnDenoise.apply_conv(s_img,setUp,stencil);
//
//
//        mask_correction.initWithValue(input_img.y_num,input_img.x_num,1,0);
//
//        int i = 0;
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(static) private(i)
//#endif
//        for (i = 0; i < sum_img.mesh.size(); ++i) {
//            mask_correction.mesh[i] = (sum_img.mesh[i] - s_img.mesh[i])/s_img.mesh[i];
//        }
//
////
////        PixelData<T> interp;
////        apr.interp_img(interp,apr.particles_intensities);
////
////        compute_mask(interp);
//
//
//    }
//
//    template<typename S>
//    void apply_correction(APR& apr,ParticleData<S>& parts){
//
//        std::vector<PixelData<float>> ds_correction;
//
//        PixelData<float> temp;
//        temp.init(mask_correction);
//        temp.copyFromMesh(mask_correction);
//
//        downsamplePyrmaid(temp, ds_correction, apr.level_max(), apr.level_min());
//
//        auto it = apr.iterator();
//
//        for (int l = it.level_min(); l <= it.level_max(); ++l) {
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(static) firstprivate(it)
//#endif
//            for (int z = 0; z < it.z_num(l); ++z) {
//                for (int x = 0; x < it.x_num(l); ++x) {
//                    for (it.begin(l,z,x); it < it.end(); it++) {
//                        parts[it] = parts[it]/(1.0 + ds_correction[l].at(it.y(),x,0));
//                    }
//                }
//
//            }
//
//        }
//
//    }
//
//    double evaluate_correction(){
//
//        double score = 0;
//        uint64_t counter = 0;
//
//        for (int i = 0; i < mask_img.mesh.size(); ++i) {
//            score += pow(mask_correction.mesh[i] - mask_img.mesh[i],2);
//            counter++;
//        }
//
//        score /= (1.0*counter);
//
//        score = log10(1/score);
//
//        std::cout << "Mask Correction: "  << score << std::endl;
//
//        return score;
//
//    }
//
//
//
//    void create_mask_bench(float noise_level,int y_num,int x_num){
//
//        mask_img.init(y_num,x_num,1);
//
//        std::default_random_engine generator;
//        std::normal_distribution<float> distribution(0.0,1.0);
//
//        for (int i = 0; i < mask_img.mesh.size(); ++i) {
//            mask_img.mesh[i] = distribution(generator)*noise_level;
//
//        }
//
//    }
//
//    template<typename T>
//    void distort_image(PixelData<T>& input_img){
//        uint64_t z = 0;
//
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(static) private(z)
//#endif
//        for (z = 0; z < input_img.z_num; ++z) {
//            for (uint64_t x = 0; x < input_img.x_num; ++x) {
//                for (uint64_t y = 0; y < input_img.y_num; ++y) {
//                    input_img.at(y,x,z) = (T) input_img.at(y,x,z)*(1 +  mask_img(y,x,0));
//                }
//            }
//        }
//
//    }
//
//
//    template<typename T>
//    void compute_sum(PixelData<T>& input_img){
//
//        sum_img.initWithValue(input_img.y_num,input_img.x_num,1,0);
//
//        for (uint64_t z = 0; z < input_img.z_num; ++z) {
//
//            uint64_t x = 0;
//
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(static) private(x)
//#endif
//            for ( x = 0; x < input_img.x_num; ++x) {
//                for (uint64_t y = 0; y < input_img.y_num; ++y) {
//                    sum_img.at(y,x,0) += input_img.at(y,x,z);
//                }
//            }
//        }
//
//        std::string image_file_name = "/Volumes/BevanT5/Denoise/sum.tif";
//        TiffUtils::saveMeshAsTiff(image_file_name, sum_img,false);
//
//    }
//
//
//    template<typename T>
//    void compute_mask(PixelData<T>& input_img){
//
//
//
//        PixelData<float> smoothed;
//        smoothed.initWithValue(input_img.y_num,input_img.x_num,1,0);
//
//
//        for (uint64_t z = 0; z < input_img.z_num; ++z) {
//
//            uint64_t x = 0;
//
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(static) private(x)
//#endif
//            for ( x = 0; x < input_img.x_num; ++x) {
//                for (uint64_t y = 0; y < input_img.y_num; ++y) {
//                    smoothed.at(y,x,0) += input_img.at(y,x,z);
//                }
//            }
//        }
//
//
//        StencilSetUp setUp;
//
//        std::vector<int> stencil_dims;
//        stencil_dims.resize(3,0);
//
//        int dim = 4;
//
//        stencil_dims[0] = dim;
//        stencil_dims[1] = dim;
//
//        setUp.setup_standard(stencil_dims);
//
//        setUp.calculate_global_index(smoothed);
//
//        setUp.stencil_l_index();
//
//        LearnDenoise learnDenoise;
//
//        Stencil<float> stencil;
//
//        learnDenoise.assemble_system(smoothed,setUp,stencil,1000,600,.05,false);
//
//        learnDenoise.apply_conv(smoothed,setUp,stencil);
//
//        std::string image_file_name = "/Volumes/BevanT5/Denoise/smooth.tif";
//        TiffUtils::saveMeshAsTiff(image_file_name, smoothed,false);
//
//        //gain_mask = (mean_img - f_fit)./f_fit;
//
//        mask_correction.initWithValue(input_img.y_num,input_img.x_num,1,0);
//
//        int i = 0;
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(static) private(i)
//#endif
//        for (i = 0; i < sum_img.mesh.size(); ++i) {
//            mask_correction.mesh[i] = (sum_img.mesh[i] - smoothed.mesh[i])/smoothed.mesh[i];
//        }
//
//        image_file_name = "/Volumes/BevanT5/Denoise/correction.tif";
//        TiffUtils::saveMeshAsTiff(image_file_name, mask_correction,false);
//
//    }
//
//
//    template<typename T>
//    void apply_correction(PixelData<T>& input_img){
//
//        uint64_t z = 0;
//
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(static) private(z)
//#endif
//        for (z = 0; z < input_img.z_num; ++z) {
//            for (uint64_t x = 0; x < input_img.x_num; ++x) {
//                for (uint64_t y = 0; y < input_img.y_num; ++y) {
//                    input_img.at(y,x,z) = (T) input_img.at(y,x,z)/(1 +  mask_correction(y,x,0));
//                }
//            }
//        }
//
//    }
//
//
//    template<typename T>
//    void apply_perfect_correction(PixelData<T>& input_img){
//
//        uint64_t z = 0;
//
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(static) private(z)
//#endif
//        for (z = 0; z < input_img.z_num; ++z) {
//            for (uint64_t x = 0; x < input_img.x_num; ++x) {
//                for (uint64_t y = 0; y < input_img.y_num; ++y) {
//                    input_img.at(y,x,z) = std::round( 1.0*input_img.at(y,x,z)/(1.0 +  mask_img(y,x,0)));
//                }
//            }
//        }
//
//    }
//
//
//
//
//};




#endif //LIBAPR_APRDENOISE_H
