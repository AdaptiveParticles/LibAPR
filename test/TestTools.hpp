//
// Created by gonciarz on 4/10/18.
//

#ifndef LIBAPR_TESTTOOLS_HPP
#define LIBAPR_TESTTOOLS_HPP


#include "data_structures/Mesh/PixelData.hpp"
#include <random>

std::string get_source_directory_apr(){
  // returns path to the directory where utils.cpp is stored

  std::string tests_directory = std::string(__FILE__);
  tests_directory = tests_directory.substr(0, tests_directory.find_last_of("\\/") + 1);

  return tests_directory;
}

/**
 * Compares mesh with provided data
 * @param mesh
 * @param data - data with [Z][Y][X] structure
 * @return true if same
 */
template<typename T>
inline bool compare(PixelData<T> &mesh, const float *data, const float epsilon) {
    size_t dataIdx = 0;
    for (int z = 0; z < mesh.z_num; ++z) {
        for (int y = 0; y < mesh.y_num; ++y) {
            for (int x = 0; x < mesh.x_num; ++x) {
                bool v = std::abs(mesh(y, x, z) - data[dataIdx]) < epsilon;
                if (v == false) {
                    std::cerr << "Mesh and expected data differ. First place at (Y, X, Z) = " << y << ", " << x
                              << ", " << z << ") " << mesh(y, x, z) << " vs " << data[dataIdx] << std::endl;
                    return false;
                }
                ++dataIdx;
            }
        }
    }
    return true;
}

template<typename T>
inline bool initFromZYXarray(PixelData<T> &mesh, const float *data) {
    size_t dataIdx = 0;
    for (int z = 0; z < mesh.z_num; ++z) {
        for (int y = 0; y < mesh.y_num; ++y) {
            for (int x = 0; x < mesh.x_num; ++x) {
                mesh(y, x, z) = data[dataIdx];
                ++dataIdx;
            }
        }
    }
    return true;
}

/**
 * Compares two meshes
 * @param expected
 * @param tested
 * @param maxNumOfErrPrinted - how many error values should be printed (-1 for all)
 * @return number of errors detected
 */
template <typename T>
inline int compareMeshes(const PixelData<T> &expected, const PixelData<T> &tested, double maxError = 0.0001, int maxNumOfErrPrinted = 3) {
    int cnt = 0;
    for (size_t i = 0; i < expected.mesh.size(); ++i) {
        if (std::abs(expected.mesh[i] - tested.mesh[i]) > maxError) {
            if (cnt < maxNumOfErrPrinted || maxNumOfErrPrinted == -1) {
                std::cout << "ERROR expected vs tested mesh: " << (float)expected.mesh[i] << " vs " << (float)tested.mesh[i] << " IDX:" << tested.getStrIndex(i) << std::endl;
            }
            cnt++;
        }
    }
    std::cout << "Number of errors / all points: " << cnt << " / " << expected.mesh.size() << std::endl;
    return cnt;
}


/**
 * compare two sets of particles (can be supplied as ParticleData, VectorData, std::vector, ...)
 * @tparam ParticleTypeA
 * @tparam ParticleTypeB
 * @param expected
 * @param tested
 * @param maxError              tolerance
 * @param maxNumOfErrPrinted    maximum number of errors to print (-1 for all)
 * @return                      number of errors
 */
template <typename ParticleTypeA, typename ParticleTypeB>
inline int64_t compareParticles(const ParticleTypeA &expected, const ParticleTypeB &tested, double maxError = 0.0001, int maxNumOfErrPrinted = 10) {
    int64_t cnt = 0;
    if(expected.size() != tested.size()) {
        std::cerr << "ERROR compareParticles: sizes differ!" << std::endl;
        cnt++;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - tested[i]) > maxError) {
            if (cnt < maxNumOfErrPrinted || maxNumOfErrPrinted == -1) {
                std::cout << "ERROR expected vs tested particle: " << (float)expected[i] << " vs " << (float)tested[i] << " IDX:" << i << std::endl;
            }
            cnt++;
        }
    }
    std::cout << "Number of errors / all points: " << cnt << " / " << expected.size() << std::endl;
    return cnt;
}


/**
 * Generates mesh with provided dims with random values in range [0, 1] * multiplier
 * @param y
 * @param x
 * @param z
 * @param multiplier
 * @return
 */
template <typename T>
inline PixelData<T> getRandInitializedMesh(int y, int x, int z, float multiplier = 2.0f, bool useIdxNumbers = false) {
    PixelData<T> m(y, x, z);
    std::cout << "Mesh info: " << m << std::endl;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
    for (size_t i = 0; i < m.mesh.size(); ++i) {
        m.mesh[i] = useIdxNumbers ? i : dist(mt) * multiplier;
    }
    return m;
}

struct TestBenchStats{

        double inf_norm=0;
        double PSNR=0;


        };

template<typename S,typename T,typename U>
TestBenchStats compare_gt(PixelData<S>& org_img,PixelData<T>& rec_img,PixelData<U>& local_scale,int b = 0,const float background = 1000){

    uint64_t z_num_o = org_img.z_num;
    uint64_t x_num_o = org_img.x_num;
    uint64_t y_num_o = org_img.y_num;

    uint64_t x_num_r = rec_img.x_num;
    uint64_t y_num_r = rec_img.y_num;

    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;

    double mean = 0;
    double inf_norm = 0;
    uint64_t counter = 0;
    double MSE = 0;
    double L1 = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(j,i,k) reduction(+: MSE) reduction(+: counter) reduction(+: mean) reduction(max: inf_norm)
#endif
    for(j = b; j < (z_num_o-b);j++){
        for(i = b; i < (x_num_o-b);i++){

            for(k = b;k < (y_num_o-b);k++){

                double scale = local_scale(k/2,i/2,j/2);

                double val = abs(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k])/scale;
                //SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] = 1000*val;

                auto gt_val = org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k];


                if(gt_val > background) {
                    MSE += pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2);
                    L1 += abs(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k]);

                    mean += val;
                    inf_norm = std::max(inf_norm, val);
                    counter++;
                }
            }
        }
    }

    mean = mean/(1.0*counter);
    MSE = MSE/(1.0*counter);
    L1 = L1/(1.0*counter);

    double MSE_var = 0;
    //calculate the variance
    double var = 0;
    counter = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(j,i,k) reduction(+: var) reduction(+: counter) reduction(+: MSE_var)
#endif
    for(j = b; j < (z_num_o-b);j++){
        for(i = b; i < (x_num_o-b);i++){

            for(k = b;k < (y_num_o-b);k++){


                auto gt_val = org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k];

                if(gt_val > background) {
                    var += pow(pow(org_img.mesh[j * x_num_o * y_num_o + i * y_num_o + k] -
                                   rec_img.mesh[j * x_num_o * y_num_o + i * y_num_o + k], 2) - MSE, 2);
                    MSE_var += pow(pow(org_img.mesh[j * x_num_o * y_num_o + i * y_num_o + k] -
                                       rec_img.mesh[j * x_num_o * y_num_o + i * y_num_o + k], 2) - MSE, 2);

                    counter++;
                }
            }
        }
    }

    //get variance
    var = var/(1.0*counter);
    MSE_var = MSE_var/(1.0*counter);

    double PSNR = 10*log10(64000.0/MSE);

    TestBenchStats outputStats;

    outputStats.PSNR = PSNR;
    outputStats.inf_norm = inf_norm;

    return outputStats;

}


#endif //LIBAPR_TESTTOOLS_HPP
