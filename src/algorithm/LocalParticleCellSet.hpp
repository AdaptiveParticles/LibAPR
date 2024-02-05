//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_LOCAL_PARTICLE_SET_HPP
#define PARTPLAY_LOCAL_PARTICLE_SET_HPP

#ifdef WIN_VS
#include <intrin.h>

// from https://github.com/llvm-mirror/libcxx/blob/9dcbb46826fd4d29b1485f25e8986d36019a6dca/include/support/win32/support.h#L106-L182
// (c) Copyright (c) 2009-2017 by the contributors listed in https://github.com/llvm-mirror/libcxx/blob/9dcbb46826fd4d29b1485f25e8986d36019a6dca/CREDITS.TXT
// Returns the number of leading 0-bits in x, starting at the most significant
// bit position. If x is 0, the result is undefined.
inline int __builtin_clzll(unsigned long long mask)
{
  unsigned long where;
// BitScanReverse scans from MSB to LSB for first set bit.
// Returns 0 if no set bit is found.
#if defined(_LIBCPP_HAS_BITSCAN64)
  if (_BitScanReverse64(&where, mask))
    return static_cast<int>(63 - where);
#else
  // Scan the high 32 bits.
  if (_BitScanReverse(&where, static_cast<unsigned long>(mask >> 32)))
    return static_cast<int>(63 -
                            (where + 32)); // Create a bit offset from the MSB.
  // Scan the low 32 bits.
  if (_BitScanReverse(&where, static_cast<unsigned long>(mask)))
    return static_cast<int>(63 - where);
#endif
  return 64; // Undefined Behavior.
}

inline int __builtin_clzl(unsigned long mask)
{
  unsigned long where;
  // Search from LSB to MSB for first set bit.
  // Returns zero if no set bit is found.
  if (_BitScanReverse(&where, mask))
    return static_cast<int>(31 - where);
  return 32; // Undefined Behavior.
}

inline int __builtin_clz(unsigned int x)
{
  return __builtin_clzl(x);
}

#endif

#include "algorithm/PullingScheme.hpp"
#include "algorithm/PullingSchemeSparse.hpp"

class LocalParticleCellSet {

public:

    APRTimer timer;

    template<typename tempType>
    inline void get_local_particle_cell_set(PullingScheme& iPullingScheme,PixelData<tempType> &local_scale_temp, PixelData<tempType> &local_scale_temp2,const APRParameters& par) {
        //  Computes the Local Particle Cell Set from a down-sampled local intensity scale (\sigma) and gradient magnitude
        //  Down-sampled due to the Equivalence Optimization

#ifdef HAVE_LIBTIFF
        if(par.output_steps){
            TiffUtils::saveMeshAsTiff(par.output_dir + "local_particle_set_level_step.tif", local_scale_temp);
        }
#endif

        int l_max = iPullingScheme.pct_level_max();
        int l_min = iPullingScheme.pct_level_min();

        timer.start_timer("pulling_scheme_fill_max_level");

        iPullingScheme.fill(l_max,local_scale_temp);
        timer.stop_timer();

        timer.start_timer("level_loop_initialize_tree");
        for(int l_ = l_max - 1; l_ >= l_min; l_--){

            //down sample the resolution level k, using a max reduction
            downsample(local_scale_temp, local_scale_temp2,
                       [](const float &x, const float &y) -> float { return std::max(x, y); },
                       [](const float &x) -> float { return x; }, true);
            //for those value of level k, add to the hash table
            iPullingScheme.fill(l_,local_scale_temp2);
            //assign the previous mesh to now be resampled.
            local_scale_temp.swap(local_scale_temp2);
        }
        timer.stop_timer();

    }

    template<typename tempType>
    inline void get_local_particle_cell_set_sparse(PullingSchemeSparse& iPullingSchemeSparse,PixelData<tempType> &local_scale_temp, PixelData<tempType> &local_scale_temp2,const APRParameters& par) {
        //  Computes the Local Particle Cell Set from a down-sampled local intensity scale (\sigma) and gradient magnitude
        //  Down-sampled due to the Equivalence Optimization

#ifdef HAVE_LIBTIFF
        if(par.output_steps){
            TiffUtils::saveMeshAsTiff(par.output_dir + "local_particle_set_level_step.tif", local_scale_temp);
        }
#endif

        int l_max = iPullingSchemeSparse.pct_level_max();;
        int l_min = iPullingSchemeSparse.pct_level_min();

        timer.start_timer("pulling_scheme_fill_max_level");

        iPullingSchemeSparse.fill(l_max,local_scale_temp);
        timer.stop_timer();

        timer.start_timer("level_loop_initialize_tree");
        for(int l_ = l_max - 1; l_ >= l_min; l_--){

            //down sample the resolution level k, using a max reduction
            downsample(local_scale_temp, local_scale_temp2,
                       [](const float &x, const float &y) -> float { return std::max(x, y); },
                       [](const float &x) -> float { return x; }, true);
            //for those value of level k, add to the hash table
            iPullingSchemeSparse.fill(l_,local_scale_temp2);
            //assign the previous mesh to now be resampled.
            local_scale_temp.swap(local_scale_temp2);
        }
        timer.stop_timer();

    }

    template<typename ImageType,typename tempType>
    inline void computeLevels(const PixelData<ImageType> &grad_temp, PixelData<tempType> &local_scale_temp, int maxLevel, float relError, float dx, float dy, float dz) {
        //divide gradient magnitude by Local Intensity Scale (first step in calculating the Local Resolution Estimate L(y), minus constants)
        timer.start_timer("compute_level_first");
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for (size_t i = 0; i < grad_temp.mesh.size(); ++i) {
            local_scale_temp.mesh[i] = grad_temp.mesh[i] / local_scale_temp.mesh[i];
        }
        timer.stop_timer();

        float min_dim = std::min(dy, std::min(dx, dz));
        float level_factor = pow(2, maxLevel) * min_dim;

        //incorporate other factors and compute the level of the Particle Cell, effectively construct LPC L_n
        timer.start_timer("compute_level_second");
        compute_level_for_array(local_scale_temp, level_factor, relError);
        timer.stop_timer();
    }



    static inline uint32_t asmlog_2(const uint32_t x) {
        if (x == 0) return 0;
        return (31 - __builtin_clz (x));
    }

    template< typename T>
    void compute_level_for_array(PixelData<T>& input, float k_factor, float rel_error) {
        //
        //  Takes the sqrt of the grad vector to caluclate the magnitude
        //
        //  Bevan Cheeseman 2016

        const float mult_const = k_factor/rel_error;

        #ifdef HAVE_OPENMP
	    #pragma omp parallel for default(shared)
        #endif
        for (size_t i = 0; i < input.mesh.size(); ++i) {
            input.mesh[i] = asmlog_2(input.mesh[i] * mult_const);
        }
    }
};


#endif //PARTPLAY_LOCAL_PARTICLE_SET_HPP
