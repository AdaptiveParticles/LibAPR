//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_LOCAL_PARTICLE_SET_HPP
#define PARTPLAY_LOCAL_PARTICLE_SET_HPP

#ifdef _MSC_VER
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

class LocalParticleCellSet {

public:
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
