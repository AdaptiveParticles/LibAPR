//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_LOCAL_PARTICLE_SET_HPP
#define PARTPLAY_LOCAL_PARTICLE_SET_HPP

#ifdef _MSC_VER
#include <intrin.h>

// from https://github.com/llvm-mirror/libcxx/blob/9dcbb46826fd4d29b1485f25e8986d36019a6dca/include/support/win32/support.h#L106-L182
// (c) Copyright (c) 2009-2017 by the contributors listed in https://github.com/llvm-mirror/libcxx/blob/9dcbb46826fd4d29b1485f25e8986d36019a6dca/CREDITS.TXT
inline int __builtin_ctzll(unsigned long long mask)
{
  unsigned long where;
// Search from LSB to MSB for first set bit.
// Returns zero if no set bit is found.
#if defined(_LIBCPP_HAS_BITSCAN64)
    (defined(_M_AMD64) || defined(__x86_64__))
  if (_BitScanForward64(&where, mask))
    return static_cast<int>(where);
#else
  // Win32 doesn't have _BitScanForward64 so emulate it with two 32 bit calls.
  // Scan the Low Word.
  if (_BitScanForward(&where, static_cast<unsigned long>(mask)))
    return static_cast<int>(where);
  // Scan the High Word.
  if (_BitScanForward(&where, static_cast<unsigned long>(mask >> 32)))
    return static_cast<int>(where + 32); // Create a bit offset from the LSB.
#endif
  return 64;
}
inline int __builtin_ctzl(unsigned long mask)
{
  unsigned long where;
  // Search from LSB to MSB for first set bit.
  // Returns zero if no set bit is found.
  if (_BitScanForward(&where, mask))
    return static_cast<int>(where);
  return 32;
}

inline int __builtin_ctz(unsigned int mask)
{
  // Win32 and Win64 expectations.
  static_assert(sizeof(mask) == 4, "");
  static_assert(sizeof(unsigned long) == 4, "");
  return __builtin_ctzl(static_cast<unsigned long>(mask));
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
