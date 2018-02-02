//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_LOCAL_PARTICLE_SET_HPP
#define PARTPLAY_LOCAL_PARTICLE_SET_HPP

#define EMPTY 0

class LocalParticleCellSet {

public:
    static inline uint32_t asmlog_2(const uint32_t x) {
        if (x == 0) return 0;
        return (31 - __builtin_clz (x));
    }

    template< typename T>
    void compute_level_for_array(MeshData<T>& input, float k_factor, float rel_error) {
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
