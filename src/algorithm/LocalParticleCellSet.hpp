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
    void compute_level_for_array(MeshData<T>& input,float k_factor,float rel_error){
        //
        //  Takes the sqrt of the grad vector to caluclate the magnitude
        //
        //  Bevan Cheeseman 2016
        //

        float mult_const = k_factor/rel_error;

        const size_t z_num = input.z_num;
        const size_t x_num = input.x_num;
        const size_t y_num = input.y_num;

        #ifdef HAVE_OPENMP
	    #pragma omp parallel for default(shared) if (z_num*x_num*y_num > 100000)
        #endif
        for(size_t j = 0; j < z_num; ++j) {
            for(size_t i = 0;i < x_num; ++i) {

                #ifdef HAVE_OPENMP
	            #pragma omp simd
                #endif
                for (size_t k = 0; k < (y_num); ++k) {
                    input.mesh[j*x_num*y_num + i*y_num + k] = asmlog_2(input.mesh[j*x_num*y_num + i*y_num + k]*mult_const);
                }
            }
        }
    }
};


#endif //PARTPLAY_LOCAL_PARTICLE_SET_HPP
