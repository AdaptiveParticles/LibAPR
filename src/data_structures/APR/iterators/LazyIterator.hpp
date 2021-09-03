//
// Created by joel on 01.09.21.
//

#ifndef APR_LAZYITERATOR_HPP
#define APR_LAZYITERATOR_HPP

#include <data_structures/APR/particles/LazyData.hpp>

#include "GenIterator.hpp"
#include "data_structures/APR/access/LazyAccess.hpp"

struct IndexRange {
    uint64_t begin;
    uint64_t end;
};

class LazyIterator: public GenIterator {

    template<typename T>
    friend class LazyData;

    template<typename T>
    friend class ParticleData;

    uint64_t current_index, begin_index, end_index;

public:

    LazyAccess* lazyAccess;

    LazyIterator() = default;

    LazyIterator(LazyAccess& access) {
        lazyAccess = &access;
        genInfo = access.genInfo;
    }

    uint64_t begin(const int level, const int z, const int x) {
        auto xz_start = lazyAccess->level_xz_vec[level] + z * x_num(level) + x;
        begin_index = lazyAccess->xz_end_vec[xz_start-1];
        end_index = lazyAccess->xz_end_vec[xz_start];
        current_index = begin_index;

        return current_index;
    }

    inline uint64_t end() const { return end_index; }

    inline uint16_t y() const { return lazyAccess->y_vec[current_index]; }

    inline void operator++ (int) { current_index++; }
    inline void operator++ () { current_index++; }
    operator uint64_t() const { return current_index; }

    void set_buffer_size(const uint64_t size) {
        lazyAccess->y_vec.set_buffer_size(size);
    }

    void load_row(const int level, const int z, const int x) {
        auto xz_start = lazyAccess->level_xz_vec[level] + z * x_num(level) + x;
        lazyAccess->y_vec.parts_start = lazyAccess->xz_end_vec[xz_start-1];
        lazyAccess->y_vec.parts_end = lazyAccess->xz_end_vec[xz_start];

        lazyAccess->y_vec.load_current_range();
    }

    void load_slice(const int level, const int z) {
        auto xz_start = lazyAccess->level_xz_vec[level] + z * x_num(level);
        lazyAccess->y_vec.parts_start = lazyAccess->xz_end_vec[xz_start-1];
        lazyAccess->y_vec.parts_end = lazyAccess->xz_end_vec[xz_start + x_num(level) - 1];

        lazyAccess->y_vec.load_current_range();
    }

    inline void load_range(const uint64_t r_begin, const uint64_t r_end) {
        lazyAccess->y_vec.load_range(r_begin, r_end);
    }

    IndexRange get_row_range(const int level, const int z, const int x) {
        auto xz_start = lazyAccess->level_xz_vec[level] + z * x_num(level) + x;
        const uint64_t row_begin = lazyAccess->xz_end_vec[xz_start-1];
        const uint64_t row_end = lazyAccess->xz_end_vec[xz_start];
        return IndexRange{row_begin, row_end};
    }

    IndexRange get_slice_range(const int level, const int z) {
        auto xz_start = lazyAccess->level_xz_vec[level] + z * x_num(level);
        const uint64_t slice_begin = lazyAccess->xz_end_vec[xz_start-1];
        const uint64_t slice_end = lazyAccess->xz_end_vec[xz_start + x_num(level) - 1];
        return IndexRange{slice_begin, slice_end};
    }
};


#endif //APR_LAZYITERATOR_HPP
