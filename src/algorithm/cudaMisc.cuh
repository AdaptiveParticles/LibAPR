#ifndef CUDAMISC_CUH
#define CUDAMISC_CUH


#include <type_traits>


/**
 * floating point output -> no rounding or under-/overflow check
 */
template<typename T>
__device__ std::enable_if_t<std::is_floating_point<T>::value, T> round(float val, size_t &errCount) {
    return val;
}

/**
 * integer output -> check for under-/overflow and round
 *
 * CUDA is not supporting std::numeric_limits<T> so this results in belows manual checking of different
 * data types range. In theory we could use --expt-relaxed-constexpr flag but since it is experimental
 * and without guarantee of long existence for now it is better to stick to belows definitions.
 */
template<typename T>
__device__  std::enable_if_t<std::is_same<T, uint8_t>::value, uint8_t> round(float val, size_t &errCount) {
    val = std::round(val);
    if (val < 0 || val > 255) { errCount++; }
    return val;
}

template<typename T>
__device__  std::enable_if_t<std::is_same<T, int8_t>::value, int8_t> round(float val, size_t &errCount) {
    val = std::round(val);
    if (val < -128 || val > 127) { errCount++; }
    return val;
}

template<typename T>
__device__  std::enable_if_t<std::is_same<T, uint16_t>::value, uint16_t> round(float val, size_t &errCount) {
    val = std::round(val);
    if (val < 0 || val > 65535) { errCount++; }
    return val;
}

template<typename T>
__device__  std::enable_if_t<std::is_same<T, int16_t>::value, int16_t> round(float val, size_t &errCount) {
    val = std::round(val);
    if (val < -32768 || val > 32767) { errCount++; }
    return val;
}

template<typename T>
__device__  std::enable_if_t<std::is_same<T, uint32_t>::value, uint32_t> round(float val, size_t &errCount) {
    val = std::round(val);
    if (val < 0 || val > 4294967295) { errCount++; }
    return val;
}

template<typename T>
__device__  std::enable_if_t<std::is_same<T, int32_t>::value, int32_t> round(float val, size_t &errCount) {
    val = std::round(val);
    if (val < -2147483648 || val > 2147483647) { errCount++; }
    return val;
}


#endif
