//
// Created by gonciarz on 8/8/18.
//

#ifndef LIBAPR_CUDAMEMORY_HPP
#define LIBAPR_CUDAMEMORY_HPP

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T>
inline T* getPinnedMemory(size_t aNumOfBytes);

template <typename T>
inline void freePinnedMemory(T *aMemory);

#endif //LIBAPR_CUDAMEMORY_HPP
