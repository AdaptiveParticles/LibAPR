#ifndef FIND_MIN_MAX_CUH
#define FIND_MIN_MAX_CUH

#include <cuda/std/limits>

/**
 * This kernel finds the minimum and maximum values in the input data array.
 * Each block processes a portion of the data and writes the minimum and maximum
 * values it finds to the resultsMin and resultsMax arrays.
 *
 * It requires 2*numOfThreads*sizeof(T) of shared memory
 *
 * @param in - input data
 * @param len - length of input data
 * @param resultsMin - output array for minimum values per block
 * @param resultsMax - output array for maximum values per block
 */
template<typename T>
__global__ void findMinMax(const T *in, const size_t len, T* resultsMin, T* resultsMax) {
    // Compute initial indices
    const int numOfThreads = blockDim.x;
    size_t idx = threadIdx.x;
    size_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Set pointers to shared memory for all needed buffers - use uint64_t to avoid alignment issues so all types
    // used as APR like uint16_t or float etc. are aligned properly
    extern __shared__ uint64_t array[];
    T *minValPerThread = reinterpret_cast<T *>(array);
    T *maxValPerThread = reinterpret_cast<T *>(array) + numOfThreads;

    // Set initial values for min and max
    minValPerThread[idx] = cuda::std::numeric_limits<T>::max();
    maxValPerThread[idx] = cuda::std::numeric_limits<T>::min();

    // Read from global memory and compute min and max
    for (size_t i = globalIdx; i < len; i += gridDim.x * blockDim.x) {
        auto val = in[i];
        if (val < minValPerThread[idx]) minValPerThread[idx] = val;
        if (val > maxValPerThread[idx]) maxValPerThread[idx] = val;
    }

    // Wait for all threads in block to finish
    __syncthreads();

    // First thread should go through the shared memory and find the global min and max
    // All that work is done only by single thread but it is fast enough to keep it simple
    if (idx == 0) {
        T globalMin = minValPerThread[0];
        T globalMax = maxValPerThread[0];
        for (int i = 1; i < numOfThreads; ++i) {
            auto vmin = minValPerThread[i];
            if (vmin < globalMin) globalMin = vmin;
            auto vmax = maxValPerThread[i];
            if (vmax > globalMax) globalMax = vmax;
        }

        // Store results to global memory
        resultsMin[blockIdx.x] = globalMin;
        resultsMax[blockIdx.x] = globalMax;
    }
}

/**
 * This kernel takes the intermediate min and max results from each block and computes the final
 * minimum and maximum values across all blocks. Results are stored in the first element of resultsMin and resultsMax.
 *
 * This kernel requires 2*numOfBlocks*sizeof(T) of shared memory.
 *
 * @param resultsMin - intermediate minimum values from each block
 * @param resultsMax - intermediate maximum values from each block
 * @param numOfBlocks - number of blocks used in 'findMinMax' kenel (size of resultsMin and resultsMax)
 */
template<typename T>
__global__ void findMinMaxFinal(T* resultsMin, T* resultsMax, int numOfBlocks) {

    // Set pointers to shared memory for all needed buffers - use uint64_t to avoid alignment issues so all types
    // used as APR like uint16_t or float etc. are aligned properly
    extern __shared__ uint64_t array2[];
    T *minValPerThread = reinterpret_cast<T *>(array2);
    T *maxValPerThread = reinterpret_cast<T *>(array2) + numOfBlocks;

    size_t idx = threadIdx.x;

    // Read all data with all threads to shared memory
    for (size_t i = idx; i < numOfBlocks; i += blockDim.x) {
        minValPerThread[i] = resultsMin[i];
        maxValPerThread[i] = resultsMax[i];
    }

    // Wait for all threads to finish
    __syncthreads();

    //First thread should go through the shared memory and find the global min and max
    if (idx == 0) {
        T globalMin = minValPerThread[0];
        T globalMax = maxValPerThread[0];
        for (int i = 1; i < numOfBlocks; ++i) {
            auto vmin = minValPerThread[i];
            if (vmin < globalMin) globalMin = vmin;
            auto vmax = maxValPerThread[i];
            if (vmax > globalMax) globalMax = vmax;
        }
        // store results to global memory
        resultsMin[0] = globalMin;
        resultsMax[0] = globalMax;
    }
}


/**
 * Compute min and max values in the cudaInput array.
 *
 * numOfBlocks and numOfThreads are computed outside of this function to allow finding the optimal values (number of SMs)
 * and allocating resultsMin and resultsMax arrays only once and then reuse.
 *
 * @param cudaInput - input data in device memory
 * @param inputDim - dimensions of the input data
 * @param aStream - cuda stream to use
 * @param resultsMin - output array for minimum value, should have numOfBlocks elements
 * @param resultsMax - output array for maximum value, should have numOfBlocks elements
 * @param numOfBlocks - number of blocks to use
 * @param numOfThreads - number of threads per block
 */
template<typename T>
void runFindMinMax(const T *cudaInput, PixelDataDim inputDim, cudaStream_t aStream, T* resultsMin, T* resultsMax, int numOfBlocks, int numOfThreads) {
    const size_t numOfElements = inputDim.size();

    findMinMax<<<numOfBlocks, numOfThreads, 2*numOfThreads*sizeof(T), aStream>>> (cudaInput, numOfElements, resultsMin, resultsMax);
    findMinMaxFinal<<<1, 1024, 2*numOfBlocks*sizeof(T), aStream>>> (resultsMin, resultsMax, numOfBlocks);
}


#endif
