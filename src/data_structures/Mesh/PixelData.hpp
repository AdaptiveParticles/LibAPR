//////////////////////////////////////////////////////////////
//
//
//  ImageGen 2016 Bevan Cheeseman
//                Krzysztof Gonciarz
//  Meshdata class for storing the image/pixel/mesh data
//
//
///////////////////////////////////////////////////////////////

#ifndef PIXEL_DATA_HPP
#define PIXEL_DATA_HPP

#include <vector>
#include <cmath>
#include <memory>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "misc/APRTimer.hpp"

#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#ifdef APR_USE_CUDA
#include "misc/CudaMemory.cuh"
#endif

struct PixelDataDim {
    size_t y;
    size_t x;
    size_t z;

    constexpr PixelDataDim(size_t y, size_t x, size_t z) : y(y), x(x), z(z) {}

    size_t size() const { return y * x * z; }
    size_t maxDimSize() const { return std::max(x, std::max(y, z)); }
    int numOfDimensions() const { return (int)(x > 1) + (int)(y > 1) + (int)(z > 1); }

    PixelDataDim operator+(const PixelDataDim &rhs) const { return {y + rhs.y, x + rhs.x, z + rhs.z}; }
    PixelDataDim operator-(const PixelDataDim &rhs) const { return {y - rhs.y, x - rhs.x, z - rhs.z}; }

    template<typename INTEGER>
    constexpr typename std::enable_if<std::is_integral<INTEGER>::value, PixelDataDim>::type
    operator+(INTEGER i) const { return {y + i, x + i, z + i}; }

    template<typename INTEGER>
    constexpr typename std::enable_if<std::is_integral<INTEGER>::value, PixelDataDim>::type
    operator-(INTEGER i) const { return *this + (-i); }

    friend bool operator==(const PixelDataDim& lhs, const PixelDataDim& rhs) {
        return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
    }
    friend bool operator!=(const PixelDataDim& lhs, const PixelDataDim& rhs) {
        return !(lhs == rhs);
    }

    friend std::ostream &operator<<(std::ostream &os, const PixelDataDim &obj) {
        os << "{" << obj.y << ", " << obj.x << ", " << obj.z << "}";
        return os;
    }
};

template <typename T>
class ArrayWrapper
{
public:
    ArrayWrapper() : iArray(nullptr), iNumOfElements(0), iNumOfAllocatedElements(0) {}
    ArrayWrapper(ArrayWrapper &&aObj) {
        iArray = aObj.iArray; aObj.iArray = nullptr;
        iNumOfElements = aObj.iNumOfElements; aObj.iNumOfElements = 0;
        iNumOfAllocatedElements = aObj.iNumOfAllocatedElements; aObj.iNumOfAllocatedElements = 0;
    }
    ArrayWrapper& operator=(ArrayWrapper&& aObj) {
        iArray = aObj.iArray; aObj.iArray = nullptr;
        iNumOfElements = aObj.iNumOfElements; aObj.iNumOfElements = 0;
        iNumOfAllocatedElements = aObj.iNumOfAllocatedElements; aObj.iNumOfAllocatedElements = 0;
        return *this;
    }

    inline void set(T *aInputArray, size_t aNumOfElements) {iArray = aInputArray; iNumOfElements = aNumOfElements;iNumOfAllocatedElements = aNumOfElements;}

    inline void resize(size_t aNumOfElements) {iNumOfElements = aNumOfElements;} //no memory management

    inline T* begin() { return (iArray); }
    inline T* end() { return (iArray + iNumOfElements); }
    inline const T* begin() const { return (iArray); }
    inline const T* end() const { return (iArray + iNumOfElements); }


    inline T& operator[](size_t idx) { return iArray[idx]; }
    inline const T& operator[](size_t idx) const { return iArray[idx]; }
    inline size_t size() const { return iNumOfElements; }
    inline size_t capacity() const { return iNumOfAllocatedElements; }

    inline T* get() {return iArray;}
    inline const T* get() const {return iArray;}

    inline void swap(ArrayWrapper<T> &aObj) {
        std::swap(iNumOfAllocatedElements, aObj.iNumOfAllocatedElements);
        std::swap(iNumOfElements, aObj.iNumOfElements);
        std::swap(iArray, aObj.iArray);
    }

    friend std::ostream & operator<<(std::ostream &os, const ArrayWrapper<T> &obj) {
        os << "ArrayWrapper: size:" << obj.size() << " elementSize:" << sizeof(T) << " sizeInBytes:" << obj.size() * sizeof(T);
        return os;
    }

private:
    ArrayWrapper(const ArrayWrapper&) = delete; // make it noncopyable
    ArrayWrapper& operator=(const ArrayWrapper&) = delete; // make it not assignable

    T *iArray;
    size_t iNumOfElements;
    size_t iNumOfAllocatedElements;
};


/**
 * Provides implementation for 1D vector with elements of given type. (interface is similar to std::vector)
 * @tparam T type of mesh elements
 */
template <typename T>
class VectorData {
public :
    using value_type = T;

    /**
     * Constructor -initialize empty
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    VectorData() = default;

    ~VectorData() = default;

    /**
     * Constructor - initialize empty using pinned memory
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    VectorData(bool usePinned){
        usePinnedMemory = usePinned;
    }

    void setUsePinnedMemory(bool usePinned){
        usePinnedMemory = usePinned;
    }

    inline uint64_t size() const{
        return vec.size();
    }

    inline T* begin(){
        return vec.begin();
    }

    inline T* end(){
        return vec.end();
    }

    inline const T* begin() const{
        return vec.begin();
    }

    inline const T* end() const{
        return vec.end();
    }

    inline T* data(){
        return vec.begin();
    }

    inline const T* data() const{
        return vec.begin();
    }

    inline T& back(){
        return vec[vec.size()-1];
    }

    /**
     * Resizes the array if there is not enough capacity. Otherwise just changes the size variable.
     * @param size
     */
    inline void resize(uint64_t size){
        if(size <= vec.capacity()){
            vec.resize(size);
        } else{
            init(size);
        }
    }

    /**
     * Resizes the array if there is not enough capacity. Otherwise just changes the size variable.
     * @param size
     * @param T aInitval fills the resized array with a given value.
     */
    inline void resize(uint64_t size,T aInitVal){
        resize(size);
        fill(aInitVal);
    }

    /**
    * initializes the array regardless if there was previosly memory allocated.
    * @param size
    */
    inline void init(uint64_t size){
        T *array = nullptr;
        if (!usePinnedMemory) {
            vecMemory.reset(new T[size]);
            array = vecMemory.get();
        }
        else {
#ifndef APR_USE_CUDA
            vecMemory.reset(new T[size]);
            array = vecMemory.get();
#else
            vecMemoryPinned.reset((T*)getPinnedMemory(size * sizeof(T)));
            array = vecMemoryPinned.get();
#endif
        }

        vec.set(array, size);
    }

    inline T& operator[](size_t index){ return vec[index]; };
    inline const T& operator[](size_t index) const { return vec[index]; };

    /**
    * Fills the array with a given value
    * @param T aInitval fills the resized array with a given value.
    */
    void fill(T aInitVal) {
        // Fill values of new buffer in parallel

#ifdef HAVE_OPENMP
#pragma omp parallel
        {
            auto threadNum = omp_get_thread_num();
            auto numOfThreads = omp_get_num_threads();
            auto chunkSize = size() / numOfThreads;
            auto begin_ = begin() + chunkSize * threadNum;
            auto end_ = (threadNum == numOfThreads - 1) ? begin() + size() : begin_ + chunkSize;
            std::fill(begin_, end_, aInitVal);
        }
#else
        std::fill(begin(),end(), aInitVal);
#endif
    }

    template <typename CopyType>
    void copy(const VectorData<CopyType>& ToCopy){
        resize(ToCopy.size());

#ifdef HAVE_OPENMP
#pragma omp parallel
        {
            auto threadNum = omp_get_thread_num();
            auto numOfThreads = omp_get_num_threads();
            auto chunkSize = size() / numOfThreads;
            auto begin_ = ToCopy.begin() + chunkSize * threadNum;
            auto end_ = (threadNum == numOfThreads - 1) ? ToCopy.begin() + size() : begin_ + chunkSize;
            auto out_ = begin() + chunkSize * threadNum;
            std::copy(begin_, end_, out_);
        }
#else
        std::copy(ToCopy.begin(),ToCopy.end(),begin());
#endif
    }

    /**
     * Swaps data of VectorData this <-> aObj
     * @param aObj
     */
    void swap(VectorData &aObj) {
        std::swap(usePinnedMemory, aObj.usePinnedMemory);
        std::swap(vecMemory, aObj.vecMemory);
        vec.swap(aObj.vec);
    }


    /**
     * Apply unary operator to each element in parallel, writing the result to VectorData 'output'.
     * @tparam S
     * @tparam UnaryOperator
     * @param output            output VectorData (can be the 'this')
     * @param op                operator to apply
     */
    template<typename S, typename UnaryOperator>
    void map(VectorData<S>& output, UnaryOperator op) {
        output.resize(size());

#ifdef HAVE_OPENMP
#pragma omp parallel
        {
            auto threadNum = omp_get_thread_num();
            auto numOfThreads = omp_get_num_threads();
            auto chunkSize = size() / numOfThreads;
            auto begin_ = begin() + chunkSize * threadNum;
            auto end_ = (threadNum == numOfThreads - 1) ? begin() + size() : begin_ + chunkSize;
            auto out_ = output.begin() + chunkSize * threadNum;
            std::transform(begin_, end_, out_, op);
        }
#else
        std::transform(begin(), end(), output.begin(), op);
#endif
    }

    template<typename S, typename UnaryOperator>
    void map(VectorData<S>& output, UnaryOperator op) const {
        output.resize(size());

#ifdef HAVE_OPENMP
#pragma omp parallel
        {
            auto threadNum = omp_get_thread_num();
            auto numOfThreads = omp_get_num_threads();
            auto chunkSize = size() / numOfThreads;
            auto begin_ = begin() + chunkSize * threadNum;
            auto end_ = (threadNum == numOfThreads - 1) ? begin() + size() : begin_ + chunkSize;
            auto out_ = output.begin() + chunkSize * threadNum;
            std::transform(begin_, end_, out_, op);
        }
#else
        std::transform(begin(), end(), output.begin(), op);
#endif
    }


    /**
     * Apply binary operator to each element of 'this' and VectorData 'parts2', writing the result to 'output'
     * @tparam S
     * @tparam U
     * @tparam BinaryOperator
     * @param parts2            another VectorData
     * @param output            output VectorData (can be 'this' or 'parts2')
     * @param op                binary operator
     */
    template<typename S, typename U, typename BinaryOperator>
    void zip(const VectorData<S>& parts2, VectorData<U>& output, BinaryOperator op) {
        output.resize(size());
#ifdef HAVE_OPENMP
#pragma omp parallel
        {
            auto threadNum = omp_get_thread_num();
            auto numOfThreads = omp_get_num_threads();
            auto chunkSize = size() / numOfThreads;
            auto begin1_ = begin() + chunkSize * threadNum;
            auto end1_ = (threadNum == numOfThreads - 1) ? begin() + size() : begin1_ + chunkSize;
            auto begin2_ = parts2.begin() + chunkSize * threadNum;
            auto out_ = output.begin() + chunkSize * threadNum;
            std::transform(begin1_, end1_, begin2_, out_, op);
        }
#else
        std::transform(begin(), end(), parts2.begin(), output.begin(), op);
#endif
    }

    template<typename S, typename U, typename BinaryOperator>
    void zip(const VectorData<S>& parts2, VectorData<U>& output, BinaryOperator op) const {
        output.resize(size());
#ifdef HAVE_OPENMP
#pragma omp parallel
        {
            auto threadNum = omp_get_thread_num();
            auto numOfThreads = omp_get_num_threads();
            auto chunkSize = size() / numOfThreads;
            auto begin1_ = begin() + chunkSize * threadNum;
            auto end1_ = (threadNum == numOfThreads - 1) ? begin() + size() : begin1_ + chunkSize;
            auto begin2_ = parts2.begin() + chunkSize * threadNum;
            auto out_ = output.begin() + chunkSize * threadNum;
            std::transform(begin1_, end1_, begin2_, out_, op);
        }
#else
        std::transform(begin(), end(), parts2.begin(), output.begin(), op);
#endif
    }


private:

    bool usePinnedMemory = false;

    std::unique_ptr<T[]> vecMemory;
    ArrayWrapper<T> vec;

#ifdef APR_USE_CUDA
    PinnedMemoryUniquePtr<T> vecMemoryPinned;
#endif
};

/**
 * Provides implementation for 3D mesh with elements of given type.
 * @tparam T type of mesh elements
 */
template <typename T>
class PixelData {
public :
    using value_type = T;

    // size of mesh and container for data
    int y_num;
    int x_num;
    int z_num;
    std::unique_ptr<T[]> meshMemory;
#ifdef APR_USE_CUDA
    PinnedMemoryUniquePtr<T> meshMemoryPinned;
#endif
    ArrayWrapper<T> mesh;
    
    uint64_t size() { return (uint64_t) x_num * y_num * z_num * sizeof(T); }
    /**
     * Constructor - initialize mesh with size of 0,0,0
     */
    PixelData() { init(0, 0, 0); }

    /**
     * Constructor - initialize initial size of mesh to provided values
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    PixelData(int aSizeOfY, int aSizeOfX, int aSizeOfZ) { init(aSizeOfY, aSizeOfX, aSizeOfZ); }

    /**
     * Constructor - creates mesh with provided dimentions initialized to aInitVal
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     * @param aInitVal - initial value of all elements
     */
    PixelData(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T aInitVal) { initWithValue(aSizeOfY, aSizeOfX, aSizeOfZ, aInitVal); }

    /**
     * Constructor - initialize initial size of mesh to provided values
     * @param aDims - PixelDataDim with length of each dimension
     */
    PixelData(PixelDataDim aDims) { init(aDims.y, aDims.x, aDims.z); }

    /**
     * Constructor - creates mesh with provided dimentions initialized to aInitVal
     * @param aDims - PixelDataDim with length of each dimension
     * @param aInitVal - initial value of all elements
     */
    PixelData(PixelDataDim aDims, T aInitVal) { initWithValue(aDims.y, aDims.x, aDims.z, aInitVal); }

    /**
     * Move constructor
     * @param aObj mesh to be moved
     */
    PixelData(PixelData &&aObj) {
        x_num = aObj.x_num;
        y_num = aObj.y_num;
        z_num = aObj.z_num;
        mesh = std::move(aObj.mesh);
        meshMemory = std::move(aObj.meshMemory);
    }

    /**
     * Move assignment operator
    * @param aObj
    */
    PixelData& operator=(PixelData &&aObj) {
        x_num = aObj.x_num;
        y_num = aObj.y_num;
        z_num = aObj.z_num;
        mesh = std::move(aObj.mesh);
        meshMemory = std::move(aObj.meshMemory);
        return *this;
    }

    /**
     * Constructor - initialize mesh with other mesh (data are copied and casted if needed).
     * @param aMesh input mesh
     */
    template<typename U>
    PixelData(const PixelData<U> &aMesh, bool aShouldCopyData, bool aUsedPinnedMemory = false) {
        init(aMesh.y_num, aMesh.x_num, aMesh.z_num, aUsedPinnedMemory);
        if (aShouldCopyData) std::copy(aMesh.mesh.begin(), aMesh.mesh.end(), mesh.begin());
    }

    /**
     * Returns dimensions of PixelData
     */
    PixelDataDim getDimension() const {
        return {static_cast<size_t>(y_num), static_cast<size_t>(x_num), static_cast<size_t>(z_num)};
    }

    /**
     * Creates copy of this mesh converting each element to new type
     * @tparam U new type of mesh
     * @return created object by value
     */
    template <typename U>
    PixelData<U> toType() const {
        PixelData<U> new_value(y_num, x_num, z_num);
        std::copy(mesh.begin(), mesh.end(), new_value.mesh.begin());
        return new_value;
    }

    /**
     * access element at provided indices with boundary checking
     * @param y
     * @param x
     * @param z
     * @return element @(y, x, z)
     */
    T& operator()(int y, int x, int z) {
        y = std::min(y, y_num-1);
        x = std::min(x, x_num-1);
        z = std::min(z, z_num-1);

        y = std::max(y, 0);
        x = std::max(x, 0);
        z = std::max(z, 0);

        size_t idx = (size_t) z * x_num * y_num + x * y_num + y;
        return mesh[idx];
    }

    /**
     * access element at provided indices without boundary checking
     * @param y
     * @param x
     * @param z
     * @return element @(y, x, z)
     */
    T& at(size_t y, size_t x, size_t z) {
        size_t idx = z * x_num * y_num + x * y_num + y;
        return mesh[idx];
    }

    /**
     * access element at provided indices without boundary checking
     * @param y
     * @param x
     * @param z
     * @return element @(y, x, z)
     */
    const T& at(size_t y, size_t x, size_t z) const {
        size_t idx = z * x_num * y_num + x * y_num + y;
        return mesh[idx];
    }

    /**
     * Copies data from aInputMesh utilizing parallel copy, requires prior initialization
     * of 'this' object (size and number of elements)
     * @tparam U type of data
     * @param aInputMesh input mesh with data
     * @param aNumberOfBlocks in how many chunks copy will be done
     */
    template<typename U>
    void copyFromMesh(const PixelData<U> &aInputMesh, unsigned int aNumberOfBlocks = 8) {
        aNumberOfBlocks = std::min((unsigned int)z_num, aNumberOfBlocks);
        unsigned int numOfElementsPerBlock = z_num/aNumberOfBlocks;

        #ifdef HAVE_OPENMP
	    #pragma omp parallel for schedule(static)
        #endif
        for (unsigned int blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
            const size_t elementSize = (size_t)x_num * y_num;
            const size_t blockSize = numOfElementsPerBlock * elementSize;
            size_t offsetBegin = blockNum * blockSize;
            size_t offsetEnd = offsetBegin + blockSize;
            if (blockNum == aNumberOfBlocks - 1) {
                // Handle tailing elements if number of blocks does not divide.
                offsetEnd = z_num * elementSize;
            }

            std::copy(aInputMesh.mesh.begin() + offsetBegin, aInputMesh.mesh.begin() + offsetEnd, mesh.begin() + offsetBegin);
        }
    }

    /**
     * Initilize mesh with provided dimensions and initial value
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     * @param aInitVal
     * NOTE: If mesh was already created only added elements (new size > old size) will be initialize with aInitVal
     */
    void initWithValue(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T aInitVal, bool aUsePinnedMemory = false) {
        y_num = aSizeOfY;
        x_num = aSizeOfX;
        z_num = aSizeOfZ;
        size_t size = (size_t)y_num * x_num * z_num;

        T *array = nullptr;
        if (!aUsePinnedMemory) {
            meshMemory.reset(new T[size]);
            array = meshMemory.get();
        }
        else {
        #ifndef APR_USE_CUDA
            meshMemory.reset(new T[size]);
            array = meshMemory.get();
        #else
            meshMemoryPinned.reset((T*)getPinnedMemory(size * sizeof(T)));
            array = meshMemoryPinned.get();
        #endif
        }

        mesh.set(array, size);

        fill(aInitVal);
    }

    void fill(T aInitVal) {
        // Fill values of new buffer in parallel

        size_t size = (size_t)y_num * x_num * z_num;
        auto array = mesh.begin();
#ifdef HAVE_OPENMP
#pragma omp parallel
        {
            auto threadNum = omp_get_thread_num();
            auto numOfThreads = omp_get_num_threads();
            auto chunkSize = size / numOfThreads;
            auto begin = array + chunkSize * threadNum;
            auto end = (threadNum == numOfThreads - 1) ? array + size : begin + chunkSize;
            std::fill(begin, end, aInitVal);
        }
#else
        std::fill(array, array + size, aInitVal);
#endif
    }


    /**
     * Initialize with provided mesh without copying.
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     * @param aArray pointer to data
     */
    void init_from_mesh(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T* aArray) {
        y_num = aSizeOfY;
        x_num = aSizeOfX;
        z_num = aSizeOfZ;
        size_t size = (size_t)y_num * x_num * z_num;

        //TODO: fix this for python wrappers?
        //meshMemory.reset(aArray);

        mesh.set(aArray, size);
    }


    /**
     * Initializes mesh with provided dimensions with default value of used type
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    void init(int aSizeOfY, int aSizeOfX, int aSizeOfZ, bool aUsePinnedMemory = false) {
        y_num = aSizeOfY;
        x_num = aSizeOfX;
        z_num = aSizeOfZ;
        size_t size = (size_t)y_num * x_num * z_num;

        T *array = nullptr;
        if (!aUsePinnedMemory) {
            meshMemory.reset(new T[size]);
            array = meshMemory.get();
        }
        else {
#ifndef APR_USE_CUDA
            meshMemory.reset(new T[size]);
            array = meshMemory.get();
#else
            meshMemoryPinned.reset((T*)getPinnedMemory(size * sizeof(T)));
            array = meshMemoryPinned.get();
#endif
        }
        mesh.set(array, size);
    }

    /**
     * Initializes mesh with provided dimensions with default value of used type, only allocating memory if more is needed then currently, otherwise just changes the dims.
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    void initWithResize(int aSizeOfY, int aSizeOfX, int aSizeOfZ, bool aUsePinnedMemory = false) {
        y_num = aSizeOfY;
        x_num = aSizeOfX;
        z_num = aSizeOfZ;
        size_t size = (size_t)y_num * x_num * z_num;

        if(size <= mesh.capacity()){
            mesh.resize(size);
        } else{
            init(aSizeOfY,aSizeOfX,aSizeOfZ,aUsePinnedMemory);
        }
    }

    /**
     * Initialize mesh with dimensions taken from provided mesh
     * @tparam S
     * @param aInputMesh
     */
    template<typename S>
    void init(const PixelData<S> &aInputMesh) {
        init(aInputMesh.y_num, aInputMesh.x_num, aInputMesh.z_num);
    }

    /**
     * Initializes mesh with size of half of provided dimensions (rounding up if not divisible by 2)
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    void initDownsampled(int aSizeOfY, int aSizeOfX, int aSizeOfZ, bool aUsePinnedMemory) {
        const int z_num_ds = ceil(1.0*aSizeOfZ/2.0);
        const int x_num_ds = ceil(1.0*aSizeOfX/2.0);
        const int y_num_ds = ceil(1.0*aSizeOfY/2.0);

        init(y_num_ds, x_num_ds, z_num_ds, aUsePinnedMemory);
    }

    /**
     * Initializes mesh with size of half of provided dimensions (rounding up if not divisible by 2) and initialize values
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     * @param aInitVal
     */
    void initDownsampled(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T aInitVal, bool aUsePinnedMemory) {
        const int z_num_ds = ceil(1.0*aSizeOfZ/2.0);
        const int x_num_ds = ceil(1.0*aSizeOfX/2.0);
        const int y_num_ds = ceil(1.0*aSizeOfY/2.0);

        initWithValue(y_num_ds, x_num_ds, z_num_ds, aInitVal, aUsePinnedMemory);
    }

    /**
     * Initializes mesh with size of half of provided mesh dimensions (rounding up if not divisible by 2)
     * @param aMesh - mesh used to get dimensions
     */
    template <typename U>
    void initDownsampled(const PixelData<U> &aMesh) {
        const int z_num_ds = ceil(1.0*aMesh.z_num/2.0);
        const int x_num_ds = ceil(1.0*aMesh.x_num/2.0);
        const int y_num_ds = ceil(1.0*aMesh.y_num/2.0);

        init(y_num_ds, x_num_ds, z_num_ds);
    }

    /**
     * Initializes mesh with size of half of provided mesh dimensions (rounding up if not divisible by 2) and initialize values
     * @param aMesh - mesh used to get dimensions
     * @param aInitVal
     */
    template <typename U>
    void initDownsampled(const PixelData<U> &aMesh, T aInitVal) {
        const int z_num_ds = ceil(1.0*aMesh.z_num/2.0);
        const int x_num_ds = ceil(1.0*aMesh.x_num/2.0);
        const int y_num_ds = ceil(1.0*aMesh.y_num/2.0);

        initWithValue(y_num_ds, x_num_ds, z_num_ds, aInitVal);
    }

    /**
     * Swaps data of meshes this <-> aObj
     * @param aObj
     */
    void swap(PixelData &aObj) {
        std::swap(x_num, aObj.x_num);
        std::swap(y_num, aObj.y_num);
        std::swap(z_num, aObj.z_num);
        meshMemory.swap(aObj.meshMemory);
        mesh.swap(aObj.mesh);
    }


    /**
     * Normalize the elements of the mesh to sum to unity. Does nothing if data type T is not floating point.
     */
    void normalize() {
        if(std::is_floating_point<T>::value) {
            float sum = std::accumulate(mesh.begin(), mesh.end(), 0.0f);
            if(std::abs(sum) > 1e-6 && sum != 1.0f) {
                std::transform(mesh.begin(), mesh.end(), mesh.begin(), [sum](T &a) { return a / sum; });
            }
        }
    }

    /**
     * Initialize in parallel 'this' mesh with aInputMesh elements modified by provided unary operation
     * NOTE: this mesh must be big enough to contain all elements from aInputMesh
     * @tparam U - type of data
     * @param aInputMesh - source data
     * @param aOp - function/lambda modifing each element of aInputMesh: [](const int &a) { return a + 5; }
     * @param aNumberOfBlocks - in how many chunks copy will be done
     */
    template<typename U, typename R>
    void copyFromMeshWithUnaryOp(const PixelData<U> &aInputMesh, R aOp, int aNumberOfBlocks = 10) {
        aNumberOfBlocks = std::min(aInputMesh.z_num, aNumberOfBlocks);
        size_t numOfElementsPerBlock = aInputMesh.z_num/aNumberOfBlocks;

        #ifdef HAVE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
            const size_t elementSize = (size_t)aInputMesh.x_num * aInputMesh.y_num;
            const size_t blockSize = numOfElementsPerBlock * elementSize;
            size_t offsetBegin = blockNum * blockSize;
            size_t offsetEnd = offsetBegin + blockSize;
            if (blockNum == aNumberOfBlocks - 1) {
                // Handle tailing elements if number of blocks does not divide.
                offsetEnd = aInputMesh.z_num * elementSize;
            }

            std::transform(aInputMesh.mesh.begin() + offsetBegin,
                           aInputMesh.mesh.begin() + offsetEnd,
                           mesh.begin() + offsetBegin,
                           aOp);
        }
    }

    /**
     * Returns string with (y, x, z) coordinates for given index (for debug purposes)
     * @param aIdx
     * @return
     */
    std::string getStrIndex(size_t aIdx) const {
        if (aIdx >= mesh.size()) return "(ErrIdx)";
        size_t z = aIdx / ((size_t) x_num * y_num);
        aIdx -= z * ((size_t) x_num * y_num);
        size_t x = aIdx / y_num;
        aIdx -= x * y_num;
        size_t y = aIdx;
        std::ostringstream outputStr;
        outputStr << "(" << y << ", " << x << ", " << z << ")";
        return outputStr.str();
    }

    /**
     * Prints X-Y or X-Z planes of mesh (for debug/test purposses - use only on small meshes)
     */
    void printMesh(int aColumnWidth, int aFloatPrecision = 10, bool aXYplanes = true) const {
        std::ios::fmtflags flagsBefore(std::cout.flags());
        std::cout << std::setw(aColumnWidth) << std::setprecision(aFloatPrecision) << std::fixed;

        if (aXYplanes) {
            for (int z = 0; z < z_num; ++z) {
                std::cout << "z=" << z << "\n";
                for (int y = 0; y < y_num; ++y) {
                    for (int x = 0; x < x_num; ++x) {
                        std::cout << std::setw(aColumnWidth) << std::setprecision(aFloatPrecision) << std::fixed << (sizeof(T) == 1 ? (int)at(y, x, z) : at(y, x, z)) << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << std::endl;
            }
        }
        else { // X-Z planes
            for (int y = 0; y < y_num; ++y) {
                std::cout << "y=" << y << "\n";
                for (int z = 0; z < z_num; ++z) {
                    for (int x = 0; x < x_num; ++x) {
                        std::cout << std::setw(aColumnWidth) << std::setprecision(aFloatPrecision) << std::fixed << (sizeof(T) == 1 ? (int)at(y, x, z) : at(y, x, z)) << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << std::endl;
            }
        }

        // Revert settings
        std::cout.flags(flagsBefore);
    }

    /**
 * Prints X-Y or X-Z planes of mesh (for debug/test purposses - use only on small meshes)
 */
    void printMeshT(int aColumnWidth, int aFloatPrecision = 10, bool aXYplanes = true) const {
        std::ios::fmtflags flagsBefore(std::cout.flags());
        std::cout << std::setw(aColumnWidth) << std::setprecision(aFloatPrecision) << std::fixed;

        if (aXYplanes) {
            for (int z = 0; z < z_num; ++z) {
                std::cout << "z=" << z << "\n";
                for (int x = 0; x < x_num; ++x) {
                    for (int y = 0; y < y_num; ++y) {
                        std::cout << std::setw(aColumnWidth) << std::setprecision(aFloatPrecision) << std::fixed << (sizeof(T) == 1 ? (int)at(y, x, z) : at(y, x, z)) << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << std::endl;
            }
        }
        else { // X-Z planes
            for (int y = 0; y < y_num; ++y) {
                std::cout << "y=" << y << "\n";
                for (int x = 0; x < x_num; ++x) {
                    for (int z = 0; z < z_num; ++z) {
                        std::cout << std::setw(aColumnWidth) << std::setprecision(aFloatPrecision) << std::fixed << (sizeof(T) == 1 ? (int)at(y, x, z) : at(y, x, z)) << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << std::endl;
            }
        }

        // Revert settings
        std::cout.flags(flagsBefore);
    }

    friend std::ostream & operator<<(std::ostream &os, const PixelData<T> &obj) {
        os << "PixelData: size(Y/X/Z)=" << obj.y_num << "/" << obj.x_num << "/" << obj.z_num << " " << obj.mesh << " ";
        return os;
    }

private:

    PixelData(const PixelData&) = delete; // make it noncopyable
    PixelData& operator=(const PixelData&) = delete; // make it not assignable

};


template<typename T, typename S, typename R, typename C>
void downsample(const PixelData<T> &aInput, PixelData<S> &aOutput, R reduce, C constant_operator, bool aInitializeOutput = false) {
    const size_t z_num = aInput.z_num;
    const size_t x_num = aInput.x_num;
    const size_t y_num = aInput.y_num;

    // downsampled dimensions twice smaller (rounded up)
    const size_t z_num_ds = ceil(z_num/2.0);
    const size_t x_num_ds = ceil(x_num/2.0);
    const size_t y_num_ds = ceil(y_num/2.0);

    APRTimer timer;
    timer.verbose_flag = false;

    if (aInitializeOutput) {
        timer.start_timer("downsample_initalize");
        aOutput.initWithResize(y_num_ds, x_num_ds, z_num_ds);
        timer.stop_timer();
    }

    timer.start_timer("downsample_loop");
    #ifdef HAVE_OPENMP
    #pragma omp parallel for default(shared)
    #endif
    for (size_t z = 0; z < z_num_ds; ++z) {
        for (size_t x = 0; x < x_num_ds; ++x) {

            // shifted +1 in original inMesh space
            const size_t shx = std::min(2*x + 1, x_num - 1);
            const size_t shz = std::min(2*z + 1, z_num - 1);

            const ArrayWrapper<T> &inMesh = aInput.mesh;
            ArrayWrapper<S> &outMesh = aOutput.mesh;

            for (size_t y = 0; y < y_num_ds; ++y) {
                const size_t shy = std::min(2*y + 1, y_num - 1);
                const size_t idx = z * x_num_ds * y_num_ds + x * y_num_ds + y;
                outMesh[idx] =  constant_operator(
                        reduce(reduce(reduce(reduce(                             // inMesh coordinates
                               inMesh[2*z * x_num * y_num + 2*x * y_num + 2*y],  // z,   x,   y
                               inMesh[2*z * x_num * y_num + shx * y_num + 2*y]), // z,   x+1, y
                               inMesh[shz * x_num * y_num + 2*x * y_num + 2*y]), // z+1, x,   y
                               inMesh[shz * x_num * y_num + shx * y_num + 2*y]), // z+1, x+1, y
                               reduce(reduce(reduce(
                               inMesh[2*z * x_num * y_num + 2*x * y_num + shy],  // z,   x,   y+1
                               inMesh[2*z * x_num * y_num + shx * y_num + shy]), // z,   x+1, y+1
                               inMesh[shz * x_num * y_num + 2*x * y_num + shy]), // z+1, x,   y+1
                               inMesh[shz * x_num * y_num + shx * y_num + shy])) // z+1, x+1, y+1
                );
            }
        }
    }
    timer.stop_timer();
}

template<typename T>
void const_upsample_img(PixelData<T>& input_us, PixelData<T>& input){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Creates a constant upsampling of an image
    //
    //

    APRTimer timer;

    timer.verbose_flag = false;

    //restrict the domain to be only as big as possibly needed

    const size_t z_num_ds = input.z_num;
    const size_t x_num_ds = input.x_num;
    const size_t y_num_ds = input.y_num;

    const size_t z_num = input_us.z_num;
    const size_t x_num = input_us.x_num;
    const size_t y_num = input_us.y_num;

    timer.start_timer("resize");

    timer.stop_timer();

    std::vector<T> temp_vec;
    temp_vec.resize(y_num_ds,0);

    timer.start_timer("up_sample_const");

    size_t j, i, k;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(j,i,k) firstprivate(temp_vec) if(z_num_ds*x_num_ds > 100)
#endif
    for(j = 0;j < z_num_ds;j++){
        for(i = 0;i < x_num_ds;i++){

            //first take into cache
            for (k = 0; k < y_num_ds;k++){
                temp_vec[k] = input.mesh[j*x_num_ds*y_num_ds + i*y_num_ds + k];
            }


            for (size_t z = 2*j; z <= std::min(2*j+1, z_num-1); ++z) {
                for (size_t x = 2*i; x <= std::min(2*i+1, x_num-1); ++x) {
                    for (size_t y = 0; y < y_num; ++y) {

                        input_us.mesh[z*x_num*y_num + x*y_num + y] = temp_vec[y/2];

                    }

                }

            }


        }
    }

    timer.stop_timer();
}


template<typename T, typename BinaryOperator, typename UnaryOperator>
void downsamplePyramid(PixelData<T> &original_image,
                       std::vector<PixelData<T>> &downsampled,
                       size_t l_max,
                       size_t l_min,
                       BinaryOperator reduction_operator,
                       UnaryOperator constant_operator) {
    downsampled.resize(l_max + 1); // each level is kept at same index
    downsampled.back().swap(original_image); // put original image at l_max index

    for (size_t level = l_max; level > l_min; --level) {
        downsample(downsampled[level], downsampled[level - 1], reduction_operator, constant_operator, true);
    }
}


template<typename T>
void downsamplePyramid(PixelData<T> &original_image, std::vector<PixelData<T>> &downsampled, size_t l_max, size_t l_min) {
    auto sum = [](const float x, const float y) -> float { return x + y; };
    auto divide_by_8 = [](const float x) -> float { return x/8.0f; };

    downsamplePyramid(original_image, downsampled, l_max, l_min, sum, divide_by_8);
}


/**
 * Padds an array performing reflection, first y,x,z - reflecting around the edge pixel.
 * @tparam T - type of data
 * @param input - source data
 * @param output - padded image
 * @param sz_y - desired padding size, will be bounded by y_num - 1
 * @param sz_x- desired padding size, will be bounded by x_num - 1
 * @param sz_z - desired padding size, will be bounded by z_num - 1
*/
template<typename T>
void paddPixels(PixelData<T> &input, PixelData<T> &output, int sz_y, int sz_x, int sz_z){

    if(input.y_num > 1){
        sz_y = std::min(sz_y, input.y_num-1);
    } else {
        sz_y = 0;
    }

    if(input.x_num > 1){
        sz_x =  std::min(sz_x, input.x_num-1);
    } else {
        sz_x = 0;
    }

    if(input.z_num > 1){
        sz_z =  std::min(sz_z, input.z_num-1);
    } else {
        sz_z = 0;
    }

    output.initWithResize(input.y_num + 2*sz_y,input.x_num + 2*sz_x,input.z_num + 2*sz_z);

    //copy across internal

    int j = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(j)
#endif
    for (j = 0; j < (int) input.z_num; ++j) {
        for (int i = 0; i < (int) input.x_num; ++i) {
            for (int k = 0; k <  (int) input.y_num; ++k) {
                output.at(k+sz_y,i+sz_x,j+sz_z)=input.at(k,i,j);
            }
        }
    }

    if(input.y_num > 1) {
        //reflect y
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(j)
#endif
        for (j = 0; j < output.z_num; ++j) {
            for (int i = 0; i < output.x_num; ++i) {

                for (int k = 0; k < sz_y; ++k) {
                    output.at(k, i, j) = output.at(2 * sz_y - k, i, j);
                }

                int idx = sz_y+2;
                for (int k = output.y_num - sz_y; k < output.y_num; ++k) {

                    output.at(k, i, j) = output.at(output.y_num - idx, i, j);
                    idx++;
                }
            }
        }
    }

    if(input.x_num > 1) {
        //reflect x
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(j)
#endif
        for (j = 0; j < output.z_num; ++j) {
            for (int i = 0; i < sz_x; ++i) {
                for (int k = 0; k < output.y_num; ++k) {
                    output.at(k, i, j) = output.at(k, 2 * sz_x - i, j);
                }
            }
            int idx = sz_x+2;
            for (int i = output.x_num - sz_x; i < output.x_num; ++i) {
                for (int k = 0; k < output.y_num; ++k) {
                    output.at(k, i, j) = output.at(k, output.x_num - idx, j);

                }
                idx++;
            }
        }
    }

    //z loops
    if(input.z_num > 1) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(j)
#endif
        for (j = 0; j < sz_z; ++j) {
            for (int i = 0; i < output.x_num; ++i) {
                for (int k = 0; k < output.y_num; ++k) {
                    output.at(k, i, j) = output.at(k, i, 2 * sz_z - j);
                }
            }
        }

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(j)
#endif
        for (int j = output.z_num - sz_z; j < output.z_num; ++j) {
            auto idx = sz_z+2 + j - (output.z_num - sz_z);
            for (int i = 0; i < output.x_num; ++i) {
                for (int k = 0; k < output.y_num; ++k) {
                    output.at(k, i, j) = output.at(k, i, output.z_num - idx);
                }
            }
        }
    }
}


/**
 * unPadds an array
 * @tparam T - type of data
 * @param input - padded source data
 * @param output - unpadded image
 * @param org_dim_y - original image y_num
 * @param org_dim_x- original image x_num
 * @param org_dim_z - original image z_num
*/
template<typename T>
void unpaddPixels(PixelData<T> &input, PixelData<T> &output, int org_dim_y, int org_dim_x, int org_dim_z) {


    output.initWithResize(org_dim_y, org_dim_x, org_dim_z);

    int sz_y = (input.y_num - org_dim_y)/2; //accounts for the resizing due to minimum dimension constraints that could occur on the first pass.
    int sz_x = (input.x_num - org_dim_x)/2;
    int sz_z = (input.z_num - org_dim_z)/2;

    int j = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(j)
#endif
    //copy across internal
    for (j = 0; j < output.z_num; ++j) {
        for (int i = 0; i < output.x_num; ++i) {
            for (int k = 0; k < output.y_num; ++k) {
                output.at(k,i,j) = input.at(k + sz_y, i + sz_x, j + sz_z);
            }
        }
    }
}






#endif //PIXEL_DATA_HPP
