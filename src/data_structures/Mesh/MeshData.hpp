//////////////////////////////////////////////////////////////
//
//
//  ImageGen 2016 Bevan Cheeseman
//                Krzysztof Gonciarz
//  Meshdata class for storing the image/mesh data
//
//
//
//
///////////////////////////////////////////////////////////////

#ifndef PARTPLAY_MESHCLASS_H
#define PARTPLAY_MESHCLASS_H

#include <vector>
#include <cmath>
#include <memory>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "../../misc/APRTimer.hpp"


template <typename T>
class ArrayWrapper
{
public:
    ArrayWrapper() : iArray(nullptr), iNumOfElements(0) {}
    ArrayWrapper(ArrayWrapper &&aObj) {
        iArray = aObj.iArray; aObj.iArray = nullptr;
        iNumOfElements = aObj.iNumOfElements; aObj.iNumOfElements = 0;
    }
    ArrayWrapper& operator=(ArrayWrapper&& aObj) {
        iArray = aObj.iArray; aObj.iArray = nullptr;
        iNumOfElements = aObj.iNumOfElements; aObj.iNumOfElements = 0;
        return *this;
    }

    inline void set(T *aInputArray, size_t aNumOfElements) {iArray = aInputArray; iNumOfElements = aNumOfElements;}

    inline T* begin() { return (iArray); }
    inline T* end() { return (iArray + iNumOfElements); }
    inline const T* begin() const { return (iArray); }
    inline const T* end() const { return (iArray + iNumOfElements); }


    inline T& operator[](size_t idx) { return iArray[idx]; }
    inline const T& operator[](size_t idx) const { return iArray[idx]; }
    inline size_t size() const { return iNumOfElements; }
    inline size_t capacity() const { return iNumOfElements; }

    inline T* get() {return iArray;}
    inline const T* get() const {return iArray;}

    inline void swap(ArrayWrapper<T> &aObj) {
        std::swap(iNumOfElements, aObj.iNumOfElements);
        std::swap(iArray, aObj.iArray);
    }

private:
    ArrayWrapper(const ArrayWrapper&) = delete; // make it noncopyable
    ArrayWrapper& operator=(const ArrayWrapper&) = delete; // make it not assignable

    T *iArray;
    size_t iNumOfElements;
};


/**
 * Provides implementation for 3D mesh with elements of given type.
 * @tparam T type of mesh elements
 */
template <typename T>
class MeshData {
public :
    // size of mesh and container for data
    size_t y_num;
    size_t x_num;
    size_t z_num;
    std::unique_ptr<T[]> meshMemory;
    ArrayWrapper<T> mesh;

    /**
     * Constructor - initialize mesh with size of 0,0,0
     */
    MeshData() { init(0, 0, 0); }

    /**
     * Constructor - initialize initial size of mesh to provided values
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    MeshData(int aSizeOfY, int aSizeOfX, int aSizeOfZ) { init(aSizeOfY, aSizeOfX, aSizeOfZ); }

    /**
     * Constructor - creates mesh with provided dimentions initialized to aInitVal
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     * @param aInitVal - initial value of all elements
     */
    MeshData(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T aInitVal) { init(aSizeOfY, aSizeOfX, aSizeOfZ, aInitVal); }

    /**
     * Move constructor
     * @param aObj mesh to be moved
     */
    MeshData(MeshData &&aObj) {
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
    MeshData& operator=(MeshData &&aObj) {
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
    MeshData(const MeshData<U> &aMesh, bool aShouldCopyData) {
        init(aMesh.y_num, aMesh.x_num, aMesh.z_num);
        if (aShouldCopyData) std::copy(aMesh.mesh.begin(), aMesh.mesh.end(), mesh.begin());
    }

    /**
     * Creates copy of this mesh converting each element to new type
     * @tparam U new type of mesh
     * @return created object by value
     */
    template <typename U>
    MeshData<U> toType() const {
        MeshData<U> new_value(y_num, x_num, z_num);
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
    T& operator()(size_t y, size_t x, size_t z) {
        y = std::min(y, y_num-1);
        x = std::min(x, x_num-1);
        z = std::min(z, z_num-1);
        size_t idx = (size_t)z * x_num * y_num + x * y_num + y;
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
    void copyFromMesh(const MeshData<U> &aInputMesh, unsigned int aNumberOfBlocks = 8) {
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
    void init(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T aInitVal) {
        y_num = aSizeOfY;
        x_num = aSizeOfX;
        z_num = aSizeOfZ;
        size_t size = (size_t)y_num * x_num * z_num;
        meshMemory.reset(new T[size]);
        T *array = meshMemory.get();
        if (array == nullptr) { std::cerr << "Could not allocate memory!" << size << std::endl; exit(-1); }
        mesh.set(array, size);

        // Fill values of new buffer in parallel
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
     * Initialize mesh with dimensions taken from provided mesh
     * @tparam S
     * @param aInputMesh
     */
    template<typename S>
    void init(const MeshData<S> &aInputMesh) {
        init(aInputMesh.y_num, aInputMesh.x_num, aInputMesh.z_num);
    }

    /**
     * Initializes mesh with provided dimensions with default value of used type
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    void init(int aSizeOfY, int aSizeOfX, int aSizeOfZ) {
        y_num = aSizeOfY;
        x_num = aSizeOfX;
        z_num = aSizeOfZ;
        size_t size = (size_t)y_num * x_num * z_num;
        meshMemory.reset(new T[size]);
        if (meshMemory.get() == nullptr) { std::cerr << "Could not allocate memory!" << size << std::endl; exit(-1); }
        mesh.set(meshMemory.get(), size);
    }

    /**
     * Initializes mesh with size of half of provided dimensions (rounding up if not divisible by 2)
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    void initDownsampled(int aSizeOfY, int aSizeOfX, int aSizeOfZ) {
        const int z_num_ds = ceil(1.0*aSizeOfZ/2.0);
        const int x_num_ds = ceil(1.0*aSizeOfX/2.0);
        const int y_num_ds = ceil(1.0*aSizeOfY/2.0);

        init(y_num_ds, x_num_ds, z_num_ds);
    }

    /**
     * Initializes mesh with size of half of provided dimensions (rounding up if not divisible by 2) and initialize values
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     * @param aInitVal
     */
    void initDownsampled(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T aInitVal) {
        const int z_num_ds = ceil(1.0*aSizeOfZ/2.0);
        const int x_num_ds = ceil(1.0*aSizeOfX/2.0);
        const int y_num_ds = ceil(1.0*aSizeOfY/2.0);

        init(y_num_ds, x_num_ds, z_num_ds, aInitVal);
    }

    /**
     * Initializes mesh with size of half of provided mesh dimensions (rounding up if not divisible by 2)
     * @param aMesh - mesh used to get dimensions
     */
    template <typename U>
    void initDownsampled(const MeshData<U> &aMesh) {
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
    void initDownsampled(const MeshData<U> &aMesh, T aInitVal) {
        const int z_num_ds = ceil(1.0*aMesh.z_num/2.0);
        const int x_num_ds = ceil(1.0*aMesh.x_num/2.0);
        const int y_num_ds = ceil(1.0*aMesh.y_num/2.0);

        init(y_num_ds, x_num_ds, z_num_ds, aInitVal);
    }

    /**
     * Swaps data of meshes this <-> aObj
     * @param aObj
     */
    void swap(MeshData &aObj) {
        std::swap(x_num, aObj.x_num);
        std::swap(y_num, aObj.y_num);
        std::swap(z_num, aObj.z_num);
        meshMemory.swap(aObj.meshMemory);
        mesh.swap(aObj.mesh);
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
    void copyFromMeshWithUnaryOp(const MeshData<U> &aInputMesh, R aOp, size_t aNumberOfBlocks = 10) {
        aNumberOfBlocks = std::min(aInputMesh.z_num, (size_t)aNumberOfBlocks);
        size_t numOfElementsPerBlock = aInputMesh.z_num/aNumberOfBlocks;

        #ifdef HAVE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
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
        size_t z = aIdx / (x_num * y_num);
        aIdx -= z * (x_num * y_num);
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
    void printMesh(int aColumnWidth, bool aXYplanes = true) const {
        if (aXYplanes) {
            for (size_t z = 0; z < z_num; ++z) {
                std::cout << "z=" << z << "\n";
                for (size_t y = 0; y < y_num; ++y) {
                    for (size_t x = 0; x < x_num; ++x) {
                        std::cout << std::setw(aColumnWidth) << at(y, x, z) << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << std::endl;
            }
        }
        else { // X-Z planes
            for (size_t y = 0; y < y_num; ++y) {
                std::cout << "y=" << y << "\n";
                for (size_t z = 0; z < z_num; ++z) {
                    for (size_t x = 0; x < x_num; ++x) {
                        std::cout << std::setw(aColumnWidth) << at(y, x, z) << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << std::endl;
            }
        }
    }

    friend std::ostream & operator<<(std::ostream &os, const MeshData<T> &obj) {
        os << "MeshData: size(Y/X/Z)=" << obj.y_num << "/" << obj.x_num << "/" << obj.z_num << " vSize:" << obj.mesh.size() << " vCapacity:" << obj.mesh.capacity() << " elementSize:" << sizeof(T);
        return os;
    }

private:

    MeshData(const MeshData&) = delete; // make it noncopyable
    MeshData& operator=(const MeshData&) = delete; // make it not assignable

};


template<typename T, typename S, typename R, typename C>
void downsample(const MeshData<T> &aInput, MeshData<S> &aOutput, R reduce, C constant_operator, bool aInitializeOutput = false) {
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
        aOutput.init(y_num_ds, x_num_ds, z_num_ds);
        timer.stop_timer();
    }

    timer.start_timer("downsample_loop");
    #ifdef HAVE_OPENMP
    #pragma omp parallel for default(shared)
    #endif
    for (size_t z = 0; z < z_num_ds; ++z) {
        for (size_t x = 0; x < x_num_ds; ++x) {

            // shifted +1 in original inMesh space
            const int64_t shx = std::min(2*x + 1, x_num - 1);
            const int64_t shz = std::min(2*z + 1, z_num - 1);

            const ArrayWrapper<T> &inMesh = aInput.mesh;
            ArrayWrapper<S> &outMesh = aOutput.mesh;

            for (size_t y = 0; y < y_num_ds; ++y) {
                const int64_t shy = std::min(2*y + 1, y_num - 1);
                const int64_t idx = z * x_num_ds * y_num_ds + x * y_num_ds + y;
                outMesh[idx] =  constant_operator(
                        reduce(reduce(reduce(reduce(reduce(reduce(reduce(        // inMesh coordinates
                               inMesh[2*z * x_num * y_num + 2*x * y_num + 2*y],  // z,   x,   y
                               inMesh[2*z * x_num * y_num + 2*x * y_num + shy]), // z,   x,   y+1
                               inMesh[2*z * x_num * y_num + shx * y_num + 2*y]), // z,   x+1, y
                               inMesh[2*z * x_num * y_num + shx * y_num + shy]), // z,   x+1, y+1
                               inMesh[shz * x_num * y_num + 2*x * y_num + 2*y]), // z+1, x,   y
                               inMesh[shz * x_num * y_num + 2*x * y_num + shy]), // z+1, x,   y+1
                               inMesh[shz * x_num * y_num + shx * y_num + 2*y]), // z+1, x+1, y
                               inMesh[shz * x_num * y_num + shx * y_num + shy])  // z+1, x+1, y+1
                );
            }
        }
    }
    timer.stop_timer();
}

template<typename T>
void downsamplePyrmaid(MeshData<T> &original_image, std::vector<MeshData<T>> &downsampled, size_t l_max, size_t l_min) {
    downsampled.resize(l_max + 1); // each level is kept at same index
    downsampled.back().swap(original_image); // put original image at l_max index

    // calculate downsampled in range (l_max, l_min]
    auto sum = [](const float x, const float y) -> float { return x + y; };
    auto divide_by_8 = [](const float x) -> float { return x/8.0; };
    for (size_t level = l_max; level > l_min; --level) {
        downsample(downsampled[level], downsampled[level - 1], sum, divide_by_8, true);
    }
}

#endif //PARTPLAY_MESHCLASS_H
