//////////////////////////////////////////////////////////////
//
//
//  ImageGen 2016 Bevan Cheeseman
//
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

#include "benchmarks/development/old_structures/structure_parts.h"
#include <tiffio.h>

struct coords3d {
    int x,y,z;

    coords3d operator *(uint16_t multiplier)
    {
        coords3d new_;
        new_.y = y * multiplier;
        new_.x = x * multiplier;
        new_.z = z * multiplier;
        return new_;
    }

    coords3d operator -(uint16_t diff)
    {
        coords3d new_;
        new_.y = this->y - diff;
        new_.x = this->x - diff;
        new_.z = this->z - diff;
        return new_;
    }

    friend bool operator <=(coords3d within, coords3d boundaries)
    {
        return within.y <= boundaries.y && within.x <= boundaries.x && within.z <= boundaries.z;
    }

    friend bool operator <(coords3d within, coords3d boundaries)
    {
        return within.y < boundaries.y && within.x < boundaries.x && within.z < boundaries.z;
    }
    
    friend bool operator ==(coords3d within, coords3d boundaries)
    {
        return within.y == boundaries.y && within.x == boundaries.x && within.z == boundaries.z;
    }

    friend std::ostream& operator<<(std::ostream& os, const coords3d& coords)
    {
        return std::cout << coords.y << " " << coords.x << " " << coords.z;
    }

    bool contains(coords3d neighbour, uint8_t multiplier)
    {
        return abs(this->x - neighbour.x) <= multiplier &&
               abs(this->y - neighbour.y) <= multiplier &&
               abs(this->z - neighbour.z) <= multiplier;
    }

};

template <typename T>
class ArrayWrapper
{
public:
    ArrayWrapper() : iArray(nullptr), iNumOfElements(-1) {}
    ArrayWrapper(ArrayWrapper &&aObj) {
        iArray = aObj.iArray; aObj.iArray = nullptr;
        iNumOfElements = aObj.iNumOfElements; aObj.iNumOfElements = -1;
    }
    ArrayWrapper& operator=(ArrayWrapper&& aObj) {
        iArray = aObj.iArray; aObj.iArray = nullptr;
        iNumOfElements = aObj.iNumOfElements; aObj.iNumOfElements = -1;
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
template <class T>
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
    MeshData() { initialize(0, 0, 0); }

    /**
     * Constructor - initialize initial size of mesh to provided values
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    MeshData(int aSizeOfY, int aSizeOfX, int aSizeOfZ) { initialize(aSizeOfY, aSizeOfX, aSizeOfZ); }

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
        initialize(aMesh.y_num, aMesh.x_num, aMesh.z_num);
        if (aShouldCopyData) std::copy(aMesh.mesh.begin(), aMesh.mesh.end(), mesh.begin());
    }

    /**
     * Creates copy of this mesh converting each element to new type
     * @tparam U new type of mesh
     * @return created object by value
     */
    template <typename U>
    MeshData<U> to_type() const {
        // TODO: currently it creates local object and returns it via copy...
        //       for small objects it's acceptable but when sizes are big it is not so cool.
        //       Should return (smart)pointer or sth.
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
     */ //#FIXME changed it to size_t
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
    T& access_no_protection(int y, int x, int z) {
        size_t idx = (size_t)z * x_num * y_num + x * y_num + y;
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
    void block_copy_data(const MeshData<U> &aInputMesh, unsigned int aNumberOfBlocks = 10) {
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
    void initialize(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T aInitVal) {
        y_num = aSizeOfY;
        x_num = aSizeOfX;
        z_num = aSizeOfZ;
        size_t size = (size_t)y_num * x_num * z_num;
        meshMemory.reset(new T[size]);
        T *array = meshMemory.get();
        if (array == nullptr) { std::cerr << "Could not allocate memory!" << size << std::endl; exit(-1); }
        mesh.set(array, size);

        // Fill values of new buffer in parallel
        // TODO: set dynamically number of threads
        #ifdef HAVE_OPENMP
        #pragma omp parallel num_threads(4)
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
     * Initialize mesh with dimensions taken from provided mesh and initializes
     * its values to 0
     * @tparam S
     * @param aInputMesh
     */
    template<typename S>
    void initialize(const MeshData<S>& aInputMesh) {
        initialize(aInputMesh.y_num, aInputMesh.x_num, aInputMesh.z_num, 0);
    }

    /**
     * Initializes mesh with provided dimensions with default value of used type
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     */
    void initialize(int aSizeOfY, int aSizeOfX, int aSizeOfZ) {
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
     * sets provided val for all elements
     * @param aSizeOfY
     * @param aSizeOfX
     * @param aSizeOfZ
     * @param aInitVal
     */
    void preallocate(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T aInitVal) {
        const int z_num_ds = ceil(1.0*aSizeOfZ/2.0);
        const int x_num_ds = ceil(1.0*aSizeOfX/2.0);
        const int y_num_ds = ceil(1.0*aSizeOfY/2.0);

        initialize(y_num_ds, x_num_ds, z_num_ds, aInitVal);
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


    template<typename U>
    void runUnaryOp(const MeshData<U> &aInputMesh, T (*aOp) (const T& aVal), unsigned int aNumberOfBlocks = 10) {
        aNumberOfBlocks = std::min((unsigned int)aInputMesh.z_num, aNumberOfBlocks);
        unsigned int numOfElementsPerBlock = aInputMesh.z_num/aNumberOfBlocks;

        #ifdef HAVE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (unsigned int blockNum = 0; blockNum < aNumberOfBlocks; ++blockNum) {
            const size_t elementSize = (size_t)aInputMesh.x_num * aInputMesh.y_num;
            const size_t blockSize = numOfElementsPerBlock * elementSize;
            size_t offsetBegin = blockNum * blockSize;
            size_t offsetEnd = offsetBegin + blockSize;
            if (blockNum == aNumberOfBlocks - 1) {
                // Handle tailing elements if number of blocks does not divide.
                offsetEnd = aInputMesh.z_num * elementSize;
            }

            std::transform(aInputMesh.mesh.begin() + offsetBegin,aInputMesh.mesh.begin() + offsetEnd, mesh.begin() + offsetBegin, aOp);
        }
    }

private:

    MeshData(const MeshData&) = delete; // make it noncopyable
    MeshData& operator=(const MeshData&) = delete; // make it not assignable

    //REMOVE_FLAG
    void write_image_tiff(std::string& filename);
    //REMOVE_FLAG
    void write_image_tiff_uint16(std::string& filename);
    //REMOVE_FLAG
    void load_image_tiff(std::string file_name,int z_start = 0, int z_end = -1);

    //REMOVE_FLAG
    void set_size(int y_num_,int x_num_,int z_num_){

        y_num = y_num_;
        x_num = x_num_;
        z_num = z_num_;
    }

    //REMOVE_FLAG
    template<typename V>
    void write_image_tiff(std::vector<V> &data, std::string &filename);
};

template<typename T>
std::ostream & operator<<(std::ostream &os, const MeshData<T> &obj)
{
    os << "MeshData: size(Y/X/Z)=" << obj.y_num << "/" << obj.x_num << "/" << obj.z_num << " vSize:" << obj.mesh.size() << " vCapacity:" << obj.mesh.capacity() << " elementSize:" << sizeof(T);
    return os;
}

template<typename T>
void MeshData<T>::load_image_tiff(std::string file_name,int z_start, int z_end){
    TIFF* tif = TIFFOpen(file_name.c_str(), "r");
    int dircount = 0;
    uint32 width;
    uint32 height;
    unsigned short nbits;
    unsigned short samples;
    void* raster;

    if (tif) {
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &nbits);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);

        do {
            dircount++;
        } while (TIFFReadDirectory(tif));
    } else {
        std::cout <<  "Could not open TIFF file." << std::endl;
        return;
    }


    if (dircount < (z_end - z_start + 1)){
        std::cout << "number of slices and start and finish inconsitent!!" << std::endl;
    }

    //Conditions if too many slices are asked for, or all slices
    if (z_end > dircount) {
        std::cout << "Not that many slices, using max number instead" << std::endl;
        z_end = dircount-1;
    }
    if (z_end < 0) {
        z_end = dircount-1;
    }


    dircount = z_end - z_start + 1;

    long ScanlineSize=TIFFScanlineSize(tif);
    long StripSize =  TIFFStripSize(tif);

    int rowsPerStrip;
    int nRowsToConvert;

    raster = _TIFFmalloc(StripSize);
    T *TBuf = (T*)raster;

    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);


    for(int i = z_start; i < (z_end+1); i++) {
        TIFFSetDirectory(tif, i);


        for (int topRow = 0; topRow < height; topRow += rowsPerStrip) {
            nRowsToConvert = (topRow + rowsPerStrip >height?height- topRow : rowsPerStrip);

            TIFFReadEncodedStrip(tif, TIFFComputeStrip(tif, topRow, 0), TBuf, nRowsToConvert*ScanlineSize);

            std::copy(TBuf, TBuf+nRowsToConvert*width, back_inserter(this->mesh));

        }


    }

    _TIFFfree(raster);


    this->z_num = dircount;
    this->y_num = width;
    this->x_num = height;


    TIFFClose(tif);
}


template<typename T> template<typename V>
void MeshData<T>::write_image_tiff(std::vector<V>& data,std::string& filename){
    //
    //
    //  Bevan Cheeseman 2015
    //
    //
    //  Code for writing tiff image to file
    //


    TIFF* tif = TIFFOpen(filename.c_str() , "w");
    uint32 width;
    uint32 height;
    unsigned short nbits;
    unsigned short samples;
    void* raster;

    //set the size
    width = this->y_num;
    height = this->x_num;
    samples = 1;
    //bit size
    nbits = sizeof(V)*8;

    int num_dir = this->z_num;

    //set up the tiff file
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, nbits);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,TIFFDefaultStripSize(tif, width*samples));

    int test_field;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &test_field);

    int ScanlineSize= (int)TIFFScanlineSize(tif);
    int StripSize =  (int)TIFFStripSize(tif);
    int rowsPerStrip;
    int nRowsToConvert;

    raster = _TIFFmalloc(StripSize);
    V *TBuf = (V*)raster;

    TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);

    int z_start = 0;
    int z_end = num_dir-1;

    int row_count = 0;

    for(int i = z_start; i < (z_end+1); i++) {
        if (i > z_start) {
            TIFFWriteDirectory(tif);

        }

        //set up the tiff file
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, nbits);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples);
        TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,TIFFDefaultStripSize(tif, width*samples));

        row_count = 0;

        for (int topRow = 0; topRow < height; topRow += rowsPerStrip) {
            nRowsToConvert = (topRow + rowsPerStrip >height?height- topRow : rowsPerStrip);

            std::copy(data.begin() + i*width*height + row_count,data.begin() + i*width*height + row_count + nRowsToConvert*width, TBuf);

            row_count += nRowsToConvert*width;

            TIFFWriteEncodedStrip(tif, TIFFComputeStrip(tif, topRow, 0), TBuf, nRowsToConvert*ScanlineSize);

        }

    }

    _TIFFfree(raster);


    TIFFClose(tif);

}

template<typename T>
void MeshData<T>::write_image_tiff_uint16(std::string& filename){
    //
    //  Converts the data to uint16t then writes it (requires creation of a complete copy of the data)
    //

    std::vector<uint16_t> data;
    data.resize(this->y_num*this->x_num*this->z_num);

    std::copy(this->mesh.begin(),this->mesh.end(),data.begin());

    MeshData::write_image_tiff<uint16_t>(data, filename);

}

template<typename T, typename S,typename L1, typename L2>
void down_sample(MeshData<T>& test_a, MeshData<S>& test_a_ds, L1 reduce, L2 constant_operator,
                 bool with_allocation = false);

template<typename T>
void const_upsample_img(MeshData<T>& input_us,MeshData<T>& input,std::vector<unsigned int>& max_dims);

template<typename T>
void MeshData<T>::write_image_tiff(std::string& filename) {
    MeshData::write_image_tiff(this->mesh,filename);
};

template<typename T, typename S,typename L1, typename L2>
void down_sample_overflow_proct(MeshData<T>& test_a, MeshData<S>& test_a_ds, L1 reduce, L2 constant_operator,
                                bool with_allocation = false );

template<typename T>
void downsample_pyrmaid(MeshData<T> &original_image,std::vector<MeshData<T>>& downsampled,unsigned int l_max, unsigned int l_min)
{
    downsampled.resize(l_max+2);
    downsampled.back().swap(original_image);

    auto sum = [](float x, float y) { return x+y; };
    auto divide_by_8 = [](float x) { return x * (1.0/8.0); };

    for (int level = l_max+1; level > l_min; level--) {
        down_sample_overflow_proct(downsampled[level], downsampled[level - 1], sum, divide_by_8, true);
    }
}

template<typename T>
void const_upsample_img(MeshData<T>& input_us,MeshData<T>& input,std::vector<unsigned int>& max_dims){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Creates a constant upsampling of an image
    //
    //
    
    Part_timer timer;
    
    timer.verbose_flag = false;
    
    //restrict the domain to be only as big as possibly needed
    
    int y_size_max = ceil(max_dims[0]/2.0)*2;
    int x_size_max = ceil(max_dims[1]/2.0)*2;
    int z_size_max = ceil(max_dims[2]/2.0)*2;
    
    const int z_num = std::min(input.z_num*2,z_size_max);
    const int x_num = std::min(input.x_num*2,x_size_max);
    const int y_num = std::min(input.y_num*2,y_size_max);
    
    const int z_num_ds_l = z_num/2;
    const int x_num_ds_l = x_num/2;
    const int y_num_ds_l = y_num/2;
    
    const int x_num_ds = input.x_num;
    const int y_num_ds = input.y_num;
    
    input_us.y_num = y_num;
    input_us.x_num = x_num;
    input_us.z_num = z_num;
    
    timer.start_timer("resize");
    
    //input_us.initialize(y_num, x_num,z_num,0);
    //input_us.mesh.resize(y_num*x_num*z_num);
    
    timer.stop_timer();
    
    std::vector<T> temp_vec;
    temp_vec.resize(y_num_ds,0);
    
    timer.start_timer("up_sample_const");
    
    unsigned int j, i, k;
    
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(j,i,k) firstprivate(temp_vec) if(z_num_ds_l*x_num_ds_l > 100)
#endif
    for(j = 0;j < z_num_ds_l;j++){
        
        for(i = 0;i < x_num_ds_l;i++){
            
//            //four passes
//            
//            unsigned int offset = j*x_num_ds*y_num_ds + i*y_num_ds;
//            //first take into cache
//            for (k = 0; k < y_num_ds_l;k++){
//                temp_vec[k] = input.mesh[offset + k];
//            }
//            
//            //(0,0)
//            
//            offset = 2*j*x_num*y_num + 2*i*y_num;
//            //then do the operations two by two
//            for (k = 0; k < y_num_ds_l;k++){
//                input_us.mesh[offset + 2*k] = temp_vec[k];
//                input_us.mesh[offset + 2*k + 1] = temp_vec[k];
//            }
//            
//            //(0,1)
//            offset = (2*j+1)*x_num*y_num + 2*i*y_num;
//            //then do the operations two by two
//            for (k = 0; k < y_num_ds_l;k++){
//                input_us.mesh[offset + 2*k] = temp_vec[k];
//                input_us.mesh[offset + 2*k + 1] = temp_vec[k];
//            }
//            
//            offset = 2*j*x_num*y_num + (2*i+1)*y_num;
//            //(1,0)
//            //then do the operations two by two
//            for (k = 0; k < y_num_ds_l;k++){
//                input_us.mesh[offset + 2*k] = temp_vec[k];
//                input_us.mesh[offset + 2*k + 1] = temp_vec[k];
//            }
//            
//            offset = (2*j+1)*x_num*y_num + (2*i+1)*y_num;
//            //(1,1)
//            //then do the operations two by two
//            for (k = 0; k < y_num_ds_l;k++){
//                input_us.mesh[offset + 2*k] = temp_vec[k];
//                input_us.mesh[offset + 2*k + 1] = temp_vec[k];
//            }
            //first take into cache
            for (k = 0; k < y_num_ds_l;k++){
                temp_vec[k] = input.mesh[j*x_num_ds*y_num_ds + i*y_num_ds + k];
            }
            
            //(0,0)
            
            //then do the operations two by two
            for (k = 0; k < y_num_ds_l;k++){
                input_us.mesh[2*j*x_num*y_num + 2*i*y_num + 2*k] = temp_vec[k];
                input_us.mesh[2*j*x_num*y_num + 2*i*y_num + 2*k + 1] = temp_vec[k];
            }
            
            //(0,1)
            
            //then do the operations two by two
            for (k = 0; k < y_num_ds_l;k++){
                input_us.mesh[(2*j+1)*x_num*y_num + 2*i*y_num + 2*k] = temp_vec[k];
                input_us.mesh[(2*j+1)*x_num*y_num + 2*i*y_num + 2*k + 1] = temp_vec[k];
            }
            
            //(1,0)
            //then do the operations two by two
            for (k = 0; k < y_num_ds_l;k++){
                input_us.mesh[2*j*x_num*y_num + (2*i+1)*y_num + 2*k] = temp_vec[k];
                input_us.mesh[2*j*x_num*y_num + (2*i+1)*y_num + 2*k + 1] = temp_vec[k];
            }
            
            //(1,1)
            //then do the operations two by two
            for (k = 0; k < y_num_ds_l;k++){
                input_us.mesh[(2*j+1)*x_num*y_num + (2*i+1)*y_num + 2*k] = temp_vec[k];
                input_us.mesh[(2*j+1)*x_num*y_num + (2*i+1)*y_num + 2*k + 1] = temp_vec[k];
            }
            
            
        }
    }
    
    timer.stop_timer();
    
    
    
    
}
template<typename T, typename S,typename L1, typename L2>
void down_sample_overflow_proct(MeshData<T>& test_a, MeshData<S>& test_a_ds, L1 reduce, L2 constant_operator,
                 bool with_allocation ){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Downsampling
    //
    //
    //  Updated method to protect over=flow of down-sampling
    //

    const int64_t z_num = test_a.z_num;
    const int64_t x_num = test_a.x_num;
    const int64_t y_num = test_a.y_num;

    const int64_t z_num_ds = (int64_t) ceil(1.0*z_num/2.0);
    const int64_t x_num_ds = (int64_t) ceil(1.0*x_num/2.0);
    const int64_t y_num_ds = (int64_t) ceil(1.0*y_num/2.0);

    Part_timer timer;
    //timer.verbose_flag = true;

    if(with_allocation) {
        timer.start_timer("downsample_initalize");

        test_a_ds.initialize(y_num_ds, x_num_ds, z_num_ds, 0);

        timer.stop_timer();
    }

    timer.start_timer("downsample_loop");
    std::vector<float> temp_vec;
    temp_vec.resize(y_num,0);

    std::vector<float> temp_vec2;
    temp_vec2.resize(y_num_ds,0);


    int64_t i, k, si_, sj_, sk_;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i,k,si_,sj_,sk_) firstprivate(temp_vec,temp_vec2)
#endif
    for(int j = 0;j < z_num_ds; j++) {


        for (i = 0; i < x_num_ds; i++) {

            si_ = std::min((int64_t)2 * i + 1, x_num - 1);
            sj_ = std::min((int64_t)2 * j + 1, z_num - 1);

            //four passes

            //first take into cache
            for (k = 0; k < y_num; k++) {
                temp_vec[k] = test_a.mesh[2 * j * x_num * y_num + 2 * i * y_num + k];
            }

            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                temp_vec2[k] = reduce(0,temp_vec[2 * k]);
                temp_vec2[k] = reduce(temp_vec2[k], temp_vec[sk_]);
            }

            //first take into cache
            for (k = 0; k < y_num; k++) {
                temp_vec[k] = test_a.mesh[2 * j * x_num * y_num + si_ * y_num + k];
            }

            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                temp_vec2[k] =
                        reduce(temp_vec2[k], temp_vec[2 * k]);
                temp_vec2[k] =
                        reduce(temp_vec2[k], temp_vec[sk_]);
            }


            //first take into cache
            for (k = 0; k < y_num; k++) {
                temp_vec[k] = test_a.mesh[sj_ * x_num * y_num + 2 * i * y_num + k];
            }


            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                temp_vec2[k] =
                        reduce(temp_vec2[k], temp_vec[2 * k]);
                temp_vec2[k] =
                        reduce(temp_vec2[k], temp_vec[sk_]);
            }

            //first take into cache
            for (k = 0; k < y_num; k++) {
                temp_vec[k] = test_a.mesh[sj_ * x_num * y_num + si_ * y_num + k];
            }


            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                temp_vec2[k] =
                        reduce(temp_vec2[k], temp_vec[2 * k]);
                temp_vec2[k] =
                        reduce(temp_vec2[k], temp_vec[sk_]);
                //final operaions
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        constant_operator(temp_vec2[k]);
            }

        }

    }

    timer.stop_timer();

}



template<typename T, typename S,typename L1, typename L2>
void down_sample(MeshData<T>& test_a, MeshData<S>& test_a_ds, L1 reduce, L2 constant_operator,
                 bool with_allocation ){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Downsampling
    //
    //

    const int64_t z_num = test_a.z_num;
    const int64_t x_num = test_a.x_num;
    const int64_t y_num = test_a.y_num;

    const int64_t z_num_ds = (int64_t) ceil(1.0*z_num/2.0);
    const int64_t x_num_ds = (int64_t) ceil(1.0*x_num/2.0);
    const int64_t y_num_ds = (int64_t) ceil(1.0*y_num/2.0);

    Part_timer timer;
    //timer.verbose_flag = true;

    if(with_allocation) {
        timer.start_timer("downsample_initalize");

        test_a_ds.initialize(y_num_ds, x_num_ds, z_num_ds, 0);

        timer.stop_timer();
    }

    timer.start_timer("downsample_loop");
    std::vector<T> temp_vec;
    temp_vec.resize(y_num,0);


    int64_t i, k, si_, sj_, sk_;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i,k,si_,sj_,sk_) firstprivate(temp_vec)
#endif
    for(int64_t j = 0;j < z_num_ds; j++) {


        for (i = 0; i < x_num_ds; i++) {

            si_ = std::min(2 * i + 1, x_num - 1);
            sj_ = std::min(2 * j + 1, z_num - 1);

            //four passes

            std::copy(test_a.mesh.begin() + 2 * j * x_num * y_num + 2 * i * y_num,test_a.mesh.begin() + 2 * j * x_num * y_num + 2 * i * y_num + y_num,temp_vec.begin());

            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] = reduce(0,temp_vec[2 * k]);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[sk_]);
            }


            std::copy(test_a.mesh.begin() + 2 * j * x_num * y_num + si_ * y_num,test_a.mesh.begin() + 2 * j * x_num * y_num + si_ * y_num + y_num,temp_vec.begin());


            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[2 * k]);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[sk_]);
            }


            std::copy(test_a.mesh.begin() + sj_ * x_num * y_num + 2 * i * y_num,test_a.mesh.begin() + sj_ * x_num * y_num + 2 * i * y_num + y_num,temp_vec.begin());

            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[2 * k]);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[sk_]);
            }

            std::copy(test_a.mesh.begin() + sj_ * x_num * y_num + si_ * y_num,test_a.mesh.begin() + sj_ * x_num * y_num + si_ * y_num + y_num,temp_vec.begin());

            //then do the operations two by two
            for (k = 0; k < y_num_ds; k++) {
                sk_ = std::min(2 * k + 1, y_num - 1);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[2 * k]);
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        reduce(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k], temp_vec[sk_]);
                //final operaions
                test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k] =
                        constant_operator(test_a_ds.mesh[j * x_num_ds * y_num_ds + i * y_num_ds + k]);
            }

        }

    }

    timer.stop_timer();

}

#endif //PARTPLAY_MESHCLASS_H
