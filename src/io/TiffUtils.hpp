/*
 * Created by Krzysztof Gonciarz 2018
 *
 * Implements mesh read/write from/to TIFF files functionality.
 */
#ifndef TIFF_UTILS_HPP
#define TIFF_UTILS_HPP


#include <string>
#include <tiffio.h>
#include <sstream>
#include <src/data_structures/Mesh/MeshData.hpp>


namespace TiffUtils {

    /**
     * Class for reading useful information from TIFF file (dimensions, type of data etc.)
     */
    class TiffInfo {
    public:
        enum class TiffType {
            TIFF_UINT8, TIFF_UINT16, TIFF_FLOAT, TIFF_INVALID
        };

        TiffInfo(const std::string &aFileName) { open(aFileName); }
        TiffInfo(TiffInfo &&obj) = default;

        ~TiffInfo() { close(); }

        std::string toString() const {
            if (iFile == nullptr) {
                return "<File not opened>";
            }
            std::ostringstream outputStr;
            outputStr << "FileName: [" << iFileName << "]";
            outputStr << ", Width/Height/Depth: " << iImgWidth << "/" << iImgHeight << "/" << iNumberOfDirectories;
            outputStr << ", SamplesPerPixel: " << iSamplesPerPixel;
            outputStr << ", Bits per sample: " << iBitsPerSample;
            outputStr << ", ImageType: ";
            switch (iType) {
                case TiffType::TIFF_UINT8:
                    outputStr << "uint8";
                    break;
                case TiffType::TIFF_UINT16:
                    outputStr << "uint16";
                    break;
                case TiffType::TIFF_FLOAT:
                    outputStr << "float";
                    break;
                default:
                    outputStr << "NOT SUPPORTED";
            }
            outputStr << ", Photometric: " << iPhotometric;
            outputStr << ", StripSize: " << TIFFStripSize(iFile);

            return outputStr.str();
        }

        bool isFileOpened() const { return iFile != nullptr; }

        TiffType iType = TiffType::TIFF_INVALID;
        TIFF *iFile = nullptr;
        std::string iFileName = "";
        uint32 iImgWidth = 0;
        uint32 iImgHeight = 0;
        uint32 iNumberOfDirectories = 0;
        unsigned short iSamplesPerPixel = 0;
        unsigned short iBitsPerSample = 0;
        unsigned short iSampleFormat = 0;
        unsigned short iPhotometric = 0;

    private:
        TiffInfo(const TiffInfo&) = delete; // make it noncopyable
        TiffInfo& operator=(const TiffInfo&) = delete; // make it not assignable

        bool open(const std::string &aFileName) {

            std::cout << "Opening file: [" << (aFileName == "" ? "null" : aFileName) << "]" << std::endl;
            iFileName = aFileName;
            iFile = TIFFOpen(aFileName.c_str(), "r");
            if (iFile == nullptr) {
                std::cout << "Could not open file [" << aFileName << "]" << std::endl;
                return false;
            }

            // ----- Img dimensions
            TIFFGetField(iFile, TIFFTAG_IMAGEWIDTH, &iImgWidth);
            TIFFGetField(iFile, TIFFTAG_IMAGELENGTH, &iImgHeight);
            iNumberOfDirectories = TIFFNumberOfDirectories(iFile);

            // -----  Img type
            TIFFGetField(iFile, TIFFTAG_SAMPLESPERPIXEL, &iSamplesPerPixel);
            TIFFGetField(iFile, TIFFTAG_BITSPERSAMPLE, &iBitsPerSample);
            if (TIFFGetField(iFile, TIFFTAG_SAMPLEFORMAT, &iSampleFormat) == 0) {
                // format is not set in file - assume uint
                iSampleFormat = SAMPLEFORMAT_UINT;
            }

            // -----  Img color scheme
            TIFFGetField(iFile, TIFFTAG_PHOTOMETRIC, &iPhotometric);

            // ----- Validation
            if (iBitsPerSample == 8 && iSampleFormat == SAMPLEFORMAT_UINT) {
                iType = TiffType::TIFF_UINT8;
            } else if (iBitsPerSample == 16 && iSampleFormat == SAMPLEFORMAT_UINT) {
                iType = TiffType::TIFF_UINT16;
            } else if (iBitsPerSample == 32 && iSampleFormat == SAMPLEFORMAT_IEEEFP) {
                iType = TiffType::TIFF_FLOAT;
            } else {
                std::cout << "Not supported type of TIFF (BitsPerSample: " << iBitsPerSample << ", SampleFromat: " << iSampleFormat << std::endl;
                iType = TiffType::TIFF_INVALID;
            }

            if (iPhotometric != PHOTOMETRIC_MINISBLACK) {
                std::cout << "Only grayscale images are supported" << std::endl;
                iType = TiffType::TIFF_INVALID;
            }

            if (iType == TiffType::TIFF_INVALID) {
                close();
                return false;
            }

            return true;
        }

        void close() {
            if (iFile != nullptr) {
                TIFFClose(iFile);
                iFile = nullptr;
            }
        }

        friend std::ostream& operator<<(std::ostream &os, const TiffInfo &obj) {
            os << obj.toString();
            return os;
        }
    };


    /**
     * Reads TIFF file to mesh
     * @tparam T type of mesh/image (uint8_t, uint16_t, float)
     * @param aFileName full absolute file name
     * @return mesh with tiff or empty mesh if reading file failed
     */
    template<typename T>
    MeshData<T> getMesh(const std::string &aFileName) {
        TiffInfo tiffInfo(aFileName);
        return getMesh<T>(tiffInfo);
    }

    /**
     * Reads TIFF file to mesh
     * @tparam T type of mesh/image (uint8_t, uint16_t, float)
     * @param aTiff TiffInfo class with opened image
     * @return mesh with tiff or empty mesh if reading file failed
     */
    template<typename T>
    MeshData<T> getMesh(const TiffInfo &aTiff) {
        if (!aTiff.isFileOpened()) return MeshData<T>();
        MeshData<T> mesh(aTiff.iImgHeight, aTiff.iImgWidth, aTiff.iNumberOfDirectories);
        getMesh(aTiff, mesh);
        return mesh;
    }

    /**
    * Reads TIFF file to provided mesh
    * @tparam T type of mesh/image (uint8_t, uint16_t, float)
    * @param aTiff TiffInfo class with opened image
    * @param aInputMesh pre-created mesh with dimensions of image from aTiff class
    * @return mesh with tiff or empty mesh if reading file failed
    */
    template<typename T>
    void getMesh(const TiffInfo &aTiff, MeshData<T> &aInputMesh) {
        if (!aTiff.isFileOpened()) return;

        // Prepeare preallocated MeshData object for TIF
        std::cout << "getMesh: " << aInputMesh << std::endl;

        // Get some more data from TIFF needed during reading
        const long stripSize = TIFFStripSize(aTiff.iFile);
        std::cout << __func__ << ": ScanlineSize=" << TIFFScanlineSize(aTiff.iFile) << " StripSize=" << stripSize << " NumberOfStrips=" << TIFFNumberOfStrips(aTiff.iFile) << std::endl;

        // Read TIF to MeshData
        size_t currentOffset = 0;
        for (int i = 0; i < aTiff.iNumberOfDirectories; ++i) {
            TIFFSetDirectory(aTiff.iFile, i);

            // read current directory
            for (tstrip_t strip = 0; strip < TIFFNumberOfStrips(aTiff.iFile); ++strip) {
                tmsize_t readLen = TIFFReadEncodedStrip(aTiff.iFile, strip, (&aInputMesh.mesh[(size_t)0] + currentOffset), (tsize_t) -1 /* read as much as possible */);
                currentOffset += readLen/sizeof(T);
            }
        }

        // Set proper dimensions (x and y are exchanged giving transpose w.r.t. original file)
        aInputMesh.z_num = aTiff.iNumberOfDirectories;
        aInputMesh.y_num = aTiff.iImgWidth;
        aInputMesh.x_num = aTiff.iImgHeight;
    }

    /**
     * Saves provided mesh as a TIFF file
     * @tparam T handled types are uint8_t, uint16_t and float
     * @param aFileName name of output TIFF file
     * @param aData mesh with data
     */
    template<typename T>
    void saveMeshAsTiff(const std::string &aFileName, const MeshData<T> &aData) {

        // Set proper dimensions (x and y are exchanged giving transpose)
        const uint32_t width = aData.y_num;
        const uint32_t height = aData.x_num;
        const uint32_t depth = aData.z_num;
        const uint16_t samplesPerPixel = 1;
        const uint16_t bitsPerSample = sizeof(T) * 8;

        size_t imgSize = (size_t)width * height * depth * sizeof(T);
        bool isBigTiff = imgSize > (2 ^ 32 - 32 * 2^10); // 4GB - 32kB headerSize (should be safe enough)
        TIFF *tif = TIFFOpen(aFileName.c_str(), isBigTiff ? "w8" : "w");

        // Set fileds needed to calculate TIFFDefaultStripSize and set proper TIFFTAG_ROWSPERSTRIP
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
        uint32_t rowsPerStrip = TIFFDefaultStripSize(tif, -1 /*width*samples*nbits/8*/);
        if (rowsPerStrip > height) rowsPerStrip = height; // max one image at a time
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);

        size_t StripSize =  (size_t)TIFFStripSize(tif);
        size_t ScanlineSize = (size_t)TIFFScanlineSize(tif);
        std::cout << __func__ << ": " << "FileName: [" << aFileName << "] " << aData << std::endl;
        std::cout << __func__ << ": ScanlineSize=" << ScanlineSize << " StripSize: " << StripSize << " NoOfStrips: " << TIFFNumberOfStrips(tif) << std::endl;

        size_t currentOffset = 0;
        for(int i = 0; i < depth; ++i) {
            TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
            TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
            TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
            TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
            TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, bitsPerSample == 32 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT);
            TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
            TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
            TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);

            size_t dataLen = ScanlineSize * height; // length of single image
            for (tstrip_t strip = 0; strip < TIFFNumberOfStrips(tif); ++strip) {
                tmsize_t writeLen = TIFFWriteEncodedStrip(tif, strip, (void *) (&aData.mesh[(size_t)0] + currentOffset), dataLen >= StripSize ? StripSize : dataLen);
                dataLen -= writeLen;
                currentOffset += writeLen/sizeof(T);
            }

            TIFFWriteDirectory(tif);
        }

        TIFFClose(tif);
    }

    /**
     * Saves provided mesh as a uint16 TIFF file
     * @tparam T handled types are uint8_t, uint16_t and float
     * @param aFileName name of output TIFF file
     * @param aData mesh with data
     */
    template<typename T>
    void saveMeshAsTiffUint16(const std::string &filename, const MeshData<T> &aData) {
        //  Converts the data to uint16t then writes it (requires creation of a complete copy of the data)
        MeshData<uint16_t> mesh16{aData, true /*copy data*/};
        saveMeshAsTiff(filename, mesh16);
    }
}

#endif
