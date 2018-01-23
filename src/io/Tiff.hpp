#ifndef TIFF_HPP
#define TIFF_HPP


#include <string>
#include <tiffio.h>
#include <sstream>
#include <src/data_structures/Mesh/MeshData.hpp>


namespace TiffUtils {

    class Tiff {
    public:
        enum class TiffType {
            TIFF_UINT8, TIFF_UINT16, TIFF_FLOAT, TIFF_INVALID
        };

        Tiff(const std::string &aFileName) { open(aFileName); }
        Tiff(Tiff &&obj) = default;

        ~Tiff() { close(); }

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

        bool isFileOpened() { return iFile != nullptr; }

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
        Tiff( const Tiff& ) = delete; // make it noncopyable

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
    };

    std::ostream& operator<<(std::ostream &os, const Tiff &obj) {
        os << obj.toString();
        return os;
    }

    template<typename T>
    MeshData<T> getMesh(const Tiff &aTiff) {
        // Prepeare preallocated MeshData object for TIF
        MeshData<T> mesh(aTiff.iImgHeight, aTiff.iImgWidth, aTiff.iNumberOfDirectories);
        std::cout << mesh << std::endl;

        // Get some more data from TIFF needed during reading
        const long stripSize = TIFFStripSize(aTiff.iFile);
        std::cout << "ScanlineSize=" << TIFFScanlineSize(aTiff.iFile) << " StripSize=" << stripSize << " NumberOfStrips=" << TIFFNumberOfStrips(aTiff.iFile) << std::endl;

        // Read TIF to MeshData
        size_t currentOffset = 0;
        for (int i = 0; i < aTiff.iNumberOfDirectories; ++i) {
            TIFFSetDirectory(aTiff.iFile, i);

            // read current directory
            for (tstrip_t strip = 0; strip < TIFFNumberOfStrips(aTiff.iFile); ++strip) {
                tmsize_t readLen = TIFFReadEncodedStrip(aTiff.iFile, strip, (&mesh.mesh[0] + currentOffset), (tsize_t) -1 /* read as much as possible */);
                currentOffset += readLen/sizeof(T);
            }
        }

        // Set proper dimensions (x and y are exchanged giving transpose w.r.t. original file)
        mesh.z_num = aTiff.iNumberOfDirectories;
        mesh.y_num = aTiff.iImgWidth;
        mesh.x_num = aTiff.iImgHeight;

        return mesh;
    }

    template<typename T>
    void saveMeshAsTiff(const std::string &aFileName, const MeshData<T> &aData) {
        TIFF *tif = TIFFOpen(aFileName.c_str() , "w8");

        // Set proper dimensions (x and y are exchanged giving transpose)
        const uint32_t width = aData.y_num;
        const uint32_t height = aData.x_num;
        const uint32_t depth = aData.z_num;
        const uint16_t samplesPerPixel = 1;
        const uint16_t bitsPerSample = sizeof(T) * 8;

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
        std::cout << "ScanlineSize=" << ScanlineSize << " StripSize: " << StripSize << " NoOfStrips: " << TIFFNumberOfStrips(tif) << std::endl;

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
                tmsize_t writeLen = TIFFWriteEncodedStrip(tif, strip, (void *) (&aData.mesh[0] + currentOffset), dataLen >= StripSize ? StripSize : dataLen);
                dataLen -= writeLen;
                currentOffset += writeLen/sizeof(T);
            }

            TIFFWriteDirectory(tif);
        }

        TIFFClose(tif);
    }
}

#endif
