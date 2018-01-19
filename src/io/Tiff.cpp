#include "Tiff.hpp"
#include <iostream>
#include <src/data_structures/Mesh/MeshData.hpp>

bool Tiff::open(const std::string &aFileName) {
    iFileName = aFileName;
    iFile = TIFFOpen(aFileName.c_str(), "r");
    if (iFile == nullptr) {
        std::cout << "Could not open file" << std::endl;
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
    }
    else if (iBitsPerSample == 16 && iSampleFormat == SAMPLEFORMAT_UINT) {
        iType = TiffType::TIFF_UINT16;
    }
    else if (iBitsPerSample == 32 && iSampleFormat == SAMPLEFORMAT_IEEEFP) {
        iType = TiffType::TIFF_FLOAT;
    }
    else {
        std::cout << "Not supported type of TIFF" << std::endl;
        iType = TiffType::TIFF_INVALID;
    }

    if (iPhotometric != PHOTOMETRIC_MINISBLACK) {
        std::cout << "Only grayscale images are supported" << std::endl;
        iType = TiffType ::TIFF_INVALID;
    }

    if (iType == TiffType::TIFF_INVALID) {
        close();
        return false;
    }

    return true;
}

void Tiff::close() {
    if (iFile != nullptr) {
        TIFFClose(iFile);
        iFile = nullptr;
    }
}

void Tiff::printInfo() {
    if (iFile == nullptr) return;

    std::cout << "----- TIFF INFO for file: [" << iFileName << "] -----" << std::endl;
    std::cout << "Width/Height/Depth: " << iImgWidth << "/" << iImgHeight << "/" << iNumberOfDirectories << std::endl;
    std::cout << "Samples per pixel: " << iSamplesPerPixel << " Bits per sample: " << iBitsPerSample << std::endl;
    std::cout << "ImageType: ";
    switch(iType) {
        case TiffType::TIFF_UINT8:
            std::cout << "uint8";
            break;
        case TiffType::TIFF_UINT16:
            std::cout << "uint16";
            break;
        case TiffType::TIFF_FLOAT:
            std::cout << "float";
            break;
        default:
            std::cout << "NOT SUPPORTED";
    }
    std::cout << std::endl;
    std::cout << "Photometric: " << iPhotometric << std::endl;
}

void Tiff::getMesh(int aStartZ, int aEndZ) {

}
