#ifndef TIFF_HPP
#define TIFF_HPP

#include <string>
#include <tiffio.h>

class Tiff {
public:
    Tiff(const std::string &aFileName) { open(aFileName); }
    ~Tiff() { close(); }

    void printInfo();

//    void write_image_tiff(std::string& filename);
//    void write_image_tiff_uint16(std::string& filename);
    void getMesh(int aStartZ = 0, int aEndZ = -1);

private:
    enum class TiffType {TIFF_UINT8, TIFF_UINT16, TIFF_FLOAT, TIFF_INVALID};

    bool open(const std::string &aFileName);
    void close();

    TiffType iType = TiffType::TIFF_INVALID;
    TIFF* iFile = nullptr;
    std::string iFileName = "";
    uint32 iImgWidth = 0;
    uint32 iImgHeight = 0;
    uint32 iNumberOfDirectories = 0;
    unsigned short iSamplesPerPixel = 0;
    unsigned short iBitsPerSample = 0;
    unsigned short iSampleFormat = 0;
    unsigned short iPhotometric = 0;
};

#endif
