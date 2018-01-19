#include <gtest/gtest.h>
#include "src/io/Tiff.hpp"

namespace {
    TEST(TiffTest, PrintInfoTest) {
        Tiff t1("/Users/gonciarz/Documents/MOSAIC/work/repo/APR_FILES/101x122x8bit.tif");
        t1.printInfo();

        Tiff t2("/Users/gonciarz/Documents/MOSAIC/work/repo/APR_FILES/102x121x16bit.tif");
        t2.printInfo();

        Tiff t3("/Users/gonciarz/Documents/MOSAIC/work/repo/APR_FILES/102x122x3xfloat.tif");
        t3.printInfo();

        Tiff t4("/Users/gonciarz/Documents/MOSAIC/work/repo/APR_FILES/big.tif");
        t4.printInfo();

        Tiff t5("/Users/gonciarz/Documents/MOSAIC/work/repo/APR_FILES/10x20x30xRGB.tif");
        t5.printInfo();

        Tiff t6("/Users/gonciarz/Documents/MOSAIC/work/repo/APR_FILES/10x20x30xRGB.tiff");
        t6.printInfo();

    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
