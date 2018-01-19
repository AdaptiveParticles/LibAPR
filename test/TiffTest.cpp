#include <gtest/gtest.h>
#include "src/io/Tiff.hpp"

namespace {
    TEST(TiffTest, PrintInfoTest) {
        Tiff t;
        t.printTiffInfo("/hahaha");
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
