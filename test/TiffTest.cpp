/*
 * Created by Krzysztof Gonciarz 2018
 */
#include <gtest/gtest.h>
#include "io/TiffUtils.hpp"

namespace {
    std::string testFilesDirectory(){
        std::string testDir = std::string(__FILE__);
        return testDir.substr(0, testDir.find_last_of("\\/") + 1);
    }

    TEST(TiffTest, LoadUint8) {
        const MeshData<uint8_t> mesh = TiffUtils::getMesh<uint8_t>(testFilesDirectory() + "files/tiffTest/4x3x2x8bit.tif");
        for (int i = 0; i < 24; ++i) {
            ASSERT_EQ(mesh.mesh[i], i + 1);
        }
    }

    TEST(TiffTest, LoadUint8PreallocatedMesh) {
        TiffUtils::TiffInfo t1(testFilesDirectory() + "files/tiffTest/4x3x2x8bit.tif");
        std::cout << t1 << std::endl;

        MeshData<uint8_t> meshIn(t1.iImgHeight, t1.iImgWidth, t1.iNumberOfDirectories);
        TiffUtils::getMesh<uint8_t>(t1, meshIn);
        for (int i = 0; i < 24; ++i) {
            ASSERT_EQ(meshIn.mesh[i], i + 1);
        }
    }

    TEST(TiffTest, LoadUint16) {
        std::string fileName = testFilesDirectory() + "files/tiffTest/3x2x4x16bit.tif";
        TiffUtils::TiffInfo t1(fileName);
        std::cout << t1 << std::endl;
        ASSERT_STREQ(t1.toString().c_str(), ("FileName: [" + fileName + "], Width/Height/Depth: 3/2/4, SamplesPerPixel: 1, Bits per sample: 16, ImageType: uint16, Photometric: 1, StripSize: 12").c_str());
        ASSERT_EQ(t1.isFileOpened(), true);

        const MeshData<uint16_t> &mesh = TiffUtils::getMesh<uint16_t>(t1);
        ASSERT_EQ(mesh.x_num, 2);
        ASSERT_EQ(mesh.y_num, 3);
        ASSERT_EQ(mesh.z_num, 4);
        ASSERT_EQ(mesh.mesh.size(), 24);
        for (int i = 0; i < 24; ++i) {
            ASSERT_EQ(mesh.mesh[i], i + 1);
        }
    }

    TEST(TiffTest, LoadFloat) {
        TiffUtils::TiffInfo t1(testFilesDirectory() + "files/tiffTest/2x4x3xfloat.tif");
        std::cout << t1 << std::endl;

        const MeshData<float> &mesh = TiffUtils::getMesh<float>(t1);
        ASSERT_EQ(mesh.x_num, 4);
        ASSERT_EQ(mesh.y_num, 2);
        ASSERT_EQ(mesh.z_num, 3);
        ASSERT_EQ(mesh.mesh.size(), 24);
        for (int i = 0; i < 24; ++i) {
            ASSERT_EQ(mesh.mesh[i], i + 1);
        }
    }

    TEST(TiffTest, NotExistingFile) {
        TiffUtils::TiffInfo t("/tmp/forSureThisFileDoesNotExists.tiff666");
        ASSERT_STREQ(t.toString().c_str(), "<File not opened>");
        ASSERT_EQ(t.isFileOpened(), false);
    }

    TEST(TiffTest, TiffSave) {
        // Test reads test tiff file and then saves it in temp directory
        // Then reads it again and compares input file and save file if same
        typedef uint16_t ImgType;
        TiffUtils::TiffInfo t(testFilesDirectory() + "files/tiffTest/3x2x4x16bit.tif");
        MeshData<ImgType> mesh = TiffUtils::getMesh<ImgType>(t);
        ASSERT_EQ(t.isFileOpened(), true);

        std::string fileName = "/tmp/testAprTiffSave" + std::to_string(time(nullptr)) + ".tif";
        TiffUtils::saveMeshAsTiff(fileName, mesh);

        TiffUtils::TiffInfo t2(fileName);
        ASSERT_EQ(t2.isFileOpened(), true);
        const MeshData<ImgType> &mesh2 = TiffUtils::getMesh<ImgType>(t2);

        ASSERT_EQ(mesh.mesh.size(), mesh2.mesh.size());
        for (size_t i = 0; i < mesh.mesh.size(); ++i)
            ASSERT_EQ(mesh.mesh[i], mesh2.mesh[i]);

        if (remove(fileName.c_str()) != 0) {
            std::cerr << "Could not remove file [" << fileName << "]" << std::endl;
        }
    }

    TEST(TiffTest, TiffSaveUint16) {
        // Test reads test tiff file and then saves it in temp directory
        // Then reads it again and compares input file and save file if same
        TiffUtils::TiffInfo t(testFilesDirectory() + "files/tiffTest/2x4x3xfloat.tif");
        ASSERT_EQ(t.isFileOpened(), true);
        MeshData<float> mesh = TiffUtils::getMesh<float>(t);

        std::string fileName = "/tmp/testAprTiffSave" + std::to_string(time(nullptr)) + ".tif";
        TiffUtils::saveMeshAsTiffUint16(fileName, mesh);

        TiffUtils::TiffInfo t2(fileName);
        ASSERT_EQ(t2.isFileOpened(), true);
        const MeshData<uint16_t> &mesh2 = TiffUtils::getMesh<uint16_t>(t2);

        ASSERT_EQ(mesh.mesh.size(), mesh2.mesh.size());
        for (size_t i = 0; i < mesh.mesh.size(); ++i)
            ASSERT_EQ(mesh.mesh[i], mesh2.mesh[i]);

        if (remove(fileName.c_str()) != 0) {
            std::cerr << "Could not remove file [" << fileName << "]" << std::endl;
        }
    }
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
