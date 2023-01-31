//
// Created by Krzysztof Gonciarz on 4/10/18.
//

#include <gtest/gtest.h>
#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/LocalIntensityScale.hpp"
#include "TestTools.hpp"


namespace {
    TEST(LocalIntensityScaleTest, 1D_Y_DIR) {
        {   // OFFSET=0

            PixelData<float> m(8, 1, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {3,6,9,12,15,18,21,24};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_y(m, 0);

            ASSERT_TRUE(compare(m, expect, 0.000001));
        }
        {   // OFFSET=1

            PixelData<float> m(8, 1, 1, 0);
            float dataIn[] = {1, 2, 3, 4, 5, 6, 7, 8};
            float expect[] = {1.5, 2, 3, 4, 5, 6, 7, 7.5};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_y(m, 1);

            ASSERT_TRUE(compare(m, expect, 0.000001));
        }
        {   // OFFSET=2 (+symmetricity check)

            PixelData<float> m(8, 1, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {6, 7.5, 9, 12, 15, 18, 19.5, 21};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_y(m, 2);

            ASSERT_TRUE(compare(m, expect, 0.000001));

            // check if data in opposite order gives same result
            float dataIn2[] = {24,21,18,15,12,9,6,3};
            float expect2[] = {21, 19.5, 18, 15,12, 9, 7.5, 6};

            initFromZYXarray(m, dataIn2);

            lis.calc_sat_mean_y(m, 2);

            ASSERT_TRUE(compare(m, expect2, 0.000001));
        }
    }

    TEST(LocalIntensityScaleTest, 1D_X_DIR) {
        {   // OFFSET=0

            PixelData<float> m(1, 8, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {3,6,9,12,15,18,21,24};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_x(m, 0);

            ASSERT_TRUE(compare(m, expect, 0.000001));
        }
        {   // OFFSET=1

            PixelData<float> m(1, 8, 1, 0);
            float dataIn[] = {1, 2, 3, 4, 5, 6, 7, 8};
            float expect[] = {1.5, 2, 3, 4, 5, 6, 7, 7.5};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_x(m, 1);

            ASSERT_TRUE(compare(m, expect, 0.000001));
        }
        {   // OFFSET=2 (+symmetricity check)

            PixelData<float> m(1, 8, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {6, 7.5, 9, 12, 15, 18, 19.5, 21};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_x(m, 2);

            ASSERT_TRUE(compare(m, expect, 0.000001));

            // check if data in opposite order gives same result
            float dataIn2[] = {24,21,18,15,12,9,6,3};
            float expect2[] = {21, 19.5, 18, 15,12, 9, 7.5, 6};

            initFromZYXarray(m, dataIn2);

            lis.calc_sat_mean_x(m, 2);

            ASSERT_TRUE(compare(m, expect2, 0.000001));
        }
    }

    TEST(LocalIntensityScaleTest, 1D_Z_DIR) {
        {   // OFFSET=0

            PixelData<float> m(1, 1, 8, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {3,6,9,12,15,18,21,24};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_z(m, 0);

            ASSERT_TRUE(compare(m, expect, 0.000001));
        }
        {   // OFFSET=1

            PixelData<float> m(1, 1, 8, 0);
            float dataIn[] = {1, 2, 3, 4, 5, 6, 7, 8};
            float expect[] = {1.5, 2, 3, 4, 5, 6, 7, 7.5};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_z(m, 1);

            ASSERT_TRUE(compare(m, expect, 0.000001));
        }
        {   // OFFSET=2 (+symmetricity check)

            PixelData<float> m(1, 1, 8, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {6, 7.5, 9, 12, 15, 18, 19.5, 21};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_z(m, 2);

            ASSERT_TRUE(compare(m, expect, 0.000001));

            // check if data in opposite order gives same result
            float dataIn2[] = {24,21,18,15,12,9,6,3};
            float expect2[] = {21, 19.5, 18, 15,12, 9, 7.5, 6};

            initFromZYXarray(m, dataIn2);

            lis.calc_sat_mean_z(m, 2);

            ASSERT_TRUE(compare(m, expect2, 0.000001));
        }
    }

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
