/*
 * Created by Krzysztof Gonciarz 2018
 */
#include <array>
#include <gtest/gtest.h>
#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/ComputeGradient.hpp"
#include <random>
#include "TestTools.hpp"

namespace {

    TEST(ComputeGradientTest, 2D_XY) {
        {   // Corner points
            PixelData<float> m(6, 6, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {1.41, 0, 4.24,
                              0, 0, 0,
                              2.82, 0, 5.65};
            // put values in corners
            m(0, 0, 0) = 2;
            m(5, 0, 0) = 4;
            m(0, 5, 0) = 6;
            m(5, 5, 0) = 8;
            PixelData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.05));
        }
        {   // In the middle
            PixelData<float> m(6, 6, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {1, 1, 0,
                              1, 0, 0,
                              0, 0, 0};
            // put values in corners
            m(1, 1, 0) = 2;
            PixelData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.01));
        }
        {   // One pixel image 1x1x1
            PixelData<float> m(1, 1, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0};
            // put values in corners
            m(0, 0, 0) = 2;
            PixelData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.01));
        }

    }

    TEST(ComputeGradientTest, Corners3D) {
        PixelData<float> m(6, 6, 4, 0);
        // expect gradient is 3x3x2 X/Y/Z plane
        float expect[] = {1.73, 0, 5.19,
                          0, 0, 0,
                          3.46, 0, 6.93,

                          8.66, 0, 12.12,
                          0, 0, 0,
                          10.39, 0, 13.85};
        // put values in corners
        m(0, 0, 0) = 2;
        m(5, 0, 0) = 4;
        m(0, 5, 0) = 6;
        m(5, 5, 0) = 8;
        m(0, 0, 3) = 10;
        m(5, 0, 3) = 12;
        m(0, 5, 3) = 14;
        m(5, 5, 3) = 16;

        PixelData<float> grad;
        grad.initDownsampled(m, 0);
        ComputeGradient cg;
        cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);

        grad.printMesh(1);
        //m.printMesh(1);

        ASSERT_TRUE(compare(grad, expect, 0.05));
    }

    /*
     *
     * This is testing that a constant image returns a constant output.
     */
    TEST(ComputeGradientTest, 2D_XY_BSPLINE__CONSTANT_Y_DIR) {
        //Float tests
        {   // values in corners and in middle
            PixelData<float> m(5, 7, 1, 100);
            // expect gradient is constant
            PixelData<float> expect(5, 7, 1, 100);

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {
            PixelData<float> m(9, 3, 1, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(9, 3, 1, 100);

            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {   // two pixel image 1x2x1
            PixelData<float> m(2, 1, 1, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(2, 1, 1, 100);

            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }

        //uint16_t tests
        {   // values in corners and in middle
            PixelData<uint16_t> m(5, 7, 1, 100);
            // expect gradient is constant
            PixelData<float> expect(5, 7, 1, 100);

            // Calculate bspline on CPU
            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {
            PixelData<uint16_t> m(9, 3, 1, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(9, 3, 1, 100);

            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {   // two pixel image 1x2x1
            PixelData<uint16_t> m(2, 1, 1, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(2, 1, 1, 100);

            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }

    }


    /*
    *
    * This is testing that a constant image returns a constant output.
    */
    TEST(ComputeGradientTest, 2D_XY_BSPLINE__CONSTANT_X_DIR) {
        //Float tests
        {   // values in corners and in middle
            PixelData<float> m(5, 7, 1, 100);
            // expect gradient is constant
            PixelData<float> expect(5, 7, 1, 100);

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_x(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {
            PixelData<float> m(9, 3, 1, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(9, 3, 1, 100);

            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_x(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {   // two pixel image 1x2x1
            PixelData<float> m(1, 2, 1, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(1, 2, 1, 100);

            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_x(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }

        //uint16_t tests
        {   // values in corners and in middle
            PixelData<uint16_t> m(5, 7, 1, 100);
            // expect gradient is constant
            PixelData<float> expect(5, 7, 1, 100);

            // Calculate bspline on CPU
            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_x(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {
            PixelData<uint16_t> m(9, 3, 1, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(9, 3, 1, 100);

            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_x(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {
            PixelData<uint16_t> m(1, 2, 1, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(2, 1, 1, 100);

            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_x(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }

    }

    /*
   *
   * This is testing that a constant image returns a constant output.
   */
    TEST(ComputeGradientTest, 2D_XY_BSPLINE__CONSTANT_Z_DIR) {
        //Float tests
        {   // values in corners and in middle
            PixelData<float> m(5, 1, 7, 100);
            // expect gradient is constant
            PixelData<float> expect(5, 1, 7, 100);

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_z(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {
            PixelData<float> m(9, 1, 3, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(9, 1, 3, 100);

            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_z(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {   // two pixel image 1x2x1
            PixelData<float> m(1, 1, 2, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(1, 1, 2, 100);

            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_z(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }

        //uint16_t tests
        {   // values in corners and in middle
            PixelData<uint16_t> m(5, 1, 7, 100);
            // expect gradient is constant
            PixelData<float> expect(5, 1, 7, 100);

            // Calculate bspline on CPU
            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_z(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {
            PixelData<uint16_t> m(9, 1, 3, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(9, 1, 3, 100);

            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_z(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }
        {
            PixelData<uint16_t> m(1, 1, 2, 100);
            // expect gradient is 3x3 X/Y plane
            PixelData<float> expect(1, 1, 2, 100);

            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_z(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect.mesh.begin(), 0.01));
        }

    }

    TEST(ComputeGradientTest, 2D_XY_BSPLINE_Z_DIR) {


        {   // values in corners and in middle
            PixelData<uint16_t> m(5, 1, 7, 100);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {104,100,101,100,109,
                              103,100,101,100,107,
                              102,100,102,100,104,
                              103,100,103,100,105,
                              107,100,102,100,109,
                              112,100,101,100,116,
                              117,100,101,100,121};

            // put values in corners
            m(2, 0, 3) = 110;
            m(0, 0, 0) = 120;
            m(4, 0, 0) = 140;
            m(0, 0, 6) = 160;
            m(4, 0, 6) = 180;

            // Calculate bspline on CPU
            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_z(mCpu, 3.0, 0.0001);

            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }


        {   // values in corners and in middle
            PixelData<float> m(1, 5, 7, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.3598212600, 0.0000000000, 0.0897887424, 0.0000000000, 0.8728340864,
                    0.2939075828, 0.0000000000, 0.1207435802, 0.0000000000, 0.6849087477,
                    0.2319568098, 0.0000000000, 0.2009553164, 0.0000000000, 0.4642915130,
                    0.3356855512, 0.0000000000, 0.2707113624, 0.0000000000, 0.4915564358,
                    0.6802870631, 0.0000000000, 0.2038949281, 0.0000000000, 0.8787427545,
                    1.2156776190, 0.0000000000, 0.1250728220, 0.0000000000, 1.5540421009,
                    1.6544119120, 0.0000000000, 0.0915138945, 0.0000000000, 2.1367254257 };
            // put values in corners
            m(0, 2, 3) = 1;
            m(0, 0, 0) = 2;
            m(0, 4, 0) = 4;
            m(0, 0, 6) = 6;
            m(0, 4, 6) = 8;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_z(mCpu, 3.0, 0.0001);

            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
        {   // single point set in the middle
            PixelData<float> m(1, 9, 3, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.0000000000, 0.2294456065, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.2193282992, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.2930246294, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000 };
            // put values in corners
            m(0, 1, 2) = 1;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_z(mCpu, 3.0, 0.0001);

            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
        {   // two pixel image 1x2x1
            PixelData<float> m(1, 1, 2, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.78,
                              0.71};
            // put values in corners
            m(0, 0, 0) = 1;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_z(mCpu, 3.0, 0.0001);

            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
    }



    TEST(ComputeGradientTest, 2D_XY_BSPLINE_X_DIR) {


        {   // values in corners and in middle
            PixelData<uint16_t> m(5, 7, 1, 100);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {104, 103, 102, 103, 107, 112, 117,
                    100, 100, 100, 100, 100, 100, 100,
                    101, 101, 102, 103, 102, 101, 101,
                    100, 100, 100, 100, 100, 100, 100,
                    109, 107, 104, 105, 109, 116, 121};
            // put values in corners
            m(2, 3, 0) = 110;
            m(0, 0, 0) = 120;
            m(4, 0, 0) = 140;
            m(0, 6, 0) = 160;
            m(4, 6, 0) = 180;

            // Calculate bspline on CPU
            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_x(mCpu, 3.0, 0.0001);



            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }


        {   // values in corners and in middle
            PixelData<float> m(5, 7, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.3598212600, 0.2939075828, 0.2319568098, 0.3356855512, 0.6802870631, 1.2156776190, 1.6544119120,
                    0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000,
                    0.0897887424, 0.1207435802, 0.2009553164, 0.2707113624, 0.2038949281, 0.1250728220, 0.0915138945,
                    0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000,
                    0.8728340864, 0.6849087477, 0.4642915130, 0.4915564358, 0.8787427545, 1.5540421009, 2.1367254257};
            // put values in corners
            m(2, 3, 0) = 1;
            m(0, 0, 0) = 2;
            m(4, 0, 0) = 4;
            m(0, 6, 0) = 6;
            m(4, 6, 0) = 8;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_x(mCpu, 3.0, 0.0001);

            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
        {   // single point set in the middle
            PixelData<float> m(9, 3, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    0.3793833256, 0.4131873846, 0.4386565983,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.0000000000, 0.0000000000};
            // put values in corners
            m(4, 1, 0) = 1;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_x(mCpu, 3.0, 0.0001);

            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
        {   // two pixel image 1x2x1
            PixelData<float> m(1, 2, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.78,
                              0.71};
            // put values in corners
            m(0, 0, 0) = 1;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_x(mCpu, 3.0, 0.0001);

            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
    }


    TEST(ComputeGradientTest, 2D_XY_BSPLINE_Y_DIR) {


        {   // values in corners and in middle
            PixelData<uint16_t> m(5, 7, 1, 100);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {105, 100, 100, 102, 100, 100, 115,
                              105, 100, 100, 103, 100, 100, 113,
                              106, 100, 100, 104, 100, 100, 113,
                              108, 100, 100, 103, 100, 100, 116,
                              110, 100, 100, 102, 100, 100, 120};
            // put values in corners
            m(2, 3, 0) = 110;
            m(0, 0, 0) = 120;
            m(4, 0, 0) = 140;
            m(0, 6, 0) = 160;
            m(4, 6, 0) = 180;

            // Calculate bspline on CPU
            PixelData<uint16_t> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);

            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }


        {   // values in corners and in middle
            PixelData<float> m(5, 7, 1, 0); 
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.47, 0.00, 0.00, 0.23, 0.00, 0.00, 1.53,
                              0.46, 0.00, 0.00, 0.26, 0.00, 0.00, 1.32,
                              0.57, 0.00, 0.00, 0.31, 0.00, 0.00, 1.30,
                              0.83, 0.00, 0.00, 0.26, 0.00, 0.00, 1.62,
                              1.04, 0.00, 0.00, 0.24, 0.00, 0.00, 1.96};
            // put values in corners
            m(2, 3, 0) = 1;
            m(0, 0, 0) = 2;
            m(4, 0, 0) = 4;
            m(0, 6, 0) = 6;
            m(4, 6, 0) = 8;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg; 
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);



            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }   
        {   // single point set in the middle
            PixelData<float> m(9, 3, 1, 0); 
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.00, 0.01, 0.00,
                              0.00, 0.03, 0.00,
                              0.00, 0.10, 0.00,
                              0.00, 0.21, 0.00,
                              0.00, 0.28, 0.00,
                              0.00, 0.20, 0.00,
                              0.00, 0.10, 0.00,
                              0.00, 0.03, 0.00,
                              0.00, 0.01, 0.00};
            // put values in corners
            m(4, 1, 0) = 1;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg; 
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);


            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }   
        {   // two pixel image 1x2x1
            PixelData<float> m(2, 1, 1, 0); 
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.78,
                              0.71};
            // put values in corners
            m(0, 0, 0) = 1;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg; 
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);

            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }   
    }   

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Y) {
        using ImgType = float;

        ImgType init[] =   {1.00, 0.00, 0.00,
                            1.00, 0.00, 6.00,
                            0.00, 6.00, 0.00,
                            6.00, 0.00, 0.00};

        ImgType expect[] = {1.00, 0.00, 2.00,
                            0.83, 1.00, 4.00,
                            1.17, 4.00, 1.00,
                            4.00, 2.00, 0.00};

        PixelData<ImgType> m(4, 3, 1);
        initFromZYXarray(m, init);

        // Calculate and compare
        ComputeGradient().calc_inv_bspline_y(m);
        ASSERT_TRUE(compare(m, expect, 0.01));
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_X) {
        using ImgType = float;

        ImgType init[] =   {0.00, 6.00, 0.00,
                            1.00, 0.00, 0.00,
                            0.00, 0.00, 1.00};

        ImgType expect[] = {2.00, 4.00, 2.00,
                            0.67, 0.16, 0.00,
                            0.00, 0.16, 0.67};

        PixelData<ImgType> m(3, 3, 1);
        initFromZYXarray(m, init);

        // Calculate and compare
        ComputeGradient().calc_inv_bspline_x(m);
        ASSERT_TRUE(compare(m, expect, 0.01));
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Z) {
        using ImgType = float;

        ImgType init[] =   {0.00, 6.00, 0.00,
                            1.00, 0.00, 0.00,
                            0.00, 0.00, 1.00};

        ImgType expect[] = {0.3333333433, 4.0000000000, 0.0000000000,
                0.6666666865, 1.0000000000, 0.1666666716,
                0.3333333433, 0.0000000000, 0.6666666865};

        PixelData<ImgType> m(1, 3, 3);
        initFromZYXarray(m, init);

        // Calculate and compare
        ComputeGradient().calc_inv_bspline_z(m);

        ASSERT_TRUE(compare(m, expect, 0.01));
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
