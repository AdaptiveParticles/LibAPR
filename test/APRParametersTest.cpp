#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include <streambuf>
#include "algorithm/APRParameters.hpp" 

class APRParametersTest : public ::testing::Test {
protected:
    // This can be used to install a custom std::cerr buffer
    std::streambuf* original_cerr;

    std::stringstream cerr_content;

    void SetUp() override {
        // Before each test, backup the stream buffer for std::cerr
        original_cerr = std::cerr.rdbuf();
        // Redirect std::cerr to our stringstream buffer or any other stream
        std::cerr.rdbuf(cerr_content.rdbuf());
    }

    void TearDown() override {
        // Restore the original buffer before exiting the test
        std::cerr.rdbuf(original_cerr);
    }

    // Helper function to check if the warning message is in the buffer
    bool hasWarningMessage(const std::string& message) {
        return cerr_content.str().find(message) != std::string::npos;
    }
};

TEST_F(APRParametersTest, ValidateParametersWarnsOnZeroSigmaTh) {
    APRParameters params;
    params.sigma_th = 0; // Set sigma_th to zero to trigger the warning

    params.validate_parameters(); // Call the validation method

    // Check if the correct warning message was logged to std::cerr
    EXPECT_TRUE(hasWarningMessage("Warning: sigma_th is set to 0"));
}