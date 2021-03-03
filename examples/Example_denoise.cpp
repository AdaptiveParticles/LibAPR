//
// Created by bevan on 29/11/2020.
//


#include "Example_denoise.hpp"

int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptionsDenoise options = read_command_line_options(argc, argv);

    // Seperated for testing.
    denoise_example(options);

}

