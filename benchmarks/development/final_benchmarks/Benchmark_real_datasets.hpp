//
// Created by cheesema on 25.01.18.
//

#ifndef PARTPLAY_BENCHMARK_REAL_DATASETS_HPP
#define PARTPLAY_BENCHMARK_REAL_DATASETS_HPP


#include "benchmarks/development/final_benchmarks/BenchHelper.hpp"

#include <functional>
#include <string>

#include "src/data_structures/APR/APR.hpp"
#include "benchmarks/development/final_benchmarks/APRBenchmark.hpp"

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    bool stats_file = false;
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);


#endif //PARTPLAY_BENCHMARK_REAL_DATASETS_HPP
