//
// Created by cheesema on 25.01.18.
//

#include "Benchmark_real_datasets.hpp"


bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_apr_iterate -i input_apr_file -d directory\"" << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;

}


int main(int argc, char **argv) {

    BenchHelper benchHelper;

    cmdLineOptions options = read_command_line_options(argc,argv);

    std::vector<std::string> file_list = benchHelper.listFiles( options.directory,".tif");
    APRBenchmark apr_benchmarks;

    for (int j = 0; j <file_list.size(); j++) {

        std::string image_file_name = options.directory + file_list[j];

        std::string image_name  = file_list[j];
        image_name.erase(image_name.find_last_of("."), std::string::npos);

        std::string stats_file_name = image_name + "_stats.txt";

        if (benchHelper.check_file_exists(options.directory + stats_file_name)) {
            //check that the stats file exists

            Proc_par old_pars;

            get_image_stats(old_pars, options.directory, stats_file_name);

            APRConverter<uint16_t> apr_converter;

            apr_converter.par.lambda = old_pars.lambda;
            apr_converter.par.rel_error = old_pars.rel_error;
            apr_converter.par.min_signal = old_pars.var_th;
            apr_converter.par.Ip_th = old_pars.I_th;

            apr_converter.par.input_image_name = image_name + ".tif";
            apr_converter.par.input_dir = options.directory;

            apr_benchmarks.benchmark_dataset(apr_converter);

        }
    }
    apr_benchmarks.analysis_data.file_name = options.directory + "analysis_data/real_data_" + apr_benchmarks.analysis_data.file_name;
    apr_benchmarks.analysis_data.write_analysis_data_hdf5();

}