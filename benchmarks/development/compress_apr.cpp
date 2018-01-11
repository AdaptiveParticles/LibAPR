//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Examples of simple iteration an access to Particle Cell, and particle information. (See Example_neigh, for neighbor access)
///
/// Usage:
///
/// (using output of Example_get_apr)
///
/// Example_apr_iterate -i input_image_tiff -d input_directory
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>

#include "benchmarks/development/compress_apr.h"

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
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
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

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    Part_timer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    APRCompress comp;

    comp.compress(apr);



//    float e = 1.6;
//    float background = 1;
//    float cnv = 65636/30000;
//    float q = .5;
//
//
//    Mesh_data<uint16_t> pc;
//    apr.interp_img(pc,apr.particles_int);
//    std::string output = options.directory + "pc.tif";
//    pc.write_image_tiff(output);
//
//
//    //Create particle datasets, once intiailized this has the same layout as the Particle Cells
//    ExtraPartCellData<float> var_scaled(apr);
//
//
//
//    //Basic serial iteration over all particles
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//        // multiple the Particle Cell type by the particle intensity (the intensity is stored as a ExtraPartCellData and therefore is no different from any additional datasets)
//        apr(var_scaled) = 2*sqrt(std::max((float) apr(apr.particles_int)-background,(float)0)/(cnv) + pow(e,2)) - 2*e;
//
//        apr(var_scaled) = apr(var_scaled)/(q);
//    }
//
//    //predition step
//
//    ExtraPartCellData<float> prediction(apr);
//
//    std::vector<unsigned int> dir = {1,3,5};
//
//    APR_iterator<uint16_t> neigh_it(apr);
//
//    //loops from lowest level to highest
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//
//        //get the minus neighbours (1,3,5)
//
//        //now we only update the neighbours, and directly access them through a neighbour iterator
//        apr.update_all_neighbours();
//
//        float counter = 0;
//        float temp = 0;
//
//        //loop over all the neighbours and set the neighbour iterator to it
//        for (int f = 0; f < dir.size(); ++f) {
//            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//            unsigned int face = dir[f];
//
//            for (int index = 0; index < apr.number_neighbours_in_direction(face); ++index) {
//                // on each face, there can be 0-4 neighbours accessed by index
//                if(neigh_it.set_neighbour_iterator(apr, face, index)){
//                    //will return true if there is a neighbour defined
//                    if(neigh_it.depth() <= apr.depth()) {
//
//                        temp += neigh_it(var_scaled);
//                        counter++;
//                    }
//
//                }
//            }
//        }
//
//        if(counter > 0){
//            apr(prediction) = apr(var_scaled) - temp/counter;
//        } else {
//            apr(prediction) = apr(var_scaled);
//        }
//
//    }
//
////    Mesh_data<float> pimg;
////
////    apr.interp_img(pimg,prediction);
////
////    std::string name = options.directory + "pimg.tif";
////    pimg.write_image_tiff(name);
////
////    apr.interp_img(pimg,var_scaled);
////
////    name = options.directory + "varscale.tif";
////    pimg.write_image_tiff(name);
//
//
//
//    ExtraPartCellData<uint16_t> symbols(apr);
//
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//
//        int16_t val = apr(prediction);
//
//        apr(symbols) = 2*(abs(val)) + (val >> 15);
//
//    }
//
//    apr.write_particles_only(options.directory,"symbols",symbols);
//
////    apr.write_particles_only(options.directory,"prediction",prediction);
////
////    apr.write_particles_only(options.directory,"var_scaled",var_scaled);
//
//    apr.write_particles_only(options.directory,"original",apr.particles_int);
//
//    ExtraPartCellData<uint16_t> symbols_max(apr);
//
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//        if(apr.depth() == apr.depth_max()) {
//            apr(symbols_max) = apr(symbols);
//        } else {
//            apr(symbols_max) = apr(apr.particles_int);
//        }
//
//
//    }
//
//    apr.write_particles_only(options.directory,"symbols_max",symbols_max);
//
//
//
//    //Convert back from symbols to signed
//
//    ExtraPartCellData<float> unsymbol(apr);
//
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//
//        int16_t negative = apr(symbols) % 2;
//
//        apr(unsymbol) = (1 - 2 * negative) * ((apr(symbols) + negative) / 2);
//
//    }
//
////    apr.interp_img(pimg,unsymbol);
////
////    name = options.directory + "un_prediction.tif";
////    pimg.write_image_tiff(name);
//
//
//    ExtraPartCellData<float> prediction_reverse(apr);
//
//    //need re predict here.
//
//
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//        if(apr.depth() < apr.depth_max()) {
//            apr(unsymbol) = apr(prediction);
//        }
//
//
//    }
//
//
//
//
//        for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//
//            //get the minus neighbours (1,3,5)
//
//            //now we only update the neighbours, and directly access them through a neighbour iterator
//            apr.update_all_neighbours();
//
//            float counter = 0;
//            float temp = 0;
//
//            //loop over all the neighbours and set the neighbour iterator to it
//            for (int f = 0; f < dir.size(); ++f) {
//                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//                unsigned int face = dir[f];
//
//                for (int index = 0; index < apr.number_neighbours_in_direction(face); ++index) {
//                    // on each face, there can be 0-4 neighbours accessed by index
//                    if(neigh_it.set_neighbour_iterator(apr, face, index)){
//                        //will return true if there is a neighbour defined
//                        if(neigh_it.depth() <= apr.depth()) {
//                            temp += neigh_it(prediction_reverse);
//                            counter++;
//                        }
//
//                    }
//                }
//            }
//
//            if(counter > 0){
//                apr(prediction_reverse) = apr(unsymbol) + temp/counter;
//            } else {
//                apr(prediction_reverse) = apr(unsymbol);
//            }
//
//        }
//
//
//
////    apr.interp_img(pimg,prediction_reverse);
////
////    name = options.directory + "after_prediction.tif";
////    pimg.write_image_tiff(name);
//
//    ExtraPartCellData<uint16_t> recon(apr);
//
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//
//        float D = q*apr(prediction_reverse) + 2*e;
//
//        if(D >= 2*e){
//            D = (pow(D,2)/4.0 - pow(e,2))*cnv + background;
//            apr(recon) = (uint16_t) D;
//        } else {
//            apr(recon) = background;
//        }
//    }
//
//
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//        if(apr.depth() == apr.depth_max()) {
//        } else {
//            apr(recon) = apr(apr.particles_int);
//        }
//    }
//
//
//    Mesh_data<uint16_t> img;
//
//    apr.interp_img(img,recon);
//    std::string name = options.directory + "decomp.tif";
//    img.write_image_tiff(name);
//
////    apr.write_particles_only(options.directory,"recon",recon);
//
//
//    float max_level = 0;
//    float total = 0;
//
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//        if(apr.depth_max() == apr.depth()){
//
//            max_level++;
//        }
//        total++;
//
//
//    }
//    std::cout << max_level/total << std::endl;
//
////
//    Mesh_data<uint8_t> level;
//    apr.interp_depth(level);
//    name = options.directory + "level.tif";
//
//    level.write_image_tiff(name);

    //    void vstCPU(float* in, float* out, int num, float offset, float conversion, float sigma)
//    {
//        //#pragma omp parallel for
//        for (int x = 0; x < num; x++)
//        {
//            out[x] = 2 * sqrtf((fmaxf(in[x] - offset, 0)) / conversion + sigma*sigma) - 2 * sigma;
//        }
//        return;
//    }
//
//    void invVstCPU(float* in, float* out, int num, float offset, float conversion, float sigma)
//    {
//        float D = 0;
//        //#pragma omp parallel for
//        for (int x = 0; x < num; x++)
//        {
//            D = in[x];
//            D = D + 2 * sigma; // remove offset
//            if (D >= 2 * sigma) {
//                out[x] = ((D*D / 4) - sigma*sigma)*conversion + offset;
//            }
//            else {
//                out[x] = offset;
//            }
//        }
//        return;
//    }
//
//    // convert signed shorts to symbols (>= 0 -> even, < 0 -> odd)
//    void symbolizeCPU(ushort* pSymbols, const short* pData, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchSrc, uint slicePitchSrc)
//    {
//        for (size_t i = 0; i < sizeX*sizeY*sizeZ; i++)
//        {
//            pSymbols[i] = 2 * abs(pData[i]) + getNegativeSign(pData[i]);
//        }
//        return;
//    }
//
//    void unsymbolizeCPU(short* pData, const ushort* pSymbols, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchDst, uint slicePitchDst)
//    {
//        int negative = 0;
//        for (size_t i = 0; i < sizeX*sizeY*sizeZ; i++)
//        {
//            negative = pSymbols[i] % 2;
//            pData[i] = (1 - 2 * negative) * ((pSymbols[i] + negative) / 2);
//        }
//        return;
//    }

}


