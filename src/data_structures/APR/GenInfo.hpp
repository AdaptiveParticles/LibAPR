//
// Created by cheesema on 2019-06-04.
//

#ifndef LIBAPR_GENINFO_HPP
#define LIBAPR_GENINFO_HPP


#include <sstream>
#include <vector>
#include <cmath>

//Note this function sets up the domain for the APR for a given input size.
class GenInfo {

    //computes a power then uses round (this was requried due to casting differences across sytems)
    double powr(uint64_t num,uint64_t pow2){
        //return (uint64_t) std::round(std::pow(num,pow2));
        return std::round(pow(num,pow2));
    }

public:
    int l_min = 0;
    int l_max = -1;
    int org_dims[3] = {0,0,0};

    uint8_t number_dimensions = 0;

    std::vector<int> x_num;
    std::vector<int> y_num;
    std::vector<int> z_num;

    uint64_t total_number_particles = 0;

    std::vector<int> level_size; // precomputation of the size of each level, used by the iterators.

    //initialize the information given the original dimensions
    void init(uint64_t y_org,uint64_t x_org,uint64_t z_org){

        org_dims[0] = y_org;
        org_dims[1] = x_org;
        org_dims[2] = z_org;

        number_dimensions = (y_org> 1) + (x_org > 1) + (z_org > 1);

        int max_dim = std::max(std::max(org_dims[1], org_dims[0]), org_dims[2]);

        int levelMax = std::max(1, (int) ceil(std::log2(max_dim)));

        // APR l_min is 1, tree l_min is 0, enabling the parent level to always exist
        l_min = 1;
        l_max = levelMax;

        y_num.resize(levelMax+1);
        x_num.resize(levelMax+1);
        z_num.resize(levelMax+1);

        level_size.resize(levelMax + 1);
        for (int k = 0; k <= levelMax; ++k) {
           level_size[k] = powr(2,levelMax - k);
        }


        for (int l = l_min; l <= l_max; ++l) {
            double cellSize = powr(2, l_max - l);
            y_num[l] = ceil(y_org / cellSize);
            x_num[l] = ceil(x_org / cellSize);
            z_num[l] = ceil(z_org / cellSize);
        }
    }

    //initialize the information given the original dimensions
    void init_tree(uint64_t y_org,uint64_t x_org,uint64_t z_org){

        org_dims[0] = y_org;
        org_dims[1] = x_org;
        org_dims[2] = z_org;

        number_dimensions = (y_org > 1) + (x_org > 1) + (z_org > 1);

        int max_dim = std::max(std::max(y_org, x_org), z_org);

        int levelMax = ceil(std::log2(max_dim));

        // APR l_min is 1, tree l_min is 0, enabling the parent level to always exist
        l_min = 0;
        l_max = std::max(levelMax-1,0);

        y_num.resize(l_max+1);
        x_num.resize(l_max+1);
        z_num.resize(l_max+1);

        level_size.resize(levelMax + 1);
        for (int k = 0; k <= levelMax; ++k) {
            level_size[k] = (int) powr(2,levelMax - k);
        }

        for (int l = l_min; l <= l_max; ++l) {
            double cellSize = powr(2, l_max - l + 1);
            y_num[l] = ceil(y_org / cellSize);
            x_num[l] = ceil(x_org / cellSize);
            z_num[l] = ceil(z_org / cellSize);
        }
    }

    friend std::ostream & operator<<(std::ostream &os, const GenInfo &gi) {
        os << "GenInfo {\n";
        os << "    Original dimensions(y/x/z): [" << gi.org_dims[0] << ", " << gi.org_dims[1] << ", " << gi.org_dims[2] << "]\n";
        os << "    Number of dimensions: " << static_cast<int>(gi.number_dimensions) << "\n";
        os << "    l_min, l_max: {" << gi.l_min << " - " << gi.l_max << "}\n";
        os << "    total number of particles: " << gi.total_number_particles << "\n";
        os << "    y_num, x_num, z_num:\n";
        for (int l = gi.l_min; l <= gi.l_max; ++l) {
            os << "        level [" << l << "] =  " << gi.y_num[l] << ", " << gi.x_num[l] << ", " << gi.z_num[l] << "\n";
        }
        os << "    level_size:\n";
        for (int l = gi.l_min; l <= gi.l_max; ++l) {
            os << "        level " << l << ": " << gi.level_size[l] << "\n";
        }
        os << "}";

        return os;
    }
};


#endif //LIBAPR_GENINFO_HPP
