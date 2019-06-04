//
// Created by cheesema on 2019-06-04.
//

#ifndef LIBAPR_GENINFO_HPP
#define LIBAPR_GENINFO_HPP


class GenInfo {

public:
    uint64_t l_min;
    uint64_t l_max;
    uint64_t org_dims[3]={0,0,0};

    uint8_t number_dimensions = 3;

    // TODO: SHould they be also saved as uint64 in HDF5? (currently int is used)
    std::vector<uint64_t> x_num;
    std::vector<uint64_t> y_num;
    std::vector<uint64_t> z_num;

    uint64_t total_number_particles;

    std::vector<unsigned int> level_size; // precomputation of the size of each level, used by the iterators.

    //initialize the information given the original dimensions
    void init(uint64_t y_org,uint64_t x_org,uint64_t z_org){

        org_dims[0] = y_org;
        org_dims[1] = x_org;
        org_dims[2] = z_org;

        number_dimensions = (y_org> 1) + (x_org > 1) + (z_org > 1);

        int max_dim = std::max(std::max(org_dims[1], org_dims[0]), org_dims[2]);
        //int min_dim = std::min(std::min(aAPR.apr_access.org_dims[1], aAPR.apr_access.org_dims[0]), aAPR.apr_access.org_dims[2]);

        int min_dim = max_dim;
        min_dim = y_org > 1 ? std::min(min_dim, (int) y_org) : min_dim;
        min_dim = x_org > 1 ? std::min(min_dim, (int) x_org) : min_dim;
        min_dim = z_org > 1 ? std::min(min_dim, (int) z_org) : min_dim;

        int levelMax = ceil(std::log2(max_dim));
        // TODO: why minimum level is forced here to be 2?
        int levelMin = std::max( (int)(levelMax - floor(std::log2(min_dim))), 1);

        l_min = levelMin;
        l_max = levelMax;

        y_num.resize(levelMax+1);
        x_num.resize(levelMax+1);
        z_num.resize(levelMax+1);

        level_size.resize(levelMax + 1);
        for (int k = 0; k <= levelMax; ++k) {
           level_size[k] = (uint64_t) pow(2,levelMax - k);
        }


        for (uint64_t l = l_min; l <= l_max; ++l) {
            double cellSize = pow(2.0, l_max - l);
            y_num[l] = (uint64_t) ceil(y_org / cellSize);
            x_num[l] = (uint64_t) ceil(x_org / cellSize);
            z_num[l] = (uint64_t) ceil(z_org / cellSize);
        }



    }

    //initialize the information given the original dimensions
    void init_tree(uint64_t y_org,uint64_t x_org,uint64_t z_org){

        org_dims[0] = y_org;
        org_dims[1] = x_org;
        org_dims[2] = z_org;

        number_dimensions = (y_org > 1) + (x_org > 1) + (z_org > 1);

        int max_dim = std::max(std::max(y_org, x_org), z_org);
        //int min_dim = std::min(std::min(aAPR.apr_access.org_dims[1], aAPR.apr_access.org_dims[0]), aAPR.apr_access.org_dims[2]);

        int min_dim = max_dim;
        min_dim = y_org > 1 ? std::min(min_dim, (int) y_org) : min_dim;
        min_dim = x_org > 1 ? std::min(min_dim, (int) x_org) : min_dim;
        min_dim = z_org > 1 ? std::min(min_dim, (int) z_org) : min_dim;

        int levelMax = ceil(std::log2(max_dim));
        // TODO: why minimum level is forced here to be 2?
        int levelMin = std::max( (int)(levelMax - floor(std::log2(min_dim))), 1);

        l_min = levelMin-1;
        l_max = levelMax-1;

        y_num.resize(l_max+1);
        x_num.resize(l_max+1);
        z_num.resize(l_max+1);

        level_size.resize(levelMax + 1);
        for (int k = 0; k <= levelMax; ++k) {
            level_size[k] = (uint64_t) pow(2,levelMax - k);
        }

        for (uint64_t l = l_min; l <= l_max; ++l) {
            double cellSize = pow(2.0, l_max - l + 1);
            y_num[l] = (uint64_t) ceil(y_org / cellSize);
            x_num[l] = (uint64_t) ceil(x_org / cellSize);
            z_num[l] = (uint64_t) ceil(z_org / cellSize);
        }

    }


};


#endif //LIBAPR_GENINFO_HPP
