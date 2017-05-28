//
// Created by cheesema on 26.05.17.
//

#ifndef PARTPLAY_APR_TIME_HPP
#define PARTPLAY_APR_TIME_HPP

#include "../Tree/ExtraPartCellData.hpp"
#include "APR.hpp"


class APR_Time {
    public:


    // parameters

    std::vector<float> t_dim;
    float dt;
    float Et;
    float Nt;
    float lt_max;
    float Sigma_t;

    float t;

    //storage variables
    std::vector<ExtraPartCellData<uint16_t>> add;
    std::vector<ExtraPartCellData<uint16_t>> remove;

    std::vector<ExtraPartCellData<float>> add_fp;
    std::vector<ExtraPartCellData<float>> update_fp;
    std::vector<ExtraPartCellData<float>> update_y;

    ExtraPartCellData<uint16_t> same_index;
    ExtraPartCellData<uint16_t> same_index_old;

    ExtraPartCellData<uint16_t> add_index;

    //init APR
    APR<float> init_APR;

    //construction/reconstruction variables
    ExtraPartCellData<float> curr_scale;
    ExtraPartCellData<uint16_t> curr_yp;
    ExtraPartCellData<float> curr_fp;

    ExtraPartCellData<float> curr_sp;
    ExtraPartCellData<float> curr_tp;

    ExtraPartCellData<uint8_t> curr_l;
    ExtraPartCellData<uint8_t> prev_l;

    ExtraPartCellData<float> prev_scale;

    ExtraPartCellData<float> prev_sp;
    ExtraPartCellData<float> prev_tp;

    APR_Time(){

    }


    void initialize(APR<float>& initial_APR,std::vector<float> t_dim_,float Et_,float Nt_,ExtraPartCellData<float>& scale){

        //set initial APR
        init_APR = initial_APR;

        //initialize all the state variables;
        add.resize(Nt_);
        add_fp.resize(Nt_);
        remove.resize(Nt_);

        update_fp.resize(Nt_);
        update_y.resize(Nt_);

        curr_scale.initialize_structure_parts(initial_APR.y_vec);
        curr_yp.initialize_structure_parts(initial_APR.y_vec);
        curr_fp.initialize_structure_parts(initial_APR.y_vec);
        curr_sp.initialize_structure_parts(initial_APR.y_vec);
        curr_tp.initialize_structure_parts(initial_APR.y_vec);

        prev_sp.initialize_structure_parts(initial_APR.y_vec);
        prev_tp.initialize_structure_parts(initial_APR.y_vec);
        prev_scale.initialize_structure_parts(initial_APR.y_vec);

        curr_l.initialize_structure_parts(initial_APR.y_vec);
        prev_l.initialize_structure_parts(initial_APR.y_vec);

        same_index.initialize_structure_parts(initial_APR.y_vec);
        same_index_old.initialize_structure_parts(initial_APR.y_vec);

        add_index.initialize_structure_parts(initial_APR.y_vec);

        Nt = Nt_;
        Et = Et_;

        t_dim = t_dim_;
        dt = (t_dim[1] - t_dim[0]/(Nt+1));

        lt_max = ceil(log2(Nt));

        Sigma_t = pow(2,lt_max)*(dt);

        t = 1;

        int z_,x_,j_,y_;

        for(uint64_t depth = (initial_APR.y_vec.depth_min);depth <= initial_APR.y_vec.depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = initial_APR.y_vec.x_num[depth];
            const unsigned int z_num_ = initial_APR.y_vec.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;


#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                    for (int j_ = 0; j_ < initial_APR.y_vec.data[depth][pc_offset].size(); ++j_) {

                        curr_yp.data[depth][pc_offset][j_] = initial_APR.y_vec.data[depth][pc_offset][j_];
                        curr_fp.data[depth][pc_offset][j_] = initial_APR.particles_int.data[depth][pc_offset][j_];
                        curr_sp.data[depth][pc_offset][j_] = 3;
                        curr_tp.data[depth][pc_offset][j_] = 1;
                        curr_scale.data[depth][pc_offset][j_] = scale.data[depth][pc_offset][j_];
                        prev_scale.data[depth][pc_offset][j_] = scale.data[depth][pc_offset][j_];
                        prev_l.data[depth][pc_offset][j_] = lt_max;
                    }


                }
            }
        }


    }


    template <class InputIterator1, class InputIterator2, class OutputIterator>
    std::vector<OutputIterator> set_intersection_index (InputIterator1 first1, InputIterator1 last1,
                                     InputIterator2 first2, InputIterator2 last2,
                                     OutputIterator result,OutputIterator result2)
    {
        auto first1_ind = first1;
        auto first2_ind = first2;

        while (first1!=last1 && first2!=last2)
        {
            if (*first1<*first2) ++first1;
            else if (*first2<*first1) ++first2;
            else {
                *result = std::distance(first1_ind,first1);
                *result2 = std::distance(first2_ind,first2);
                ++result; ++first1; ++first2;++result2;
            }
        }

        std::vector<OutputIterator> results = {result,result2};

        return results;
    }

    template <class InputIterator1, class InputIterator2, class OutputIterator>
    OutputIterator set_difference_indx (InputIterator1 first1, InputIterator1 last1,
                                   InputIterator2 first2, InputIterator2 last2,
                                   OutputIterator result)
    {
        auto first1_ind = first1;
       // *result = std::distance(first1_ind,first1)

        while (first1!=last1 && first2!=last2)
        {
            if (*first1<*first2) { *result = std::distance(first1_ind,first1); ++result; ++first1; }
            else if (*first2<*first1) ++first2;
            else { ++first1; ++first2; }
        }

        while (first1!=last1) {
            *result = std::distance(first1_ind,first1);
            ++result; ++first1;
        }
        return result;

       // return std::copy(first1,last1,result);
    }

    void calc_apr_diff(APR<float>& apr_c){
        //
        //  Bevan Cheeseman 2017
        //
        //  Set diff
        //

        int z_,x_,j_,y_;

        add[t].initialize_structure_parts_empty(apr_c.y_vec);
        add_fp[t].initialize_structure_parts_empty(apr_c.y_vec);
        remove[t].initialize_structure_parts_empty(apr_c.y_vec);


        int same = 0;
        int add_total = 0;
        int rem_total = 0;
        int tt=0;


        std::vector<uint16_t>::iterator it;
        std::vector<uint16_t>::iterator it_f;

        std::vector<std::vector<uint16_t>::iterator> its;

        //intersection loop

        for(uint64_t depth = (apr_c.y_vec.depth_min);depth <= apr_c.y_vec.depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = apr_c.y_vec.x_num[depth];
            const unsigned int z_num_ = apr_c.y_vec.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;


//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                    if(apr_c.y_vec.data[depth][pc_offset].size() > 0) {

                        same_index.data[depth][pc_offset].resize(std::max(curr_yp.data[depth][pc_offset].size(),
                                          apr_c.y_vec.data[depth][pc_offset].size()));
                        same_index_old.data[depth][pc_offset].resize(std::max(curr_yp.data[depth][pc_offset].size(),
                                                                          apr_c.y_vec.data[depth][pc_offset].size()));

                        its = set_intersection_index(curr_yp.data[depth][pc_offset].begin(),
                                            curr_yp.data[depth][pc_offset].end(),
                                            apr_c.y_vec.data[depth][pc_offset].begin(),
                                            apr_c.y_vec.data[depth][pc_offset].end(), same_index_old.data[depth][pc_offset].begin(),same_index.data[depth][pc_offset].begin());

                        same_index.data[depth][pc_offset].resize(its[1] - same_index.data[depth][pc_offset].begin());
                        same_index_old.data[depth][pc_offset].resize(its[0] - same_index_old.data[depth][pc_offset].begin());


                    } else {
                        same_index.data[depth][pc_offset].resize(0);
                        same_index_old.data[depth][pc_offset].resize(0);
                    }
                }
            }
        }

        //remove loop

        for(uint64_t depth = (apr_c.y_vec.depth_min);depth <= apr_c.y_vec.depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = apr_c.y_vec.x_num[depth];
            const unsigned int z_num_ = apr_c.y_vec.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;


//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                    if(curr_yp.data[depth][pc_offset].size() > 0) {

                        remove[t].data[depth][pc_offset].resize(std::max(curr_yp.data[depth][pc_offset].size(),
                                                                          apr_c.y_vec.data[depth][pc_offset].size()));

                        it = std::set_difference(curr_yp.data[depth][pc_offset].begin(), curr_yp.data[depth][pc_offset].end(),apr_c.y_vec.data[depth][pc_offset].begin(),apr_c.y_vec.data[depth][pc_offset].end(), remove[t].data[depth][pc_offset].begin());

                        remove[t].data[depth][pc_offset].resize(it - remove[t].data[depth][pc_offset].begin());




                    }
                }
            }
        }

        //add loop

        for(uint64_t depth = (apr_c.y_vec.depth_min);depth <= apr_c.y_vec.depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = apr_c.y_vec.x_num[depth];
            const unsigned int z_num_ = apr_c.y_vec.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;


//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                    if(apr_c.y_vec.data[depth][pc_offset].size() > 0) {



                        add_index.data[depth][pc_offset].resize(std::max(curr_yp.data[depth][pc_offset].size(),
                                                                      apr_c.y_vec.data[depth][pc_offset].size()));

                        it = set_difference_indx(apr_c.y_vec.data[depth][pc_offset].begin(),
                                                 apr_c.y_vec.data[depth][pc_offset].end(),curr_yp.data[depth][pc_offset].begin(),
                                                 curr_yp.data[depth][pc_offset].end(),
                                            add_index.data[depth][pc_offset].begin());

                        add_index.data[depth][pc_offset].resize(it - add_index.data[depth][pc_offset].begin());

                        std::vector<uint16_t> test;

                        test.resize(std::max(curr_yp.data[depth][pc_offset].size(),
                                                                         apr_c.y_vec.data[depth][pc_offset].size()));

                        it = std::set_difference(apr_c.y_vec.data[depth][pc_offset].begin(),
                                            apr_c.y_vec.data[depth][pc_offset].end(),curr_yp.data[depth][pc_offset].begin(),
                                            curr_yp.data[depth][pc_offset].end(),
                                            test.begin());

                        test.resize(it - test.begin());





                    } else {
                        add_index.data[depth][pc_offset].resize(0);
                    }
                }
            }
        }


        //now construct the add_fp and set add to y
        for(uint64_t depth = (apr_c.y_vec.depth_min);depth <= apr_c.y_vec.depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = apr_c.y_vec.x_num[depth];
            const unsigned int z_num_ = apr_c.y_vec.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;


//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                    add_fp[t].data[depth][pc_offset].resize(add_index.data[depth][pc_offset].size());
                    add[t].data[depth][pc_offset].resize(add_index.data[depth][pc_offset].size());

                    for (int i = 0; i < add_index.data[depth][pc_offset].size(); ++i) {


                        //add_fp[t].data[depth][pc_offset][i] = apr_c.particles_int.data[depth][pc_offset][add_index.data[depth][pc_offset][i]];




                    }

                    for (int i = 0; i < add_index.data[depth][pc_offset].size(); ++i) {

                        int index = add_index.data[depth][pc_offset][i];

                        int y_sz = apr_c.y_vec.data[depth][pc_offset].size();

                        if(index >= y_sz){
                            int stop = 1;
                        }

                        //add[t].data[depth][pc_offset][i] = apr_c.y_vec.data[depth][pc_offset][add_index.data[depth][pc_offset][i]];

                    }


                }
            }
        }



        float addq = add_index.structure_size();
        float removeq =remove[t].structure_size();
        float sames = same_index.structure_size();
        float total_parts = apr_c.y_vec.structure_size();
        float total_2 = addq + sames;
        float total_old = curr_yp.structure_size();

        std::cout << "add: " << addq << std::endl;
        std::cout << "remove: " << removeq << std::endl;
        std::cout << "same: " << sames << std::endl;
        std::cout << "total parts: " << total_parts << std::endl;
        std::cout << "total parts 2: " << total_2 << std::endl;


    }

    void calc_ip_updates(APR<float>& apr_c){

        //
        //  Calculate the time pulling scheme
        //
        //



        //////////////////
        ////
        ////    First compute l_t
        ////
        //////////////////

        int z_,x_,j_,y_;
        //check the same indices

        for(uint64_t depth = (apr_c.y_vec.depth_min);depth <= apr_c.y_vec.depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = apr_c.y_vec.x_num[depth];
            const unsigned int z_num_ = apr_c.y_vec.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;


//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;

                        float L_t;
                        float l_t;

                        curr_l.data[depth][pc_offset].resize(apr_c.y_vec.data[depth][pc_offset].size());

                        for (int i = 0; i < same_index.data[depth][pc_offset].size(); ++i) {

                            float sigma_c = curr_scale.data[depth][pc_offset][same_index.data[depth][pc_offset][i]];
                            float sigma_p = prev_scale.data[depth][pc_offset][same_index_old.data[depth][pc_offset][i]];
                            float f_diff = abs(curr_fp.data[depth][pc_offset][same_index_old.data[depth][pc_offset][i]] - apr_c.particles_int.data[depth][pc_offset][same_index.data[depth][pc_offset][i]]);

                            if(f_diff == 0) {
                                l_t = 1.0;

                            } else {
                                L_t = Et*dt*(sigma_c + sigma_p)/(f_diff*2);

                                l_t = ceil(log2(Sigma_t/L_t));

                                l_t = std::max(l_t,1.0f);

                                l_t = std::min(l_t,lt_max);



                            }

                            curr_l.data[depth][pc_offset][same_index.data[depth][pc_offset][i]] = l_t;
                            curr_l.data[depth][pc_offset][same_index.data[depth][pc_offset][i]] = 1;
                        }



                }
            }
        }


        //now construct the add_fp and set add to y
        for(uint64_t depth = (apr_c.y_vec.depth_min);depth <= apr_c.y_vec.depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = apr_c.y_vec.x_num[depth];
            const unsigned int z_num_ = apr_c.y_vec.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;


//#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_*z_ + x_;



                    for (int i = 0; i < add_index.data[depth][pc_offset].size(); ++i) {

                        int index = add_index.data[depth][pc_offset][i];

                        int curr_sz = curr_l.data[depth][pc_offset].size();

                        int y_sz = apr_c.y_vec.data[depth][pc_offset].size();

                        if(index >= curr_l.data[depth][pc_offset].size()){
                            int stop = 1;
                        }


                        //curr_l.data[depth][pc_offset][index] = 2;

                    }


                }
            }
        }



    }




    void integrate_new_t(APR<float>& apr_c,ExtraPartCellData<float>& scale,int t_){
        //
        //  Computes the APR+t pulling scheme and diff scheme
        //
        //  Bevan Cheeseman 2017
        //
        //

        t = t_;

        calc_apr_diff(apr_c);

        std::swap(curr_scale.data,scale.data);

        calc_ip_updates(apr_c);



        // end of time step change over the variables;
        std::swap(curr_yp,apr_c.y_vec);
        std::swap(curr_fp,apr_c.particles_int);

        std::swap(curr_scale.data,prev_scale.data);

        std::swap(curr_l.data,prev_l.data);
        std::swap(curr_tp.data,prev_tp.data);
        std::swap(curr_sp.data,prev_sp.data);


    }



};


#endif //PARTPLAY_APR_TIME_HPP
