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

    ExtraPartCellData<float> parts_recon;
    ExtraPartCellData<float> parts_recon_prev;


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

        parts_recon.initialize_structure_parts(initial_APR.particles_int);
        parts_recon_prev.initialize_structure_parts(initial_APR.particles_int);

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
                        prev_l.data[depth][pc_offset][j_] = 1;
                        prev_sp.data[depth][pc_offset][j_] = 3;

                        parts_recon_prev.data[depth][pc_offset][j_] = initial_APR.particles_int.data[depth][pc_offset][j_];
                        parts_recon.data[depth][pc_offset][j_] = initial_APR.particles_int.data[depth][pc_offset][j_];

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


                        add_fp[t].data[depth][pc_offset][i] = apr_c.particles_int.data[depth][pc_offset][add_index.data[depth][pc_offset][i]];




                    }

                    for (int i = 0; i < add_index.data[depth][pc_offset].size(); ++i) {


                        add[t].data[depth][pc_offset][i] = apr_c.y_vec.data[depth][pc_offset][add_index.data[depth][pc_offset][i]];

                    }


                }
            }
        }






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
                        curr_sp.data[depth][pc_offset].resize(apr_c.y_vec.data[depth][pc_offset].size());
                        curr_tp.data[depth][pc_offset].resize(apr_c.y_vec.data[depth][pc_offset].size());

                        parts_recon.data[depth][pc_offset].resize(apr_c.y_vec.data[depth][pc_offset].size());

                        for (int i = 0; i < same_index.data[depth][pc_offset].size(); ++i) {

                            float sigma_c = curr_scale.data[depth][pc_offset][same_index.data[depth][pc_offset][i]];
                            float sigma_p = prev_scale.data[depth][pc_offset][same_index_old.data[depth][pc_offset][i]];
                            float f_diff = abs(curr_fp.data[depth][pc_offset][same_index_old.data[depth][pc_offset][i]] - apr_c.particles_int.data[depth][pc_offset][same_index.data[depth][pc_offset][i]]);

                            if(f_diff == 0) {
                                l_t = 1.0;

                            } else {
                                L_t = Et*dt*(sigma_c)/(f_diff);

                                l_t = ceil(log2(Sigma_t/L_t));

                                l_t = std::max(l_t,1.0f);

                                l_t = std::min(l_t,lt_max);



                            }

                            curr_l.data[depth][pc_offset][same_index.data[depth][pc_offset][i]] = l_t;

                        }



                }
            }
        }



        update_fp[t].initialize_structure_parts_empty(apr_c.y_vec);
        update_y[t].initialize_structure_parts_empty(apr_c.y_vec);




        //////////////////////////////
        ////
        ////    Pulling Scheme Loop
        ////
        ////
        //////////////////////////////

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



                    for (int i = 0; i < same_index.data[depth][pc_offset].size(); ++i) {

                        int old_index = same_index_old.data[depth][pc_offset][i];
                        int new_index = same_index.data[depth][pc_offset][i];


                        float curr_lt = curr_l.data[depth][pc_offset][new_index];
                        float prev_lt = prev_l.data[depth][pc_offset][old_index];

                        if(curr_lt > prev_lt){

                            curr_tp.data[depth][pc_offset][new_index] = t;
                            curr_sp.data[depth][pc_offset][new_index] = 1;

                            update_fp[t].data[depth][pc_offset].push_back(apr_c.particles_int.data[depth][pc_offset][new_index]);
                            update_y[t].data[depth][pc_offset].push_back(apr_c.y_vec.data[depth][pc_offset][new_index]);


                            parts_recon.data[depth][pc_offset][new_index] = apr_c.particles_int.data[depth][pc_offset][new_index];

                        } else {

                            float t_int = pow(2,lt_max-prev_lt);

                            if((t - prev_tp.data[depth][pc_offset][old_index]) >= t_int){

                                float status_p = prev_sp.data[depth][pc_offset][old_index];

                                if(status_p == 1){
                                    // seed, same level, pad with boundary
                                    curr_sp.data[depth][pc_offset][new_index] = 2;
                                    curr_l.data[depth][pc_offset][new_index] = prev_l.data[depth][pc_offset][old_index];
                                    curr_tp.data[depth][pc_offset][new_index] = t;

                                    update_fp[t].data[depth][pc_offset].push_back(apr_c.particles_int.data[depth][pc_offset][new_index]);
                                    update_y[t].data[depth][pc_offset].push_back(apr_c.y_vec.data[depth][pc_offset][new_index]);

                                    parts_recon.data[depth][pc_offset][new_index] = apr_c.particles_int.data[depth][pc_offset][new_index];


                                } else if (status_p == 2){
                                    // boundary, same level, pad with filler
                                    curr_sp.data[depth][pc_offset][new_index] = 3;
                                    curr_l.data[depth][pc_offset][new_index] = prev_l.data[depth][pc_offset][old_index];
                                    curr_tp.data[depth][pc_offset][new_index] = t;

                                    update_fp[t].data[depth][pc_offset].push_back(apr_c.particles_int.data[depth][pc_offset][new_index]);
                                    update_y[t].data[depth][pc_offset].push_back(apr_c.y_vec.data[depth][pc_offset][new_index]);

                                    parts_recon.data[depth][pc_offset][new_index] = apr_c.particles_int.data[depth][pc_offset][new_index];

                                } else {
                                    // filler, go up a level
                                    curr_sp.data[depth][pc_offset][new_index] = 3;
                                    curr_l.data[depth][pc_offset][new_index] = prev_l.data[depth][pc_offset][old_index]-1;
                                    curr_tp.data[depth][pc_offset][new_index] = t;

                                    update_fp[t].data[depth][pc_offset].push_back(apr_c.particles_int.data[depth][pc_offset][new_index]);
                                    update_y[t].data[depth][pc_offset].push_back(apr_c.y_vec.data[depth][pc_offset][new_index]);

                                    parts_recon.data[depth][pc_offset][new_index] = apr_c.particles_int.data[depth][pc_offset][new_index];

                                }



                            } else {

                                curr_tp.data[depth][pc_offset][new_index] = prev_tp.data[depth][pc_offset][old_index];
                                curr_sp.data[depth][pc_offset][new_index] = prev_tp.data[depth][pc_offset][old_index];
                                curr_l.data[depth][pc_offset][new_index] = prev_l.data[depth][pc_offset][old_index];

                                parts_recon.data[depth][pc_offset][new_index] = parts_recon_prev.data[depth][pc_offset][old_index];

                            }

                            if(curr_lt == prev_lt){
                                // stay the same set to seed.
                                curr_sp.data[depth][pc_offset][new_index]=1;


                            }


                        }

                    }

                }
            }
        }


        //////////////////////////////
        ////
        ////    Update those where the apr structure changed
        ////
        ////
        //////////////////////////////



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



                        curr_l.data[depth][pc_offset][index] = lt_max;

                        curr_sp.data[depth][pc_offset][index] = 1;

                        curr_tp.data[depth][pc_offset][index] = t;

                        parts_recon.data[depth][pc_offset][index] = apr_c.particles_int.data[depth][pc_offset][index];


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

        std::swap(parts_recon_prev.data,parts_recon.data);


    }



};


#endif //PARTPLAY_APR_TIME_HPP
