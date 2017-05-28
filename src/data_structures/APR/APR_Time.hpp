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

    //init APR
    APR<float> init_APR;

    //construction/reconstruction variables
    ExtraPartCellData<float> curr_scale;
    ExtraPartCellData<uint16_t> curr_yp;
    ExtraPartCellData<float> curr_fp;

    ExtraPartCellData<float> curr_sp;
    ExtraPartCellData<float> curr_tp;



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

        same_index.initialize_structure_parts(initial_APR.y_vec);

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
                    }



                }
            }
        }


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


//        for(uint64_t depth = (apr_c.y_vec.depth_min);depth <= apr_c.y_vec.depth_max;depth++) {
//            //loop over the resolutions of the structure
//            const unsigned int x_num_ = apr_c.y_vec.x_num[depth];
//            const unsigned int z_num_ = apr_c.y_vec.z_num[depth];
//
//            const unsigned int x_num_min_ = 0;
//            const unsigned int z_num_min_ = 0;
//
//
////#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
//            for (z_ = z_num_min_; z_ < z_num_; z_++) {
//                //both z and x are explicitly accessed in the structure
//
//                for (x_ = x_num_min_; x_ < x_num_; x_++) {
//
//                    const unsigned int pc_offset = x_num_*z_ + x_;
//
//                    same_index.data[depth][pc_offset].resize(0);
//
//                    int j_old = 0;
//                    int j_old_max = curr_yp.data[depth][pc_offset].size();
//
//                    float y_old=999999999;
//                    float y_new=0;
//
//                    for (int j_ = 0; j_ < apr_c.y_vec.data[depth][pc_offset].size(); ++j_) {
//                        if(j_old < j_old_max) {
//                            y_old = curr_yp.data[depth][pc_offset][j_old];
//                        }
//                        y_new = apr_c.y_vec.data[depth][pc_offset][j_];
//
//                        if(y_old == y_new) {
//                            //still the same do nothing
//
//                            //add to the same index
//                            same_index.data[depth][pc_offset].push_back(j_);
//
//
//                            if (j_old < j_old_max) {
//
//                                j_old++;
//                            }
//
//
//
//                        } else {
//                            if(y_old < y_new){
//
//                                if(y_old < y_new){
//                                    remove[t].data[depth][pc_offset].push_back(y_old);
//
//                                }
//                                while(y_old < y_new & (j_old+1) < j_old_max){
//
//                                    j_old++;
//                                    y_old = curr_yp.data[depth][pc_offset][j_old];
//
//                                    if(y_old < y_new) {
//                                        remove[t].data[depth][pc_offset].push_back(y_old);
//
//                                    }
//
//
//                                }
//
//                                if(y_old == y_new){
//                                    same_index.data[depth][pc_offset].push_back(j_);
//
//                                    j_old++;
//
//                                }
//
//
//                            } else {
//                                add[t].data[depth][pc_offset].push_back(y_new);
//                                add_fp[t].data[depth][pc_offset].push_back(apr_c.particles_int.data[depth][pc_offset][j_]);
//
//                            }
//
//                        }
//
//
//
//                    }
//
//
//
//                }
//            }
//        }



        int same = 0;
        int add_total = 0;
        int rem_total = 0;
        int tt=0;


        std::vector<uint16_t>::iterator it;
        std::vector<uint16_t>::iterator it_f;

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

                        it = std::set_intersection(curr_yp.data[depth][pc_offset].begin(),
                                            curr_yp.data[depth][pc_offset].end(),
                                            apr_c.y_vec.data[depth][pc_offset].begin(),
                                            apr_c.y_vec.data[depth][pc_offset].end(), same_index.data[depth][pc_offset].begin());

                        same_index.data[depth][pc_offset].resize(it - same_index.data[depth][pc_offset].begin());

                        int sz = same_index.data[depth][pc_offset].size();



                    } else {
                        same_index.data[depth][pc_offset].resize(0);
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

                        add[t].data[depth][pc_offset].resize(std::max(curr_yp.data[depth][pc_offset].size(),
                                                                      apr_c.y_vec.data[depth][pc_offset].size()));

                        it = std::set_difference(apr_c.y_vec.data[depth][pc_offset].begin(),
                                                 apr_c.y_vec.data[depth][pc_offset].end(),curr_yp.data[depth][pc_offset].begin(),
                                                 curr_yp.data[depth][pc_offset].end(),
                                                  add[t].data[depth][pc_offset].begin());

                        add[t].data[depth][pc_offset].resize(it - add[t].data[depth][pc_offset].begin());


                    }
                }
            }
        }



        float addq = add[t].structure_size();
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


    void integrate_new_t(APR<float>& apr_c,ExtraPartCellData<float>& scale,int t_){
        //
        //  Computes the APR+t pulling scheme and diff scheme
        //
        //  Bevan Cheeseman 2017
        //
        //

        t = t_;

        calc_apr_diff(apr_c);

        std::swap(curr_yp,apr_c.y_vec);
        std::swap(curr_fp,apr_c.particles_int);

//
//        %% Handle Change in Sampling
//
//        nt = apr_pt.nt;
//
//        [apr_pt.add{nt},indx_a] = setdiff(apr.y_p,apr_pt.curr_y);
//
//        same_c = logical(ones(size(apr.y_p)));
//        same_c(indx_a) = false;
//
//        apr_pt.add_fp{nt} = apr.f_p(indx_a);
//
//        [apr_pt.remove{nt},indx_r] = setdiff(apr_pt.curr_y,apr.y_p);
//
//        same_p =  logical(ones(size(apr_pt.curr_y)));
//        same_p(indx_r) = false;
//
//        %% Handle Adaptive Sampling Through Time.
//
//                                            %check all those that are not the above!^^
//
//                                                                                   y_c = apr.y_p(same_c);
//        y_p = apr_pt.curr_y(same_p);
//
//        f_c = apr.f_p(same_c);
//        f_p = apr_pt.curr_f(same_p);
//
//        %% Compute l
//        scale = (apr.scale(1) + apr_pt.curr_scale)/2;
//        L = (apr_pt.Et*scale*apr_pt.dt)./(abs(f_p-f_c));
//        lt = compute_time_level(L,apr_pt);
//
//        % previous variables
//        lt_p = apr_pt.curr_lt(same_p);
//        status_p = apr_pt.curr_s(same_p);
//        t_p = apr_pt.curr_t(same_p);
//
//        lt_c = apr_pt.curr_lt(same_p);
//        status_c = apr_pt.curr_s(same_p);
//        t_c = apr_pt.curr_t(same_p);
//
//        for i = 1:length(lt)
//        if(lt(i) > lt_p(i))
//        %propogate new solution for this point
//                t_c(i) = nt;
//        status_c(i) = 1;
//        lt_c(i) = lt(i);
//
//        apr_pt.update_fp{nt}(end+1) = f_c(i);
//        apr_pt.update_y{nt}(end+1) = y_c(i);
//
//        else
//
//
//        if (nt - t_p(i)) >= 2^(apr_pt.lt_max-lt_p(i))
//        if(status_p(i) == 1)
//        %seed, same level, pad with boundary
//        status_c(i) = 2;
//        lt_c(i) = lt_p(i);
//        t_c(i) = nt;
//
//        apr_pt.update_fp{nt}(end+1) = f_c(i);
//        apr_pt.update_y{nt}(end+1) = y_c(i);
//
//        elseif(status_p(i) == 2)
//        %boundary, same level
//                lt_c(i) = lt_p(i);
//        status_c(i) = 3;
//        t_c(i) = nt;
//
//        apr_pt.update_fp{nt}(end+1) = f_c(i);
//        apr_pt.update_y{nt}(end+1) = y_c(i);
//
//        else
//        %filler, go up a level
//        status_c(i) = 3;
//        lt_c(i) = lt_p(i)-1;
//        t_c(i) = nt;
//        apr_pt.update_fp{nt}(end+1) = f_c(i);
//        apr_pt.update_y{nt}(end+1) = y_c(i);
//
//        end
//
//        else
//        % do nothing
//                    lt_c(i) = lt_p(i);
//        status_c(i) = status_p(i);
//        t_c(i) = t_p(i);
//        end
//
//        if(lt(i) == lt_p(i))
//        %stay the same, and set to seed
//        status_c(i) = 1;
//        lt_c(i) = lt(i);
//        end
//                end
//
//        end
//
//        % figure(gcf);plot(y_c,lt);hold on
//                                        % xlim([-10,10])
//        % ylim([0,apr_pt.lt_max+2])
//        % %plot(apr.y_p,apr.f_p);
//        % drawnow();
//        % hold off
//               % pause(0.1)
//
//               %% update for next time step
//
//        apr_pt.nt = nt + 1;
//
//        apr_pt.curr_y = apr.y_p;
//
//        apr_pt.curr_f = apr.f_p;
//
//        apr_pt.curr_scale = apr.scale(1);
//
//        apr_pt.curr_lt = zeros(size(apr_pt.curr_f));
//        apr_pt.curr_s = zeros(size(apr_pt.curr_f));
//        apr_pt.curr_t = zeros(size(apr_pt.curr_f));
//
//        %% Update those the fixed space cells
//
//        apr_pt.curr_lt(same_c) = lt_c;
//
//        apr_pt.curr_s(same_c) = status_c;
//
//        apr_pt.curr_t(same_c) = t_c;
//
//        %% Update those that changed
//
//        apr_pt.curr_lt(~same_c) = apr_pt.lt_max;
//
//        apr_pt.curr_s(~same_c) = 1;
//
//        apr_pt.curr_t(~same_c) = nt;




    }



};


#endif //PARTPLAY_APR_TIME_HPP
