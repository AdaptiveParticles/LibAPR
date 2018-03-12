//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_APR_TIMER_HPP
#define PARTPLAY_APR_TIMER_HPP

#include <vector>
#include <chrono>
#include <iostream>
#include <string>
#ifdef HAVE_OPENMP
#include "omp.h"

class APRTimer{
//
//
//  Bevan Cheeseman 2016
//
//  Just to be used for timing stuff, and recording the results \hoho
//
//

public:

    std::vector<double> timings;
    std::vector<std::string> timing_names;

    int timer_count;

    double t1;
    double t2;

    bool verbose_flag; //turn to true if you want all the functions to write out their timings to terminal

    APRTimer(){
        timer_count = 0;
        timings.resize(0);
        timing_names.resize(0);
        verbose_flag = false;
    }


    void start_timer(std::string timing_name){
        timing_names.push_back(timing_name);
        t1 = omp_get_wtime();
    }

    void stop_timer(){
        t2 = omp_get_wtime();

        timings.push_back(t2-t1);


        if (verbose_flag){
            //output to terminal the result
            std::cout << timing_names.back() << " took "
                      << t2-t1
                      << " seconds\n";
        }
        timer_count++;
    }
};

#else

class APRTimer {
//
//
//  Bevan Cheeseman 2016
//
//  Just to be used for timing stuff, and recording the results \hoho
//
//

public:

    std::vector<double> timings;
    std::vector<std::string> timing_names;

    int timer_count;

    std::chrono::system_clock::time_point t1_internal;
    std::chrono::system_clock::time_point t2_internal;

    float t1;
    float t2;

    bool verbose_flag; //turn to true if you want all the functions to write out their timings to terminal

    APRTimer() {
        timer_count = 0;
        timings.resize(0);
        timing_names.resize(0);
        verbose_flag = false;
        t1=0;
    }

    void start_timer(std::string timing_name){
        timing_names.push_back(timing_name);

        t1_internal = std::chrono::system_clock::now();
    }


    void stop_timer(){
        t2_internal = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = t2_internal - t1_internal;
        t2 = elapsed_seconds.count();

        timings.push_back(elapsed_seconds.count());

        if (verbose_flag){
            //output to terminal the result
            std::cout <<  (timing_names.back()) << " took " << std::to_string(elapsed_seconds.count()) << " seconds" << std::endl;
        }
        timer_count++;
    }
};

#endif

#endif //PARTPLAY_APR_TIMER_HPP
