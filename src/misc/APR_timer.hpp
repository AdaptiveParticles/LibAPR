//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_APR_TIMER_HPP
#define PARTPLAY_APR_TIMER_HPP

#include <vector>
#include <iostream>
#include "omp.h"

class APR_timer{
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

APR_timer(){
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
        std::cout <<  timing_names[timer_count] << " took "
                  << t2-t1
                  << " seconds\n";
    }
    timer_count++;
}



};

#endif //PARTPLAY_APR_TIMER_HPP
