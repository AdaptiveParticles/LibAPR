//
//
//  Bevan Cheeseman 2015
//
//  Part Play Library
//
//  Basic data structures for the particles
//
//
//  structure_parts.h
//  PartPlay
//
//  Created by cheesema on 11/22/15.
//  Copyright (c) 2015 imagep. All rights reserved.
//

#ifndef PARTPLAY_STRUCTURE_PARTS_H
#define PARTPLAY_STRUCTURE_PARTS_H

#include <algorithm>
#ifdef _WINDOWS
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include <cmath>
#include <iostream>
#include <map>
#include <stdint.h>
#include <string>
#include <vector>
#include <algorithm>

#include "omp.h"
#include "../io/parameters.h"

class Part_map;
template <typename T>
class Part_data;
class Particle;
class Cell_id;
class Cell_index;
class Cell_id{
    //
    // cell id class for organizing the particle cells
    //
    //
    
public:
    
    uint16_t x,y,z;
    uint8_t k;
    
#ifdef data_2D
    
    Cell_id() : y(), x(), k() {};
    Cell_id(unsigned int y,unsigned int x,unsigned int k) : y(y), x(x), k(k) {};
    
#else
    
    Cell_id() : y(), x(), z(), k() {};
    Cell_id(unsigned int y,unsigned int x,unsigned int z,unsigned int k) : y(y), x(x),z(z), k(k) {};
    
#endif
    
};
class Cell_index{
    //
    // cell id class for organizing the particle cells
    //
    //
    
public:
    
    int first,last,cindex;
    // first and last indices for particle data, and cindex index for cell level data
    
    
    Cell_index(){};
    
};
class Particle{
public:
    unsigned int x,y,z;
    uint8_t k;
    
    Particle():x(0),y(0),z(0),k(0){};
    
    Particle(int x,int y,int z,uint8_t k):x(x),y(y),z(z),k(k){};
    
    
};
class Part_timer{
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
    
    Part_timer(){
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


template <typename T>
class Part_data{
    //particle data
    
public:
    
    
    std::vector<T> data;
    std::string data_name;
    std::string data_type;
    
    Part_data(){
            }
    
    //init
    Part_data(std::string data_name_in,int data_size){
        data_name=data_name_in;
        data.resize(data_size);
    }
    
    
};
template<typename key, typename val>
struct my_hash_map
{
    //
    //  Template class for the hash table or, ordered map I use.
    //
    
    
    //typedef google::dense_hash_map<key,val> type;
    //typedef std::unordered_map<key,val> type;
    typedef std::map<key,val> type;
};
class Part_map{
    //hash map structures for organisizing the data.
public:
    //vector of hash-maps for every resolution level
    std::vector<my_hash_map<Cell_id,unsigned int>::type> pl;
    int k_max;
    int k_min;
    
    //counters
    int curr_total_cells;
    
    Part_map():k_max(0),k_min(0),curr_total_cells(0){};
    
    //containers for the cell datasets and access
    std::vector<Cell_id> cells;
    std::vector<Cell_index> cell_indices;
    
    void init_part_map(int k_min_,int k_max_){
        //initializes the maping structures
        pl.resize(k_max_+1);
        k_max = k_max_;
        k_min = k_min_;
    }
    
    void init_part_map(){
        //initializes the maping structures
        pl.resize(k_max+1);
    }
    
    void add_cell_to_map(Cell_id new_cell_id,Cell_index new_cell_index){
        //adds a cell into the datastructures
        int k_ = new_cell_id.k;

        curr_total_cells++; //add cell

        new_cell_index.cindex = curr_total_cells - 1;

        cells.push_back(new_cell_id);

        cell_indices.push_back(new_cell_index);

        //add it to its structures
        pl[k_][new_cell_id] = curr_total_cells - 1;
        
    }
    
    //commented out because to check existence you will still need to compare with the end address of the map
    //    my_hash_map<Cell_id,Cell_index*>::type::iterator search_cell(Cell_id new_cell_id){
    //        //allows you to search for a cell and returns its iterator if its successful otherwise,
    //
    //        auto cell_it = pl[new_cell_id.k].find(new_cell_id);
    //
    //
    //        return cell_it;
    //    }
    
};
class Data_set_id{
public:
    void* vec;
    unsigned int data_index;
    std::string data_type;
    int print_flag;
    
    Data_set_id(void* vec,int data_index,std::string data_type,int print_flag):vec(vec),data_index(data_index),data_type(data_type),print_flag(print_flag){}
    Data_set_id(){}
};


class Data_manager{

    
protected:
    
    //data containers
    std::vector<Part_data<bool>> part_data_bool;
    std::vector<Part_data<float>> part_data_float;
    std::vector<Part_data<uint16_t>> part_data_uint16;
    std::vector<Part_data<int16_t>> part_data_int16;
    std::vector<Part_data<int>> part_data_int;
    std::vector<Part_data<uint8_t>> part_data_uint8;
    std::vector<Part_data<int8_t>> part_data_int8;
    std::vector<Part_data<std::string>> part_data_string;
    
public:
    
    Data_manager(){};
    
    my_hash_map<std::string,Data_set_id>::type part_data_list;
    
    void create_bool_dataset(std::string data_set_name,int data_set_size){
        //creates a particle dataset
        Part_data<bool> new_data_set(data_set_name,data_set_size);
        new_data_set.data_type = "bool";
        part_data_bool.push_back(new_data_set);
        Data_set_id data_id(&part_data_bool,(unsigned int)(part_data_bool.size()-1),"bool",0);
        part_data_list[data_set_name] =  data_id;
    }
    
    void create_float_dataset(std::string data_set_name,int data_set_size){
        //creates a particle dataset
        Part_data<float> new_data_set(data_set_name,data_set_size);
        new_data_set.data_type = "float";
        part_data_float.push_back(new_data_set);
        Data_set_id data_id(&part_data_float,(unsigned int)(part_data_float.size()-1),"float",0);
        part_data_list[data_set_name] =  data_id;
    }
    
    void create_uint16_dataset(std::string data_set_name,int data_set_size){
        //creates a particle dataset
        Part_data<uint16_t> new_data_set(data_set_name,data_set_size);
        new_data_set.data_type = "uint16_t";
        part_data_uint16.push_back(new_data_set);
        Data_set_id data_id(&part_data_uint16,(unsigned int)(part_data_uint16.size()-1),"uint16_t",0);
        part_data_list[data_set_name] =  data_id;
    }
    
    void create_int16_dataset(std::string data_set_name,int data_set_size){
        //creates a particle dataset
        Part_data<int16_t> new_data_set(data_set_name,data_set_size);
        new_data_set.data_type = "int16_t";
        part_data_int16.push_back(new_data_set);
        Data_set_id data_id(&part_data_int16,(unsigned int)(part_data_int16.size()-1),"int16_t",0);
        part_data_list[data_set_name] =  data_id;
    }
    
    void create_int_dataset(std::string data_set_name,int data_set_size){
        //creates a particle dataset
        Part_data<int> new_data_set(data_set_name,data_set_size);
        new_data_set.data_type = "int";
        part_data_int.push_back(new_data_set);
        Data_set_id data_id(&part_data_int,(unsigned int)(part_data_int.size()-1),"int",0);
        part_data_list[data_set_name] =  data_id;
    }
    
    void create_uint8_dataset(std::string data_set_name,int data_set_size){
        //creates a particle dataset
        Part_data<uint8_t> new_data_set(data_set_name,data_set_size);
        new_data_set.data_type = "uint8_t";
        part_data_uint8.push_back(new_data_set);
        Data_set_id data_id(&part_data_uint8,(unsigned int)(part_data_uint8.size()-1),"uint8_t",0);
        part_data_list[data_set_name] =  data_id;
    }
    
    void create_int8_dataset(std::string data_set_name,int data_set_size){
        //creates a particle dataset
        Part_data<int8_t> new_data_set(data_set_name,data_set_size);
        new_data_set.data_type = "int8_t";
        part_data_int8.push_back(new_data_set);
        Data_set_id data_id(&part_data_int8,(unsigned int)(part_data_int8.size()-1),"int8_t",0);
        part_data_list[data_set_name] =  data_id;
    }
    
    void create_string_dataset(std::string data_set_name,int data_set_size){
        //creates a particle dataset
        Part_data<std::string> new_data_set(data_set_name,data_set_size);
        new_data_set.data_type = "string";
        part_data_string.push_back(new_data_set);
        Data_set_id data_id(&part_data_string,(unsigned int)(part_data_string.size()-1),"string",0);
        part_data_list[data_set_name] =  data_id;
    }
    
    


    template<typename T>
    Part_data<T>* get_data_ref(std::string data_set_name){
        //returns a pointer to the dataset
        Data_set_id data_id;
        
        
        auto data_id_ref = part_data_list.find(data_set_name);
        
        if (data_id_ref != part_data_list.end()) {
            //dataset doesn't exist
            
            data_id = data_id_ref->second;
            
            std::vector<Part_data<T>>* main_ptr = static_cast<std::vector<Part_data<T>>*>(data_id.vec);
            
            return &((*main_ptr)[data_id.data_index]);
            
        } else {
            //dataset doesn't exist
            return nullptr;
            
        }
        
        
    }




};




////////////////////////////////////////////////////////////////////////////////////////
//
//  Particle Representation: Main Data Structure
//
//////////////////////////////////////////////////////////////////////////////////////
class Part_rep: public Data_manager {
    //
    //
    //  Main data structure for the APR.
    //
    //
    
public:
    
    //part_list map
    Part_map pl_map;
    
    
    
    //processing parameters
    Proc_par pars;
    
    float len_scale;
    
    //original image dimensions
    std::vector<int> org_dims;
    
    Part_data<uint16_t> Ip;
    Part_data<uint8_t> status;
    
    Part_data<uint16_t>* x;
    Part_data<uint16_t>* y;
    Part_data<uint16_t>* z;
    
    Part_data<uint8_t>* k;
    
    int num_parts;
    
    Part_timer timer;
    
    
    //init
    Part_rep(): Data_manager() {
        org_dims.resize(3,0);
    }
    
    Part_rep(int y_num,int x_num,int z_num): Data_manager() {

        initialize(y_num, x_num, z_num);

        pl_map.init_part_map();
    }

    void initialize(int y_num,int x_num,int z_num) {
        org_dims.resize(3,0);

        org_dims[0] = y_num;
        org_dims[1] = x_num;
        org_dims[2] = z_num;

        int max_dim;
        int min_dim;

        if(z_num == 1) {
            max_dim = (std::max(org_dims[1], org_dims[0]));
            min_dim = (std::min(org_dims[1], org_dims[0]));
        }
        else{
            max_dim = std::max(std::max(org_dims[1], org_dims[0]), org_dims[2]);
            min_dim = std::min(std::min(org_dims[1], org_dims[0]), org_dims[2]);
        }


        int k_max_ = ceil(M_LOG2E*log(max_dim)) - 1;
        int k_min_ = std::max( (int)(k_max_ - floor(M_LOG2E*log(min_dim)) + 1),2);

        pl_map.k_max = k_max_;
        pl_map.k_min = k_min_;

        //needs to be changed for anisotropic sampling
        len_scale = pars.dx*pow(2,k_max_+1);
    }

    unsigned long long int get_active_cell_num(){
        return std::count_if(status.data.begin(), status.data.end(), [](uint8_t & p) {
            return p > 1;
        });
    }
    
    unsigned long long int get_cell_num(){
        return pl_map.cells.size();
    }
    
    unsigned long long int get_part_num(){
        return Ip.data.size();
    }
    
    int get_num_parts_cell(uint8_t status);
    
    void get_part_coords_from_cell(Cell_id curr_cell_id,uint8_t status,std::vector<std::vector<int>>& co_ords,int k_max,int& num_parts);
    
    void get_all_part_co_ords();
    
};
//get the co_ordinates of all the particles

namespace std {

    template <>
    struct hash<Cell_id>
    {
        std::size_t operator()(const Cell_id& c_id) const
        {
            using std::size_t;
            using std::hash;
            using std::string;

            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:

            return ((hash<uint8_t>()(c_id.k)
                     ^ (hash<uint16_t>()(c_id.x) << 1)) >> 1)
                   ^ (hash<uint16_t>()(c_id.y) << 1) ^ (hash<uint16_t>()(c_id.z) << 1);
        }
    };

}

bool operator==(const Cell_id & lhs, const Cell_id & rhs);
bool operator<(const Cell_id & lhs, const Cell_id & rhs);

void get_all_part_k(Part_rep& p_rep,Part_data<uint8_t>& k_vec);
void get_all_part_type(Part_rep& p_rep,Part_data<uint8_t>& type_vec);
void get_cell_properties(Part_rep& p_rep,Part_data<uint16_t>& y_coords_cell,Part_data<uint16_t>& x_coords_cell,
                         Part_data<uint16_t>& z_coords_cell,Part_data<uint8_t>& k_vec,Part_data<uint8_t>& type_vec,
                         int type_or_status = 1,int raw_coords = 0);


#endif
