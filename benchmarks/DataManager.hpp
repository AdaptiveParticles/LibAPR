
#ifndef DataManager
#define DataManager

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <ctime>
#include <vector>
#include <map>

class Data_set_id{
public:
    void* vec;
    unsigned int data_index;
    std::string data_type;
    int print_flag;
    
    Data_set_id(void* vec,int data_index,std::string data_type,int print_flag):vec(vec),data_index(data_index),data_type(data_type),print_flag(print_flag){}
    Data_set_id(){}
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
    //  Template class for the hash table or, ordered map_inplace I use.
    //

    typedef std::map<key,val> type;
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
    
    Data_manager(){

    };

    void clearAll() {
        part_data_list.clear();
        part_data_bool.clear();
        part_data_float.clear();
        part_data_uint16.clear();
        part_data_int16.clear();
        part_data_int.clear();
        part_data_uint8.clear();
        part_data_int8.clear();
        part_data_string.clear();
    }
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
#endif