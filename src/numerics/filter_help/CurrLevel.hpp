///////////////////
//
//  Bevan Cheeseman 2016
//
//  Class for iterating over neighbour arrays
//
///////////////

#ifndef PARTPLAY_CURRENTLEVEL_HPP
#define PARTPLAY_CURRENTLEVEL_HPP
// type T data structure base type

#include "../../data_structures/Tree/PartCellStructure.hpp"
#include "../../data_structures/Tree/ParticleDataNew.hpp"

template<typename V,typename T>
class CurrentLevel {
    
public:
    
    T status;
    T part_offset;
    T pc_offset;
    
    T j_num;
    
    T depth;
    
    T x;
    T z;
    T j;
    
    T y;
    
    T x_num;
    T z_num;
    
    T depth_max;
    T depth_min;
    
    uint16_t node_val;
    T y_num;
    
    uint64_t curr_key;
    
    T type;
    
    PartCellNeigh<uint64_t> neigh_part_keys;
    
    CurrentLevel(){
        depth = 0;
        x = 0;
        z = 0;
        j = 0;
        pc_offset = 0;
        y = 0;
        j_num = 0;
        x_num = 0;
        z_num = 0;
        status = 0;
        node_val = 0;
        
        
    };
    
    CurrentLevel(PartCellData<T>& pc_data){
        depth = 0;
        x = 0;
        z = 0;
        j = 0;
        pc_offset = 0;
        y = 0;
        j_num = 0;
        x_num = 0;
        z_num = 0;
        status = 0;
        node_val = 0;
        
        depth_max = pc_data.depth_max;
        depth_min = pc_data.depth_min;
        
    };
    
    template<typename U>
    void set_new_depth(T depth_,ParticleDataNew<U, T>& part_data){
        
        depth = depth_;
        x_num = part_data.access_data.x_num[depth];
        z_num = part_data.access_data.z_num[depth];
        
        part_data.access_data.pc_key_set_depth(curr_key,depth);
        
    }
    
    
    void init(T init_key,PartCellData<T>& pc_data){
        
        pc_key init_pc_key;
        init_pc_key.update_cell(init_key);
        
        
        depth = init_pc_key.depth;
        x_num = pc_data.x_num[depth];
        z_num = pc_data.z_num[depth];
        
        x = init_pc_key.x;
        z = init_pc_key.z;
        
        pc_offset = x_num*z + x;
        j_num = pc_data.data[depth][pc_offset].size();
        part_offset = 0;
        y = 0;
        
        j = init_pc_key.j;
        
        curr_key = init_key;
        
        node_val = pc_data.data[depth][pc_offset][j];
        
        if(!(node_val&1)){
            type = 1;
            
        } else {
            type = 0;
        }
        
    }

    
    
    template<typename U>
    void init(T x_,T z_,T j_,T depth_,ParticleDataNew<U, T>& part_data){
        
        depth = depth_;
        x_num = part_data.access_data.x_num[depth];
        z_num = part_data.access_data.z_num[depth];
        
        part_data.access_data.pc_key_set_depth(curr_key,depth);
        
        x = x_;
        z = z_;
        
        pc_offset = x_num*z + x;
        j_num = part_data.access_data.data[depth][pc_offset].size();
        part_offset = 0;
        y = 0;
        
        part_data.access_data.pc_key_set_x(curr_key,x);
        part_data.access_data.pc_key_set_z(curr_key,z);
        
        j = j_;
        
        part_data.access_data.pc_key_set_j(curr_key,j_);
        
        node_val = part_data.access_data.data[depth][pc_offset][j_];
        
        
        if(!(node_val&1)){
            type = 1;
            
            status = part_data.access_node_get_status(node_val);
            
            y++;
            
            //seed offset accoutns for which (x,z) you are doing
            part_offset = part_data.access_node_get_part_offset(node_val);
            
        } else {
            type = 0;
            y += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
            y--;

        }
        
    }
    
    template<typename U>
    bool move_cell(unsigned int dir,unsigned int index, ParticleDataNew<U, T>& part_data,PartCellData<uint64_t>& pc_data){
        
        bool edge_domain = false;
        
        // get neigh
        update_neigh(dir,pc_data);
        
        uint64_t neigh_key=0;
        
        if(neigh_part_keys.neigh_face[dir].size() > index){
            neigh_key = neigh_part_keys.neigh_face[dir][index];
            
        } else if(neigh_part_keys.neigh_face[dir].size() > 0) {
            neigh_key = neigh_part_keys.neigh_face[dir][0];
        }
        
        if(neigh_key > 0){
            
            curr_key = neigh_key;
            
            int depth_prev = depth;
            
            int curr_y = y;
            
            pc_data.get_coordinates_cell(curr_y,curr_key,x,z,y,depth,status);
            
            if(depth_prev != depth){
                x_num = pc_data.x_num[depth];
                z_num = pc_data.z_num[depth];
            }
            
            pc_offset = x_num*z + x;
            j_num = pc_data.data[depth][pc_offset].size();
            
            j = pc_data.pc_key_get_j(curr_key);
            
            node_val = part_data.access_data.data[depth][pc_offset][j];
            
            status = part_data.access_node_get_status(node_val);
            
            part_offset = part_data.access_node_get_part_offset(node_val);

        } else {
            edge_domain = true;
        }
        
        return edge_domain;
    }
    
    
    template<typename U>
    bool move_cell(unsigned int dir,unsigned int index,PartCellData<uint64_t>& pc_data){
        
        bool edge_domain = false;
        
        // get neigh
        update_neigh(dir,pc_data);
        
        uint64_t neigh_key=0;
        
        if(neigh_part_keys.neigh_face[dir].size() > index){
            neigh_key = neigh_part_keys.neigh_face[dir][index];
            
        } else if(neigh_part_keys.neigh_face[dir].size() > 0) {
            neigh_key = neigh_part_keys.neigh_face[dir][0];
        }
        
        if(neigh_key > 0){
            
            curr_key = neigh_key;
            
            int depth_prev = depth;
            
            int curr_y = y;
            
            pc_data.get_coordinates_cell(curr_y,curr_key,x,z,y,depth,status);
            
            if(depth_prev != depth){
                x_num = pc_data.x_num[depth];
                z_num = pc_data.z_num[depth];
            }
            
            pc_offset = x_num*z + x;
            j_num = pc_data.data[depth][pc_offset].size();
            
            j = pc_data.pc_key_get_j(curr_key);
            
            
        } else {
            edge_domain = true;
        }
        
        return edge_domain;
    }
    
    void update_neigh(unsigned int dir,PartCellData<uint64_t>& pc_data){
        uint64_t node_val_pc = pc_data.data[depth][pc_offset][j];
        
//        pc_key d_key;
//        d_key.update_cell(curr_key);
//        
//        node_key d_node;
//        d_node.update_node(node_val_pc);
        
        
        pc_data.get_neighs_face(curr_key,node_val_pc,dir,neigh_part_keys);
    }
    
        
    template<typename U>
    U update_and_get_neigh_int(unsigned int dir,ParticleDataNew<U, T>& part_data,PartCellData<uint64_t>& pc_data){
        V node_val_pc = pc_data[depth][pc_offset][j];
        
        pc_data.get_neighs_face(curr_key,node_val_pc,dir,neigh_part_keys);
        
        U part_int=0;
        int counter=0;
        //loop over the nieghbours
        for(int n = 0; n < neigh_part_keys.neigh_face[dir].size();n++){
            // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
            uint64_t neigh_key = neigh_part_keys.neigh_face[dir][n];
            
            if(neigh_key > 0){
                //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                uint64_t part_node = part_data.access_data.get_val(neigh_key);
                uint64_t part_offset = part_data.access_node_get_part_offset(part_node);
                part_data.access_data.pc_key_set_index(neigh_key,part_offset);
                part_int += part_data.particle_data.get_part(neigh_key);
                counter++;
            }
            
            part_int = part_int/counter;
            
        }
        
        return part_int;
        
    }
    
    template<typename U>
    U get_neigh_int(unsigned int dir,ParticleDataNew<U, T>& part_data,PartCellData<uint64_t>& pc_data){
        //
        //  Need to have already gotten your neighbours using an update neighbour routine
        //
        
        U part_int=0;
        int counter=0;
        //loop over the nieghbours
        for(int n = 0; n < neigh_part_keys.neigh_face[dir].size();n++){
            // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
            uint64_t neigh_key = neigh_part_keys.neigh_face[dir][n];
            
            if(neigh_key > 0){
                //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                uint64_t part_node = part_data.access_data.get_val(neigh_key);
                uint64_t part_offset = part_data.access_node_get_part_offset(part_node);
                part_data.access_data.pc_key_set_index(neigh_key,part_offset);
                part_int += part_data.particle_data.get_part(neigh_key);
                counter++;
            }
            
            part_int = part_int/counter;
            
        }
        
        return part_int;
        
    }
    
    
    void update_all_neigh(PartCellData<uint64_t>& pc_data){
        uint64_t node_val_pc = pc_data.data[depth][pc_offset][j];
        
        pc_data.get_neighs_all(curr_key,node_val_pc,neigh_part_keys);
        
        
    }
    
    
    template<typename U>
    void set_new_xz(T x_,T z_,ParticleDataNew<U, T>& part_data){
        
        x = x_;
        z = z_;
        
        pc_offset = x_num*z + x;
        j_num = part_data.access_data.data[depth][pc_offset].size();
        part_offset = 0;
        y = 0;
        
        part_data.access_data.pc_key_set_z(curr_key,x);
        part_data.access_data.pc_key_set_z(curr_key,z);
        
        
    }
    
    template<typename U>
    bool new_j(T j_,ParticleDataNew<U, T>& part_data){
        j = j_;
        
        part_data.access_data.pc_key_set_j(curr_key,j_);
        
        node_val = part_data.access_data.data[depth][pc_offset][j_];
        
        //returns if it is a cell or not
        return !(node_val&1);
        
    }
    
    template<typename U>
    inline U& get_part(ExtraPartCellData<U>& particle_data){
        return particle_data.data[depth][pc_offset][part_offset];
    }
    
    template<typename U>
    inline U& get_val(ExtraPartCellData<U>& particle_data){
        return particle_data.data[depth][pc_offset][j];
    }
    
    template<typename U>
    U& get_part(ParticleDataNew<U, T>& part_data){
        return part_data.particle_data.data[depth][pc_offset][part_offset];
    }

    pc_key get_key(){
        pc_key curr_key;
        curr_key.depth_p = depth;
        curr_key.y_p = y;
        curr_key.x_p = x;
        curr_key.z_p = z;
        return curr_key;
    }
    
    
    template<typename U>
    void update_cell(ParticleDataNew<U, T>& part_data){
        
        status = part_data.access_node_get_status(node_val);
        
        y++;
        
        //seed offset accoutns for which (x,z) you are doing
        part_offset = part_data.access_node_get_part_offset(node_val);
        
    }
    
    
    void update_gap(){
        
        y += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
        y--;
    }
    
    

    

private:
    
    
    //offsets
    
    
    
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP