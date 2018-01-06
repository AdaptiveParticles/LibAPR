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
    
    T node_val;

    T y_num;
    
    uint64_t curr_key;
    
    T type;

    uint64_t counter;
    
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
        curr_key = 0;
        
        
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
    CurrentLevel(ParticleDataNew<U, T>& part_data){
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

        depth_max = part_data.depth_max;
        depth_min = part_data.depth_min;

    };
    
    template<typename U>
    void set_new_depth(T depth_,ParticleDataNew<U, T>& part_data){
        
        depth = depth_;
        x_num = part_data.access_data.x_num[depth];
        z_num = part_data.access_data.z_num[depth];
        
        part_data.access_data.pc_key_set_depth(curr_key,depth);
        
    }

    void set_new_depth(T depth_,PartCellData<T>& pc_data){
        // updates the key with the new current depth
        depth = depth_;
        x_num = pc_data.x_num[depth];
        z_num = pc_data.z_num[depth];

        pc_data.pc_key_set_depth(curr_key,depth);

    }

    void update_cell(PartCellData<T>& pc_data){

        status = pc_data.get_status(node_val);

        pc_data.pc_key_set_status(curr_key,status);

        y++;

    }


    void update_gap(){

        y += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
        y--;
    }

    void update_gap(PartCellData<T>& pc_data){

        y = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
        y--; //set the y_coordinate to the value before the next coming up in the structure
    }


    void init(T init_key,PartCellData<T>& pc_data){

        curr_key = 0;

        pc_key init_pc_key;
        init_pc_key.update_cell(init_key);

        depth = init_pc_key.depth;
        x_num = pc_data.x_num[depth];
        z_num = pc_data.z_num[depth];
        
        x = init_pc_key.x;
        z = init_pc_key.z;

        status = init_pc_key.status;
        
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

    void init(PartCellData<T>& pc_data){

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
        counter = 1;
        y_num = 0;
        type = 0;

        depth_max = pc_data.depth_max;
        depth_min = pc_data.depth_min;


        depth = pc_data.depth_min;
        x_num = pc_data.x_num[depth];
        z_num = pc_data.z_num[depth];
        z_num = pc_data.y_num[depth];

        x = 0;
        z = 0;

        pc_offset = x_num*z + x;
        j_num = pc_data.data[depth][pc_offset].size();
        part_offset = 0;
        y = 0;

        j = 0;

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


    uint64_t init_iterate(PartCellData<T>& pc_data){
        // intialize iterator

        counter=1;

        depth_min = pc_data.depth_min;
        depth_max = pc_data.depth_max;

        set_new_depth(pc_data.depth_min,pc_data);
        set_new_x(0,pc_data);
        set_new_z(0,pc_data);
        update_j(pc_data,0);

        move_to_next_pc(pc_data);

        return counter;

    }

    bool move_to_next_pc(PartCellData<T>& pc_data){
        //move to new particle cell, if it reaches the end it returns false

        iterate_forward(pc_data);

        while((node_val&1) & (counter!=0)){
            iterate_forward(pc_data);
        }

        return (counter != 0);

    }

    bool move_to_next_pc(PartCellData<T>& pc_data,unsigned int depth){
        //move to new particle cell, if it reaches the end it returns false

        iterate_forward(pc_data,depth);

        while((node_val&1) & (counter!=0)){
            iterate_forward(pc_data,depth);
        }

        return (counter != 0);

    }


    uint64_t init_iterate(PartCellData<T>& pc_data,unsigned int depth){

        counter=1;

        depth_min = pc_data.depth_min;
        depth_max = pc_data.depth_max;

        set_new_depth(depth,pc_data);
        set_new_x(0,pc_data);
        set_new_z(0,pc_data);
        update_j(pc_data,0);

        move_to_next_pc(pc_data,depth);

        return counter;

    }

    uint64_t iterate_forward(PartCellData<T>& pc_data){
        // iterate forward

        if(j < (j_num-1)){
            //move j
            update_j(pc_data,j+1);
            counter++;

        } else {
            if(x < (x_num-1)){
                set_new_x(x+1,pc_data);
                update_j(pc_data,0);
                counter++;
                y=0;

            } else{
                if(z < (z_num-1)) {
                    set_new_z(z+1,pc_data);
                    set_new_x(0,pc_data);
                    update_j(pc_data,0);
                    counter++;
                    y=0;

                } else{

                    if(depth < depth_max){

                        set_new_depth(depth+1,pc_data);
                        set_new_z(0,pc_data);
                        set_new_x(0,pc_data);
                        update_j(pc_data,0);
                        counter++;
                        y=0;

                    } else {
                        counter = 0;
                        return 0;
                    }
                }

            }


        }

        if(!(node_val&1)){
            update_cell(pc_data);
        } else{
            update_gap(pc_data);
        }

        return counter;
    }

    uint64_t iterate_forward(PartCellData<T>& pc_data,unsigned int depth){
        // iterate forward

        if(j < (j_num-1)){
            //move j
            update_j(pc_data,j+1);
            counter++;

        } else {
            if(x < (x_num-1)){
                set_new_x(x+1,pc_data);
                update_j(pc_data,0);
                counter++;
                y=0;

            } else{
                if(z < (z_num-1)) {
                    set_new_z(z+1,pc_data);
                    set_new_x(0,pc_data);
                    update_j(pc_data,0);
                    counter++;
                    y=0;

                } else{

                    counter = 0;
                    return 0;

                }

            }


        }

        if(!(node_val&1)){
            update_cell(pc_data);
        } else{
            update_gap(pc_data);
        }

        return counter;
    }





    template<typename U>
    bool move_cell(unsigned int dir,unsigned int index,PartCellData<U>& pc_data){
        
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
    void update_get_neigh_dir(ExtraPartCellData<U>& parts,PartCellData<uint64_t>& pc_data,std::vector<U>& neigh_val,unsigned int dir){
        //
        //  Gets the neighbours in one of the six directions (dir) (y+,y_,x+,x-,z+,z-)
        //
        //

        pc_data.get_neighs_face(curr_key,node_val,dir,neigh_part_keys);

        neigh_val.resize(0);

        //loop over the nieghbours
        for(int n = 0; n < neigh_part_keys.neigh_face[dir].size();n++){
            // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
            uint64_t neigh_key = neigh_part_keys.neigh_face[dir][n];

            if(neigh_key > 0){
                //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)

                neigh_val.push_back(parts.get_val(neigh_key));

            }

        }

    }


    void update_neigh_all(PartCellData<uint64_t>& pc_data) {
        //
        //  Loops over and returns a vector with vectors  of the particles in each of the 6 directions
        //

        //get the neighbours
        pc_data.get_neighs_all(curr_key, node_val, neigh_part_keys);
    }

    void update_neigh_dir(PartCellData<uint64_t>& pc_data,unsigned int dir) {
        //
        //  Loops over and returns a vector with vectors  of the particles in each of the 6 directions
        //

        pc_data.get_neighs_face(curr_key, node_val,dir, neigh_part_keys);

    }


    template<typename U>
    void update_and_get_neigh_all(ExtraPartCellData<U>& parts,PartCellData<uint64_t>& pc_data,std::vector<std::vector<U>>& neigh_val){
        //
        //  Loops over and returns a vector with vectors  of the particles in each of the 6 directions
        //
        //


        //get the neighbours
        pc_data.get_neighs_all(curr_key,node_val,neigh_part_keys);

        U part_int=0;
        float counter=0;

        neigh_val.resize(6);

        //loop over the 6 directions
        for (int dir = 0; dir < 6; ++dir) {

            neigh_val[dir].resize(0);

            //loop over the nieghbours
            for(int n = 0; n < neigh_part_keys.neigh_face[dir].size();n++){
                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                uint64_t neigh_key = neigh_part_keys.neigh_face[dir][n];


                if(neigh_key > 0){
                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)

                    neigh_val[dir].push_back(parts.get_val(neigh_key));

                }

            }

        }


    }



    template<typename U>
    void update_and_get_neigh_all_avg(ExtraPartCellData<U>& parts,PartCellData<uint64_t>& pc_data,std::vector<std::vector<U>>& neigh_val){
        //
        //  Loops over and returns a vector with the average of the particles in each of the 6 directions
        //
        //
        // [+y,-y,+x,-x,+z,-z]
        //  [0,1,2,3,4,5]


        //get the neighbours
        pc_data.get_neighs_all(curr_key,node_val,neigh_part_keys);

        U part_int=0;
        float counter=0;

        neigh_val.resize(6);

        //loop over the 6 directions
        for (int dir = 0; dir < 6; ++dir) {

            neigh_val[dir].resize(0);

            //loop over the nieghbours
            for(int n = 0; n < neigh_part_keys.neigh_face[dir].size();n++){
                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                uint64_t neigh_key = neigh_part_keys.neigh_face[dir][n];

                part_int = 0;
                counter = 0;

                if(neigh_key > 0){
                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)

                    part_int += parts.get_val(neigh_key);
                    counter++;

                }

            }
            if(counter>0) {
                neigh_val[dir].push_back(part_int / counter);
            }
        }


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
    

    void set_new_xz(T x_,T z_,PartCellData<T>& pc_data){
        
        x = x_;
        z = z_;
        
        pc_offset = x_num*z + x;
        j_num = pc_data.data[depth][pc_offset].size();
        part_offset = 0;
        y = 0;

        pc_data.pc_key_set_x(curr_key,x);
        pc_data.pc_key_set_z(curr_key,z);

    }

    void set_new_x(T x_,PartCellData<T>& pc_data){

        x = x_;

        pc_offset = x_num*z + x;
        j_num = pc_data.data[depth][pc_offset].size();
        part_offset = 0;
        y = 0;

        pc_data.pc_key_set_x(curr_key,x);

    }

    void set_new_z(T z_,PartCellData<T>& pc_data){

        z = z_;

        pc_offset = x_num*z + x;
        j_num = pc_data.data[depth][pc_offset].size();
        part_offset = 0;
        y = 0;

        pc_data.pc_key_set_z(curr_key,z);

    }

    template<typename U>
    void set_new_xz(T x_,T z_,ParticleDataNew<U, T>& part_data){

        x = x_;
        z = z_;

        pc_offset = x_num*z + x;
        j_num = part_data.access_data.data[depth][pc_offset].size();
        part_offset = 0;
        y = 0;

        part_data.access_data.pc_key_set_x(curr_key,x);
        part_data.access_data.pc_key_set_z(curr_key,z);


    }
    

    bool new_j(T j_,PartCellData<T>& pc_data){
        j = j_;

        pc_data.pc_key_set_j(curr_key,j_);
        
        node_val = pc_data.data[depth][pc_offset][j_];
        
        //returns if it is a cell or not
        return !(node_val&1);
        
    }

    bool update_j(PartCellData<T>& pc_data,T j_){
        j = j_;

        pc_data.pc_key_set_j(curr_key,j_);

        node_val = pc_data.data[depth][pc_offset][j_];

        if(j_==0){
            y = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
            y--; //set the y_coordinate to the value before the next coming up in the structure
        }

        //returns if it is a cell or not
        return !(node_val&1);

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


    

    

private:
    
    
    //offsets
    
    
    
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP