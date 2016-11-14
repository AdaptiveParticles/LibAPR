///////////////////
//
//  Bevan Cheeseman 2016
//
//  Parent Child Cell Access Structure for Parent (Ghost) nodes (Can be used for doing operations that use the APR as an octtree)
//
///////////////

#ifndef PARTPLAY_PARENT_HPP
#define PARTPLAY_PARENT_HPP

#include "PartCellNeigh.hpp"
#include "PartCellData.hpp"

// Parent / Child Relation Nodes (parent_info)
#define PARENT_MASK ((((uint64_t)1) << 12) - 1) << 0
#define PARENT_SHIFT 0

// First set of children
#define CHILD0_MASK ((((uint64_t)1) << 13) - 1) << 12 // y0,x0,z0
#define CHILD0_SHIFT 12

#define CHILD2_MASK ((((uint64_t)1) << 13) - 1) << 25 // y0,x1,z0
#define CHILD2_SHIFT 25

#define CHILD4_MASK ((((uint64_t)1) << 13) - 1) << 38 // y0,x0,z1
#define CHILD4_SHIFT 38

#define CHILD6_MASK ((((uint64_t)1) << 13) - 1) << 51 // y0,x1,z1
#define CHILD6_SHIFT 51

// Second set of children
#define CHILD1_MASK ((((uint64_t)1) << 13) - 1) << 12 // y0,x0,z0
#define CHILD1_SHIFT 12

#define CHILD3_MASK ((((uint64_t)1) << 13) - 1) << 25 // y0,x1,z0
#define CHILD3_SHIFT 25

#define CHILD5_MASK ((((uint64_t)1) << 13) - 1) << 38 // y0,x0,z1
#define CHILD5_SHIFT 38

#define CHILD7_MASK ((((uint64_t)1) << 13) - 1) << 51 // y0,x1,z1
#define CHILD7_SHIFT 51

// CHILD STATUS (To be used with the regular part data access strategy) (neigh_info)

#define REAL_CHILDREN 2
#define GHOST_CHILDREN 1

#define CHILD0_REAL_MASK ((((uint64_t)1) << 1) - 1) << 0
#define CHILD0_REAL_SHIFT 0

#define CHILD1_REAL_MASK ((((uint64_t)1) << 1) - 1) << 1
#define CHILD1_REAL_SHIFT 1

#define CHILD2_REAL_MASK ((((uint64_t)1) << 1) - 1) << 2
#define CHILD2_REAL_SHIFT 2

#define CHILD3_REAL_MASK ((((uint64_t)1) << 1) - 1) << 3
#define CHILD3_REAL_SHIFT 3

#define CHILD4_REAL_MASK ((((uint64_t)1) << 1) - 1) << 4
#define CHILD4_REAL_SHIFT 4

#define CHILD5_REAL_MASK ((((uint64_t)1) << 1) - 1) << 5
#define CHILD5_REAL_SHIFT 5

#define CHILD6_REAL_MASK ((((uint64_t)1) << 1) - 1) << 6
#define CHILD6_REAL_SHIFT 6

#define CHILD7_REAL_MASK ((((uint64_t)1) << 1) - 1) << 7
#define CHILD7_REAL_SHIFT 7

// type T data structure base type
template <typename T>
class PartCellParent {
    
public:
    
    
    
    template<typename S>
    T find_partcell(T& x,T& y,T& z,PartCellStructure<S,T>& pc_struct){
        //
        //  Given x,y,z will find the responsible particle cell
        //
        //
        
        T curr_key = 0;
        
        T factor =pow(2,pc_struct.depth_max + 1 - pc_struct.depth_min);
        //calculate on min layer
        T y_min = y/factor;
        T x_min = x/factor;
        
        T z_min = z/factor;
        
        if( (y_min > pc_struct.y_num[pc_struct.depth_min]) | (x_min > pc_struct.x_num[pc_struct.depth_min]) | (z_min > pc_struct.z_num[pc_struct.depth_min])  ){
            return 0; //out of bounds
        }
        
        T j = neigh_info.get_j_from_y(x_min,z_min,pc_struct.depth_min,y_min);
        
        
        if (j == 0){
            // the parent wasn't found so it must be in a real seed at lowest resolution
            T j_pc = pc_struct.pc_data.get_j_from_y(x_min,z_min,pc_struct.depth_min,y_min);
            if(j_pc == 0){
                return 0;
            } else {
                
                T depth_min = pc_struct.depth_min;
                pc_struct.pc_data.set_details_cell(curr_key,x_min,z_min,j_pc,depth_min);
                
                return curr_key;
            }
            
        }
        
        
        
        
        neigh_info.pc_key_set_x(curr_key,x_min);
        neigh_info.pc_key_set_z(curr_key,z_min);
        neigh_info.pc_key_set_j(curr_key,j);
        neigh_info.pc_key_set_depth(curr_key,pc_struct.depth_min);
        
        std::vector<uint64_t> children_keys;
        std::vector<uint64_t> children_flag;
        uint64_t index;
        uint64_t y_curr;
        uint64_t x_curr;
        uint64_t z_curr;
        
        uint64_t child_y;
        uint64_t child_x;
        uint64_t child_z;
        uint64_t child_depth;
        
        for(int i = pc_struct.depth_min; i < pc_struct.depth_max; i++){
            
            get_children_keys(curr_key,children_keys,children_flag);
            
            factor =pow(2,pc_struct.depth_max + 1 - i - 1);
            //calculate on min layer
            y_curr = y/factor;
            x_curr = x/factor;
            
            z_curr = z/factor;
            
            index = 4*(z_curr&1) + 2*(x_curr&1) + (y_curr&1);
            
            curr_key = children_keys[index];
            
            get_child_coordinates_cell(children_keys,index,y_curr/2,child_y,child_x,child_z,child_depth);
            
            curr_key = children_keys[index];
            
            if (children_flag[index] == 1){
                //found the cell;
                break;
                
            }
        }
        
        return curr_key;
        
    }
    
    
    T get_parent_j(const T& node_val_parent_info){
        //returns the parent j
        return ((node_val_parent_info & PARENT_MASK) >> PARENT_SHIFT);
    }
    
    T get_parent_key(const T& node_val_parent_info,const T& curr_key){
        //returns the parent access key
        
        T parent_key = 0;
        if ((parent_info.pc_key_get_depth(curr_key)-1) >= parent_info.depth_min){
            parent_info.pc_key_set_x(parent_key,parent_info.pc_key_get_x(curr_key)/2);
            parent_info.pc_key_set_z(parent_key,parent_info.pc_key_get_z(curr_key)/2);
            parent_info.pc_key_set_depth(parent_key,parent_info.pc_key_get_depth(curr_key)-1);
            parent_info.pc_key_set_j(parent_key,get_parent_j(node_val_parent_info));
        }
        return parent_key;
    }
    
    T get_child_real_ind(const T& node_val_parent_info2,T index){
        //
        //  Returns whether or not the child is a real particle cell or a ghost
        //
        
        constexpr uint64_t child_real_mask[8] = {CHILD0_REAL_MASK,CHILD1_REAL_MASK,CHILD2_REAL_MASK,CHILD3_REAL_MASK,CHILD4_REAL_MASK,CHILD5_REAL_MASK,CHILD6_REAL_MASK,CHILD7_REAL_MASK};
        constexpr uint64_t child_real_shift[8] =  {CHILD0_REAL_SHIFT,CHILD1_REAL_SHIFT,CHILD2_REAL_SHIFT,CHILD3_REAL_SHIFT,CHILD4_REAL_SHIFT,CHILD5_REAL_SHIFT,CHILD6_REAL_SHIFT,CHILD7_REAL_SHIFT};
        
        return ((node_val_parent_info2 & child_real_mask[index]) >> child_real_shift[index]);
        
        
    }
    
    void get_children_keys(const T& curr_key,std::vector<T>& children_keys,std::vector<T>& children_flag){
        //
        //  Returns the children keys from the structure
        //
        //  Ordering follows particle ordering ((0,0,0),(1,0,0),(0,1,0),(1,1,0),..) y,x then z
        //
        
        constexpr uint64_t child_mask[8] = {CHILD0_MASK,CHILD1_MASK,CHILD2_MASK,CHILD3_MASK,CHILD4_MASK,CHILD5_MASK,CHILD6_MASK,CHILD7_MASK};
        constexpr uint64_t child_shift[8] =  {CHILD0_SHIFT,CHILD1_SHIFT,CHILD2_SHIFT,CHILD3_SHIFT,CHILD4_SHIFT,CHILD5_SHIFT,CHILD6_SHIFT,CHILD7_SHIFT};
        
        T parent_val = parent_info.get_val(curr_key);
        T parent_val2 = parent_info2.get_val(curr_key);
        
        children_keys.resize(8,0);
        children_flag.resize(8,0);
        
        T child_x = parent_info.pc_key_get_x(curr_key)*2;
        T child_z = parent_info.pc_key_get_z(curr_key)*2;
        T child_depth = parent_info.pc_key_get_depth(curr_key)+1;
        
        uint64_t child_index = 0;
        
        //loop over and set variables
        for(int p = 0; p < 8;p++){
            
            if(p&1){
                //odd
                child_index = ((parent_val2 & child_mask[p]) >> child_shift[p]);
            } else {
                child_index = ((parent_val & child_mask[p]) >> child_shift[p]);
            }
            
            if(child_index > 0){
            
                parent_info.pc_key_set_x(children_keys[p],child_x + parent_info.seed_part_x[p]);
                parent_info.pc_key_set_z(children_keys[p],child_z + parent_info.seed_part_z[p]);
                parent_info.pc_key_set_depth(children_keys[p],child_depth);
                children_flag[p] = get_child_real_ind(parent_val2,p); //indicates which structure the child is in
                parent_info.pc_key_set_j(children_keys[p],child_index);
            }
        }
        
    }
    
    void get_child_coordinates_cell(std::vector<T>& cell_children,T index,T current_y,T& child_y,T& child_x,T& child_z,T& child_depth){
        //
        //  Get the coordinates for a child cell
        //
        
        T child = cell_children[index];
        
        
        if(child > 0){
            child_x = parent_info.pc_key_get_x(child);
            child_z = parent_info.pc_key_get_z(child);
            child_depth = parent_info.pc_key_get_depth(child);
            child_y = 2*current_y + parent_info.seed_part_y[index];
        } else {
            child_y = 0;
            child_x = 0;
            child_z = 0;
            child_depth = 0;
        }
        
        
    };
    
    void get_neighs_parent_all(const uint64_t& curr_key,uint64_t node_val,PartCellNeigh<uint64_t>& neigh_keys){
        // Selects the neighbour in the correct direction
        
        neigh_keys.curr = curr_key;
        
        neigh_keys.neigh_face[0].resize(0);
        neigh_keys.neigh_face[1].resize(0);
        neigh_keys.neigh_face[2].resize(0);
        neigh_keys.neigh_face[3].resize(0);
        neigh_keys.neigh_face[4].resize(0);
        neigh_keys.neigh_face[5].resize(0);
        
        get_neighs_face_parent_t<0>(curr_key,node_val,neigh_keys.neigh_face[0]);
        get_neighs_face_parent_t<1>(curr_key,node_val,neigh_keys.neigh_face[1]);
        get_neighs_face_parent_t<2>(curr_key,node_val,neigh_keys.neigh_face[2]);
        get_neighs_face_parent_t<3>(curr_key,node_val,neigh_keys.neigh_face[3]);
        get_neighs_face_parent_t<4>(curr_key,node_val,neigh_keys.neigh_face[4]);
        get_neighs_face_parent_t<5>(curr_key,node_val,neigh_keys.neigh_face[5]);
        
    }
    
    
    PartCellParent(){};
    
    template<typename S>
    PartCellParent(PartCellStructure<S,T>& pc_struct){
        initialize_parents(pc_struct);
    };
    
    PartCellData<T> neigh_info;
    PartCellData<T> parent_info;
    PartCellData<T> parent_info2;
    
    
private:
    //counter for number of parent cells, should be set on intialization
    uint64_t number_parent_cells;
    
    void set_parent_relationships(){
        //
        //  Sets the parent relationships
        //
        //
        //
        
        Part_timer timer;
        timer.verbose_flag = true;
        uint64_t z_;
        uint64_t x_;
        uint64_t j_;
        
        timer.start_timer("Get parent child");
        
        unsigned int y_parent;
        uint64_t j_parent;
       
        
        uint64_t node_val;
        uint64_t y_coord;
        
        
        for(int i = (parent_info.depth_min+1);i <= parent_info.depth_max;i++){
            
            const unsigned int x_num_ = parent_info.x_num[i];
            const unsigned int z_num_ = parent_info.z_num[i];
            
            const unsigned int x_num_parent = parent_info.x_num[i-1];
            
            
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,y_coord) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < (z_num_);z_++){
                
                for(x_ = 0;x_ < (x_num_);x_++){
                    
                    const size_t z_parent = (z_)/2;
                    const size_t x_parent = (x_)/2;
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t offset_pc_data_parent = x_num_parent*z_parent + x_parent;
                    
                    
                    //initialization
                    y_coord = (neigh_info.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT ;
                    y_parent = (neigh_info.data[i-1][offset_pc_data_parent][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                    
                    j_parent = 1;
    
                    if (neigh_info.data[i-1][offset_pc_data_parent].size() == 1){
                        //set to max so its not checked
                        y_parent = 64000;
                    }
                    
                    y_coord--;
                    
                    const size_t j_num = neigh_info.data[i][offset_pc_data].size();
                    const size_t j_num_parent = neigh_info.data[i-1][offset_pc_data_parent].size();
                    
                    
                    if((x_ == 5) & (z_ == 1) & (i == 4)){
                        int stop = 1;
                    }
                    
                    for(j_ = 1;j_ < j_num;j_++){
                        
                        // Parent relation
                        
                        node_val = neigh_info.data[i][offset_pc_data][j_];
                        
                        
                        
                        
                        
                        if (node_val&1){
                            //get the index gap node
                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_coord--;
                            
                        } else {
                            //normal node
                            y_coord++;
                            
                            while ((y_parent < y_coord/2) & (j_parent < (j_num_parent-1))){
                                
                                j_parent++;
                                node_val = neigh_info.data[i-1][offset_pc_data_parent][j_parent];
                                
                                if (node_val&1){
                                    //get the index gap node
                                    y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                    j_parent++;
                                    
                                } else {
                                    //normal node
                                    y_parent++;
                                }
                            }
                            
                            
                            if (y_coord/2 == y_parent){
                                //set the link to the parent
                                
                                parent_info.data[i][offset_pc_data][j_] |= (j_parent << PARENT_SHIFT);
                                
                                //symmetric
                                if(y_coord == y_parent*2){
                                    //only add parent once
                                    //need to choose the correct one... formulae
                                    if(z_ == z_parent*2){
                                        
                                        if(x_ == x_parent*2){
                                            parent_info.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD0_SHIFT);
                                        } else {
                                            parent_info.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD2_SHIFT);
                                        }
                                        
                                    } else {
                                        if(x_ == x_parent*2){
                                            parent_info.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD4_SHIFT);
                                        } else {
                                            parent_info.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD6_SHIFT);
                                        }
                                        
                                    }
                                    
                                } else {
                                    if(z_ == z_parent*2){
                                        
                                        if(x_ == x_parent*2){
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD1_SHIFT);
                                        } else {
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD3_SHIFT);
                                        }
                                        
                                    } else {
                                        if(x_ == x_parent*2){
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD5_SHIFT);
                                        } else {
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD7_SHIFT);
                                        }
                                        
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    
    void set_parent_real_relationships(PartCellData<T>& pc_data){
        //
        //  Sets the parent relationships
        //
        //
        //
        
        Part_timer timer;
        timer.verbose_flag = true;
        uint64_t z_;
        uint64_t x_;
        uint64_t j_;
        
        timer.start_timer("Get parent child real");
        
        unsigned int y_parent;
        uint64_t j_parent;
        
        
        uint64_t node_val;
        uint64_t y_coord;
        
        //first loop over the nodes and set the indicators all to 0
        
       
        
        //
        //  loops simultaneously over the real structure and the neigh parent structure, to create the real links, and set the correct indicator variable.
        //
        
        
        for(int i = (pc_data.depth_min+1);i <= pc_data.depth_max;i++){
            
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            
            const unsigned int x_num_parent = parent_info.x_num[i-1];
            
            //Don't parallize this loop due to race condition on variable read/write
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,y_coord) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < (z_num_);z_++){
                
                for(x_ = 0;x_ < (x_num_);x_++){
                    
                    const size_t z_parent = (z_)/2;
                    const size_t x_parent = (x_)/2;
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t offset_pc_data_parent = x_num_parent*z_parent + x_parent;
                    
                    
                    //initialization
                    y_coord = (pc_data.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT ;
                    y_parent = (neigh_info.data[i-1][offset_pc_data_parent][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                    
                    j_parent = 1;
                    
                    if (neigh_info.data[i-1][offset_pc_data_parent].size() == 1){
                        //set to max so its not checked
                        y_parent = 64000;
                    }
                    
                    y_coord--;
                    
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    const size_t j_num_parent = neigh_info.data[i-1][offset_pc_data_parent].size();
                    
                    for(j_ = 1;j_ < j_num;j_++){
                        
                        // Parent relation
                        
                        node_val = pc_data.data[i][offset_pc_data][j_];
                        
                        if (node_val&1){
                            //get the index gap node
                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_coord--;
                            
                        } else {
                            //normal node
                            y_coord++;
                            
                            while ((y_parent < y_coord/2) & (j_parent < (j_num_parent-1))){
                                
                                j_parent++;
                                node_val = neigh_info.data[i-1][offset_pc_data_parent][j_parent];
                                
                                if (node_val&1){
                                    //get the index gap node
                                    y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                    j_parent++;
                                    
                                } else {
                                    //normal node
                                    y_parent++;
                                }
                            }
                            
                            
                            if (y_coord/2 == y_parent){
                                //set the link to the parent
                                
                                //symmetric
                                if(y_coord == y_parent*2){
                                    //setting the parent child links and indicators
                                    if(z_ == z_parent*2){
                                        
                                        if(x_ == x_parent*2){
                                            parent_info.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD0_SHIFT);
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (((uint64_t)1) << CHILD0_REAL_SHIFT);
                                        } else if(x_ == (x_parent*2+1)) {
                                            parent_info.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD2_SHIFT);
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (((uint64_t)1) << CHILD2_REAL_SHIFT);
                                        }
                                        
                                    } else if(z_ == (z_parent*2+1)) {
                                        if(x_ == x_parent*2){
                                            parent_info.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD4_SHIFT);
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (((uint64_t)1) << CHILD4_REAL_SHIFT);
                                        } else if(x_ == (x_parent*2+1)) {
                                            parent_info.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD6_SHIFT);
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (((uint64_t)1)  << CHILD6_REAL_SHIFT);
                                        }
                                        
                                    }
                                    
                                } else if(y_coord == (y_parent*2+1)) {
                                    //setting the parent child indicators
                                    if(z_ == z_parent*2){
                                        
                                        if(x_ == x_parent*2){
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD1_SHIFT);
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (((uint64_t)1) << CHILD1_REAL_SHIFT);
                                        } else if(x_ == (x_parent*2+1)) {
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD3_SHIFT);
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (((uint64_t)1) << CHILD3_REAL_SHIFT);
                                        }
                                        
                                    } else if(z_ == (z_parent*2+1)){
                                        if(x_ == x_parent*2){
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD5_SHIFT);
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (((uint64_t)1) << CHILD5_REAL_SHIFT);
                                        } else if(x_ == (x_parent*2+1)) {
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (j_ << CHILD7_SHIFT);
                                            parent_info2.data[i-1][offset_pc_data_parent][j_parent] |= (((uint64_t)1) << CHILD7_REAL_SHIFT);
                                        }
                                        
                                    }
                                    
                                }
                            } else {
                                //std::cout << "BUG" << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    
    void set_neighbor_relationships(uint8_t face){
        //
        //  Neighbour function for different face for parent nodes (does not go up and down levels)
        //
        //
        //
        const uint64_t depth_mask_dir[6] = {YP_DEPTH_MASK,YM_DEPTH_MASK,XP_DEPTH_MASK,XM_DEPTH_MASK,ZP_DEPTH_MASK,ZM_DEPTH_MASK};
        const uint64_t depth_shift_dir[6] =  {YP_DEPTH_SHIFT,YM_DEPTH_SHIFT,XP_DEPTH_SHIFT,XM_DEPTH_SHIFT,ZP_DEPTH_SHIFT,ZM_DEPTH_SHIFT};
        
        const uint64_t index_shift_dir[6] = {YP_INDEX_SHIFT,YM_INDEX_SHIFT,XP_INDEX_SHIFT,XM_INDEX_SHIFT,ZP_INDEX_SHIFT,ZM_INDEX_SHIFT};
        
        
        //variables for neighbour search loops
        const uint8_t x_start_vec[6] = {0,0,0,1,0,0};
        const uint8_t x_stop_vec[6] = {0,0,1,0,0,0};
        
        const uint8_t z_start_vec[6] = {0,0,0,0,0,1};
        const uint8_t z_stop_vec[6] = {0,0,0,0,1,0};
        
        //replication of above
        const int8_t x_offset_vec[6] = {0,0,1,-1,0,0};
        const int8_t z_offset_vec[6] = {0,0,0,0,1,-1};
        
        
        //x/z variables
        const uint64_t x_start = x_start_vec[face];
        const uint64_t x_stop = x_stop_vec[face];
        const uint64_t z_start = z_start_vec[face];
        const uint64_t z_stop = z_stop_vec[face];
        const int8_t x_offset = x_offset_vec[face];
        const int8_t z_offset = z_offset_vec[face];
        const uint64_t index_shift_0 = index_shift_dir[face];
        const uint64_t depth_shift_0 = depth_shift_dir[face];
        
        const uint64_t depth_mask_0 = depth_mask_dir[face];
        
        
        //y variables
        Part_timer timer;
        timer.verbose_flag = true;
        uint64_t z_;
        uint64_t x_;
        uint64_t j_;
        
        timer.start_timer("Get neighbours dir: " + std::to_string(face));
        
        unsigned int y_neigh;
        unsigned int y_parent;
        uint64_t j_parent;
        uint64_t j_neigh;
        
        uint64_t node_val;
        uint64_t y_coord;
        
        if (face > 1){
            
            for(int i = (neigh_info.depth_min);i <= neigh_info.depth_max;i++){
                
                const unsigned int x_num_ = neigh_info.x_num[i];
                const unsigned int z_num_ = neigh_info.z_num[i];
                
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord) if(z_num_*x_num_ > 100)
                for(z_ = z_start;z_ < (z_num_-z_stop);z_++){
                    
                    for(x_ = x_start;x_ < (x_num_-x_stop);x_++){
                        
                        const size_t z_neigh = (z_+z_offset);
                        const size_t x_neigh = (x_+x_offset);
                        
                        const size_t offset_pc_data = x_num_*z_ + x_;
                        const size_t offset_pc_data_neigh = x_num_*z_neigh + x_neigh;
                        
                        //initialization
                        y_coord = (neigh_info.data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_neigh = (neigh_info.data[i][offset_pc_data_neigh][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        j_neigh = 1;
                        
                        if (neigh_info.data[i][offset_pc_data_neigh].size() == 1){
                            //set to max so its not checked
                            y_neigh = 64000;
                        }
                        
                        y_coord--;
                        
                        const size_t j_num = neigh_info.data[i][offset_pc_data].size();
                        const size_t j_num_neigh = neigh_info.data[i][offset_pc_data_neigh].size();
                        
                        for(j_ = 1;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = neigh_info.data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_coord--;
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                while ((y_neigh < y_coord) & (j_neigh < (j_num_neigh-1))){
                                    
                                    j_neigh++;
                                    node_val = neigh_info.data[i][offset_pc_data_neigh][j_neigh];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_neigh = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_neigh++;
                                        
                                    } else {
                                        //normal node
                                        y_neigh++;
                                    }
                                }
                                
                                
                                if(y_coord == y_neigh){
                                    neigh_info.data[i][offset_pc_data][j_] |= (j_neigh << index_shift_0);
                                    neigh_info.data[i][offset_pc_data][j_]&= -((depth_mask_0)+1);
                                    neigh_info.data[i][offset_pc_data][j_] |= (LEVEL_SAME << depth_shift_0);
                                    
                                } else {
                                    //std::cout << "BUG" << std::endl;
                                }
                            }
                        }
                    }
                    
                }
                
            }
            
        }
    }
    
    template<typename S>
    void initialize_parents(PartCellStructure<S,T>& pc_struct){
        //
        //
        //  From a given particle structure of active cells create
        //
        //
        
        //Initialize the structures
        
        parent_info.depth_max = pc_struct.pc_data.depth_max-1;
        parent_info.depth_min = pc_struct.pc_data.depth_min;
        
        parent_info2.depth_max = pc_struct.pc_data.depth_max-1;
        parent_info2.depth_min = pc_struct.pc_data.depth_min;
        
        neigh_info.depth_max = pc_struct.pc_data.depth_max-1;
        neigh_info.depth_min = pc_struct.pc_data.depth_min;
        
        parent_info.z_num.resize(parent_info.depth_max+1);
        parent_info.x_num.resize(parent_info.depth_max+1);
        
        parent_info.data.resize(parent_info.depth_max+1);
        
        neigh_info.z_num.resize(neigh_info.depth_max+1);
        neigh_info.x_num.resize(neigh_info.depth_max+1);
        
        neigh_info.data.resize(neigh_info.depth_max+1);
        
        for(int i = neigh_info.depth_min;i <= neigh_info.depth_max;i++){
            neigh_info.z_num[i] = pc_struct.pc_data.z_num[i];
            neigh_info.x_num[i] = pc_struct.pc_data.x_num[i];
            neigh_info.data[i].resize(neigh_info.z_num[i]*neigh_info.x_num[i]);
        }
        
        for(int i = parent_info.depth_min;i <= parent_info.depth_max;i++){
            parent_info.z_num[i] = pc_struct.pc_data.z_num[i];
            parent_info.x_num[i] = pc_struct.pc_data.x_num[i];
            parent_info.data[i].resize(parent_info.z_num[i]*parent_info.x_num[i]);
        }
        
        parent_info2.z_num = parent_info.z_num;
        parent_info2.x_num = parent_info.x_num;
        
        //temporary array needs to be created for efficient creation of the parent structure
        std::vector<std::vector<uint8_t>> parent_map;
        
        parent_map.resize(neigh_info.depth_max+1);
        
        for(int i = parent_info.depth_min;i <= parent_info.depth_max;i++){
           
            parent_map[i].resize(pc_struct.x_num[i]*pc_struct.y_num[i]*pc_struct.z_num[i]);
        }
        
        
        //now loop over the structure
        uint64_t y_coord; //keeps track of y coordinate
        //initialize
        uint64_t node_val;
        
        uint64_t x_;
        uint64_t z_;
        uint64_t j_;
        uint64_t curr_key = 0;
       

        
        for(int i = (pc_struct.pc_data.depth_max);i > pc_struct.pc_data.depth_min;i--){
            
            const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
            const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
            
            
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,y_coord) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                //set the key values
                pc_struct.pc_data.pc_key_set_z(curr_key,z_);
                pc_struct.pc_data.pc_key_set_depth(curr_key,i);
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    pc_struct.pc_data.pc_key_set_x(curr_key,x_);
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    //number of nodes on the level
                    const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                    
                    uint64_t parent_x;
                    uint64_t parent_y;
                    uint64_t parent_z;
                    
                    y_coord = 0;
                    
                    uint64_t depth;
                    
                    
                    
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        //this value encodes the state and neighbour locations of the particle cell
                        node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                        
                        if (!(node_val&1)){
                            y_coord++; //iterate y
                            
                            parent_x = x_/2;
                            parent_z = z_/2;
                            parent_y = y_coord/2;
                            depth = i - 1;
                            
                            if(parent_map[depth][parent_z*pc_struct.y_num[depth]*pc_struct.x_num[depth] + parent_x*pc_struct.y_num[depth] + parent_y] < 2){
                                
                                parent_map[depth][parent_z*pc_struct.y_num[depth]*pc_struct.x_num[depth] + parent_x*pc_struct.y_num[depth] + parent_y] = 2;
                                
                                parent_x = parent_x/2;
                                parent_z = parent_z/2;
                                parent_y = parent_y/2;
                                depth = depth - 1;
                                
                                if(depth > pc_struct.pc_data.depth_min){
                                    
                                    while(parent_map[depth][parent_z*pc_struct.y_num[depth]*pc_struct.x_num[depth] + parent_x*pc_struct.y_num[depth] + parent_y] ==0 ){
                                        
                                        parent_map[depth][parent_z*pc_struct.y_num[depth]*pc_struct.x_num[depth] + parent_x*pc_struct.y_num[depth] + parent_y] = 1;
                                        
                                        parent_x = parent_x/2;
                                        parent_z = parent_z/2;
                                        parent_y = parent_y/2;
                                        depth = depth - 1;
                                        
                                        if  (depth < pc_struct.pc_data.depth_min){
                                            break;
                                        }
                                        
                                    }
                                }
                                
                                
                            }
                            
                            
                            
                        } else {
                            //This is a gap node
                            
                            //Gap nodes store the next and previous coodinate
                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                        }
                        
                    }
                }
            }
        }
        
//        
//        Mesh_data<uint8_t> temp;
//        for(int i = neigh_info.depth_min; i <= neigh_info.depth_max;i++){
//            temp.y_num = pc_struct.y_num[i];
//            temp.x_num = pc_struct.x_num[i];
//            temp.z_num = pc_struct.z_num[i];
//            temp.mesh = parent_map[i];
//            
//            debug_write(temp,"parent_" + std::to_string(i));
//            
//        }
//        
        
        
        uint16_t curr_index;
        uint8_t status;
        uint8_t prev_ind = 0;
        uint64_t y_;
        
        // Construct the parent info
        
        for(int i = pc_struct.pc_data.depth_min;i < (pc_struct.pc_data.depth_max);i++){
            
            const unsigned int x_num_ = pc_struct.x_num[i];
            const unsigned int z_num_ = pc_struct.z_num[i];
            const unsigned int y_num_ = pc_struct.y_num[i];
            
            
//#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    size_t first_empty = 0;
                    const size_t offset_part_map = x_*y_num_ + z_*y_num_*x_num_;
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    curr_index = 0;
                    prev_ind = 0;
                    
                    //first value handle the duplication of the gap node
                    
                    status = parent_map[i][offset_part_map];
                    if((status> 0)){
                        first_empty = 0;
                    } else {
                        first_empty = 1;
                    }
                    
                    for(y_ = 0;y_ < y_num_;y_++){
                        
                        status = parent_map[i][offset_part_map + y_];
                        
                        if(status> 0){
                            curr_index+= 1 + prev_ind;
                            prev_ind = 0;
                        } else {
                            prev_ind = 1;
                        }
                    }
                    
                    if(curr_index == 0){
                        neigh_info.data[i][offset_pc_data].resize(1); //always first adds an extra entry for intialization and extra info
                        parent_info.data[i][offset_pc_data].resize(1); //always first adds an extra entry for intialization and extra info
                    } else {
                        
                        neigh_info.data[i][offset_pc_data].resize(curr_index + 2 - first_empty,0); //gap node to begin, already finishes with a gap node
                        parent_info.data[i][offset_pc_data].resize(curr_index + 2 - first_empty,0); //gap node to begin, already finishes with a gap node
                        
                    }
                }
            }
            
        }
        
        //copy across the data
        parent_info2.data.resize(parent_info.data.size());
        //copy over to parent info2
        for(int i = 0;i < parent_info.data.size();i++){
            
            parent_info2.data[i].resize(parent_info.data[i].size());
            
            for(int j = 0;j < parent_info.data[i].size();j++){
                parent_info2.data[i][j].resize(parent_info.data[i][j].size(),0);
            }
        }
    
        
        uint64_t prev_coord = 0;
        
        
        for(int i = neigh_info.depth_min;i <= neigh_info.depth_max;i++){
            
            const unsigned int x_num_ = pc_struct.x_num[i];
            const unsigned int z_num_ = pc_struct.z_num[i];
            const unsigned int y_num_ = pc_struct.  y_num[i];
            
#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind,prev_coord) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    
                    const size_t offset_part_map = x_*y_num_ + z_*y_num_*x_num_;
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    curr_index = 0;
                    prev_ind = 1;
                    prev_coord = 0;
                    
                    //initialize the first values type
                    neigh_info.data[i][offset_pc_data][0] = TYPE_GAP_END;
                    
                    for(y_ = 0;y_ < y_num_;y_++){
                        
                        status = parent_map[i][offset_part_map + y_];
                        
                        if(status> 0){
                            
                            curr_index++;
                            
                            //set starting type
                            if(prev_ind == 1){
                                //gap node
                                //set type
                                neigh_info.data[i][offset_pc_data][curr_index-1] = TYPE_GAP;
                                neigh_info.data[i][offset_pc_data][curr_index-1] |= (y_ << NEXT_COORD_SHIFT);
                                neigh_info.data[i][offset_pc_data][curr_index-1] |= ( prev_coord << PREV_COORD_SHIFT);
                                neigh_info.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                                neigh_info.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);
                                
                                curr_index++;
                            }
                            prev_coord = y_;
                            //set type
                            neigh_info.data[i][offset_pc_data][curr_index-1] = TYPE_PC;
                            
                            //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                            neigh_info.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                            neigh_info.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                            neigh_info.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                            neigh_info.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);
                            
                            //set the status
                            switch(status){
                                case GHOST_CHILDREN:
                                {
                                    neigh_info.data[i][offset_pc_data][curr_index-1] |= (GHOST_CHILDREN <<STATUS_SHIFT);
                                    break;
                                }
                                case REAL_CHILDREN:
                                {
                                    neigh_info.data[i][offset_pc_data][curr_index-1] |= (REAL_CHILDREN<<STATUS_SHIFT);
                                    break;
                                }
                            }
                            
                            prev_ind = 0;
                        } else {
                            //store for setting above
                            if(prev_ind == 0){
                                //prev_coord = y_;
                            }
                            
                            prev_ind = 1;
                            
                        }
                    }
                    
                    //Initialize the last value GAP END indicators to no neighbour
                    neigh_info.data[i][offset_pc_data][neigh_info.data[i][offset_pc_data].size()-1] = TYPE_GAP_END;
                    neigh_info.data[i][offset_pc_data][neigh_info.data[i][offset_pc_data].size()-1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                    neigh_info.data[i][offset_pc_data][neigh_info.data[i][offset_pc_data].size()-1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);
                }
            }
            
        }

        //
        //  Now set the neighbours for the neigh_info
        //
        
        set_neighbor_relationships(0);
        set_neighbor_relationships(1);
        set_neighbor_relationships(2);
        set_neighbor_relationships(3);
        set_neighbor_relationships(4);
        set_neighbor_relationships(5);
        
        //now set the parent child relationships
        set_parent_relationships();
        
        set_parent_real_relationships(pc_struct.pc_data);
        
        
        //lastly calculate number of cells
        
        
        T num_cells = 0;
        
        for(int i = neigh_info.depth_min;i <= neigh_info.depth_max;i++){
            
            const unsigned int x_num_ = neigh_info.x_num[i];
            const unsigned int z_num_ = neigh_info.z_num[i];
            
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val) reduction(+:num_cells)
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = neigh_info.data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        node_val = neigh_info.data[i][offset_pc_data][j_];
                        
                        if (!(node_val&1)){
                            //in this loop there is a cell
                            num_cells++;
                        }
                    }
                    
                }
            }
        }
        
        number_parent_cells = num_cells;
        
        
    }
    
    
    template<uint64_t face>
    void get_neighs_face_parent_t(const uint64_t& curr_key,uint64_t node_val,std::vector<uint64_t>& neigh_keys){
        //
        //  Bevan Cheeseman (2016)
        //
        //  Get all the nieghbours in direction face for parent structure
        //
        /** Get neighbours of a cell in one of the direction
         *
         *  @param curr_key    input: current key, output: neighbour key
         *  @param face        direction to follow. Possible values are [0,5]
         *                     They stand for [+y,-y,+x,-x,+z,-z] //change this ordering.. (y+ y-) are different,
         */
        //
        
        
        constexpr uint64_t index_mask_dir[6] = {YP_INDEX_MASK,YM_INDEX_MASK,XP_INDEX_MASK,XM_INDEX_MASK,ZP_INDEX_MASK,ZM_INDEX_MASK};
        constexpr uint64_t index_shift_dir[6] = {YP_INDEX_SHIFT,YM_INDEX_SHIFT,XP_INDEX_SHIFT,XM_INDEX_SHIFT,ZP_INDEX_SHIFT,ZM_INDEX_SHIFT};
        
        constexpr int8_t von_neumann_y_cells[6] = { 1,-1, 0, 0, 0, 0};
        constexpr int8_t von_neumann_x_cells[6] = { 0, 0, 1,-1, 0, 0};
        constexpr int8_t von_neumann_z_cells[6] = { 0, 0, 0, 0, 1,-1};
        
        uint64_t neigh_j;
        
        uint64_t neigh_key;
        
        // +-y direction is different
        if(face < 2){
            
            neigh_key = curr_key;
            
            neigh_info.pc_key_offset_j(neigh_key,von_neumann_y_cells[face]);
            
            node_val = neigh_info.get_val(neigh_key);
            
            if(!(node_val&1)){
                //same level
                neigh_keys.push_back(neigh_key);
                
                return;
            } else {
                return;
            }
        }
        
        //dir
        neigh_j =  neigh_info.node_get_val(node_val,index_mask_dir[face],index_shift_dir[face]);
        
        if (neigh_j > 0){
            //same level return single neighbour
            neigh_key = curr_key;
        
            neigh_info.pc_key_set_j(neigh_key, neigh_j);
            neigh_info.pc_key_offset_x(neigh_key,von_neumann_x_cells[face]);
            neigh_info.pc_key_offset_z(neigh_key,von_neumann_z_cells[face]);
        
            neigh_keys.push_back(neigh_key);
        }
        
    }
    
    
};



#endif //PARTPLAY_PARTKEY_HPP