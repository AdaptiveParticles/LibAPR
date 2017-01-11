///////////////////
//
//  Bevan Cheeseman 2016
//
//  Class for iterating over neighbour arrays
//
///////////////

#ifndef PARTPLAY_NEIGHOFFSET_HPP
#define PARTPLAY_NEIGHOFFSET_HPP
// type T data structure base type

#define SAME_LEVEL 1
#define PARENT_LEVEL 2
#define CHILD_LEVEL 3


#include "../../data_structures/Tree/PartCellStructure.hpp"
#include "CurrLevel.hpp"
#include "NeighIterator.hpp"

template<typename V,typename T>
class NeighOffset {
    
private:
    
    const unsigned int dir;
    NeighIterator<V,T> neigh_same;
    NeighIterator<V,T> neigh_parent;
    
    NeighIterator<V,T> neigh_child_0;
    NeighIterator<V,T> neigh_child_1;
    NeighIterator<V,T> neigh_child_2;
    NeighIterator<V,T> neigh_child_3;
    
public:
    
    
    
    NeighOffset(unsigned int dir): dir(dir),neigh_same(0,dir),neigh_parent(1,dir),neigh_child_0(-1,dir),neigh_child_1(-2,dir),neigh_child_2(-3,dir),neigh_child_3(-4,dir){
        
        
        
    };
    
    
    template<typename U>
    void set_new_depth(CurrentLevel<U,T>& curr_level,ParticleDataNew<U, T>& part_data){
        
        neigh_same.set_new_depth(curr_level.depth,part_data);
        
        neigh_parent.set_new_depth(curr_level.depth,part_data);
        
        neigh_child_0.set_new_depth(curr_level.depth,part_data);
        neigh_child_1.set_new_depth(curr_level.depth,part_data);
        neigh_child_2.set_new_depth(curr_level.depth,part_data);
        neigh_child_3.set_new_depth(curr_level.depth,part_data);
        
    }
    
    template<typename U>
    void set_new_row(CurrentLevel<U,T>& curr_level,ParticleDataNew<U, T>& part_data){
        
        if (neigh_same.isactive_depth()) {
            neigh_same.set_new_row(curr_level,part_data);
        }
        
        if (neigh_parent.isactive_depth()) {
            neigh_parent.set_new_row(curr_level,part_data);
        }
        
        if (neigh_child_0.isactive_depth()) {
            neigh_child_0.set_new_row(curr_level,part_data);
            neigh_child_1.set_new_row(curr_level,part_data);
            neigh_child_2.set_new_row(curr_level,part_data);
            neigh_child_3.set_new_row(curr_level,part_data);
        }
        
    }
    
    template<typename U>
    void iterate(CurrentLevel<U,T>& curr_level,ParticleDataNew<U, T>& part_data){
        
        if (neigh_same.isactive_depth()) {
            neigh_same.iterate(curr_level.y,part_data);
        }
        
        if (neigh_parent.isactive_depth()) {
            neigh_parent.iterate(curr_level.y,part_data);
        }
        
        if (neigh_child_0.isactive_depth()) {
            neigh_child_0.iterate(curr_level.y,part_data);
            neigh_child_1.iterate(curr_level.y,part_data);
            neigh_child_2.iterate(curr_level.y,part_data);
            neigh_child_3.iterate(curr_level.y,part_data);
        }
        
        
    }
    
    
    template<typename S>
    S get_part(ExtraPartCellData<S>& p_data){
        
        if(neigh_same.current_flag == 1){
            return neigh_same.get_part(p_data);
            
        } else if (neigh_parent.current_flag == 1){
            return neigh_parent.get_part(p_data);
            
        } else if (neigh_child_0.current_flag == 1){
            return neigh_child_0.get_part(p_data);
        }
        
        return 0;
        
        
    }
    
    
//    template<typename S>
//    S get_part(const ExtraPartCellData<std::vector<S>>& p_data){
//        
//        if(current_flag ==SAME_LEVEL){
//            
//            if(part_xz == 0){
//                //iterating non seed particles
//                if (status_same == SEED){
//                    return seed_reduce_operator_same(p_data);
//                } else {
//                    return p_data.data[depth_same][pc_offset_same][0][part_offset_same];
//                }
//                
//                
//            } else {
//                //iterating seed particles
//                
//                if (status_same == SEED){
//                    return p_data.data[depth_same][pc_offset_same][part_xz_same][part_offset_same];
//                } else {
//                    return p_data.data[depth_same][pc_offset_same][0][part_offset_same];
//                }
//                
//            }
//            
//            
//        } else if (current_flag == PARENT_LEVEL){
//            
//            if(status_parent == SEED){
//                
//                
//                return p_data.data[depth_parent][pc_offset_parent][part_xz_parent][part_offset_parent + part_offset_1_parent];
//                
//                
//                
//            } else {
//                return p_data.data[depth_parent][pc_offset_parent][0][part_offset_parent];
//                
//                
//            }
//            
//            
//        } else if (current_flag == CHILD_LEVEL){
//            return p_data.data[depth_child][pc_offset_child][0][part_offset_child];
//        }
//        else {
//            return 0;
//        }
//        
//        
//        
//    }

//
//    template<typename U>
//    U seed_reduce_operator_same(const ExtraPartCellData<std::vector<U>>& p_data){
//        
//        U temp =p_data.data[depth_same][pc_offset_same][part_xz_1_same][part_offset_same + part_offset_1_same];
//        temp +=p_data.data[depth_same][pc_offset_same][part_xz_2_same][part_offset_same + part_offset_2_same];
//        temp +=p_data.data[depth_same][pc_offset_same][part_xz_3_same][part_offset_same + part_offset_3_same];
//        temp +=p_data.data[depth_same][pc_offset_same][part_xz_4_same][part_offset_same + part_offset_4_same];
//        return temp;
//        
//        
//    }
//    
//    template<typename U>
//    U seed_reduce_operator_child(const ExtraPartCellData<std::vector<U>>& p_data){
//        
//        U temp =p_data.data[depth_same][pc_offset_same][part_xz_1_same][part_offset_same + part_offset_1_same];
//        temp +=p_data.data[depth_same][pc_offset_same][part_xz_2_same][part_offset_same + part_offset_2_same];
//        temp +=p_data.data[depth_same][pc_offset_same][part_xz_3_same][part_offset_same + part_offset_3_same];
//        temp +=p_data.data[depth_same][pc_offset_same][part_xz_4_same][part_offset_same + part_offset_4_same];
//        return temp;
//        
//        
//    }
    

    
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP