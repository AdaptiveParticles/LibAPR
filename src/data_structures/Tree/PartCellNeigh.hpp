///////////////////
//
//  Bevan Cheeseman 2016
//
//  Class for storing and accessing cell or particle neighbours
//
///////////////

#ifndef PARTPLAY_PARTCELLNEIGH_HPP
#define PARTPLAY_PARTCELLNEIGH_HPP
 // type T data structure base type

#include "PartCellData.hpp"

#define NUM_FACES 6

template<typename T>
class PartCellNeigh {
    
public:
    
    std::vector<std::vector<T>> neigh_face; //the neighbours arranged by face
    
    T curr; //current cell or particle
    
    PartCellNeigh(){
        neigh_face.resize(NUM_FACES);
        curr = 0;
    };
    
    void get_coordinates_cell(T face,T index,T current_y,T& neigh_y,T& neigh_x,T& neigh_z,T& neigh_depth,PartCellData<T>& pc_data){
        //
        //  Get the coordinates for a cell
        //
        
        T neigh = neigh_face[face][index];
        
        if(neigh > 0){
            neigh_x = pc_data.pc_key_get_x(neigh);
            neigh_z = pc_data.pc_key_get_z(neigh);
            neigh_depth = pc_data.pc_key_get_depth(neigh);
            
            T curr_depth = pc_data.pc_key_get
            
            if(neigh_depth == curr_depth){
                //neigh is on same layer
                neigh_y = current_y + pc_data.von_neumann_y_cells[face];
            }
            else if (neigh_depth > curr_depth){
                //neigh is on parent layer
                neigh_y = (current_y + pc_data.von_neumann_y_cells[face])/2;
            }
            else{
                //neigh is on child layer
                neigh_y = (current_y + pc_data.von_neumann_y_cells[face])*2 + pc_data.neigh_child_y_offsets[face][index];
            }
            
            
        } else {
            neigh_y = 0;
            neigh_x = 0;
            neigh_z = 0;
            neigh_depth = 0;
        }
        
        
    }
    
    void get_coordinates_part(T current_y,T& neigh_y,T& neigh_x,T& neigh_z,T& neigh_depth){
        //
        //  Get the coordinates for a particle
        //
        //
        
    }
    
    
private:
    
};

#endif //PARTPLAY_PARTNEIGH_HPP