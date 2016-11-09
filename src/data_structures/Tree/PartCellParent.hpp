///////////////////
//
//  Bevan Cheeseman 2016
//
//  Access class for accessing particle information from PartCellStructure
//
///////////////

#ifndef PARTPLAY_PARENT_HPP
#define PARTPLAY_PARENT_HPP

#include "PartCellNeigh.hpp"
#include "PartCellData.hpp"


// Parent / Child Relation Nodes
#define PARENT_MASK ((((uint64_t)1) << 12) - 1) << 0
#define PARENT_SHIFT 0

#define CHILD1_MASK ((((uint64_t)1) << 13) - 1) << 12
#define CHILD1_SHIFT 12

#define CHILD1_MASK ((((uint64_t)1) << 13) - 1) << 25
#define CHILD1_SHIFT 25

#define CHILD1_MASK ((((uint64_t)1) << 13) - 1) << 38
#define CHILD1_SHIFT 38

#define CHILD1_MASK ((((uint64_t)1) << 13) - 1) << 51
#define CHILD1_SHIFT 51



// type T data structure base type
template <typename T>
class PartCellParent {
    
public:
    
    /*
     * Size of the model in number of particles that fit and number of possible indices
     */
    
    /*
     * Number of layers without the root and the contents.
     */
    
    PartCellParent(){};
    
    PartCellData<T> neigh_info;
    PartCellData<T> parent_info;
    
    
private:
    
    
    
};

#endif //PARTPLAY_PARTKEY_HPP