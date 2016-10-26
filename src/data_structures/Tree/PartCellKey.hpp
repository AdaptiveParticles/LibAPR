///////////////////
//
//  Bevan Cheeseman 2016
//
//  Access class for accessing particle information from PartCellStructure
//
///////////////

#ifndef PARTPLAY_PARTCELLKEY_HPP
#define PARTPLAY_PARTCELLKEY_HPP
 // type T data structure base type
class PartCellKey {
    
public:
    
    /*
     * Size of the model in number of particles that fit and number of possible indices
     */
    
    /*
     * Number of layers without the root and the contents.
     */
    uint64_t depth;
    uint64_t z;
    uint64_t x;
    uint64_t j;
    
    PartCellKey(){};
    
    
    
private:
    
};

#endif //PARTPLAY_PARTKEY_HPP