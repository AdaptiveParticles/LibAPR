///////////////////
//
//  Bevan Cheeseman 2016
//
//  Access class for accessing particle information from PartCellStructure
//
///////////////

#ifndef PARTPLAY_PARTKEY_HPP
#define PARTPLAY_PARTKEY_HPP
// type T data structure base type
class PartKey {
    
public:
    
    /*
     * Size of the model in number of particles that fit and number of possible indices
     */
    
    /*
     * Number of layers without the root and the contents.
     */
    unsigned int depth;
    unsigned int z;
    unsigned int x;
    unsigned int j;
    unsigned int index;
    
    PartKey(){};
    
    
private:
    
    
    
};

#endif //PARTPLAY_PARTKEY_HPP