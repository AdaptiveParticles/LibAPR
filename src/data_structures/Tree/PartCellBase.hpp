///////////////////
//
//  Bevan Cheeseman 2016
//
//  Part cell base class, specifies interface
//
///////////////

#ifndef PARTPLAY_PARTCELLBASE_HPP
#define PARTPLAY_PARTCELLBASE_HPP
template <typename T,typename S> // type T is the image type, type S is the data structure base type
class PartCellBase {

public:
    
    /*
     * Size of the model in number of particles that fit and number of possible indices
     */
    
    
    /*
     * Number of layers without the root and the contents.
     */
    uint8_t depth;
    
    
    PartCellBase(){};
    
    //returns the global co_ordinates of a particle cell
    
    
    //returns the status of a particle cell
    inline uint8_t get_status(S node) const;
    
private:
    

    
};

#endif //PARTPLAY_PARTCELLBASE_HPP