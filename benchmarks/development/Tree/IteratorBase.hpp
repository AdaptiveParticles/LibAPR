//////////////////////
//
//  Bevan Cheeseman 2016
//
//  Base Iterator Class
//
//////////////////////

#ifndef ITERATORBASE_HPP
#define ITERATORBASE_HPP

template <typename T>
class BaseIterator{
    
public:
    
       /*
     * Iterator definitions
     */
    typedef uint64_t value_type;
    typedef ptrdiff_t difference_type;
    typedef uint64_t* pointer;
    typedef uint64_t& reference;
    typedef std::forward_iterator_tag iterator_category;
    
    // Constructor for creating the end of any iteration.
    BaseIterator(){};
    

    BaseIterator &operator=(BaseIterator&& iterator)
    {
        // Be careful for move when using
        this -> ~BaseIterator();
        new (this) BaseIterator (std::move(iterator));
        
        return *this;
    }
    
    bool operator==(const BaseIterator &iterator)
    {
        return current == iterator.current;
    }
    
    bool operator!=(const BaseIterator &iterator)
    {
        return current != iterator.current;
    }
    
    
    uint64_t& operator*();

    uint64_t *operator->();

    BaseIterator &operator++();


    BaseIterator operator++(int);


    BaseIterator end();

    coords3d get_current_coords();


    
private:
    
    
    
    
    
    
    
 
    
    uint64_t current;
    uint8_t level;
    Tree<T> &tree;
    coords3d current_coords;
};

#endif //ITORATORBASE_HPP
