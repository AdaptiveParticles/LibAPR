%module libapr

//%include "stl.i"
%include "std_vector.i"
//%include "swig_vector.i"
%include "stdint.i"

using namespace std;
namespace std {
//%template(IntegerVec) std::vector<uint16_t>;
//%template(ShortVec) std::vector<uint8_t>;
%template(U16Vec) std::vector<uint16_t>;
%template(U8Vec) std::vector<uint8_t>;
}

/*%typemap(javabase) std::vector<uint8_t> "java.util.AbstractList<Integer>"
%typemap(javainterface) std::vector<uint8_t> "java.util.RandomAccess"
%typemap(javacode) std::vector<uint8_t> %{
  public Integer get(int idx) {
    return get_impl(idx);
  }
  public Integer size() {
    return (int)size_impl();
  }
  public Integer set(int idx, Short d) {
    Integer old = get_impl(idx);
    set_impl(idx, d.intValue());
    return old;
  }

%}

%typemap(javabase) std::vector<uint16_t> "java.util.AbstractList<Integer>"
%typemap(javainterface) std::vector<uint16_t> "java.util.RandomAccess"
%typemap(javacode) std::vector<uint16_t> %{
  public Integer get(int idx) {
    return get_impl(idx);
  }
  public int size() {
    return (int)size_impl();
  }
  public Integer set(int idx, Integer d) {
    Integer old = get_impl(idx);
    set_impl(idx, d.intValue());
    return old;
  }

%}
*/

%include "std_string.i"

%rename(equals) operator==;
%rename(less_than) operator<;

%{
#include "src/data_structures/particle_map.hpp"
#include "src/data_structures/structure_parts.h"
#include "src/algorithm/gradient.hpp"
#include "src/algorithm/pipeline.h"
%}

%include "src/data_structures/particle_map.hpp"
%include "src/data_structures/structure_parts.h"
%include "src/algorithm/gradient.hpp"
%include "src/algorithm/pipeline.h"

%template(PartDataUint16) Part_data<uint16_t>;
%template(PartDataUint8) Part_data<uint8_t>;
