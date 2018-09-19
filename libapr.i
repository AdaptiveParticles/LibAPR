%module apr

%include "std_vector.i"
%include "stdint.i"
%include "typemaps.i"
%include "cpointer.i"

using namespace std;
namespace std {
%template(U8Vec) std::vector<uint8_t>;
%template(UVec) std::vector<unsigned int>;
%template(U16Vec) std::vector<uint16_t>;
%template(U16VecVec) vector< vector<uint16_t> >;
%template(U16VecVecVec) vector< vector< vector<uint16_t> > >;
}

%include "std_string.i"

%rename(equals) operator==;
%rename(less_than) operator<;

%ignore Part_timer;
%ignore APRIteratorOld;

%typemap(javain) APR<uint16_t>& apr "getCPtrAndAddReference($javainput)"
%typemap(javacode) APRIterator<uint16_t> %{
  // ensure premature GC doesn't happen by storing a reference to the APR
  // in-class
  private static java.util.ArrayList<APRStd> aprReferences = new java.util.ArrayList<APRStd>();
  private static long getCPtrAndAddReference(APRStd a) {
    aprReferences.add(a);
    return APRStd.getCPtr(a);
  }
%}

%typemap(javacode) ExtraParticleData<float> %{
  // ensure premature GC doesn't happen by storing a reference to the APR
  // in-class
  private static java.util.ArrayList<APRStd> aprReferences = new java.util.ArrayList<APRStd>();
  private static long getCPtrAndAddReference(APRStd a) {
    aprReferences.add(a);
    return APRStd.getCPtr(a);
  }
%}

%typemap(javacode) ExtraParticleData<uint16_t> %{
  // ensure premature GC doesn't happen by storing a reference to the APR
  // in-class
  private static java.util.ArrayList<APRStd> aprReferences = new java.util.ArrayList<APRStd>();
  private static long getCPtrAndAddReference(APRStd a) {
    aprReferences.add(a);
    return APRStd.getCPtr(a);
  }
%}

%typemap(javacode) ExtraParticleData<std::vector<float>> %{
  // ensure premature GC doesn't happen by storing a reference to the APR
  // in-class
  private static java.util.ArrayList<APRStd> aprReferences = new java.util.ArrayList<APRStd>();
  private static long getCPtrAndAddReference(APRStd a) {
    aprReferences.add(a);
    return APRStd.getCPtr(a);
  }
%}

%{
#include "src/data_structures/APR/APR.hpp"
#include "src/numerics/APRNumerics.hpp"
%}

%include "src/data_structures/Mesh/PixelData.hpp"
%include "src/data_structures/APR/APR.hpp"
%include "src/data_structures/APR/APRIterator.hpp"
%include "src/numerics/APRNumerics.hpp"
%include "src/data_structures/APR/ExtraParticleData.hpp"
%include "src/data_structures/APR/ExtraPartCellData.hpp"

%pointer_class(uint16_t, UInt16Pointer);

%template(FloatVec) std::vector<float>;
%template(FloatVecVec) std::vector< std::vector<float> >;
%template(FloatVecVecVec) std::vector< std::vector< std::vector<float> > >;

%naturalvar FloatVec;
%naturalvar FloatVecVec;
%naturalvar FloatVecVec;

%template(read_parts_only_std) APR::read_parts_only<uint16_t>;

%template(compute_gradient_vector_std) APRNumerics::compute_gradient_vector<uint16_t>;
%template(compute_gradient_vector_float) APRNumerics::compute_gradient_vector<float>;

//%template(get_particle) ExtraParticleData::get_particle<uint16_t>;
//%template(set_particle) ExtraParticleData::set_particle<uint16_t>;

%template(ExtraParticleData) ExtraParticleData::ExtraParticleData<uint16_t>;
%template(ExtraParticleData) ExtraParticleData::ExtraParticleData<float>;

%template(ExtraParticleDataStd) ExtraParticleData<uint16_t>;
%template(ExtraParticleDataFloat) ExtraParticleData<float>;
%template(ExtraParticleDataFloatVec) ExtraParticleData<std::vector<float>>;

%template(ExtraPartCellDataStd) ExtraPartCellData<uint16_t>;
%template(ExtraPartCellDataFloat) ExtraPartCellData<float>;

%template(APRFloat) APR<float>;
%template(APRIteratorStd) APRIterator;
%template(APRStd) APR<uint16_t>;


