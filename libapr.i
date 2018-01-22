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

%{
#include "src/data_structures/APR/APR.hpp"
%}

%include "src/data_structures/Mesh/MeshData.hpp"
%include "src/data_structures/APR/APR.hpp"
%include "src/data_structures/APR/APRIterator.hpp"
%include "src/data_structures/APR/ExtraParticleData.hpp"
%include "src/data_structures/APR/ExtraPartCellData.hpp"

%pointer_class(uint16_t, UInt16Pointer);

%template(FloatVec) std::vector<float>;
%template(FloatVecVec) std::vector< std::vector<float> >;
%template(FloatVecVecVec) std::vector< std::vector< std::vector<float> > >;

%template(get_particle) ExtraParticleData::get_particle<uint16_t>;
%template(set_particle) ExtraParticleData::set_particle<uint16_t>;

%template(ExtraParticleData) ExtraParticleData::ExtraParticleData<uint16_t>;
%template(ExtraParticleData) ExtraParticleData::ExtraParticleData<float>;

%template(ExtraParticleDataStd) ExtraParticleData<uint16_t>;
%template(ExtraParticleDataFloat) ExtraParticleData<float>;

%template(ExtraPartCellDataStd) ExtraPartCellData<uint16_t>;
%template(ExtraPartCellDataFloat) ExtraPartCellData<float>;

%template(APRFloat) APR<float>;
%template(APRIteratorStd) APRIterator<uint16_t>;
%template(APRStd) APR<uint16_t>;

