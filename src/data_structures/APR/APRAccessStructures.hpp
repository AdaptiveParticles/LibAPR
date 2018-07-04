#ifndef __APR_ACCESSS_STRUCTURES__
#define __APR_ACCESSS_STRUCTURES__

#include <map>
#include <vector>

struct ParticleCell {
    uint16_t x,y,z,level,type;
    uint64_t pc_offset,global_index;
};

struct YGap_map {
    uint16_t y_end;
    uint64_t global_index_begin;
};

struct ParticleCellGapMap{
    std::map<uint16_t,YGap_map> map;
};

struct MapIterator{
    std::map<uint16_t,YGap_map>::iterator iterator;
    uint64_t pc_offset;
    uint16_t level;
};

struct LocalMapIterators{
    std::vector<MapIterator>  same_level;
    std::vector<std::vector<MapIterator>>  child_level;
    std::vector<MapIterator>  parent_level;

    LocalMapIterators(){
        //initialize them to be set to pointing to no-where
        MapIterator init;
        init.pc_offset = -1;
        init.level = -1;

        same_level.resize(6,init);
        parent_level.resize(6,init);
        child_level.resize(6);
        for (int i = 0; i < 6; ++i) {
            child_level[i].resize(4,init);
        }
    }
};

struct MapStorageData{
    std::vector<uint16_t> y_begin;
    std::vector<uint16_t> y_end;
    std::vector<uint64_t> global_index;
    std::vector<uint16_t> z;
    std::vector<uint16_t> x;
    std::vector<uint8_t> level;
    std::vector<uint16_t> number_gaps;
};


#endif