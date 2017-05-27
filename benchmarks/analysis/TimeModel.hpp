//
// Created by cheesema on 26.05.17.
//

#ifndef PARTPLAY_TIMEMODEL_HPP
#define PARTPLAY_TIMEMODEL_HPP


class TimeModel {

public:

    //movement of objects
    std::vector<float> move_speed;
    std::vector<std::vector<float>> location;

    //direction change
    std::vector<float> theta;
    std::vector<float> phi;
    std::vector<float> direction_speed;

    //direction change
    std::vector<float> intensity_speed;

    Genrand_uni gen_rand;

    int num_objects;

    TimeModel(int num_objects): num_objects(num_objects) {

        move_speed.resize(num_objects);
        location.resize(num_objects);
        theta.resize(num_objects);
        phi.resize(num_objects);
        direction_speed.resize(num_objects);
        intensity_speed.resize(num_objects);

        for (int i = 0; i < num_objects; ++i) {
            location[i].resize(3);
        }
    }




};


#endif //PARTPLAY_TIMEMODEL_HPP
