//
// Created by Ulrik Guenther on 20/02/17.
//

#ifndef PARTPLAY_OBJECT_H
#define PARTPLAY_OBJECT_H

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <vector>

class Object {

protected:
    glm::mat4 model = glm::diagonal4x4(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    glm::vec3 position;
    glm::fquat rotation = glm::fquat(1.0f, 0.0f, 0.0f, 0.0f);

    std::vector<Object>* children = new std::vector<Object>();

public:
    Object(glm::vec3 position = glm::vec3(0.0f), glm::fquat rotation = glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));
    ~Object();

    glm::vec3 getPosition();
    glm::fquat getRotation();
    glm::mat4* getModel();
    std::vector<Object>* getChildren();
};

#endif //PARTPLAY_OBJECT_H
