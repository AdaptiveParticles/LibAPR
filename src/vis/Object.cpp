//
// Created by Ulrik Guenther on 20/02/17.
//

#include "Object.h"

Object::Object(glm::vec3 position, glm::fquat rotation) {

}

glm::vec3 Object::getPosition() {
    return glm::vec3();
}

glm::fquat Object::getRotation() {
    return glm::fquat();
}

std::vector<Object> *Object::getChildren() {
    return children;
}

Object::~Object() {
    delete children;
}

glm::mat4 *Object::getModel() {
    model = glm::diagonal4x4(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    model *= glm::toMat4(rotation);
    model *= glm::translate(position);

    return &model;
}
