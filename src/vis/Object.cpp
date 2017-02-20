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
    return &model;
}
