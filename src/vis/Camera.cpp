//
// Created by Ulrik Guenther on 20/02/17.
//

#include "Camera.h"

Camera::Camera(glm::vec3 position, glm::fquat rotation) : Object(position, rotation) {}

Camera* Camera::setPerspectiveCamera(float aspectRatio, float fov, float nearPlane, float farPlane) {
    this->projection = glm::perspective(fov, aspectRatio, nearPlane, farPlane);
    return this;
}

Camera* Camera::setOrthographicCamera(float nearPlane, float farPlane) {
    this->projection = glm::orthoRH(-1.0f, 1.0f, -1.0f, 1.0f, nearPlane, farPlane);
    return this;
}

glm::mat4 *Camera::getView() {
    view = glm::diagonal4x4(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    view *= glm::toMat4(rotation);
    view *= glm::translate(this->position);

    return &view;
}
