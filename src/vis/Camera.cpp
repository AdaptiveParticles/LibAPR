//
// Created by Ulrik Guenther on 20/02/17.
//

#include "Camera.h"

Camera::Camera(glm::vec3 position, glm::fquat rotation) : Object(position, rotation) { this->position = position; }

Camera* Camera::setPerspectiveCamera(float aspectRatio, float fov, float nearPlane, float farPlane) {
    this->fov = fov;
    this->nearPlaneDistance = nearPlane;
    this->farPlaneDistance = farPlane;

    this->projection = glm::perspective(fov, aspectRatio, nearPlane, farPlane);
    return this;
}

Camera* Camera::setOrthographicCamera(float nearPlane, float farPlane) {
    this->nearPlaneDistance = nearPlane;
    this->farPlaneDistance = farPlane;

    this->projection = glm::orthoRH(-1.0f, 1.0f, -1.0f, 1.0f, nearPlane, farPlane);
    return this;
}

glm::mat4 *Camera::getView() {
    view = glm::diagonal4x4(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    view *= glm::toMat4(rotation);
    view *= glm::translate(this->position);

    setCoordinateSystem();

    return &view;
}

void Camera::setCoordinateSystem() {
    this->forward = glm::normalize(glm::vec3(view[0][2], view[1][2], view[2][2]));
    this->right =   glm::normalize(glm::vec3(view[0][0], view[1][0], view[2][0]));
    this->up =      glm::normalize(glm::vec3(view[0][1], view[1][1], view[2][1]));
}
