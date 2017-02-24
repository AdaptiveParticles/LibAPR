//
// Created by Ulrik Guenther on 20/02/17.
//

#ifndef PARTPLAY_CAMERA_H
#define PARTPLAY_CAMERA_H

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include "Object.h"

class Camera : Object {
    friend class RaytracedObject;

protected:
    glm::mat4 view;
    glm::mat4 projection;

    glm::vec3 target = glm::vec3(0.0f);
    glm::vec3 up = glm::vec3(0.0f);
    glm::vec3 forward = glm::vec3(0.0f);
    glm::vec3 right = glm::vec3(0.0f);

    float nearPlaneDistance = 1.0f;
    float farPlaneDistance = 1000.0f;
    float fov = 50.0f;

public:
    Camera(glm::vec3 position, glm::fquat rotation);

    Camera* setPerspectiveCamera(float aspectRatio, float fov, float nearPlane, float farPlane);
    Camera* setOrthographicCamera(float nearPlane, float farPlane);

    glm::mat4* getView();
    void setCoordinateSystem();
    inline glm::mat4* getProjection() { return &this->projection; };
};


#endif //PARTPLAY_CAMERA_H
