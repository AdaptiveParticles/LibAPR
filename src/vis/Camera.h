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

protected:
    glm::mat4 view;
    glm::mat4 projection;

public:
    Camera(glm::vec3 position, glm::fquat rotation);

    Camera* setPerspectiveCamera(float aspectRatio, float fov, float nearPlane, float farPlane);
    Camera* setOrthographicCamera(float nearPlane, float farPlane);

    inline glm::mat4* getView() { return &this->view; };
    inline glm::mat4* getProjection() { return &this->projection; };
};


#endif //PARTPLAY_CAMERA_H
