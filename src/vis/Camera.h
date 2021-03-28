//
// Created by Ulrik Guenther on 20/02/17.
//

#ifndef PARTPLAY_CAMERA_H
#define PARTPLAY_CAMERA_H

#define GLM_ENABLE_EXPERIMENTAL

#include "Object.h"

#ifdef WIN_COMPILE
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API
#endif

class LIBRARY_API Camera : Object {
    friend class RaytracedObject;

protected:
    glm::mat4 view;
    glm::mat4 projection;

    bool targeted = false;

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

    glm::mat4* getView();
    void setTargeted(glm::vec3 t);
    void setUntargeted();
    void setCoordinateSystem();
    inline glm::mat4* getProjection() { return &this->projection; };

    Camera *setOrthographicCamera(unsigned int width, unsigned int height, float nearPlane, float farPlane);
};


#endif //PARTPLAY_CAMERA_H
