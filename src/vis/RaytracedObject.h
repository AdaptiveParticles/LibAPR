//
// Created by Ulrik Guenther on 20/02/17.
//

#ifndef PARTPLAY_RAYTRACEDOBJECT_H
#define PARTPLAY_RAYTRACEDOBJECT_H


#include "Object.h"
#include "Camera.h"
#include <glm/gtc/matrix_access.hpp>

#ifdef WIN_COMPILE
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API
#endif

class LIBRARY_API RaytracedObject : public Object {

protected:
    glm::vec3 extent_min;
    glm::vec3 extent_max;

    std::pair<bool, glm::vec3> intersect(glm::vec3 origin, glm::vec3 direction);

public:
    RaytracedObject(glm::vec3 position, glm::fquat rotation);

    std::pair<glm::vec3, glm::vec3> rayForObserver(glm::mat4 inverse_projection, glm::mat4 modelview, unsigned int imageSizeX, unsigned int imageSizeY, unsigned int x,
                                                   unsigned int y);

    std::pair<bool, glm::vec3>
    rayOriginForCoordinates(Camera& observer, glm::ivec2 coordinates, unsigned int imageSizeX, unsigned int imageSizeY);

    void setExtent(glm::vec3 min, glm::vec3 max) { this->extent_min = min; this->extent_max = max; };

    glm::vec2
    worldToScreen(glm::mat4 mvp, glm::vec3 worldPosition, unsigned int imageSizeX, unsigned int imageSizeY);
};

#endif //PARTPLAY_RAYTRACEDOBJECT_H
