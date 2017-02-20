//
// Created by Ulrik Guenther on 20/02/17.
//

#ifndef PARTPLAY_RAYTRACEDOBJECT_H
#define PARTPLAY_RAYTRACEDOBJECT_H


#include "Object.h"
#include "Camera.h"

class RaytracedObject : public Object {

protected:

public:
    RaytracedObject(glm::vec3 position, glm::fquat rotation);

    std::pair<glm::vec3, glm::vec3> rayForObserver(glm::mat4 inverse_projection, glm::mat4 modelview, unsigned int imageSizeX, unsigned int imageSizeY, unsigned int x,
                                                   unsigned int y);
};

#endif //PARTPLAY_RAYTRACEDOBJECT_H
