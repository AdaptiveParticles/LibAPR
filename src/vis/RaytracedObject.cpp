//
// Created by Ulrik Guenther on 20/02/17.
//

#include "RaytracedObject.h"
#include <glm/gtc/matrix_access.hpp>

RaytracedObject::RaytracedObject(glm::vec3 position, glm::fquat rotation) : Object(position, rotation) {}

std::pair<glm::vec3, glm::vec3>
RaytracedObject::rayForObserver(glm::mat4 inverse_projection, glm::mat4 inverse_modelview, unsigned int imageSizeX, unsigned int imageSizeY, unsigned int x,
                                unsigned int y) {
    float u = (x / (float)imageSizeX)*2.0f - 1.0f;
    float v = (y / (float)imageSizeY)*2.0f - 1.0f;

    glm::vec4 front = glm::vec4(u, v, -1.0f, 1.0f);
    glm::vec4 back = glm::vec4(u, v, 1.0f, 1.0f);

    glm::vec4 origin0 = glm::vec4(
        glm::dot(front, glm::vec4(glm::column(inverse_projection, 0))),
        glm::dot(front, glm::vec4(glm::column(inverse_projection, 1))),
        glm::dot(front, glm::vec4(glm::column(inverse_projection, 2))),
        glm::dot(front, glm::vec4(glm::column(inverse_projection, 3)))
    );
    origin0 *= 1.0f/origin0.w;

    glm::vec4 origin = glm::vec4(
            glm::dot(origin0, glm::vec4(glm::column(inverse_modelview, 0))),
            glm::dot(origin0, glm::vec4(glm::column(inverse_modelview, 1))),
            glm::dot(origin0, glm::vec4(glm::column(inverse_modelview, 2))),
            glm::dot(origin0, glm::vec4(glm::column(inverse_modelview, 3)))
    );
    origin *= 1.0f/origin.w;

    glm::vec4 direction0 = glm::vec4(
            glm::dot(back, glm::vec4(glm::column(inverse_projection, 0))),
            glm::dot(back, glm::vec4(glm::column(inverse_projection, 1))),
            glm::dot(back, glm::vec4(glm::column(inverse_projection, 2))),
            glm::dot(back, glm::vec4(glm::column(inverse_projection, 3)))
    );
    direction0 *= 1.0f/direction0.w;

    direction0 -= origin0;
    direction0 = glm::normalize(direction0);

    glm::vec4 direction = glm::vec4(
            glm::dot(direction0, glm::vec4(glm::column(inverse_modelview, 0))),
            glm::dot(direction0, glm::vec4(glm::column(inverse_modelview, 1))),
            glm::dot(direction0, glm::vec4(glm::column(inverse_modelview, 2))),
            0.0f
    );

    return std::pair<glm::vec3, glm::vec3>(origin, direction);
}
