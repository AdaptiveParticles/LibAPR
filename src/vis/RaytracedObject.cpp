//
// Created by Ulrik Guenther on 20/02/17.
//

#include "RaytracedObject.h"
#include <glm/gtc/matrix_access.hpp>
#include <iostream>
#include <algorithm>

RaytracedObject::RaytracedObject(glm::vec3 position, glm::fquat rotation) : Object(position, rotation) { this->position = position; this->rotation = rotation; }

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

void printV(glm::vec3 v) {
    std::cout << v.x << ", " << v.y << ", " << v.z << std::endl;
}

std::pair<bool, glm::vec3> RaytracedObject::rayOriginForCoordinates(Camera &observer, glm::ivec2 coordinates, unsigned int imageSizeX,
                                                   unsigned int imageSizeY) {

    observer.getView();
    observer.setCoordinateSystem();

    glm::vec3 view = glm::normalize(observer.target - observer.position);
    glm::vec3 h = glm::normalize(glm::cross(observer.up, view));
    glm::vec3 v = glm::cross(h, view);

    float lengthV = tanf(observer.fov/2.0f) * observer.nearPlaneDistance;
    float lengthH = lengthV * (imageSizeX/imageSizeY);

    v *= lengthV;
    h *= lengthH;

    float posX = (coordinates.x - imageSizeX/2.0f)/(imageSizeX/2.0f);
    float posY = -(coordinates.y - imageSizeY/2.0f)/(imageSizeY/2.0f);

    glm::vec3 worldPos = observer.position + view * observer.nearPlaneDistance + h * posX + v * posY;
    glm::vec3 worldDir = glm::normalize(worldPos - observer.position);

//    printV(worldPos);
//    printV(worldDir);

    return intersect(worldPos, worldDir);
}

std::pair<bool, glm::vec3> RaytracedObject::intersect(glm::vec3 origin, glm::vec3 direction) {

    glm::vec4 min = *this->getModel() * glm::vec4(extent_min, 1.0f);
    glm::vec4 max = *this->getModel() * glm::vec4(extent_max, 1.0f);

    glm::vec3 invDir = glm::vec3(1.0f/direction.x, 1.0f/direction.y, 1.0f/direction.z);

    float t1 = (min.x - origin.x) * invDir.x;
    float t2 = (max.x - origin.x) * invDir.x;
    float t3 = (min.y - origin.y) * invDir.y;
    float t4 = (max.y - origin.y) * invDir.y;
    float t5 = (min.z - origin.z) * invDir.z;
    float t6 = (max.z - origin.z) * invDir.z;

    float tmin = std::max(std::max(std::min(t1, t1), std::min(t3, t4)), std::min(t5, t6));
    float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

    if(tmax < 0.0f) {
        return std::pair<bool, glm::vec3>(false, origin + tmax * direction);
    }

    if(tmin > tmax) {
        return std::pair<bool, glm::vec3>(false, origin + tmax * direction);
    }

    return std::pair<bool, glm::vec3>(true, origin + tmin * direction);
}

glm::vec2 RaytracedObject::worldToScreen(glm::mat4 mvp, glm::vec3 worldPosition, unsigned int imageSizeX, unsigned int imageSizeY) {
    glm::vec4 clip = mvp * glm::vec4(worldPosition, 1.0);
    glm::vec2 ndc = glm::vec2(clip.x/clip.w, clip.y/clip.w);

//    printV(glm::vec3(ndc, 1.0f));

    glm::vec2 result = (ndc - glm::vec2(1.0f))/2.0f;
    result.x *= imageSizeX;
    result.y *= imageSizeY;

    return result;
}
