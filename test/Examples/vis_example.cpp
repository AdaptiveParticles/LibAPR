//
// Created by Ulrik Guenther on 20/02/17.
//

#include <iostream>
#include "../../src/vis/Camera.h"
#include "../../src/vis/RaytracedObject.h"
#ifdef _WINDOWS
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include <chrono>

// uncomment to see output of ray origins/directions
//#define DEBUG_OUTPUT

int main(int argc, char **argv) {


    unsigned int imageWidth = 1280;
    unsigned int imageHeight = 720;

    // perspective camera, sitting at 5 units down from the origin on the positive Z axis,
    // with no rotation applied, facing down negative Z axis
    Camera cam = Camera(glm::vec3(0.0f, 0.0f, 20.0f*imageHeight), glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));
    cam.setPerspectiveCamera(1.0f, (float) (50.0f / 180.0f * M_PI), 1.0f, 15.0f);

    //Camera cam = Camera(glm::vec3(imageHeight*.5, imageWidth*.5, 2.0f*imageHeight), glm::fquat(0.0f, 1.0f, 0.0f, 0.0f));
    //cam.setPerspectiveCamera((float)imageWidth/(float)imageHeight, (float) (60.0f / 180.0f * M_PI), 0.5f, 70.0f);


    // ray traced object, sitting on the origin, with no rotation applied
    RaytracedObject o = RaytracedObject(glm::vec3(0.0f, 0.0f, 0.0f), glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));
    o.setExtent(glm::vec3(-5.0f, -5.0f, -5.0f), glm::vec3(5.0f, 5.0f, 5.0f));


    auto start = std::chrono::high_resolution_clock::now();
    glm::mat4 inverse_projection = glm::inverse(*cam.getProjection());
    glm::mat4 inverse_modelview = glm::inverse((*cam.getView()) * (*o.getModel()));


    const glm::mat4 mvp = (*cam.getProjection()) * (*cam.getView());


    std::cout << "Generating " << imageWidth * imageHeight << " origin/direction rays..." << std::endl;

#ifdef DEBUG_OUTPUT
    std::cout << "x,y\t(origin) / (direction)" << std::endl;
#endif

    for (unsigned int i = 0; i < imageWidth; i++) {
        for (unsigned int j = 0; j < imageHeight; j++) {
            std::pair<glm::vec3, glm::vec3> ray = o.rayForObserver(inverse_projection,
                                                                   inverse_modelview,
                                                                   imageWidth, imageHeight, i, j);

#ifdef DEBUG_OUTPUT
            std::cout << i << "," << j << "\t" << ray.first.x << " " << ray.first.y << " " << ray.first.z << " / " << ray.second.x << " " << ray.second.y << " " << ray.second.z << std::endl;
#endif
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    std::cout << "Calculating " << imageWidth * imageHeight << " rays took " << duration.count() << "s" << std::endl;

    std::cout << "Generating " << imageWidth * imageHeight << " picking points ..." << std::endl;

    start = std::chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < imageWidth; i++) {
        for (unsigned int j = 0; j < imageHeight; j++) {
            std::pair<bool, glm::vec3> hit = o.rayOriginForCoordinates(cam, glm::ivec2(i, j), imageWidth, imageHeight);

#ifdef DEBUG_OUTPUT
            if(hit.first) {
                std::cout << i << "," << j << "\t" << hit.second.x << " " << hit.second.y << " " << hit.second.z << std::endl;
            }
#endif
        }
    }

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;

    std::cout << "Calculating " << imageWidth * imageHeight << " origins took " << duration.count() << "s" << std::endl;

    glm::vec2 pos = o.worldToScreen(mvp, glm::vec3(0.1f, 0.2f, 20.0f), imageWidth, imageHeight);

    std::cout << pos.x << "/" << pos.y << std::endl;

    // Debug Test

    int i = 123;
    int j = 224;

    std::pair<bool, glm::vec3> hit = o.rayOriginForCoordinates(cam, glm::ivec2(i, j), imageWidth, imageHeight);

    std::pair<glm::vec3, glm::vec3> ray = o.rayForObserver(inverse_projection,
                                                           inverse_modelview,
                                                           imageWidth, imageHeight, i, j);

    float x_ = hit.second.x;
    float y_ = hit.second.y;
    float z_ = hit.second.z;

    glm::vec2 pos1 = o.worldToScreen(mvp, glm::vec3(hit.second.x, hit.second.y, hit.second.z), imageWidth, imageHeight);

    glm::vec2 pos2 = o.worldToScreen(mvp, glm::vec3(ray.second.x, ray.second.y, ray.second.z), imageWidth, imageHeight);

    glm::vec2 pos3 = o.worldToScreen(mvp, glm::vec3(ray.first.x, ray.first.y, ray.first.z), imageWidth, imageHeight);



    return 0;
}

