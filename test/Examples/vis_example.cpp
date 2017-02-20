//
// Created by Ulrik Guenther on 20/02/17.
//

#include <iostream>
#include "../../src/vis/Camera.h"
#include "../../src/vis/RaytracedObject.h"

// uncomment to see output of ray origins/directions
//#define DEBUG_OUTPUT

int main(int argc, char** argv) {

    // perspective camera, sitting at 5 units down from the origin on the positive Z axis,
    // with no rotation applied, facing down negative Z axis
    Camera cam = Camera(glm::vec3(0.0f, 0.0f, 15.0f), glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));
    cam.setPerspectiveCamera(1.0f, (float) (50.0f / 180.0f * M_PI), 0.1f, 15.0f);

    // ray traced object, sitting on the origin, with no rotation applied
    RaytracedObject o = RaytracedObject(glm::vec3(0.0f, 0.0f, 0.0f), glm::fquat(1.0f, 0.0f, 0.0f, 0.0f));

    unsigned int imageWidth = 1024;
    unsigned int imageHeight = 1024;

    auto start = std::chrono::high_resolution_clock::now();
    glm::mat4 inverse_projection = glm::inverse(*cam.getProjection());
    glm::mat4 modelview = (*cam.getView()) * (*o.getModel());

    std::cout << "Generating " << imageWidth*imageHeight << " origin/direction rays..." << std::endl;

#ifdef DEBUG_OUTPUT
    std::cout << "x,y\t(origin) / (direction)" << std::endl;
#endif

    for(unsigned int i = 0; i < imageWidth; i++) {
        for(unsigned int j = 0; j < imageHeight; j++) {
            std::pair<glm::vec3, glm::vec3> ray = o.rayForObserver(inverse_projection, *cam.getView(), imageWidth, imageHeight, i, j);

#ifdef DEBUG_OUTPUT
            std::cout << i << "," << j << "\t" << ray.first.x << " " << ray.first.y << " " << ray.first.z << " / " << ray.second.x << " " << ray.second.y << " " << ray.second.z << std::endl;
#endif
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    std::cout << "Calculating " << imageWidth*imageHeight << " rays took " << duration.count() << "s" << std::endl;

    return 0;
}

