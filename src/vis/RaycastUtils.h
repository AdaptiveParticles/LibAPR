//
// Created by bevan on 28/03/2021.
//

#ifndef APR_RAYCASTUTILS_H
#define APR_RAYCASTUTILS_H

#ifdef WIN_COMPILE
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API
#endif

struct LIBRARY_API GlmObjectsContainer;

LIBRARY_API void initObjects(GlmObjectsContainer* &glmObjects, int imageWidth, int imageHeight, float radius, float theta, float x0, float y0,
                 float z0, float x0f, float y0f, float z0f, float phi, float phi_s);

LIBRARY_API void killObjects(GlmObjectsContainer* &glmObjects);
LIBRARY_API void getPos(GlmObjectsContainer* &glmObjects, int &dim1, int &dim2, float x_actual, float y_actual, float z_actual, int x_num, int y_num);

#endif //APR_RAYCASTUTILS_H
