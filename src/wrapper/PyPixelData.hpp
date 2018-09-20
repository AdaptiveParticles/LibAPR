//
// Created by Joel Jonsson on 29.06.18.
//

#ifndef LIBAPR_PYPIXELDATA_HPP
#define LIBAPR_PYPIXELDATA_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "data_structures/Mesh/PixelData.hpp"

namespace py = pybind11;

// -------- Utility classes to be wrapped in python ----------------------------
/**
 * Currently only using this class to return PixelData objects to python as arrays without copy. It could be made more
 * complete with constructor methods and other functionality, but I don't think it is necessary.
 *
 * @tparam T type of mesh elements
 */
template<typename T>
class PyPixelData {

    PixelData<T> image;

public:
    PyPixelData() {}

    PyPixelData(PixelData<T> &aInput) {
        image.swap(aInput);
    }

    /**
     * @return pointer to the mesh data
     */
    T *data() {return image.mesh.get();}

    /**
     * @return width of the domain (number of columns)
     */
    int width() const {return image.x_num;}

    /**
     * @return height of the domain (number of rows)
     */
    int height() const {return image.y_num;}

    /**
     * @return depth of the domain
     */
    int depth() const {return image.z_num;}
};

template<typename DataType>
void AddPyPixelData(pybind11::module &m, const std::string &aTypeString) {
    using PixelDataType = PyPixelData<DataType>;
    std::string typeStr = "PixelData" + aTypeString;
    py::class_<PixelDataType>(m, typeStr.c_str(), py::buffer_protocol())
            .def("width", &PixelDataType::width, "Returns number of columns (x)")
            .def("height", &PixelDataType::height, "Returns number of rows (y)")
            .def("depth", &PixelDataType::depth, "Returns the depth (z)")
            .def_buffer([](PixelDataType &a) -> py::buffer_info{
                return py::buffer_info(
                        a.data(),
                        sizeof(DataType),
                        py::format_descriptor<DataType>::format(),
                        3,
                        {a.width(), a.height(), a.depth()},
                        {sizeof(DataType) * a.height(), sizeof(DataType), sizeof(DataType) * a.height() * a.width()}
                );
            });
}

#endif //LIBAPR_PYPIXELDATA_HPP
