//
// Created by Krzysztof Gonciarz on 5/7/18.
//

#include <pybind11/pybind11.h>

#include "ConfigAPR.h"
#include "data_structures/APR/APR.hpp"

namespace py = pybind11;

// -------- Check if properly configured in CMAKE -----------------------------
#ifndef APR_MODULE_NAME
#error "Name of APR module (python binding) is not defined!"
#endif

// TODO: If more classes added wrappers should be moved to seperate files, only
//       module definition shold be kept here
// -------- Utility classes to be wrapped in python ----------------------------
template <typename T>
class AprToImg {
    MeshData <T> reconstructedImage;

public:
    AprToImg () {}
    void read(const std::string &aAprFileName) {
        APR <T> apr;
        apr.read_apr(aAprFileName);
        ReconPatch r;
        APRReconstruction().interp_image_patch(apr, reconstructedImage, apr.particles_intensities, r);
    }

    T *data() {return reconstructedImage.mesh.get();}
    int height() const {return reconstructedImage.x_num;}
    int width() const {return reconstructedImage.y_num;}
    int depth() const {return reconstructedImage.z_num;}
};

// -------- Templated wrapper -------------------------------------------------
template <typename DataType>
void AddAprToImg(pybind11::module &m, const std::string &aTypeString) {
    using AprType = AprToImg<DataType>;
    std::string typeStr = "Apr" + aTypeString;
    py::class_<AprType>(m, typeStr.c_str(), py::buffer_protocol())
            .def(py::init())
            .def("read", &AprType::read, "Method to read HDF5 APR files")
            .def("width", &AprType::width, "Returns number of columns (x)")
            .def("height", &AprType::height, "Returns number of rows (y)")
            .def("depth", &AprType::depth, "Returns depth (z)")
            .def_buffer([](AprType &a) -> py::buffer_info{
                return py::buffer_info(
                        a.data(),
                        sizeof(DataType),
                        py::format_descriptor<DataType>::format(),
                        3,
                        {a.depth(), a.height(), a.width()},
                        {sizeof(DataType) * a.width() * a.height(), sizeof(DataType) * a.width(), sizeof(DataType)}
                );
            });
}


// -------- Definition of python module ---------------------------------------
PYBIND11_MODULE(APR_MODULE_NAME, m) {
    m.doc() = "python binding for LibAPR library";
    m.attr("__version__") = pybind11::str(ConfigAPR::APR_VERSION);

    AddAprToImg<uint8_t>(m, "Byte");
    AddAprToImg<uint16_t>(m, "Short");
    AddAprToImg<float>(m, "Float");
}
