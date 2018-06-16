//
// Created by Krzysztof Gonciarz on 5/7/18.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ConfigAPR.h"
#include "data_structures/APR/APR.hpp"

namespace py = pybind11;

// -------- Check if properly configured in CMAKE -----------------------------
#ifndef APR_PYTHON_MODULE_NAME
#error "Name of APR module (python binding) is not defined!"
#endif

// TODO: If more classes added wrappers should be moved to seperate files, only
//       module definition shold be kept here
// -------- Utility classes to be wrapped in python ----------------------------
template <typename T>
class AprToImg {
    PixelData <T> originalImage;

    APR <T> apr;

    PixelData <T> reconstructedImage;

public:
    AprToImg () {}
    void read(const std::string &aAprFileName) {
        apr.read_apr(aAprFileName);
        //ReconPatch r;
        //APRReconstruction().interp_image_patch(apr, reconstructedImage, apr.particles_intensities, r);
    }

    T* pc_recon() {
        APRReconstruction().interp_img(apr, reconstructedImage, apr.particles_intensities);

        return reconstructedImage.mesh.get();
    }

    bool readArr(py::handle src, bool convert) {

        /* Some sanity checks ... */
        if (!convert && !py::array_t<T>::check_(src)) {
            std::cout << "failed type check" << std::endl;
            return false;
        }

        auto buf = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
        if (!buf) {
            std::cout << "could not read buffer" << std::endl;
            return false;
        }

        auto dims = buf.ndim();
        if (dims != 3) {
            std::cout << "failed dimension check" << std::endl;
            return false;
        }

        /* read in python array to originalImage */
        originalImage.init(buf.shape()[0], buf.shape()[1], buf.shape()[2]);

        for(int i=0; i<originalImage.mesh.size(); ++i) {
            originalImage.mesh[i] = buf.data()[i] + 1;
        }

        return true;
    }

    T *data() {return originalImage.mesh.get();}
    int height() const {return originalImage.x_num;}
    int width() const {return originalImage.y_num;}
    int depth() const {return originalImage.z_num;}
};

// -------- Templated wrapper -------------------------------------------------
template <typename DataType>
void AddAprToImg(pybind11::module &m, const std::string &aTypeString) {
    using AprType = AprToImg<DataType>;
    std::string typeStr = "Apr" + aTypeString;
    py::class_<AprType>(m, typeStr.c_str(), py::buffer_protocol())
            .def(py::init())
            .def("read", &AprType::read, "Method to read HDF5 APR files")
            .def("reconstruct", &AprType::pc_recon, "returns an image reconstructed from the APR")
            .def("width", &AprType::width, "Returns number of columns (x)")
            .def("height", &AprType::height, "Returns number of rows (y)")
            .def("depth", &AprType::depth, "Returns depth (z)")
            .def("readArr", &AprType::readArr, "reads in a python array")
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
PYBIND11_MODULE(APR_PYTHON_MODULE_NAME, m) {
    m.doc() = "python binding for LibAPR library";
    m.attr("__version__") = pybind11::str(ConfigAPR::APR_VERSION);

    AddAprToImg<uint8_t>(m, "Byte");
    AddAprToImg<uint16_t>(m, "Short");
    AddAprToImg<float>(m, "Float");
}
