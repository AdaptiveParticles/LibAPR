//
// Created by Joel Jonsson on 29.06.18.
//

#ifndef LIBAPR_PYAPR_HPP
#define LIBAPR_PYAPR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ConfigAPR.h"
#include "data_structures/APR/APR.hpp"

#include "PyPixelData.hpp"

namespace py = pybind11;

// -------- Utility classes to be wrapped in python ----------------------------
template <typename T>
class PyAPR {
    template<typename> friend class PyPixelData;
    APR <T> apr;

public:

    PyAPR () {}

    /**
     * Reads in the given HDF5 APR file.
     *
     * @param aAprFileName
     */
    void read_apr(const std::string &aAprFileName) {
        apr.read_apr(aAprFileName);
    }

    // TODO: add more versions of write_apr, with compression options etc?
    /**
     * Writes the APR to a HDF5 file without(?) compression.
     *
     * @param aOutputFile
     */
    void write_apr(const std::string &aOutputFile) {
        apr.write_apr("", aOutputFile);
    }

    /**
     * Returns the piecewise constant reconstruction from the APR instance as a PyPixelData object. This can be cast
     * into a numpy array without copy using 'arr = numpy.array(obj, copy=False)'.
     *
     * @return PyPixelData holding the reconstructed image
     */
    PyPixelData<T> pc_recon() {

        PixelData<T> reconstructedImage;

        APRReconstruction().interp_img(apr, reconstructedImage, apr.particles_intensities);

        /*
        // this creates a copy...
        return py::array_t<T>({reconstructedImage.x_num, reconstructedImage.y_num, reconstructedImage.z_num},
                         {sizeof(T) * reconstructedImage.y_num * reconstructedImage.x_num, sizeof(T), sizeof(T) * reconstructedImage.y_num},
                         reconstructedImage.mesh.get());
        */

        //this does not copy, and can be cast to numpy.array on python side without copy (set copy=False)
        return PyPixelData<T>(reconstructedImage);
    }

    /**
     * Returns the smooth reconstruction from the APR instance as a PyPixelData object. This can be cast into a numpy
     * array without copy using 'arr = numpy.array(obj, copy=False)'.
     *
     * @return PyPixelData holding the reconstructed image
     */
    PyPixelData<T> smooth_recon() {

        PixelData<T> reconstructedImage;

        APRReconstruction().interp_parts_smooth(apr, reconstructedImage, apr.particles_intensities);

        return PyPixelData<T>(reconstructedImage);
    }

    /**
     * Sets the parameters for the APR conversion.
     *
     * @param par pyApr.APRParameters object
     */
    void set_parameters(const py::object &par) {

        if( py::isinstance<APRParameters>(par) ) {
            apr.parameters = par.cast<APRParameters>();
        } else {
            throw std::invalid_argument("Input has to be a pyApr.APRParameters object.");
        }
    }

    /**
     * Computes the APR from the input python array.
     *
     * @param input image as python (numpy) array
     */
    void get_apr_from_array(py::array &input) {

        auto buf = input.request();


        // Some checks, may need some polishing
        if( buf.ptr == nullptr ) {
            std::cerr << "Could not pass buffer in call to apr_from_array" << std::endl;
        }

        if ( !input.writeable() ) {
            std::cerr << "Input array must be writeable" << std::endl;
        }

        if( !py::isinstance<py::array_t<T>>(input) ) {
            throw std::invalid_argument("Conflicting types. Make sure the input array is of the same type as the AprType instance.");
        }

        auto *ptr = static_cast<T*>(buf.ptr);

        PixelData<T> input_img;

        //TODO: fix memory/ownership passing or just revert to copying?
        input_img.init_from_mesh(buf.shape[2], buf.shape[1], buf.shape[0], ptr); // may lead to memory issues

        apr.get_apr(input_img);
    }

    /**
     * Reads in the provided tiff file and computes its APR. Note: parameters for the APR conversion should be set
     * before by using set_parameters.
     *
     * @param aInputFile path to the tiff image file
     */
    void get_apr_from_file(const std::string &aInputFile) {
        const TiffUtils::TiffInfo aTiffFile(aInputFile);

        apr.parameters.input_dir = "";
        apr.parameters.input_image_name = aInputFile;
        apr.get_apr();
    }

};

// -------- Templated wrapper -------------------------------------------------
template <typename DataType>
void AddPyAPR(pybind11::module &m, const std::string &aTypeString) {
    using AprType = PyAPR<DataType>;
    std::string typeStr = "Apr" + aTypeString;
    py::class_<AprType>(m, typeStr.c_str())
            .def(py::init())
            .def("read_apr", &AprType::read_apr, "Method to read HDF5 APR files")
            .def("write_apr", &AprType::write_apr, "Writes the APR instance to a HDF5 file")
            .def("reconstruct", &AprType::pc_recon, py::return_value_policy::move, "returns the piecewise constant image reconstruction as a python array")
            .def("reconstruct_smooth", &AprType::smooth_recon, py::return_value_policy::move, "returns a smooth image reconstruction as a python array")
            .def("set_parameters", &AprType::set_parameters, "Set parameters for APR conversion")
            .def("get_apr_from_array", &AprType::get_apr_from_array, "Construct APR from input array (no copy)")
            .def("get_apr_from_file", &AprType::get_apr_from_file, "Construct APR from input .tif image");
}

#endif //LIBAPR_PYAPR_HPP
