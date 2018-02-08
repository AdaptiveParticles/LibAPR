# LibAPR - The Adaptive Particle Representation Library

Library for producing and processing on the Adaptive Particle Representation (APR).

<img src="./docs/apr_lowfps_lossy.gif?raw=true">

Labeled Zebrafish nuclei: Gopi Shah, Huisken Lab ([MPI-CBG](https://www.mpi-cbg.de), Dresden and [Morgridge Institute for Research](https://morgridge.org/research/medical-engineering/huisken-lab/), Madison); see also [Schmid et al., _Nature Communications_ 2017](https://www.nature.com/articles/ncomms3207)

## Dependencies

* HDF5 1.8.20 or higher
* OpenMP > 3.0 (optional, but suggested)
* CMake 3.6 or higher
* LibTIFF 5.0 or higher
* SWIG 3.0.12 (optional, for generating Java wrappers)

## Building

The repository requires sub-modules, so the repository needs to be cloned recursively:

```
git clone --recursive https://github.com/cheesema/LibAPR
```

If you need to update your clone at any point later, run

```
git pull
git submodule update
```

### Building on Linux

On Ubuntu, install the `cmake`, `build-essential`, `libhdf5-dev` and `libtiff5-dev` packages (on other distributions, refer to the documentation there, the package names will be similar). OpenMP support is provided by the GCC compiler installed as part of the `build-essential` package.

In the directory of the cloned repository, run

```
mkdir build
cd build
cmake ..
make
```

This will create the `libapr.so` library in the `build` directory, as well as all of the examples.

### Building on OSX

On OSX, install the `cmake`, `hdf5` and `libtiff`  [homebrew](https://brew.sh) packages and have the [Xcode command line tools](http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/) installed.

If you want to compile with OpenMP support, also install the `llvm` package, as the clang version shipped by Apple currently does not support OpenMP.

In the directory of the cloned repository, run

```
mkdir build
cd build
cmake ..
make
```

This will create the `libapr.so` library in the `build` directory, as well as all of the examples.

In case you want to use the homebrew-installed clang, modify the call to `cmake` above to

```
CC="/usr/local/opt/llvm/bin/clang" CXX="/usr/local/opt/llvm/bin/clang++" LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib" CPPFLAGS="-I/usr/local/opt/llvm/include" cmake ..
```


### Building on Windows

__Compilation only works with mingw64/clang or the Intel C++ Compiler, with Intel C++ being the recommended way__

You need to have Visual Studio 2017 installed, with [the community edition](https://www.visualstudio.com/downloads/) being sufficient. LibAPR does not compile correctly with the default Visual Studio compiler, so you also need to have the [Intel C++ Compiler, 18.0 or higher](https://software.intel.com/en-us/c-compilers) installed. [`cmake`](https://cmake.org/download/) is also a requirement.

Furthermore, you need to have HDF5 installed (binary distribution download at [The HDF Group](http://hdfgroup.org) and LibTIFF (source download from [SimpleSystems](http://www.simplesystems.org/libtiff/). LibTIFF needs to be compiled via `cmake`. LibTIFF's install target will then install the library into `C:\Program Files\tiff`.

In the directory of the cloned repository, run:

```
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -DTIFF_INCLUDE_DIR="C:/Program Files/tiff/include" -DTIFF_LIBRARY="C:/Program Files/tiff/lib/tiff.lib " -DHDF5_ROOT="C:/Program Files/HDF_Group/HDF5/1.8.17"  -T "Intel C++ Compiler 18.0" ..
cmake --build . --config Debug
```

This will set the appropriate hints for Visual Studio to find both LibTIFF and HDF5. This will create the `apr.dll` library in the `build/Debug` directory, as well as all of the examples. If you need a `Release` build, run `cmake --build . --config Release` from the `build` directory.

## Examples and Documentation
There are nine basic examples, that show how to generate and compute with the APR:

| Example | How to ... |
|:--|:--|
| [Example_get_apr](./test/Examples/Example_get_apr.cpp) | create an APR file from a TIFF. |
| [Example_apr_iterate](./test/Examples/Example_apr_iterate.cpp) | iterate through a given APR file. |
| [Example_neighbour_access](./test/Examples/Example_neighbour_access.cpp) | access particle and face neighbours. |
| [Example_compress_apr](./test/Examples/Example_compress_apr.cpp) |  additionally compress the intensities stored in an APR file. |
| [Example_compute_gradient](./test/Examples/Example_compute_gradient.cpp) | compute a gradient based on the stored particles in an APR file. |
| [Example_produce_paraview_file](./test/Examples/Example_produce_paraview_file.cpp) | produce a file for visualisation in ParaView. |
| [Example_random_access](./test/Examples/Example_random_access.cpp) | perform random access operations on particles. |
| [Example_ray_cast](./test/Examples/Example_ray_cast.cpp) | perform a maximum intensity projection ray cast directly on the APR data structures read from an APR file. |
| [Example_reconstruct_image](./test/Examples/Example_reconstruct_image.cpp) | reconstruct an pixel image from an APR file. |

For tutorial on how to use the examples, and explanation of data-structures see [the library guide](./docs/lib_guide.pdf).

## Coming soon

* more examples for APR-based filtering and segmentation
* more convenient use from CMake projects
* deployment of the Java wrappers to Maven Central so they can be used in your project directly
* support for loading the APR in [Fiji](https://fiji.sc), including [scenery-based](https://github.com/scenerygraphics/scenery) 3D rendering

## Citing this work

If you use this library in an academic context, please cite the following paper:

* Cheeseman, Günther, Susik, Gonciarz, Sbalzarini: _Forget Pixels: Adaptive Particle Representation of Fluorescence Microscopy Images_ (in preparation)