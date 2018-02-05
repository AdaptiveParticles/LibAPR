# The Adaptive Particle Representation Library

Library for processing the Adaptive Particle Representation (APR).

## Dependencies:

* HDF5 library installed and the library linked/included (libhdf5-dev)
* OpenMP > 3.0
* CMake > 3.6
* LibTIFF > 5.0
* [Blosc](https://github.com/Blosc/c-blosc), now included with the repository

## Building

The repository requires sub-modules. These can be included by either cloneing the repository using:
```
git clone --recursive
```
or after cloning the repository typing:
```
git submodule init
git submodule update
```

### OSX compilation

OSX currently ships with an older version of clang that does not support OpenMP. A more current version (3.8+) has to be installed, e.g. via homebrew:

```
brew install llvm
```

Be aware that Apple's clang does not support OpenMP out of the box, so you might need to install another clang version via homebrew. For CMake to then pick this one up, run

```
mkdir build
cd build
CC="/usr/local/opt/llvm/bin/clang" CXX="/usr/local/opt/llvm/bin/clang++" LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib" CPPFLAGS="-I/usr/local/opt/llvm/include" cmake ..
```


### Windows compilation

__Compilation only works with mingw64/clang or Visual Studio/Intel C++ Compiler, due to Visual Studio's lack of support for current OpenMP versions, with Intel C++ being the recommended way__

For Windows, APR needs to have HDF5 installed (get it from [The HDF Group](http://hdfgroup.org) and LibTIFF (get it from [SimpleSystems](http://www.simplesystems.org/libtiff/). HDF5 can be installed just from the binary distribution, LibTIFF needs to be compiled via CMake. LibTIFF's install target will then install the library into `C:\Program Files\tiff`.

For Windows, some additional flags for CMake are needed:

```
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DTIFF_INCLUDE_DIR="C:/Program Files/tiff/include" -DTIFF_LIBRARY="C:/Program Files/tiff/lib/tiff.lib " -DHDF5_ROOT="C:/Program Files/HDF_Group/HDF5/1.8.17"  -T "Intel C++ Compiler 17.0" ..
```

This will set the appropriate hints for Visual Studio to find both LibTIFF and HDF5.

### Linux Compilation

Compilation (out of source), on OSX/Linux:

```
mkdir build
cd build
cmake ..
```
### Development

To include tests pass TESTS flag to CMAKE!

```
   cmake -H. -DTESTS=1 -Bbuild ..
```
## Examples and Documentation
There are nine basic examples, that show how to generate and compute with the APR. 

The examples can be found in test/Examples/. First see Example_get_apr for transforming a tiff image into an APR to be used by the other examples.

For tutorial on how to use the examples, and explanation of data-structures see documentation/lib_guide.

Segmentation and pixel filtering examples coming soon..
