# The Adaptive Particle Representation Library

Library for processing the Adaptive Particle Representation (APR).

## Dependencies:

* HDF5 library installed and the library linked/included (libhdf5-dev)
* OpenMP > 3.0
* CMake > 3.6
* LibTIFF > 5.0
* [Blosc](https://github.com/Blosc/c-blosc), now included with the repository

## Building

### OSX preliminaries

OSX currently ships with an older version of clang that does not support OpenMP. A more current version (3.8+) has to be installed, e.g. via homebrew:

```
brew install llvm
```

### Windows preliminaries

__Compilation only works with mingw64/clang or Visual Studio/Intel C++ Compiler, due to Visual Studio's lack of support for current OpenMP versions, with Intel C++ being the recommended way__

For Windows, APR needs to have HDF5 installed (get it from [The HDF Group](http://hdfgroup.org) and LibTIFF (get it from [SimpleSystems](http://www.simplesystems.org/libtiff/). HDF5 can be installed just from the binary distribution, LibTIFF needs to be compiled via CMake. LibTIFF's install target will then install the library into `C:\Program Files\tiff`.

### Compilation

Compilation (out of source), on OSX/Linux:

```
mkdir build
cd build
cmake ..
```

Be aware that Apple's clang does not support OpenMP out of the box, so you might need to install another clang version via homebrew. For CMake to then pick this one up, run

```
CC="/usr/local/opt/llvm/bin/clang" CXX="/usr/local/opt/llvm/bin/clang++" LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib" CPPFLAGS="-I/usr/local/opt/llvm/include" cmake ..
```

For Windows, some additional flags for CMake are needed:

```
cmake -G "Visual Studio 14 2015 Win64" -DTIFF_INCLUDE_DIR="C:/Program Files/tiff/include" -DTIFF_LIBRARY="C:/Program Files/tiff/lib/tiff.lib " -DHDF5_ROOT="C:/Program Files/HDF_Group/HDF5/1.8.17"  -T "Intel C++ Compiler 17.0" ..
```

This will set the appropriate hints for Visual Studio to find both LibTIFF and HDF5.

### Development
An additional requirements for development and testing is the Google test library. To install it, 

* for debian/ubuntu users:

```
    sudo apt-get install libgtest-dev
    cd /usr/src/gtest
    sudo cmake .
    sudo make
    sudo mv libg* /usr/lib/
```

* for OSX users, clone the repository at https://github.com/google googletest, then within the repo:
  
```
mkdir build
cd build
cmake ..
make
make install
```

Tests are stored in a submodule. Run these commands in order to run tests:

```
    git submodule init
    git submodule update
    cd APR_tests
    git lfs pull
    cd ..
    ./install_tests.sh
```

If the script does not work, open it and check what is wrong. There are only few lines there!

Remember to pass TESTS flag to CMAKE!

```
   cmake -H. -DTESTS=1 -Bbuild ..
```

## Benchmarks

Requires SynImageGen Library

cmake -H. -DTESTS=1 -Bbuild.. -DBENCHMARKS=1 -DSynImage_PATH="PATH_TO_SYNIMAGEGEN"
