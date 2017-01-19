# PartPlay

Library for processing on APR representation

## Dependencies:

* HDF5 library installed and the library linked/included (libhdf5-dev)
* CMake
* tiffio (libtiff5-dev debian/ubuntu)
* Blosc (http://blosc.org/) (https://github.com/Blosc/c-blosc) and (https://github.com/Blosc/hdf5-blosc)

## Building

### OSX preliminaries

OSX currently ships with an older version of clang that does not support OpenMP. A more current version (3.8+) has to be installed, e.g. via homebrew:

```
brew install llvm
```

All further cmake commands then have to be prepended by

```
CC="/usr/local/opt/llvm/bin/clang" CXX="/usr/local/opt/llvm/bin/clang++"
LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
CPPFLAGS="-I/usr/local/opt/llvm/include" CXXFLAGS="-std=c++14"
```

### Compilation

Compilation (out of source):

```
   mkdir build
   cd build
   cmake -H. -Bbuild ..
```

Developer dependencies (optional):

* Google test library installed. For debian/ubuntu users:

```
    sudo apt-get install libgtest-dev
    cd /usr/src/gtest
    sudo cmake .
    sudo make
    sudo mv libg* /usr/lib/
```

For OSX users, clone the repository at https://github.com/google/googletest, then within the repo :
```
mkdir build
cd build
cmake ..
make
make install
```

## How to run tests?

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

