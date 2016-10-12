# PartPlay

Library for processing on APR representation

Dependencies:

* HDF5 library installed and the library linked/included (libhdf5-dev
* CMake
* tiffio (libtiff5-dev debian/ubuntu)

Compilation (out of source):

```
   mkdir build
   cd build
   cmake -H. -Bbuild ..`
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
How to run tests?

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

