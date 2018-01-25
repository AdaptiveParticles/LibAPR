# Running with address sanitation

To use clang's sanitizer, build with 
```
cmake -DCMAKE_C_COMPILER="clang" -DCMAKE_CXX_COMPILER="clang++" -DDISABLE_OPENMP=1 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-fsanitize=address" -DCMAKE_CXX_FLAGS="-fsanitize=address" -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address -shared-libasan" -DCMAKE_MODULE_LINKER_FLAGS="-fsanitize=address -shared-libasan" ..
```

Be careful to use the right compiler, as GCC's and clangs ASan libraries are mutually incompatible. If `-shared-libasan` is not specified with clang, it'll default to the static library version, which cannot be used for building a DSO if the launching program is not also compiled with ASan.

This will not build in the required ASan lib, but requires to use `LD_PRELOAD` to use it then, as in:

```
LD_PRELOAD=/usr/lib/clang/5.0.1/lib/linux/libclang_rt.asan-x86_64.so ./Example_apr_iterate
```

Or to run one of the Java tests in libapr:
```
LD_PRELOAD=/usr/lib/clang/5.0.1/lib/linux/libclang_rt.asan-x86_64.so java -cp ".:target/dependency/junit-4.12.jar:target/apr-0.3.0-SNAPSHOT.jar:target/apr-0.3.0-SNAPSHOT-natives-linux.jar:target/apr-0.3.0-SNAPSHOT-tests.jar:target/dependency/hamcrest-core-1.3.jar:" -Dapr.testfile=/home/ulrik/Code/AdaptiveParticleRepresentation/test/files/Apr/sphere_120/sphere_apr.h5  org.junit.runner.JUnitCore de.mpicbg.mosaic.apr.tests.TestAPRIterate
```

