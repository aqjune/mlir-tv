# MLIR-TV project

## How to build MLIR-TV

We take the path of a user's local IREE repo to detect MLIR's src and build directory.

Prerequisites: [CMake](https://cmake.org/download/)(>=3.15),
[IREE](https://github.com/google/iree),
[Z3-solver](https://github.com/Z3Prover/z3),
[smt-switch](https://github.com/makaimann/smt-switch),
[Python3](https://www.python.org/downloads/)(>=3.9)

```bash
mkdir build
cd build
# If you want to build a release version, please add -DCMAKE_BUILD_TYPE=RELEASE
cmake -DIREE_DIR=<dir/to/iree> \
      -DIREE_BUILD_DIR=<dir/to/iree-build> \
      -DZ3_DIR=<dir/to/z3-install> \
      -DSMT_SWITCH_DIR=<dir/to/smt-switch/install> \
      ..
cmake --build .
```

## How to run MLIR-TV
Run the built `mlir-tv` executable as following:
```bash
mlir-tv <.mlir before opt> <.mlir after opt>`
# ex: ./build/mlir-tv \
#        tests/opts/conv2d_to_img2col/nhwc_filter.src.mlir \
#        tests/opts/conv2d_to_img2col/nhwc_filter.tgt.mlir -smt-to=5000
```

## How to test MLIR-TV
```bash
cd build
# A detailed log is written to build/Testing/Temporary/LastTest.log
# If you want detailed output on the terminal, please add -V
ctest -R Unit
ctest -R Opts # Test IR transformation passes
ctest -R Litmus # Test litmus only
```
