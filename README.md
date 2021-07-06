# IREE-TV project

## How to build IREE-TV
Prerequisites: [CMake](https://cmake.org/download/)(>=3.15),
[IREE](https://github.com/google/iree),
[Z3-solver](https://github.com/Z3Prover/z3),
[Python3.9](https://www.python.org/downloads/) or above

```bash
mkdir build
cd build
# If you want to build a release version, please add -DCMAKE_BUILD_TYPE=RELEASE
cmake -DIREE_DIR=<dir/to/iree> \
      -DIREE_BUILD_DIR=<dir/to/iree-build> \
      -DZ3_INCLUDE_DIR=<dir/to/z3-install/include> \
      -DZ3_LIBRARY_DIR=<dir/to/z3-install/lib> \
      ..
cmake --build .
```

## How to run IREE-TV
Run the built `iree-tv` executable as following:
```bash
iree-tv <.mlir before opt> <.mlir after opt>`
# ex: ./build/iree-tv \
#        examples/conv2d_to_img2col/conv2d_to_img2col.mlir \
#        examples/conv2d_to_img2col/conv2d_to_img2col_opt.mlir -smt-to=5000
```

## How to test IREE-TV
```bash
cd build
# If you want detailed output on the terminal, please add -V
ctest -R Unit
# A detailed log is written to build/Testing/Temporary/LastTest.log
ctest -R passes # Test IR transformation passes
```
