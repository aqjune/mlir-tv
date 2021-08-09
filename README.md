# MLIR-TV project

MLIR-TV is an SMT-based translation validation framework for MLIR.
Currently this is in an experimental stage.

## How to build MLIR-TV

Prerequisites: [CMake](https://cmake.org/download/)(>=3.15),
[MLIR](https://github.com/llvm/llvm-project),
[Python3](https://www.python.org/downloads/)(>=3.9)  
Solvers (at least one of them must be used): [Z3-solver](https://github.com/Z3Prover/z3)

- Installation of MLIR: please follow [this instruction](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm) & run `cmake --build . --target install`

```bash
mkdir build
cd build
# -DUSE_LIBC is OFF by default. Set it to ON iff the MLIR is built using libc++
cmake -DMLIR_DIR=<dir/to/mlir-install> \
      -DZ3_DIR=<dir/to/z3-install> \
      [-DUSE_LIBC=ON|OFF] \
      [-DCMAKE_BUILD_TYPE=DEBUG|RELEASE] \
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
