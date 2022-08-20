# MLIR-TV project

MLIR-TV is an SMT-based translation validation framework for MLIR.
This project is inspired by [Alive2](https://github.com/aliveToolkit/alive2), an SMT-based bounded translation validation framework for LLVM IR.
MLIR-TV focuses on supporting dialects that are tailored for compiling machine learning applications.

## How to build MLIR-TV

Prerequisites: [CMake](https://cmake.org/download/)(>=3.15),
[MLIR](https://github.com/llvm/llvm-project),
[Python3](https://www.python.org/downloads/),  
Solvers (at least one of them must be used):
[z3-4.8.13](https://github.com/Z3Prover/z3/releases/tag/z3-4.8.13) ,
[cvc5-0.0.3(limited support)](https://github.com/cvc5/cvc5/releases/tag/cvc5-0.0.3)

You will need to build & install MLIR.
Please follow LLVM's [Getting Started](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm), and run `cmake --build . --target install`.
If you already have your MLIR built but found that you are not sudo priviledge that is to install, you can update the `CMAKE_INSTALL_PREFIX` variable via
`cmake -DCMAKE_INSTALL_PREFIX=<your local path> ../llvm` and run the install command.

You will also need to build & install Z3.
Please [build Z3 using CMake](https://github.com/Z3Prover/z3/blob/master/README-CMake.md) and install it to somewhere designated by `CMAKE_INSTALL_PREFIX`.

```bash
mkdir build
cd build

# At least one of -DZ3_ROOT and -DCVC5_ROOT should be set. Build will fail otherwise.
# -DUSE_LIBC is OFF by default. Set it to ON iff the MLIR (and CVC5) is linked with libc++
cmake -DMLIR_ROOT=<dir/to/mlir-install> \
      [-DZ3_ROOT=<dir/to/z3-install>] \
      [-DCVC5_ROOT=<dir/to/cvc5-install>] \
      [-DUSE_LIBC=ON|OFF] \
      [-DCMAKE_BUILD_TYPE=Debug|Release] \
      ..
cmake --build . -- -j
```

If you're seeing `error: pack expansion does not contain any unexpanded parameter packs` during compilation, please try the solution suggested at https://github.com/llvm/llvm-project/issues/55010.

## How to run MLIR-TV

MLIR-TV takes two .mlir files that contain MLIR functions of identical signatures.
Run the built `mlir-tv` executable as following:
```bash
mlir-tv <.mlir before opt> <.mlir after opt>
# ex: ./build/mlir-tv \
#        tests/opts/conv2d_to_img2col/nhwc_filter.src.mlir \
#        tests/opts/conv2d_to_img2col/nhwc_filter.tgt.mlir -smt-to=5000
```

## How to test MLIR-TV
```bash
cd build
# A detailed log is written to build/Testing/Temporary/LastTest.log
# If you want detailed output on the terminal, please add -V
ctest -R Opts # Test IR transformation passes
ctest -R Long # Test passes that take a lot of time
ctest -R Litmus # Test litmus only
```

## Contributions

We appreciate any kind of contributions to this project!
