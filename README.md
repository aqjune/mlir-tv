# MLIR-TV project

MLIR-TV is an SMT-based translation validation framework for MLIR.
This project is inspired by [Alive2](https://github.com/aliveToolkit/alive2), an SMT-based bounded translation validation framework for LLVM IR.
MLIR-TV focuses on supporting dialects that are tailored for compiling machine learning applications.

## How to build MLIR-TV

Prerequisites: [CMake](https://cmake.org/download/)(>=3.13),
[MLIR](https://github.com/llvm/llvm-project),
[Python3](https://www.python.org/downloads/),  
Solvers (at least one of them must be used):
[z3-4.8.13](https://github.com/Z3Prover/z3/releases/tag/z3-4.8.13) ,
[cvc5-0.0.3(limited support)](https://github.com/cvc5/cvc5/releases/tag/cvc5-0.0.3)

Optional prerequisites: [Ninja](https://ninja-build.org/)  

You will need to build & install MLIR.
Please follow LLVM's [Getting Started](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm), and run `cmake --build . --target install`.
If you already have your MLIR built but found that you are not sudo priviledge that is to install, you can update the `CMAKE_INSTALL_PREFIX` variable via
`cmake -DCMAKE_INSTALL_PREFIX=<your local path> ../llvm` and run the install command.

You will also need to build & install Z3.
Please [build Z3 using CMake](https://github.com/Z3Prover/z3/blob/master/README-CMake.md) and install it to somewhere designated by `CMAKE_INSTALL_PREFIX`.

```bash
cmake -Bbuild \
      # We recommend you use Ninja if you have it on your system
      [-GNinja] \
      # At least one of USE_Z3 and USE_cvc5 should be set to ON. Build will fail otherwise.
      [-DUSE_Z3=ON|OFF] \
      [-DUSE_cvc5=ON|OFF] \
      # Use <dep>_ROOT variables when CMake fails to locate dependencies on its own
      [-DMLIR_ROOT=/mlir/installation/path] \
      [-DZ3_ROOT=/z3/installation/path] \
      [-Dcvc5_ROOT=/cvc5/installation/path] \
      # Set -USE_LIBC to ON iff the MLIR (and cvc5) is linked with libc++
      [-DUSE_LIBC=ON|OFF] \
      [-DCMAKE_BUILD_TYPE=Debug|Release] \
      ..
# You may omit -j if you're using Ninja
cmake --build build --target mlir-tv -j
```

## How to run MLIR-TV

MLIR-TV takes two .mlir files that contain MLIR functions of identical signatures.
Run the built `mlir-tv` executable as following:
```bash
mlir-tv <.mlir before opt> <.mlir after opt>
# ex: ./build/mlir-tv \
#        tests/opts/conv2d-to-img2col/nhwc_filter.src.mlir \
#        tests/opts/conv2d-to-img2col/nhwc_filter.tgt.mlir -smt-to=5000
```

## How to test MLIR-TV
```bash
cd build
# A detailed log will be written to build/Testing/Temporary/LastTest.log
# If you want detailed output on the terminal, please add -V
ctest -R Opts # Test IR transformation passes
ctest -R Long # Test passes that take a lot of time
ctest -R Litmus # Test litmus only
```

## Contributions

We appreciate any kind of contributions to this project!
